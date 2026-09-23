"""Deterministic visible text overlays for the injection experiments."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from textwrap import wrap

from PIL import Image, ImageColor, ImageDraw, ImageFont


@dataclass(frozen=True)
class InjectionVisualConfig:
    """Fixed first-pass visibility parameters."""

    font_size: int = 56
    font_color: str = "#D00000"
    background_color: str | None = None
    border_color: str | None = None
    border_width: int = 2
    opacity: int = 255
    padding: int = 16
    margin: int = 24
    max_width_ratio: float = 0.90
    position: str = "top"
    font_path: str = "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf"


DEFAULT_INJECTION_CONFIG = InjectionVisualConfig()


def _load_font(config: InjectionVisualConfig) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    try:
        return ImageFont.truetype(config.font_path, config.font_size)
    except OSError:
        return ImageFont.load_default()


def _wrap_text(
    draw: ImageDraw.ImageDraw,
    text: str,
    font: ImageFont.ImageFont,
    max_text_width: int,
) -> list[str]:
    lines: list[str] = []
    for paragraph in str(text).splitlines() or [""]:
        words = paragraph.split()
        if not words:
            lines.append("")
            continue
        current = ""
        for word in words:
            candidate = word if not current else f"{current} {word}"
            left, _, right, _ = draw.textbbox((0, 0), candidate, font=font)
            if current and right - left > max_text_width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def inject_text_overlay(
    image: Image.Image,
    injection_text: str,
    config: InjectionVisualConfig = DEFAULT_INJECTION_CONFIG,
) -> Image.Image:
    """Return a same-size RGB screenshot with a visible top text overlay."""

    if not isinstance(image, Image.Image):
        raise TypeError("image must be a PIL.Image.Image")
    if not str(injection_text).strip():
        raise ValueError("injection_text must not be empty")
    if config.position != "top":
        raise ValueError("The first fixed configuration supports position='top' only")

    base = image.convert("RGB").copy()
    overlay = Image.new("RGBA", base.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    font = _load_font(config)
    max_box_width = int(base.width * config.max_width_ratio)
    max_text_width = max_box_width - 2 * config.padding
    lines = _wrap_text(draw, injection_text.strip(), font, max_text_width)
    line_heights = [
        draw.textbbox((0, 0), line or " ", font=font)[3]
        - draw.textbbox((0, 0), line or " ", font=font)[1]
        for line in lines
    ]
    line_gap = max(8, config.font_size // 4)
    text_height = sum(line_heights) + line_gap * max(0, len(lines) - 1)
    box_width = min(
        max_box_width,
        max(
            2 * config.padding + 1,
            max(
                draw.textbbox((0, 0), line or " ", font=font)[2]
                - draw.textbbox((0, 0), line or " ", font=font)[0]
                for line in lines
            )
            + 2 * config.padding,
        ),
    )
    box_height = text_height + 2 * config.padding
    x0 = max(config.margin, (base.width - box_width) // 2)
    y0 = config.margin
    x1 = min(base.width - config.margin, x0 + box_width)
    y1 = y0 + box_height
    y = y0 + config.padding
    for line, line_height in zip(lines, line_heights):
        draw.text(
            (x0 + config.padding, y),
            line,
            font=font,
            fill=ImageColor.getrgb(config.font_color) + (config.opacity,),
        )
        y += line_height + line_gap

    return Image.alpha_composite(base.convert("RGBA"), overlay).convert("RGB")
