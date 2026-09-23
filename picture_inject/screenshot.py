#!/usr/bin/env python3
"""Create vertically cropped clean Mind2Web screenshots.

The original clean screenshots remain unchanged. This script writes a new
cropped copy for each of the 370 clean samples, keeping the full width and
cropping only along the vertical axis.
"""

from __future__ import annotations

import argparse
import json
import math
from io import BytesIO
from pathlib import Path
from statistics import mean
from typing import Any

import pyarrow.parquet as pq
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_DATASET = Path(
    "/root/data/datasets/mind2web/test_domain-00000-of-00011-26c55c12cbbcdc8e.parquet"
)
DEFAULT_SOURCE_MANIFEST = PROJECT_ROOT / "data" / "clean_screenshots" / "manifest.jsonl"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "data" / "clean"
VERTICAL_CROP_CONFIG = {
    "bbox_top_padding_min": 220,
    "bbox_top_padding_max": 760,
    "bbox_top_padding_span_multiplier": 3.0,
    "bbox_bottom_padding_min": 420,
    "bbox_bottom_padding_max": 1280,
    "bbox_bottom_padding_span_multiplier": 5.0,
    "bbox_min_height": 960,
    "fallback_min_height": 1080,
    "fallback_max_height": 2200,
    "fallback_height_ratio": 0.34,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Crop the exported clean screenshots vertically.")
    parser.add_argument("--dataset", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--source-manifest", type=Path, default=DEFAULT_SOURCE_MANIFEST)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, records: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def clamp(value: int, lower: int, upper: int) -> int:
    return max(lower, min(upper, value))


def decode_screenshot(screenshot: Any, dataset_path: Path) -> Image.Image:
    if not isinstance(screenshot, dict):
        raise ValueError(f"Unexpected screenshot structure: {type(screenshot)!r}")

    image_bytes = screenshot.get("bytes")
    image_path = screenshot.get("path")
    if image_bytes:
        if isinstance(image_bytes, memoryview):
            image_bytes = image_bytes.tobytes()
        with Image.open(BytesIO(bytes(image_bytes))) as image:
            return image.convert("RGB")

    if image_path:
        candidate = Path(image_path)
        if not candidate.is_absolute():
            candidate = dataset_path.parent / candidate
        if candidate.exists():
            with Image.open(candidate) as image:
                return image.convert("RGB")

    raise ValueError("Screenshot has neither usable embedded bytes nor a readable path")


def parse_boxes(pos_candidates: Any) -> list[tuple[float, float, float, float]]:
    boxes: list[tuple[float, float, float, float]] = []
    for candidate in pos_candidates or []:
        try:
            payload = json.loads(candidate) if isinstance(candidate, str) else candidate
            attributes = payload.get("attributes", {})
            if isinstance(attributes, str):
                attributes = json.loads(attributes)
            rect = str(attributes.get("bounding_box_rect", "")).strip()
            if not rect:
                continue
            coords = [float(piece) for piece in rect.split(",")]
            if len(coords) >= 4:
                x, y, w, h = coords[:4]
                boxes.append((x, y, w, h))
        except Exception:
            continue
    return boxes


def crop_from_boxes(image: Image.Image, boxes: list[tuple[float, float, float, float]]) -> tuple[Image.Image, list[int]]:
    width, height = image.size
    top = min(y for _, y, _, _ in boxes)
    bottom = max(y + h for _, y, _, h in boxes)
    span = max(1.0, bottom - top)

    top_pad = min(
        VERTICAL_CROP_CONFIG["bbox_top_padding_max"],
        max(
            VERTICAL_CROP_CONFIG["bbox_top_padding_min"],
            int(round(span * VERTICAL_CROP_CONFIG["bbox_top_padding_span_multiplier"])),
        ),
    )
    bottom_pad = min(
        VERTICAL_CROP_CONFIG["bbox_bottom_padding_max"],
        max(
            VERTICAL_CROP_CONFIG["bbox_bottom_padding_min"],
            int(round(span * VERTICAL_CROP_CONFIG["bbox_bottom_padding_span_multiplier"])),
        ),
    )
    crop_top = clamp(int(math.floor(top - top_pad)), 0, max(0, height - 1))
    crop_bottom = clamp(int(math.ceil(bottom + bottom_pad)), crop_top + 1, height)

    min_height = min(
        height,
        max(VERTICAL_CROP_CONFIG["bbox_min_height"], int(round(span + 900))),
    )
    if crop_bottom - crop_top < min_height:
        deficit = min_height - (crop_bottom - crop_top)
        crop_top = max(0, crop_top - deficit // 2)
        crop_bottom = min(height, crop_bottom + deficit - deficit // 2)
        if crop_bottom - crop_top < min_height:
            crop_top = max(0, crop_bottom - min_height)

    cropped = image.crop((0, crop_top, width, crop_bottom))
    return cropped, [0, crop_top, width, crop_bottom]


def crop_from_progress(
    image: Image.Image,
    target_index: int,
    action_count: int,
) -> tuple[Image.Image, list[int]]:
    width, height = image.size
    if height <= 1:
        return image.copy(), [0, 0, width, height]

    progress = 0.0
    if action_count > 1:
        progress = clamp(target_index, 0, action_count - 1) / float(action_count - 1)

    center_ratio = 0.18 + 0.64 * progress
    crop_height = min(
        height,
        max(
            VERTICAL_CROP_CONFIG["fallback_min_height"],
            min(
                VERTICAL_CROP_CONFIG["fallback_max_height"],
                int(round(height * VERTICAL_CROP_CONFIG["fallback_height_ratio"])),
            ),
        ),
    )
    center_y = int(round(height * center_ratio))
    crop_top = clamp(center_y - crop_height // 2, 0, max(0, height - crop_height))
    crop_bottom = crop_top + crop_height
    if crop_bottom > height:
        crop_bottom = height
        crop_top = max(0, height - crop_height)

    cropped = image.crop((0, crop_top, width, crop_bottom))
    return cropped, [0, crop_top, width, crop_bottom]


def main() -> None:
    args = parse_args()
    if not args.dataset.is_file():
        raise FileNotFoundError(f"Dataset not found: {args.dataset}")
    if not args.source_manifest.is_file():
        raise FileNotFoundError(f"Source manifest not found: {args.source_manifest}")

    source_rows = read_jsonl(args.source_manifest)
    if not source_rows:
        raise RuntimeError("Source manifest is empty")

    table = pq.read_table(
        args.dataset,
        columns=["action_uid", "action_reprs", "pos_candidates", "target_action_index", "confirmed_task", "screenshot"],
    )
    parquet_rows = table.to_pylist()
    if len(source_rows) != len(parquet_rows):
        raise RuntimeError(
            f"Manifest rows ({len(source_rows)}) and parquet rows ({len(parquet_rows)}) do not match"
        )

    args.output_dir.mkdir(parents=True, exist_ok=True)

    cropped_manifest: list[dict[str, Any]] = []
    stats = {
        "bbox_crop": 0,
        "fallback_progress": 0,
        "unchanged_height": 0,
    }
    height_deltas: list[int] = []

    for index, (source_row, parquet_row) in enumerate(zip(source_rows, parquet_rows)):
        sample_id = str(source_row["sample_id"])
        if sample_id != str(parquet_row["action_uid"]):
            raise RuntimeError(
                f"Sample id mismatch at row {index}: manifest={sample_id}, parquet={parquet_row['action_uid']}"
            )

        source_image_path = Path(str(source_row["clean_image_path"]))
        if not source_image_path.is_file():
            raise FileNotFoundError(f"Source image missing: {source_image_path}")

        with Image.open(source_image_path) as image:
            image = image.convert("RGB")
            original_width, original_height = image.size

            boxes = parse_boxes(parquet_row.get("pos_candidates"))
            target_index = int(str(parquet_row.get("target_action_index", 0)).strip())
            action_count = len(parquet_row.get("action_reprs") or [])

            if boxes:
                cropped, crop_box = crop_from_boxes(image, boxes)
                crop_mode = "bbox_union"
                stats["bbox_crop"] += 1
            else:
                cropped, crop_box = crop_from_progress(image, target_index, action_count)
                crop_mode = "progress_fallback"
                stats["fallback_progress"] += 1

            output_name = f"{source_image_path.stem}_crop.png"
            output_path = args.output_dir / output_name
            cropped.save(output_path)

        cropped_width, cropped_height = cropped.size
        if cropped_height == original_height:
            stats["unchanged_height"] += 1
        height_deltas.append(original_height - cropped_height)

        cropped_manifest.append(
            {
                "row_index": int(source_row["row_index"]),
                "parquet_row_index": int(source_row["parquet_row_index"]),
                "sample_id": sample_id,
                "user_task": str(source_row["user_task"]),
                "previous_actions": list(source_row.get("previous_actions") or []),
                "ground_truth_action": str(source_row.get("ground_truth_action", "")),
                "source_clean_image_path": source_row["clean_image_path"],
                "cropped_image_path": str(output_path),
                "original_size": [original_width, original_height],
                "cropped_size": [cropped_width, cropped_height],
                "crop_box": crop_box,
                "crop_mode": crop_mode,
                "bbox_count": len(boxes),
                "target_action_index": target_index,
                "action_count": action_count,
                "source_screenshot_path": source_row.get("source_screenshot_path"),
                "confirmed_task": str(parquet_row.get("confirmed_task", "")),
            }
        )
        print(f"[{index + 1}/{len(source_rows)}] {output_path.name} {crop_mode} {original_height}->{cropped_height}", flush=True)

    manifest_path = args.output_dir / "manifest.jsonl"
    summary_path = args.output_dir / "summary.json"
    write_jsonl(manifest_path, cropped_manifest)

    summary = {
        "dataset": str(args.dataset),
        "source_manifest": str(args.source_manifest),
        "output_dir": str(args.output_dir),
        "manifest_path": str(manifest_path),
        "records": len(cropped_manifest),
        "crop_modes": stats,
        "crop_config": VERTICAL_CROP_CONFIG,
        "average_height_reduction": round(mean(height_deltas), 2) if height_deltas else 0,
    }
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
