from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml
from PIL import Image
from torchvision import transforms

from src.models import build_model


PROJECT_DIR = Path(__file__).resolve().parent

CIFAR10_CLASSES = [
    "airplane",
    "automobile",
    "bird",
    "cat",
    "deer",
    "dog",
    "frog",
    "horse",
    "ship",
    "truck",
]


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def load_network_weights(network: torch.nn.Module, checkpoint_path: Path, device: torch.device) -> None:
    ckpt = torch.load(checkpoint_path, map_location=device)

    state_dict = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

    new_state_dict = {}
    for key, value in state_dict.items():
        if key.startswith("network."):
            new_key = key[len("network."):]
            new_state_dict[new_key] = value
        elif key.startswith("model."):
            new_key = key[len("model."):]
            new_state_dict[new_key] = value
        else:
            new_state_dict[key] = value

    missing, unexpected = network.load_state_dict(new_state_dict, strict=False)

    if missing:
        print("Warning: missing keys:")
        print(missing)

    if unexpected:
        print("Warning: unexpected keys:")
        print(unexpected)


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict one image with a trained CIFAR10 model")
    parser.add_argument("--image", type=str, required=True, help="Path to input image")
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to trained .ckpt file")
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["simple_cnn", "cnn", "custom_cnn", "resnet18", "resnet50", "vgg16", "densenet121"],
    )
    parser.add_argument("--config", type=str, default="configs/cifar.yaml")
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()

    image_path = resolve_path(args.image, PROJECT_DIR)
    checkpoint_path = resolve_path(args.checkpoint, PROJECT_DIR)
    config_path = resolve_path(args.config, PROJECT_DIR)

    cfg = load_yaml(config_path)
    model_cfg = cfg["model"]
    simple_cfg = model_cfg.get("simple_cnn", {})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    network = build_model(
        args.model,
        num_classes=10,
        simple_cnn_channels=tuple(simple_cfg.get("channels", [64, 128, 256])),
        simple_cnn_dropout=float(simple_cfg.get("dropout", 0.25)),
    )

    load_network_weights(network, checkpoint_path, device)
    network.to(device)
    network.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.4914, 0.4822, 0.4465),
            std=(0.2470, 0.2435, 0.2616),
        ),
    ])

    image = Image.open(image_path).convert("RGB")
    x = transform(image).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = network(x)
        probs = torch.softmax(logits, dim=1)[0]

    pred_idx = int(torch.argmax(probs).item())
    pred_class = CIFAR10_CLASSES[pred_idx]
    pred_prob = float(probs[pred_idx].item())

    print("=" * 50)
    print(f"Image: {image_path}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Model: {args.model}")
    print(f"Predicted class: {pred_class}")
    print(f"Confidence: {pred_prob:.4f}")
    print("=" * 50)

    topk = min(args.topk, len(CIFAR10_CLASSES))
    top_probs, top_indices = torch.topk(probs, k=topk)

    print(f"Top-{topk} predictions:")
    for rank, (prob, idx) in enumerate(zip(top_probs, top_indices), start=1):
        idx = int(idx.item())
        print(f"{rank}. {CIFAR10_CLASSES[idx]:>10s}: {float(prob.item()):.4f}")


if __name__ == "__main__":
    main()
