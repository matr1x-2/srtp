from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data import CIFAR10DataModule
from src.lightning_compat import L, LearningRateMonitor, ModelCheckpoint, TensorBoardLogger
from src.lit_module import ViTClassifier
from src.model import VisionTransformer


PROJECT_DIR = Path(__file__).resolve().parent


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_path(value: str, base_dir: Path) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="训练一个用于 CIFAR10 分类的 ViT 风格 Transformer Encoder")
    parser.add_argument("--config", type=str, default="configs/vit_cifar10.yaml")
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--patch_size", type=int, default=None)
    parser.add_argument("--embed_dim", type=int, default=None)
    parser.add_argument("--depth", type=int, default=None)
    parser.add_argument("--num_heads", type=int, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_DIR)
    cfg = load_yaml(config_path)

    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["learning_rate"] = args.lr
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.patch_size is not None:
        cfg["model"]["patch_size"] = args.patch_size
    if args.embed_dim is not None:
        cfg["model"]["embed_dim"] = args.embed_dim
    if args.depth is not None:
        cfg["model"]["depth"] = args.depth
    if args.num_heads is not None:
        cfg["model"]["num_heads"] = args.num_heads
    if args.precision is not None:
        cfg["runtime"]["precision"] = args.precision

    seed = int(cfg["project"]["seed"])
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data_cfg = cfg["data"]
    data_module = CIFAR10DataModule(
        data_dir=str(resolve_path(data_cfg["data_dir"], PROJECT_DIR)),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        val_split=int(data_cfg["val_split"]),
        seed=seed,
    )

    model_cfg = cfg["model"]
    network = VisionTransformer(
        image_size=int(model_cfg["image_size"]),
        patch_size=int(model_cfg["patch_size"]),
        in_channels=int(model_cfg["in_channels"]),
        num_classes=int(model_cfg["num_classes"]),
        embed_dim=int(model_cfg["embed_dim"]),
        depth=int(model_cfg["depth"]),
        num_heads=int(model_cfg["num_heads"]),
        mlp_ratio=float(model_cfg["mlp_ratio"]),
        dropout=float(model_cfg["dropout"]),
        emb_dropout=float(model_cfg["emb_dropout"]),
    )

    train_cfg = cfg["train"]
    lit_model = ViTClassifier(
        network=network,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optimizer_name=str(train_cfg["optimizer"]),
        scheduler_name=str(train_cfg["scheduler"]),
        max_epochs=int(train_cfg["max_epochs"]),
    )

    log_cfg = cfg["logging"]
    logger = TensorBoardLogger(
        save_dir=str(resolve_path(log_cfg["save_dir"], PROJECT_DIR)),
        name=log_cfg.get("name", "vit-cifar10"),
        version=log_cfg.get("version"),
    )

    ckpt_cfg = cfg["checkpoint"]
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(resolve_path(ckpt_cfg["dirpath"], PROJECT_DIR)),
        filename="{epoch}-{val_acc:.4f}",
        monitor=ckpt_cfg.get("monitor", "val_acc"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 1)),
        save_last=True,
    )

    trainer = L.Trainer(
        max_epochs=int(train_cfg["max_epochs"]),
        accelerator=cfg["runtime"].get("accelerator", "auto"),
        devices=cfg["runtime"].get("devices", "auto"),
        precision=cfg["runtime"].get("precision", "32-true"),
        deterministic=cfg["runtime"].get("deterministic", "warn"),
        logger=logger,
        callbacks=[checkpoint_callback, LearningRateMonitor(logging_interval="epoch")],
        log_every_n_steps=int(log_cfg.get("log_every_n_steps", 20)),
        fast_dev_run=args.fast_dev_run,
    )

    print(f"Config: {config_path}")
    print(
        "Train Settings: "
        f"epochs={train_cfg['max_epochs']}, "
        f"batch_size={data_cfg['batch_size']}, "
        f"lr={train_cfg['learning_rate']}, "
        f"patch_size={model_cfg['patch_size']}, "
        f"embed_dim={model_cfg['embed_dim']}, "
        f"depth={model_cfg['depth']}, "
        f"heads={model_cfg['num_heads']}"
    )

    trainer.fit(lit_model, datamodule=data_module)
    trainer.test(lit_model, datamodule=data_module, ckpt_path="best")


if __name__ == "__main__":
    main()
