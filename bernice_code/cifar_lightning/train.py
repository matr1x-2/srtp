from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch
import yaml

from src.data import CIFARDataModule
from src.lit_module import CIFARClassifier
from src.lightning_compat import L, LearningRateMonitor, ModelCheckpoint, TensorBoardLogger
from src.models import build_model


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
    parser = argparse.ArgumentParser(description="CIFAR10/100 training with Lightning")
    parser.add_argument("--config", type=str, default="configs/cifar.yaml")
    parser.add_argument("--dataset", type=str, choices=["cifar10", "cifar100"], default=None)
    parser.add_argument(
        "--model",
        type=str,
        choices=["simple_cnn", "cnn", "custom_cnn", "resnet18", "resnet50", "vgg16", "densenet121"],
        default=None,
    )
    parser.add_argument("--max_epochs", type=int, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--weight_decay", type=float, default=None)
    parser.add_argument("--optimizer", type=str, default=None)
    parser.add_argument("--scheduler", type=str, default=None)
    parser.add_argument("--data_dir", type=str, default=None)
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--val_split", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--accelerator", type=str, default=None)
    parser.add_argument("--devices", type=str, default=None)
    parser.add_argument("--precision", type=str, default=None)
    parser.add_argument("--deterministic", type=str, choices=["true", "false", "warn"], default=None)
    parser.add_argument("--fast_dev_run", action="store_true")
    return parser.parse_args()


def parse_deterministic(value: Any) -> bool | str:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered == "true":
            return True
        if lowered == "false":
            return False
        if lowered == "warn":
            return "warn"
    if value is None:
        return "warn"
    return bool(value)


def main() -> None:
    args = parse_args()
    config_path = resolve_path(args.config, PROJECT_DIR)
    cfg = load_yaml(config_path)

    if args.model is not None:
        cfg["model"]["name"] = args.model

    # Apply model-specific training defaults first,
    # then allow CLI args to override them.
    model_name_for_override = str(cfg["model"]["name"]).lower().strip()
    model_overrides = cfg.get("train_overrides", {}).get(model_name_for_override, {})
    if model_overrides:
        cfg["train"].update(model_overrides)

    if args.dataset is not None:
        cfg["data"]["dataset"] = args.dataset
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs
    if args.batch_size is not None:
        cfg["data"]["batch_size"] = args.batch_size
    if args.lr is not None:
        cfg["train"]["learning_rate"] = args.lr
    if args.weight_decay is not None:
        cfg["train"]["weight_decay"] = args.weight_decay
    if args.optimizer is not None:
        cfg["train"]["optimizer"] = args.optimizer
    if args.scheduler is not None:
        cfg["train"]["scheduler"] = args.scheduler
    if args.data_dir is not None:
        cfg["data"]["data_dir"] = args.data_dir
    if args.num_workers is not None:
        cfg["data"]["num_workers"] = args.num_workers
    if args.val_split is not None:
        cfg["data"]["val_split"] = args.val_split
    if args.seed is not None:
        cfg["project"]["seed"] = args.seed
    if args.accelerator is not None:
        cfg["runtime"]["accelerator"] = args.accelerator
    if args.devices is not None:
        cfg["runtime"]["devices"] = args.devices
    if args.precision is not None:
        cfg["runtime"]["precision"] = args.precision
    if args.deterministic is not None:
        cfg["runtime"]["deterministic"] = args.deterministic

    seed = int(cfg["project"]["seed"])
    L.seed_everything(seed, workers=True)
    torch.set_float32_matmul_precision("high")

    data_cfg = cfg["data"]
    data_module = CIFARDataModule(
        dataset_name=data_cfg["dataset"],
        data_dir=str(resolve_path(data_cfg["data_dir"], PROJECT_DIR)),
        batch_size=int(data_cfg["batch_size"]),
        num_workers=int(data_cfg["num_workers"]),
        val_split=int(data_cfg["val_split"]),
        seed=seed,
        random_rotation=float(data_cfg.get("random_rotation", 10)),
        random_erasing_p=float(data_cfg.get("random_erasing_p", 0.25)),
    )

    model_cfg = cfg["model"]
    simple_cfg = model_cfg.get("simple_cnn", {})
    network = build_model(
        model_cfg["name"],
        num_classes=data_module.num_classes,
        simple_cnn_channels=tuple(simple_cfg.get("channels", [64, 128, 256])),
        simple_cnn_dropout=float(simple_cfg.get("dropout", 0.25)),
    )

    train_cfg = cfg["train"]
    lit_model = CIFARClassifier(
        network=network,
        learning_rate=float(train_cfg["learning_rate"]),
        weight_decay=float(train_cfg["weight_decay"]),
        optimizer_name=str(train_cfg.get("optimizer", "adamw")),
        scheduler_name=str(train_cfg.get("scheduler", "cosine")),
        max_epochs=int(train_cfg["max_epochs"]),
    )

    log_cfg = cfg["logging"]
    dataset_name = data_cfg["dataset"].lower()
    model_name = model_cfg["name"].lower()
    logger = TensorBoardLogger(
        save_dir=str(resolve_path(log_cfg["save_dir"], PROJECT_DIR)),
        name=f"{log_cfg.get('name', 'cifar')}-{dataset_name}-{model_name}",
        version=log_cfg.get("version"),
    )

    ckpt_cfg = cfg["checkpoint"]
    checkpoint_dir = resolve_path(ckpt_cfg["dirpath"], PROJECT_DIR) / dataset_name / model_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch}-{val_acc:.4f}",
        monitor=ckpt_cfg.get("monitor", "val_acc"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 1)),
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval="epoch")

    runtime_cfg = cfg["runtime"]
    deterministic_mode = parse_deterministic(runtime_cfg.get("deterministic", "warn"))
    trainer = L.Trainer(
        max_epochs=int(train_cfg["max_epochs"]),
        accelerator=runtime_cfg.get("accelerator", "auto"),
        devices=runtime_cfg.get("devices", "auto"),
        precision=runtime_cfg.get("precision", "32-true"),
        deterministic=deterministic_mode,
        logger=logger,
        callbacks=[checkpoint_callback, lr_monitor],
        log_every_n_steps=int(log_cfg.get("log_every_n_steps", 20)),
        fast_dev_run=args.fast_dev_run,
    )

    print(f"Dataset: {dataset_name}")
    print(f"Model: {model_name}")
    print(f"Config: {config_path}")
    print(
        "Train Settings: "
        f"epochs={train_cfg['max_epochs']}, "
        f"optimizer={train_cfg.get('optimizer', 'adamw')}, "
        f"lr={train_cfg['learning_rate']}, "
        f"weight_decay={train_cfg['weight_decay']}, "
        f"scheduler={train_cfg.get('scheduler', 'cosine')}"
    )

    trainer.fit(lit_model, datamodule=data_module)

    test_ckpt_path: str | None = None if args.fast_dev_run else "best"
    trainer.test(lit_model, datamodule=data_module, ckpt_path=test_ckpt_path)


if __name__ == "__main__":
    main()
