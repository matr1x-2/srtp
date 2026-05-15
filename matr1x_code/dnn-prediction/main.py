from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import lightning as L
import torch
import yaml
from lightning.pytorch.callbacks import ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from torch import nn

from dataset import MNISTDataModule
from model import CNNNet, MLPNet


class DigitClassifier(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        learning_rate: float,
        weight_decay: float,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["network"])

        self.network = network
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = torch.argmax(logits, dim=1)
        acc = (preds == y).float().mean()
        return loss, acc

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        loss, acc = self._shared_step(batch)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_acc", acc, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, acc = self._shared_step(batch)
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def test_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        loss, acc = self._shared_step(batch)
        self.log("test_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        self.log("test_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )


def load_config(config_path: str) -> dict[str, Any]:
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def build_network(model_cfg: dict[str, Any]) -> nn.Module:
    model_name = model_cfg["name"].lower()
    if model_name == "mlp":
        mlp_cfg = model_cfg.get("mlp", {})
        return MLPNet(
            hidden_dims=tuple(mlp_cfg.get("hidden_dims", [256, 128])),
            dropout=float(mlp_cfg.get("dropout", 0.2)),
        )

    if model_name == "cnn":
        cnn_cfg = model_cfg.get("cnn", {})
        return CNNNet(
            conv_channels=tuple(cnn_cfg.get("conv_channels", [32, 64])),
            fc_hidden_dim=int(cnn_cfg.get("fc_hidden_dim", 128)),
            dropout=float(cnn_cfg.get("dropout", 0.3)),
        )

    raise ValueError(f"不支持的模型类型: {model_name}，请使用 mlp 或 cnn")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="MNIST 分类: PyTorch + Lightning")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/config.yaml",
        help="配置文件路径",
    )
    parser.add_argument(
        "--model",
        type=str,
        choices=["mlp", "cnn"],
        default=None,
        help="覆盖配置中的模型类型",
    )
    parser.add_argument(
        "--max_epochs",
        type=int,
        default=None,
        help="覆盖配置中的训练轮数",
    )
    parser.add_argument(
        "--fast_dev_run",
        action="store_true",
        help="快速调试模式，仅跑一个 batch",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    if args.model is not None:
        cfg["model"]["name"] = args.model
    if args.max_epochs is not None:
        cfg["train"]["max_epochs"] = args.max_epochs

    seed = int(cfg.get("project", {}).get("seed", 42))
    L.seed_everything(seed, workers=True)

    data_cfg = cfg.get("data", {})
    data_module = MNISTDataModule(
        data_dir=data_cfg.get("data_dir", "./data"),
        batch_size=int(data_cfg.get("batch_size", 64)),
        num_workers=int(data_cfg.get("num_workers", 4)),
        val_split=int(data_cfg.get("val_split", 5000)),
    )

    network = build_network(cfg["model"])
    train_cfg = cfg.get("train", {})
    lit_model = DigitClassifier(
        network=network,
        learning_rate=float(train_cfg.get("learning_rate", 1e-3)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
    )

    log_cfg = cfg.get("logging", {})
    model_name = cfg["model"]["name"].lower()
    logger = TensorBoardLogger(
        save_dir=log_cfg.get("save_dir", "./logs"),
        name=f"{log_cfg.get('name', 'mnist')}-{model_name}",
        version=log_cfg.get("version", None),
    )

    ckpt_cfg = cfg.get("checkpoint", {})
    checkpoint_dir = Path(ckpt_cfg.get("dirpath", "./checkpoints")) / model_name
    checkpoint_callback = ModelCheckpoint(
        dirpath=str(checkpoint_dir),
        filename="{epoch}-{val_acc:.4f}",
        monitor=ckpt_cfg.get("monitor", "val_acc"),
        mode=ckpt_cfg.get("mode", "max"),
        save_top_k=int(ckpt_cfg.get("save_top_k", 1)),
        save_last=True,
    )

    runtime_cfg = cfg.get("runtime", {})
    trainer = L.Trainer(
        max_epochs=int(train_cfg.get("max_epochs", 10)),
        accelerator=runtime_cfg.get("accelerator", "auto"),
        devices=runtime_cfg.get("devices", "auto"),
        precision=runtime_cfg.get("precision", "32-true"),
        deterministic=bool(runtime_cfg.get("deterministic", True)),
        logger=logger,
        callbacks=[checkpoint_callback],
        log_every_n_steps=int(log_cfg.get("log_every_n_steps", 10)),
        fast_dev_run=args.fast_dev_run,
    )

    trainer.fit(lit_model, datamodule=data_module)

    test_ckpt_path: str | None = "best"
    if args.fast_dev_run or int(ckpt_cfg.get("save_top_k", 1)) == 0:
        test_ckpt_path = None
    trainer.test(lit_model, datamodule=data_module, ckpt_path=test_ckpt_path)


if __name__ == "__main__":
    main()
