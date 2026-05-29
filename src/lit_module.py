from __future__ import annotations

import torch
from torch import nn

from .lightning_compat import L


class ViTClassifier(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 3e-4,
        weight_decay: float = 0.05,
        optimizer_name: str = "adamw",
        scheduler_name: str = "cosine",
        max_epochs: int = 30,
    ) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=["network"])
        self.network = network
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x)

    def _shared_step(self, batch: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        x, y = batch
        logits = self(x)
        loss = self.criterion(logits, y)
        preds = logits.argmax(dim=1)
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

    def on_train_epoch_end(self) -> None:
        metrics = self.trainer.callback_metrics

        def _fmt(name: str) -> str:
            value = metrics.get(name)
            if value is None:
                return "NA"
            return f"{float(value):.4f}"

        print(
            f"[epoch {self.current_epoch}] "
            f"train_loss={_fmt('train_loss')} "
            f"train_acc={_fmt('train_acc')} "
            f"val_loss={_fmt('val_loss')} "
            f"val_acc={_fmt('val_acc')}"
        )

    def configure_optimizers(self):
        optimizer_name = self.hparams.optimizer_name.lower()

        if optimizer_name == "adamw":
            optimizer = torch.optim.AdamW(
                self.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=self.hparams.weight_decay,
            )
        elif optimizer_name == "sgd":
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.hparams.learning_rate,
                momentum=0.9,
                weight_decay=self.hparams.weight_decay,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        if self.hparams.scheduler_name.lower() == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.hparams.max_epochs)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }
        return optimizer
