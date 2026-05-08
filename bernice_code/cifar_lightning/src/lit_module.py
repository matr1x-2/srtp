from __future__ import annotations

import torch
from torch import nn

from .lightning_compat import L


class CIFARClassifier(L.LightningModule):
    def __init__(
        self,
        network: nn.Module,
        learning_rate: float = 1e-3,
        weight_decay: float = 5e-4,
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
                nesterov=True,
            )
        else:
            raise ValueError(f"Unsupported optimizer: {self.hparams.optimizer_name}")

        scheduler_name = self.hparams.scheduler_name.lower()
        if scheduler_name in {"", "none"}:
            return optimizer
        if scheduler_name == "cosine":
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=max(1, int(self.hparams.max_epochs)),
            )
            return {
                "optimizer": optimizer,
                "lr_scheduler": {"scheduler": scheduler, "interval": "epoch"},
            }

        raise ValueError(f"Unsupported scheduler: {self.hparams.scheduler_name}")

