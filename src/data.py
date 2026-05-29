from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .lightning_compat import L


class CIFAR10DataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "../data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: int = 5000,
        seed: int = 42,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed

        self.mean = (0.4914, 0.4822, 0.4465)
        self.std = (0.2470, 0.2435, 0.2616)

        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        root = Path(self.data_dir)
        datasets.CIFAR10(root=root, train=True, download=True)
        datasets.CIFAR10(root=root, train=False, download=True)

    def _train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def _eval_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
            ]
        )

    def setup(self, stage: Optional[str] = None) -> None:
        root = Path(self.data_dir)

        if stage in (None, "fit"):
            full_train = datasets.CIFAR10(
                root=root,
                train=True,
                transform=self._train_transform(),
            )
            full_val = datasets.CIFAR10(
                root=root,
                train=True,
                transform=self._eval_transform(),
            )
            train_size = len(full_train) - self.val_split
            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(len(full_train), generator=generator).tolist()
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]
            self.train_dataset = Subset(full_train, train_indices)
            self.val_dataset = Subset(full_val, val_indices)

        if stage in (None, "test"):
            self.test_dataset = datasets.CIFAR10(
                root=root,
                train=False,
                transform=self._eval_transform(),
            )

    def _loader_kwargs(self, shuffle: bool) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "num_workers": self.num_workers,
            "shuffle": shuffle,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))
