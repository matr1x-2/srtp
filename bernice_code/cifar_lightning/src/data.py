from __future__ import annotations

from pathlib import Path
from typing import Optional

import torch
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

from .lightning_compat import L


class CIFARDataModule(L.LightningDataModule):
    def __init__(
        self,
        dataset_name: str = "cifar10",
        data_dir: str = "../data",
        batch_size: int = 128,
        num_workers: int = 4,
        val_split: int = 5000,
        seed: int = 42,
        random_rotation: float = 10.0,
        random_erasing_p: float = 0.25,
    ) -> None:
        super().__init__()
        self.dataset_name = dataset_name.lower()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split
        self.seed = seed
        self.random_rotation = random_rotation
        self.random_erasing_p = random_erasing_p

        if self.dataset_name == "cifar10":
            self.num_classes = 10
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
            self.dataset_cls = datasets.CIFAR10
        elif self.dataset_name == "cifar100":
            self.num_classes = 100
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
            self.dataset_cls = datasets.CIFAR100
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        self.train_dataset: Optional[Subset] = None
        self.val_dataset: Optional[Subset] = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        root = Path(self.data_dir)
        self.dataset_cls(root=root, train=True, download=True)
        self.dataset_cls(root=root, train=False, download=True)

    def _train_transform(self) -> transforms.Compose:
        return transforms.Compose(
            [
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(self.random_rotation),
                transforms.ToTensor(),
                transforms.Normalize(self.mean, self.std),
                transforms.RandomErasing(p=self.random_erasing_p),
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
            train_aug = self.dataset_cls(
                root=root,
                train=True,
                download=False,
                transform=self._train_transform(),
            )
            train_plain = self.dataset_cls(
                root=root,
                train=True,
                download=False,
                transform=self._eval_transform(),
            )

            total_size = len(train_aug)
            if self.val_split <= 0 or self.val_split >= total_size:
                raise ValueError("val_split must be in [1, len(train_dataset) - 1]")

            generator = torch.Generator().manual_seed(self.seed)
            indices = torch.randperm(total_size, generator=generator).tolist()
            train_size = total_size - self.val_split
            train_indices = indices[:train_size]
            val_indices = indices[train_size:]

            self.train_dataset = Subset(train_aug, train_indices)
            self.val_dataset = Subset(train_plain, val_indices)

        if stage in (None, "test"):
            self.test_dataset = self.dataset_cls(
                root=root,
                train=False,
                download=False,
                transform=self._eval_transform(),
            )

    def _loader_kwargs(self, shuffle: bool) -> dict:
        kwargs = {
            "batch_size": self.batch_size,
            "shuffle": shuffle,
            "num_workers": self.num_workers,
            "pin_memory": torch.cuda.is_available(),
        }
        if self.num_workers > 0:
            kwargs["persistent_workers"] = True
        return kwargs

    def train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise RuntimeError("Train dataset is not ready. Call setup('fit') first.")
        return DataLoader(self.train_dataset, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        if self.val_dataset is None:
            raise RuntimeError("Val dataset is not ready. Call setup('fit') first.")
        return DataLoader(self.val_dataset, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        if self.test_dataset is None:
            raise RuntimeError("Test dataset is not ready. Call setup('test') first.")
        return DataLoader(self.test_dataset, **self._loader_kwargs(shuffle=False))

