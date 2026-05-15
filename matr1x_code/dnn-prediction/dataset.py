from __future__ import annotations

from typing import Optional

import lightning as L
import torch
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import datasets, transforms


class MNISTDataModule(L.LightningDataModule):
    def __init__(
        self,
        data_dir: str = "./data",
        batch_size: int = 64,
        num_workers: int = 4,
        val_split: int = 5000,
    ) -> None:
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.val_split = val_split

        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ]
        )

        self.mnist_train: Optional[Dataset] = None
        self.mnist_val: Optional[Dataset] = None
        self.mnist_test: Optional[Dataset] = None

    def prepare_data(self) -> None:
        datasets.MNIST(root=self.data_dir, train=True, download=True)
        datasets.MNIST(root=self.data_dir, train=False, download=True)

    def setup(self, stage: Optional[str] = None) -> None:
        if stage in (None, "fit"):
            full_train = datasets.MNIST(
                root=self.data_dir,
                train=True,
                transform=self.transform,
            )
            train_size = len(full_train) - self.val_split
            if train_size <= 0:
                raise ValueError("val_split 过大，导致训练集大小小于等于 0")

            generator = torch.Generator().manual_seed(42)
            self.mnist_train, self.mnist_val = random_split(
                full_train,
                [train_size, self.val_split],
                generator=generator,
            )

        if stage in (None, "test"):
            self.mnist_test = datasets.MNIST(
                root=self.data_dir,
                train=False,
                transform=self.transform,
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
        if self.mnist_train is None:
            raise RuntimeError("train dataset 未初始化，请先调用 setup('fit')")
        return DataLoader(self.mnist_train, **self._loader_kwargs(shuffle=True))

    def val_dataloader(self) -> DataLoader:
        if self.mnist_val is None:
            raise RuntimeError("val dataset 未初始化，请先调用 setup('fit')")
        return DataLoader(self.mnist_val, **self._loader_kwargs(shuffle=False))

    def test_dataloader(self) -> DataLoader:
        if self.mnist_test is None:
            raise RuntimeError("test dataset 未初始化，请先调用 setup('test')")
        return DataLoader(self.mnist_test, **self._loader_kwargs(shuffle=False))
