import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models

import pytorch_lightning as L
from torchmetrics.classification import MulticlassAccuracy

# =========================
# 1. 数据集部分
# =========================
class CIFARDataModule(L.LightningDataModule):
    def __init__(self, dataset_name="cifar10", batch_size=128, num_workers=0):
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.num_workers = num_workers

        if dataset_name == "cifar10":
            self.num_classes = 10
            self.mean = (0.4914, 0.4822, 0.4465)
            self.std = (0.2470, 0.2435, 0.2616)
            self.dataset_class = datasets.CIFAR10
        elif dataset_name == "cifar100":
            self.num_classes = 100
            self.mean = (0.5071, 0.4867, 0.4408)
            self.std = (0.2675, 0.2565, 0.2761)
            self.dataset_class = datasets.CIFAR100
        else:
            raise ValueError("dataset_name 只能是 cifar10 或 cifar100")

    def setup(self, stage=None):
        # 训练集使用数据增强
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        # 验证集和测试集不做随机增强
        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(self.mean, self.std),
        ])

        full_train_aug = self.dataset_class(
            root="./data",
            train=True,
            download=True,
            transform=train_transform,
        )

        full_train_plain = self.dataset_class(
            root="./data",
            train=True,
            download=True,
            transform=test_transform,
        )

        self.test_dataset = self.dataset_class(
            root="./data",
            train=False,
            download=True,
            transform=test_transform,
        )

        # CIFAR 训练集一共 50000 张
        # 这里 45000 张训练，5000 张验证
        total_size = len(full_train_aug)
        indices = torch.randperm(total_size).tolist()

        train_size = int(total_size * 0.9)
        train_indices = indices[:train_size]
        val_indices = indices[train_size:]

        self.train_dataset = Subset(full_train_aug, train_indices)
        self.val_dataset = Subset(full_train_plain, val_indices)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )


# =========================
# 2. 自己写一个简单 CNN
# =========================
class SimpleCNN(nn.Module):
    def __init__(self, num_classes):
        super().__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )

        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


# =========================
# 3. 根据名字选择模型
# =========================
def get_model(model_name, num_classes):
    model_name = model_name.lower()

    if model_name == "simple_cnn":
        model = SimpleCNN(num_classes)

    elif model_name == "resnet18":
        model = models.resnet18(weights=None, num_classes=num_classes)

        # CIFAR 图片是 32x32，比 ImageNet 小，所以改一下开头
        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    elif model_name == "resnet50":
        model = models.resnet50(weights=None, num_classes=num_classes)

        model.conv1 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.maxpool = nn.Identity()

    elif model_name == "vgg16":
        model = models.vgg16(weights=None, num_classes=num_classes)

    elif model_name == "densenet121":
        model = models.densenet121(weights=None, num_classes=num_classes)

        # 同样适配 CIFAR 的 32x32 小图片
        model.features.conv0 = nn.Conv2d(
            3, 64, kernel_size=3, stride=1, padding=1, bias=False
        )
        model.features.pool0 = nn.Identity()

    else:
        raise ValueError("模型名错误")

    return model


# =========================
# 4. Lightning 训练模块
# =========================
class CIFARClassifier(L.LightningModule):
    def __init__(self, model_name, num_classes, lr=1e-3):
        super().__init__()
        self.save_hyperparameters()

        self.model = get_model(model_name, num_classes)
        self.loss_fn = nn.CrossEntropyLoss()

        self.train_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.val_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")
        self.test_acc = MulticlassAccuracy(num_classes=num_classes, average="micro")

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.train_acc(preds, labels)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True)
        self.log("train_acc", self.train_acc, prog_bar=True, on_epoch=True)

        return loss

    def validation_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.val_acc(preds, labels)

        self.log("val_loss", loss, prog_bar=True, on_epoch=True)
        self.log("val_acc", self.val_acc, prog_bar=True, on_epoch=True)

    def test_step(self, batch, batch_idx):
        images, labels = batch

        logits = self(images)
        loss = self.loss_fn(logits, labels)

        preds = torch.argmax(logits, dim=1)
        self.test_acc(preds, labels)

        self.log("test_loss", loss, prog_bar=True)
        self.log("test_acc", self.test_acc, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.lr)
        return optimizer


# =========================
# 5. 主函数
# =========================
def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="cifar10",
                        choices=["cifar10", "cifar100"])

    parser.add_argument("--model", type=str, default="simple_cnn",
                        choices=["simple_cnn", "resnet18", "resnet50", "vgg16", "densenet121"])

    parser.add_argument("--max_epochs", type=int, default=3)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--num_workers", type=int, default=0)

    args = parser.parse_args()

    data_module = CIFARDataModule(
        dataset_name=args.dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    model = CIFARClassifier(
        model_name=args.model,
        num_classes=data_module.num_classes,
        lr=args.lr,
    )

    trainer = L.Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        log_every_n_steps=10,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)


if __name__ == "__main__":
    main()