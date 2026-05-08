from __future__ import annotations

from typing import Callable, Iterable

import torch.nn as nn
from torchvision import models


class SimpleCIFARNet(nn.Module):
    def __init__(
        self,
        num_classes: int,
        channels: Iterable[int] = (64, 128, 256),
        dropout: float = 0.25,
    ) -> None:
        super().__init__()
        channel_list = list(channels)
        if not channel_list:
            raise ValueError("channels must not be empty")

        layers: list[nn.Module] = []
        in_channels = 3
        for out_channels in channel_list:
            layers.extend(
                [
                    nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
                    nn.BatchNorm2d(out_channels),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(kernel_size=2),
                    nn.Dropout2d(p=dropout),
                ]
            )
            in_channels = out_channels

        self.features = nn.Sequential(*layers, nn.AdaptiveAvgPool2d((1, 1)))
        self.classifier = nn.Linear(in_channels, num_classes)

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        return self.classifier(x)


def _instantiate(factory: Callable[..., nn.Module]) -> nn.Module:
    try:
        return factory(weights=None)
    except TypeError:  # pragma: no cover
        return factory(pretrained=False)


def _build_resnet(factory: Callable[..., nn.Module], num_classes: int) -> nn.Module:
    model = _instantiate(factory)
    model.conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.maxpool = nn.Identity()
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def _build_vgg16(num_classes: int) -> nn.Module:
    model = _instantiate(models.vgg16)
    model.classifier[6] = nn.Linear(model.classifier[6].in_features, num_classes)
    return model


def _build_densenet121(num_classes: int) -> nn.Module:
    model = _instantiate(models.densenet121)
    model.features.conv0 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False)
    model.features.pool0 = nn.Identity()
    model.classifier = nn.Linear(model.classifier.in_features, num_classes)
    return model


def build_model(model_name: str, num_classes: int, simple_cnn_channels=(64, 128, 256), simple_cnn_dropout: float = 0.25) -> nn.Module:
    name = model_name.lower().strip()

    if name in {"simple_cnn", "cnn", "custom_cnn"}:
        return SimpleCIFARNet(
            num_classes=num_classes,
            channels=simple_cnn_channels,
            dropout=simple_cnn_dropout,
        )
    if name == "resnet18":
        return _build_resnet(models.resnet18, num_classes)
    if name == "resnet50":
        return _build_resnet(models.resnet50, num_classes)
    if name == "vgg16":
        return _build_vgg16(num_classes)
    if name == "densenet121":
        return _build_densenet121(num_classes)

    raise ValueError(f"Unsupported model: {model_name}")

