import torch
from torch import nn


class CNNNet(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        conv_channels: tuple[int, int] = (32, 64),
        fc_hidden_dim: int = 128,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        if len(conv_channels) != 2:
            raise ValueError("conv_channels 需要包含两个通道数")

        c1, c2 = conv_channels
        self.features = nn.Sequential(
            nn.Conv2d(1, c1, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
            nn.Conv2d(c1, c2, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(c2 * 7 * 7, fc_hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(p=dropout),
            nn.Linear(fc_hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        return self.classifier(x)
