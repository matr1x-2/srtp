from typing import Sequence

import torch
from torch import nn


class MLPNet(nn.Module):
    def __init__(
        self,
        input_dim: int = 28 * 28,
        num_classes: int = 10,
        hidden_dims: Sequence[int] = (256, 128),
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        if not hidden_dims:
            raise ValueError("hidden_dims 不能为空")

        layers = []
        in_features = input_dim
        for hidden_dim in hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_features, hidden_dim),
                    nn.ReLU(inplace=True),
                    nn.Dropout(p=dropout),
                ]
            )
            in_features = hidden_dim

        layers.append(nn.Linear(in_features, num_classes))
        self.classifier = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.flatten(x, start_dim=1)
        return self.classifier(x)
