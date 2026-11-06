from typing import List, Optional

import torch
import torch.nn as nn

from .utils import get_activation


class MLP(nn.Module):
    def __init__(
        self,
        in_channels,
        out_channels,
        hidden_layers: List[int] = [256, 256],
        activation: str = "relu",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.hidden_layers = hidden_layers
        self.activation = activation

        trunk_layers = []
        dims = [in_channels] + hidden_layers
        for i in range(len(dims) - 1):
            trunk_layers.append(nn.Linear(dims[i], dims[i + 1]))
            trunk_layers.append(get_activation(activation))
        self.trunk = nn.Sequential(*trunk_layers)
        self.output_head = nn.Linear(dims[-1], out_channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.output_head(self.trunk(x))
