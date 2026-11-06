from dataclasses import dataclass
from typing import Optional

from torch import nn
import torch

from ..utils import get_activation


@dataclass
class MotionDiscriminatorConfig:
    conv_hidden_dim: int = 256
    kernel_size: int = 15
    stride: int = 1
    padding: int = 7
    activation: str = "relu"


class MotionDiscriminator(nn.Module):
    """
    Retargeted motion needs to fall on the motion manifold of the target skeleton. 
    Compress the feature size to 1.
    """

    def __init__(
        self,
        input_dim: int,
        conv_hidden_dim: Optional[int] = None,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 7,
        activation: str = "relu",
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_dim,
            out_channels=conv_hidden_dim // 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv3 = nn.Conv1d(
            in_channels=conv_hidden_dim // 2,
            out_channels=1,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, W, J+1, 4]]
        """
        B, W = x.shape[:2]
        
        x = x.reshape(B, W, -1).permute(0, 2, 1) # [B, (J+1)*4, W]

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        x = self.conv3(x)
        return x.mean(dim=-1)


@dataclass
class ReferenceMotionDiscriminatorConfig:
    hidden_dim: int = 256
    kernel_size: int = 15


class ReferenceMotionDiscriminator(nn.Module):
    """
    PAN reference-Mixamo positional patch discriminator. Reads raw bone vectors with the absolute root,
    and returns one sigmoid score per temporal patch.
    """

    def __init__(self, input_dim: int, hidden_dim: int = 256, kernel_size: int = 15):
        super().__init__()

        self.conv1 = nn.Conv1d(input_dim, hidden_dim, kernel_size=kernel_size)
        self.conv2 = nn.Conv1d(hidden_dim, hidden_dim // 2, kernel_size=1)
        self.conv3 = nn.Conv1d(hidden_dim // 2, 1, kernel_size=1)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, J, 3]
        returns: [B, T - kernel_size + 1], keeping both axes even for B = 1
        """
        x = x.flatten(2).transpose(1, 2)  # [B, J * 3, T]

        x = self.activation(self.conv1(x))
        x = self.activation(self.conv2(x))
        return torch.sigmoid(self.conv3(x)).squeeze(1)
