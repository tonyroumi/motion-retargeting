from dataclasses import dataclass

from torch import nn
import torch


@dataclass
class MotionDiscriminatorConfig:
    hidden_dim: int = 256
    kernel_size: int = 15


class MotionDiscriminator(nn.Module):
    """
    PAN positional patch discriminator. Reads raw bone vectors with the absolute root,
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
