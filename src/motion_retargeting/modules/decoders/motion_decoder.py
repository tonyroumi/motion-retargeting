from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn

from ..utils import get_activation


@dataclass
class MotionDecoderConfig:
    conv_hidden_dim: int = 32
    kernel_size: int = 15
    stride: int = 1
    padding: int = 7
    upsample_mode: str = "linear"
    activation: str = "relu"


class MotionDecoder(nn.Module):
    """
    Upscale and and restore the temporal and spatial dimension.
    """

    def __init__(
        self,
        input_dim: int,
        conv_hidden_dim: Optional[int] = None,
        conv_out_dim: Optional[int] = None,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 7,
        upsample_mode: str = "linear",
        activation: str = "relu",
    ):
        super().__init__()

        self.ups = nn.Upsample(
            scale_factor=2,
            mode=upsample_mode,
            align_corners=False,
        )
        self.conv1 = nn.Conv1d(
            in_channels=input_dim,
            out_channels=conv_hidden_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
        self.conv2 = nn.Conv1d(
            in_channels=conv_hidden_dim,
            out_channels=conv_out_dim,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )

        self.activation = get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T_ds, num_bodies, D]
        """
        B, T_ds, N, D = x.shape

        x = x.permute(0, 2, 3, 1) # [B, num_bodies, D, T]
        x = x.reshape(B, N*D, T_ds) # [B, num_bodies * D, T]

        x = self.conv1(self.ups(x))
        x = self.activation(x)
        x = self.conv2(self.ups(x))

        x = x.permute(0, 2, 1) # [B, T, (J+1)*4]
        return x.reshape(B, T_ds*4, -1, 4)
