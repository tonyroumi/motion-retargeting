from dataclasses import dataclass

import torch
from torch import nn

from ..encoders.motion_encoder import MaskedConv1d


@dataclass
class MotionDecoderConfig:
    kernel_size: int = 15
    upsample_mode: str = "linear"


class MotionDecoder(nn.Module):
    """
    PAN decoder: two x2 upsamplings, each with a masked reflect-padded conv. The
    first conv stays within each part; the second maps parts to joint and root-velocity channels, and has no
    activation or quaternion normalization after it.
    """

    def __init__(
        self,
        part_connectivity: torch.Tensor,
        output_connectivity: torch.Tensor,
        kernel_size: int = 15,
        upsample_mode: str = "linear",
    ):
        super().__init__()

        self.ups = nn.Upsample(scale_factor=2, mode=upsample_mode, align_corners=False)
        self.conv1 = MaskedConv1d(part_connectivity, kernel_size)
        self.conv2 = MaskedConv1d(output_connectivity, kernel_size)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T_ds, P, D], the fused motion and skeleton codes
        returns: [B, T, 4J + 4]
        """
        x = x.flatten(2, 3).transpose(1, 2)  # [B, P * D, T_ds]

        x = self.activation(self.conv1(self.ups(x)))
        x = self.conv2(self.ups(x))

        return x.transpose(1, 2)
