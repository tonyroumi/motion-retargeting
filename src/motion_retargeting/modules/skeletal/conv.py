"""
Skeletal-temporal convolution: each edge aggregates its neighbor edges over time.
"""

from dataclasses import dataclass
from typing import List, Optional

import torch
import torch.nn.functional as F
from torch import nn

from .base import SkeletalBase
from .linear import SkeletalLinear


@dataclass
class SkeletalConvConfig:
    in_channels_per_joint: int
    out_channels_per_joint: int
    offset_in_channels_per_joint: int = 0  # 0: no static offset branch
    bias: bool = True
    kernel_size: int = 15
    stride: int = 1
    padding: int = 7
    padding_mode: str = "reflect"
    dilation: int = 1
    groups: int = 1


class SkeletalConv(SkeletalBase):
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
        out_channels_per_joint: int,
        offset_in_channels_per_joint: int = 0,
        bias: bool = True,
        kernel_size: int = 15,
        stride: int = 1,
        padding: int = 7,
        padding_mode: str = "reflect",
        dilation: int = 1,
        groups: int = 1,
    ):
        super().__init__(adj_list, in_channels_per_joint)

        num_edges = len(self.adj)

        self.in_channels = in_channels_per_joint * num_edges
        self.out_channels_per_joint = out_channels_per_joint
        self.out_channels = out_channels_per_joint * num_edges

        self.stride = stride
        self.padding = (padding, padding)
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.weight = nn.Parameter(torch.zeros(self.out_channels, self.in_channels, kernel_size))
        self.bias = nn.Parameter(torch.zeros(self.out_channels)) if bias else None
        self.register_buffer("mask", torch.zeros_like(self.weight), persistent=False)

        if offset_in_channels_per_joint:
            self.offset_encoder = SkeletalLinear(self.adj, offset_in_channels_per_joint, self.out_channels)

        self.init_weights()

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        x:      [B, in_channels, T]
        offset: [B, offset_in_channels, 1], static features added to every frame of the output
        """
        output = F.conv1d(
            F.pad(x, pad=self.padding, mode=self.padding_mode),
            self.weight * self.mask,
            self.bias,
            stride=self.stride,
            dilation=self.dilation,
            groups=self.groups,
        )

        if offset is not None:
            output = output + self.offset_encoder(offset).unsqueeze(-1) / 100

        return output
