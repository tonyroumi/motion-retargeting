"""
Skeletal-linear layer, projecting static offset features into the motion branch.
"""

from typing import List

import torch
import torch.nn.functional as F
from torch import nn

from .base import SkeletalBase


class SkeletalLinear(SkeletalBase):
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
        out_channels: int,
    ):
        super().__init__(adj_list, in_channels_per_joint)

        num_edges = len(self.adj)

        self.in_channels = in_channels_per_joint * num_edges
        self.out_channels = out_channels
        self.out_channels_per_joint = out_channels // num_edges

        self.weight = nn.Parameter(torch.zeros(self.out_channels, self.in_channels))
        self.bias = nn.Parameter(torch.zeros(self.out_channels))
        self.register_buffer("mask", torch.zeros(self.out_channels, self.in_channels), persistent=False)

        self.init_weights()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, in_channels, ...] -> [B, out_channels] """
        return F.linear(x.reshape(x.shape[0], -1), self.weight * self.mask, self.bias)
