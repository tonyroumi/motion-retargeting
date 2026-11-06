"""
Skeletal unpooling: copies each pooled region's features back to the edges merged into it.
"""

from typing import List

import torch
from torch import nn


class SkeletalUnpool(nn.Module):
    def __init__(
        self,
        pooled_edges: List[List[int]],
        channels_per_edge: int,
    ):
        super().__init__()

        num_in = len(pooled_edges)
        num_out = sum(len(group) for group in pooled_edges)

        weight = torch.zeros(num_out * channels_per_edge, num_in * channels_per_edge)
        idx = torch.arange(channels_per_edge)
        for i, group in enumerate(pooled_edges):
            for j in group:
                weight[j * channels_per_edge + idx, i * channels_per_edge + idx] = 1

        self.register_buffer("weight", weight, persistent=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, pooled rows * channels, T] -> [B, rows * channels, T] """
        return self.weight @ x
