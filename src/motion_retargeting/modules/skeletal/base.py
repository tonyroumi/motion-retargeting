"""
Base class for skeletal linear and convolution layers.

The skeleton is an adjacency list where each row holds the neighbor edges of one edge.
The weight matrix mirrors this structure, expanded across the channel dimensions, and a
fixed mask zeros out the weights of edges that are not neighbors.
"""

import math
from typing import List

import torch
from torch import nn


class SkeletalBase(nn.Module):
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
    ):
        super().__init__()

        self.adj = adj_list

        # Map each neighbor edge to its channel range.
        # E.g. (in_channels_per_joint = 2)
        # adj = {0: [0, 1], 1: [0]}
        # expanded_adj_list = {0: [0, 1, 2, 3], 1: [0, 1]}
        self.expanded_adj_list = [
            [
                k * in_channels_per_joint + i
                for k in neighbors
                for i in range(in_channels_per_joint)
            ]
            for neighbors in self.adj
        ]

    @torch.no_grad()
    def init_weights(self) -> None:
        """
        Mark each edge's neighbor channels as valid in the mask, and Kaiming uniform initialize
        only those weights to keep the variance stable across weights and bias.
        """
        for edge, neighbors in enumerate(self.expanded_adj_list):
            start = edge * self.out_channels_per_joint
            end = (edge + 1) * self.out_channels_per_joint

            self.mask[start:end, neighbors, ...] = 1

            weight = torch.zeros_like(self.weight[start:end, neighbors, ...])
            nn.init.kaiming_uniform_(weight, a=math.sqrt(5))
            self.weight[start:end, neighbors, ...] = weight

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
