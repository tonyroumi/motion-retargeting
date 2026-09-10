from .base import SkeletalBase

from typing import List
import torch
import torch.nn.functional as F

class SkeletalLinear(SkeletalBase):
    """
    Skeletal-linear layer.

    This layer is used to project offset features for intercept with dynamic branch.
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
        out_channels: int
    ):
        super().__init__(adj_list, in_channels_per_joint)
        
        self.J = len(self.adj)

        self.in_channels = in_channels_per_joint * self.J
        self.out_channels = out_channels 
        # To project offset features to motion features (includes global pos)
        self.out_channels_per_joint = out_channels // self.J

        self.weight = torch.zeros(self.out_channels, self.in_channels)
        self.bias = torch.zeros(self.out_channels)

        self.mask = torch.zeros(self.out_channels, self.in_channels)

        super()._init_weights()

    def forward(self, x: torch.Tensor):
        weight_masked = self.weight * self.mask
        x = x.reshape(x.shape[0], -1)

        out = F.linear(x, weight_masked, self.bias)
        return out
