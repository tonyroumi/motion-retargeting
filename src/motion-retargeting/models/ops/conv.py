from .base import SkeletalBase
from .linear import SkeletalLinear

from typing import List, Optional

import torch
import torch.nn.functional as F

class SkeletalConv(SkeletalBase):
    """
    Skeletal-temporal convolution.

    The weights of this layer are created to learn how each edge aggregates information from its
    neighboring edges over time. 
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
        out_channels_per_joint: int,
        bias: bool,
        kernel_size: int,
        stride: int,
        padding: int,
        padding_mode: str,
        dilation: int,
        groups: int,
        offset_in_channels_per_joint: int = 0,
    ):
        super().__init__(adj_list, in_channels_per_joint)

        self.J = len(self.adj)

        self.in_channels = in_channels_per_joint * self.J
        self.out_channels_per_joint = out_channels_per_joint
        self.out_channels = out_channels_per_joint * self.J

        # self.with_bias = bias
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = (padding, padding)
        self.padding_mode = padding_mode
        self.dilation = dilation
        self.groups = groups

        self.weight = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)
        self.bias = torch.zeros(self.out_channels) if bias else self.register_parameter('bias', None)

        self.mask = torch.zeros(self.out_channels, self.in_channels, self.kernel_size)

        if offset_in_channels_per_joint:
            self.offset_encoder = SkeletalLinear(self.adj, offset_in_channels_per_joint, self.out_channels)
        
        super()._init_weights()
        
    
    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        weight_masked = self.weight * self.mask

        padded_x = F.pad(x, pad=self.padding, mode=self.padding_mode)
        output = F.conv1d(padded_x, 
                          weight_masked, 
                          self.bias, 
                          stride=self.stride,
                          dilation=self.dilation, 
                          groups=self.groups) 

        if offset is not None:
            offset_out = self.offset_encoder(offset)
            offset_out = offset_out.reshape(offset_out.shape + (1, ))
            output += offset_out / 100

        return output
