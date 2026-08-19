"""
Base class for skeletal linear and pooling operations.

We represent the skeleton as an adjacency list where each index contains neighbor edges
up to a certain distance d. The weight matrix is built to mirror this adjacency structure 
but expanded across channel dimensions. 
"""

from typing import List

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalBase(nn.Module):
    def __init__(
        self,
        adj_list: List[List[int]],
        in_channels_per_joint: int,
    ):
       super().__init__()
       self.adj = adj_list

       # Map each neighbor edge to its channel range. 
       # E.x. (in_channels_per_joint = 2)
       # adj = {0: [0,1], 1: [0]} 
       # expanded_adj_list = {0: [0,1,2,3], 1: [0,1]}

       self.expanded_adj_list = [
            [k * in_channels_per_joint + i 
            for k in neighbor 
            for i in range(in_channels_per_joint)]
            for neighbor in self.adj
       ]

    def _init_weights(self):
        """ 
        After we have mapped each neighbor edge to its channel range, we assign ones in the mask at
        those positions to mark valid connections. This mask, zeros out weights not connected via
        adjacency list. 

        Kaiming uniform initialization to prevent exploding / vanishing gradients.
        Ensures variance stability among all weights and bias.         
        """
        for edge, nbors in enumerate(self.expanded_adj_list):
            start_idx = edge * self.out_channels_per_joint
            end_idx = (edge + 1) * self.out_channels_per_joint

            self.mask[start_idx:end_idx, nbors, ...] = 1

            tmp = torch.zeros_like(self.weight[start_idx:end_idx, nbors, ...])
            nn.init.kaiming_uniform_(tmp, a=math.sqrt(5))
            self.weight[start_idx:end_idx, nbors, ...] = tmp
        
        self.weight = nn.Parameter(self.weight)
        self.mask = nn.Parameter(self.mask, requires_grad=False)

        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in)
            nn.init.uniform_(self.bias, -bound, bound)
            self.bias = nn.Parameter(self.bias)
