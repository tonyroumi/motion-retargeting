from src.models.ops import SkeletalConv, SkeletalPooling, PoolingInfo

from omegaconf import DictConfig
from typing import Dict, List, Any
import torch
import torch.nn as nn

class SkeletalDiscBlock(nn.Module):
    """
    One skeletal discriminator block : SkeletalConv -> BatchNorm1D -> AvgPooling -> LeakyReLU
    """
    def __init__(
        self, 
        adj_list: List[List], 
        edge_list: List[List], 
        conv_params: DictConfig, 
        norm: bool = True, 
        activation: bool = True
    ):
        super().__init__()

        self.conv = SkeletalConv(adj_list, **conv_params)

        out_channels = conv_params["out_channels_per_joint"]*len(adj_list)
        self.norm = nn.BatchNorm1d(out_channels)  if norm else nn.Identity()

        self.pool = SkeletalPooling(edge_list, conv_params["out_channels_per_joint"])
        self.act = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, x: torch.Tensor):
        y = self.norm(self.conv(x))
        y = self.act(self.pool(y))
        return y

class SkeletalDiscriminator(nn.Module):
    def __init__(
        self,
        pooled_info: List[PoolingInfo],
        discriminator_params: DictConfig
    ):
        super().__init__()

        self.pooled_info = pooled_info
        self.discriminator_params = discriminator_params

        self._init_blocks()
    
    def _init_blocks(self):
        self.block1 = SkeletalDiscBlock(
            adj_list=self.pooled_info[0].adj_list,
            edge_list=self.pooled_info[0].edge_list,
            **(self.discriminator_params["block1"]))

        self.block2 = SkeletalDiscBlock(
            adj_list=self.pooled_info[1].adj_list,
            edge_list=self.pooled_info[1].edge_list,
            **(self.discriminator_params["block2"]))
    
    def forward(self, x: torch.Tensor):
        y = self.block1(x.transpose(1,2))
        y = self.block2(y)

        return torch.sigmoid(y).squeeze()
