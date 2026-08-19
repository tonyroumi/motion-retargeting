from .encoder import PoolingInfo

from omegaconf import DictConfig
from typing import List, Optional
import torch
import torch.nn as nn

from src.models.ops import SkeletalConv, SkeletalUnpool

class SkeletalDecBlock(nn.Module):
    """
    One skeletal decoder block : UpS -> SkeletalUnpool -> SkeletalConv -> Optional[LeakyReLU]
    """
    def __init__(
        self,
        adj_list: List[List[int]],
        pooled_edges: List[List[int]],
        channels_per_edge: int,
        conv_params: DictConfig,
        activation: bool = True
    ):
        super().__init__()

        self.UpS = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.unpool = SkeletalUnpool(pooled_edges=pooled_edges, channels_per_edge=channels_per_edge) 
        self.conv = SkeletalConv(adj_list=adj_list, **conv_params)
        self.act = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):      
        y = self.unpool(self.UpS(x))
        y = self.conv(y, offset)
        y = self.act(y)
        return y

class SkeletalDecoder(nn.Module):
    def __init__(
        self,
        pooled_info: List[PoolingInfo],
        params: DictConfig
    ):
        super().__init__()

        self.pooled_info = pooled_info
        self.params = params

        self._init_blocks()
    
    def _init_blocks(self):
        self.block1 = SkeletalDecBlock(
            adj_list=self.pooled_info[1].adj_list,
            pooled_edges=self.pooled_info[2].pooled_edges,
            **(self.params["block1"]))

        self.block2 = SkeletalDecBlock(
            adj_list=self.pooled_info[0].adj_list,
            pooled_edges=self.pooled_info[1].pooled_edges,
            **(self.params["block2"]))
        
    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None):
        y = self.block1(x, offset=offset[1] if offset else None)
        y = self.block2(y, offset=offset[0] if offset else None)
        return y[:, :-1, :] # Discard padded global row
