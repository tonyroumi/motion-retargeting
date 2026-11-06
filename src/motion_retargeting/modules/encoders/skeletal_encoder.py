"""
Skeleton-aware encoder, reducing features on the skeleton graph towards its primal skeleton.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..skeletal import PoolingInfo, SkeletalConv, SkeletalConvConfig, SkeletalPooling


@dataclass
class SkeletalEncoderBlockConfig:
    conv_params: SkeletalConvConfig
    pool_features: bool = True
    last_pool: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SkeletalEncoderBlockConfig:
        params = dict(cfg)
        return cls(conv_params=SkeletalConvConfig(**params.pop("conv_params")), **params)


@dataclass
class SkeletalEncoderConfig:
    block1: SkeletalEncoderBlockConfig = field(default_factory=lambda: SkeletalEncoderBlockConfig(
        conv_params=SkeletalConvConfig(
            in_channels_per_joint=4,
            out_channels_per_joint=8,
            offset_in_channels_per_joint=3,
            stride=2,
        ),
    ))
    block2: SkeletalEncoderBlockConfig = field(default_factory=lambda: SkeletalEncoderBlockConfig(
        conv_params=SkeletalConvConfig(
            in_channels_per_joint=8,
            out_channels_per_joint=16,
            offset_in_channels_per_joint=6,
            stride=2,
        ),
        last_pool=True,
    ))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SkeletalEncoderConfig:
        return cls(
            block1=SkeletalEncoderBlockConfig.from_dict(cfg["block1"]),
            block2=SkeletalEncoderBlockConfig.from_dict(cfg["block2"]),
        )


class SkeletalEncBlock(nn.Module):
    """ SkeletalConv -> Optional[SkeletalPooling] -> LeakyReLU """

    def __init__(
        self,
        adj_list: List[List[int]],
        edge_list: List[List[int]],
        conv_params: Dict[str, Any],
        pool_features: bool = True,
        last_pool: bool = False,
    ):
        super().__init__()

        self.conv = SkeletalConv(adj_list, **conv_params)
        # The pooled graph is needed even when the block's own features are not pooled.
        self.pool = SkeletalPooling(
            edge_list,
            channels_per_edge=conv_params["out_channels_per_joint"],
            last_pool=last_pool,
        )
        self.activation = nn.LeakyReLU(negative_slope=0.2)

        self.pooling_info = self.pool.pooled_info
        self.pool_features = pool_features

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.conv(x, offset=offset)
        if self.pool_features:
            y = self.pool(y)
        return self.activation(y)


class SkeletalEncoder(nn.Module):
    def __init__(
        self,
        adj_list: List[List[int]],
        edge_list: List[List[int]],
        block1: Dict[str, Any],
        block2: Dict[str, Any],
        return_all: bool = False,
    ):
        super().__init__()

        self.return_all = return_all

        self.block1 = SkeletalEncBlock(adj_list, edge_list, **block1)
        self.block2 = SkeletalEncBlock(
            self.block1.pooling_info.adj_list,
            self.block1.pooling_info.edge_list,
            **block2,
        )

        self.pooling_hierarchy = [
            PoolingInfo(adj_list, edge_list),
            self.block1.pooling_info,
            self.block2.pooling_info,
        ]

    def forward(
        self,
        x: torch.Tensor,
        offset: Optional[List[torch.Tensor]] = None,
    ) -> torch.Tensor | List[torch.Tensor]:
        """
        x:      [B, rows * channels, T]
        offset: static features of each block, as returned by a `return_all` encoder
        """
        y1 = self.block1(x, offset=offset[0] if offset else None)
        y2 = self.block2(y1, offset=offset[1] if offset else None)

        return [x, y1, y2] if self.return_all else y2
