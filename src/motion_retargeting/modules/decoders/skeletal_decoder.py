"""
Skeleton-aware decoder, restoring the temporal and skeletal resolution of a `SkeletalEncoder`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import torch
from torch import nn

from ..skeletal import PoolingInfo, SkeletalConv, SkeletalConvConfig, SkeletalUnpool


@dataclass
class SkeletalDecoderBlockConfig:
    channels_per_edge: int
    conv_params: SkeletalConvConfig
    activation: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SkeletalDecoderBlockConfig:
        params = dict(cfg)
        return cls(conv_params=SkeletalConvConfig(**params.pop("conv_params")), **params)


@dataclass
class SkeletalDecoderConfig:
    block1: SkeletalDecoderBlockConfig = field(default_factory=lambda: SkeletalDecoderBlockConfig(
        channels_per_edge=16,
        conv_params=SkeletalConvConfig(
            in_channels_per_joint=16,
            out_channels_per_joint=8,
            offset_in_channels_per_joint=6,
        ),
    ))
    block2: SkeletalDecoderBlockConfig = field(default_factory=lambda: SkeletalDecoderBlockConfig(
        channels_per_edge=8,
        conv_params=SkeletalConvConfig(
            in_channels_per_joint=8,
            out_channels_per_joint=4,
            offset_in_channels_per_joint=3,
        ),
        activation=False,
    ))

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SkeletalDecoderConfig:
        return cls(
            block1=SkeletalDecoderBlockConfig.from_dict(cfg["block1"]),
            block2=SkeletalDecoderBlockConfig.from_dict(cfg["block2"]),
        )


class SkeletalDecBlock(nn.Module):
    """ Upsample -> SkeletalUnpool -> SkeletalConv -> Optional[LeakyReLU] """

    def __init__(
        self,
        adj_list: List[List[int]],
        pooled_edges: List[List[int]],
        channels_per_edge: int,
        conv_params: Dict[str, Any],
        activation: bool = True,
    ):
        super().__init__()

        self.upsample = nn.Upsample(scale_factor=2, mode="linear", align_corners=False)
        self.unpool = SkeletalUnpool(pooled_edges, channels_per_edge)
        self.conv = SkeletalConv(adj_list, **conv_params)
        self.activation = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, x: torch.Tensor, offset: Optional[torch.Tensor] = None) -> torch.Tensor:
        y = self.unpool(self.upsample(x))
        return self.activation(self.conv(y, offset=offset))


class SkeletalDecoder(nn.Module):
    def __init__(
        self,
        pooling_hierarchy: List[PoolingInfo],
        block1: Dict[str, Any],
        block2: Dict[str, Any],
    ):
        super().__init__()

        self.block1 = SkeletalDecBlock(
            pooling_hierarchy[1].adj_list,
            pooling_hierarchy[2].pooled_edges,
            **block1,
        )
        self.block2 = SkeletalDecBlock(
            pooling_hierarchy[0].adj_list,
            pooling_hierarchy[1].pooled_edges,
            **block2,
        )

    def forward(self, x: torch.Tensor, offset: Optional[List[torch.Tensor]] = None) -> torch.Tensor:
        """
        x:      [B, pooled rows * channels, T_ds]
        offset: static features per block, as returned by a `return_all` encoder
        """
        y = self.block1(x, offset=offset[1] if offset else None)
        return self.block2(y, offset=offset[0] if offset else None)
