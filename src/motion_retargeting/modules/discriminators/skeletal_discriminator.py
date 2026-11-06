"""
Skeleton-aware discriminator, scoring per-region and per-frame realism on the skeleton graph.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch
from torch import nn

from ..skeletal import PoolingInfo, SkeletalConv, SkeletalConvConfig, SkeletalPooling


@dataclass
class SkeletalDiscriminatorBlockConfig:
    conv_params: SkeletalConvConfig
    norm: bool = True
    activation: bool = True

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SkeletalDiscriminatorBlockConfig:
        params = dict(cfg)
        return cls(conv_params=SkeletalConvConfig(**params.pop("conv_params")), **params)


@dataclass
class SkeletalDiscriminatorConfig:
    block1: SkeletalDiscriminatorBlockConfig = field(default_factory=lambda: SkeletalDiscriminatorBlockConfig(
        conv_params=SkeletalConvConfig(
            in_channels_per_joint=3,
            out_channels_per_joint=6,
            bias=False,
            stride=2,
        ),
    ))
    block2: SkeletalDiscriminatorBlockConfig = field(default_factory=lambda: SkeletalDiscriminatorBlockConfig(
        conv_params=SkeletalConvConfig(
            in_channels_per_joint=6,
            out_channels_per_joint=12,
            kernel_size=16,
            stride=2,
            padding=0,
        ),
        norm=False,
        activation=False,
    ))
    # Train each block's convolution mask and pooling matrix too, as the SAN reference does by accident.
    train_structure: bool = False

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SkeletalDiscriminatorConfig:
        return cls(
            block1=SkeletalDiscriminatorBlockConfig.from_dict(cfg["block1"]),
            block2=SkeletalDiscriminatorBlockConfig.from_dict(cfg["block2"]),
            train_structure=cfg.get("train_structure", False),
        )


class SkeletalDiscBlock(nn.Module):
    """ SkeletalConv -> Optional[BatchNorm1d] -> SkeletalPooling -> Optional[LeakyReLU] """

    def __init__(
        self,
        adj_list: List[List[int]],
        edge_list: List[List[int]],
        conv_params: Dict[str, Any],
        norm: bool = True,
        activation: bool = True,
    ):
        super().__init__()

        channels_per_edge = conv_params["out_channels_per_joint"]

        self.conv = SkeletalConv(adj_list, **conv_params)
        self.norm = nn.BatchNorm1d(channels_per_edge * len(adj_list)) if norm else nn.Identity()
        self.pool = SkeletalPooling(edge_list, channels_per_edge)
        self.activation = nn.LeakyReLU(negative_slope=0.2) if activation else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.pool(self.norm(self.conv(x))))


class SkeletalDiscriminator(nn.Module):
    def __init__(
        self,
        pooling_hierarchy: List[PoolingInfo],
        block1: Dict[str, Any],
        block2: Dict[str, Any],
        train_structure: bool = False,
    ):
        super().__init__()

        self.block1 = SkeletalDiscBlock(
            pooling_hierarchy[0].adj_list,
            pooling_hierarchy[0].edge_list,
            **block1,
        )
        self.block2 = SkeletalDiscBlock(
            pooling_hierarchy[1].adj_list,
            pooling_hierarchy[1].edge_list,
            **block2,
        )

        if train_structure:
            for block in (self.block1, self.block2):
                for module, name in ((block.conv, "mask"), (block.pool, "weight")):
                    value = getattr(module, name)
                    delattr(module, name)
                    module.register_parameter(name, nn.Parameter(value))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, T, rows * channels] -> realism in [0, 1], [B, pooled rows * channels, T_ds] """
        return torch.sigmoid(self.block2(self.block1(x.transpose(1, 2))))
