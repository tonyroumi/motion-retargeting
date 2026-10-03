from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import torch
from torch import nn

from ..mlp import MLP
from ..positional_encoding import PositionalEncoding
from ..utils import get_activation


@dataclass
class MotionEncoderConfig:
    embed_dim: int = 64
    attn_num_heads: int = 4
    embed_hidden_dims: List[int] = field(default_factory=lambda: [256, 256])
    conv_hidden_dim: int = 32
    kernel_size: int = 15
    stride: int = 2
    padding: int = 7
    activation: str = "relu"
    positional_encoding_basis: float = 10000.0
    positional_encoding_max_len: int = 5000


class MotionEncoder(nn.Module):
    """
    Embed (3 linear layers) -> add positional encoding -> prepend learnable
    body tokens -> two self-attention layers -> two 1D conv layers.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        num_bodies: int,
        joint_body_mask: torch.Tensor,
        attn_num_heads: int = 4,
        embed_hidden_dims: Optional[Sequence[int]] = [256, 256],
        conv_hidden_dim: int = 32,
        kernel_size: int = 15,
        stride: int = 2,
        padding: int = 7,
        activation: str = "relu",
        positional_encoding_basis: float = 10000.0,
        positional_encoding_max_len: int = 5000,
    ):
        super().__init__()

        self.num_bodies = num_bodies

        self.register_buffer("joint_body_mask", joint_body_mask, persistent=False)

        self.embedding = MLP(
            in_channels=input_dim,
            out_channels=embed_dim,
            hidden_layers=list(embed_hidden_dims),
            activation=activation,
        )

        self.positional_encoding = PositionalEncoding(
            d=embed_dim,
            basis=positional_encoding_basis,
            max_len=positional_encoding_max_len,
        )

        self.body_tokens = nn.Parameter(torch.randn(1, num_bodies, embed_dim))

        self.attention1 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=attn_num_heads, batch_first=True
        )
        self.attention2 = nn.MultiheadAttention(
            embed_dim=embed_dim, num_heads=attn_num_heads, batch_first=True
        )

        self.temporal_convs1 = nn.ModuleList([
            nn.Conv1d(
                in_channels=embed_dim,
                out_channels=conv_hidden_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            for _ in range(num_bodies)
        ])
        self.temporal_convs2 = nn.ModuleList([
            nn.Conv1d(
                in_channels=conv_hidden_dim,
                out_channels=embed_dim,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
            )
            for _ in range(num_bodies)
        ])
        self.activation = get_activation(activation)

    def apply_temporal_convs(self, x, convs):
        outputs = [
            conv(x[:, i, :, :])
            for i, conv in enumerate(convs)
        ]
        return torch.stack(outputs, dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x:    [B, T, J, 4]
        mask: optional attention mask for the post-prepend sequence length
              (num_bodies + T). Boolean (True = blocked) or additive float,
              broadcastable against [B, attn_num_heads, L, L].
        """
        B, W, J, _ = x.shape

        x = self.embedding(x)
        x = self.positional_encoding(x)

        D = x.shape[-1]

        body_tokens = self.body_tokens.expand(B, W, -1, -1)
        x = torch.cat([body_tokens, x], dim=2)  # [B, T, num_bodies + J, embed_dim]

        x = x.reshape(B * W, J + self.num_bodies, -1)  # [B*T, num_bodies + J, embed_dim]

        x, _ = self.attention1(x, x, x, attn_mask=self.joint_body_mask)
        x, _ = self.attention2(x, x, x, attn_mask=self.joint_body_mask)

        x = x.reshape(B, W, J + self.num_bodies, -1)  # [B, T, num_bodies + J, embed_dim]

        # Keep only body tokens
        body_tokens = x[:, :, :self.num_bodies, :] 

        body_tokens = body_tokens.permute(0, 2, 3, 1) # [B, num_bodies, D, T]

        body_tokens = self.activation(
            self.apply_temporal_convs(body_tokens, self.temporal_convs1)
        )

        body_tokens = self.activation(
            self.apply_temporal_convs(body_tokens, self.temporal_convs2)
        )

        return body_tokens.permute(0, 3, 1, 2) # [B, T_ds, num_bodies, D]
