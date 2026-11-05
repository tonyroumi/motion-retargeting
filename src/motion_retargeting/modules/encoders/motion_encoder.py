from dataclasses import dataclass, field
import math
from typing import List, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

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


@dataclass
class ReferenceMotionEncoderConfig:
    embed_dim: int = 32
    embed_hidden_dim: int = 512
    num_layers: int = 2
    num_heads: int = 2
    feedforward_dim: int = 256
    dropout: float = 0.2
    part_token_std: float = 0.1
    kernel_size: int = 15
    positional_encoding_basis: float = 10000.0
    positional_encoding_max_len: int = 5000


class MaskedConv1d(nn.Conv1d):
    """
    Reflect-padded 1D conv over `connectivity` [out_channels, in_channels] (bool). The mask is a
    persistent buffer, so a checkpoint carries the connectivity its weights were trained with.
    """

    def __init__(self, connectivity: torch.Tensor, kernel_size: int, stride: int = 1):
        out_channels, in_channels = connectivity.shape
        super().__init__(in_channels, out_channels, kernel_size, stride=stride)
        self.reflect_padding = (kernel_size - 1) // 2
        self.register_buffer("mask", connectivity[..., None].to(self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        padded = F.pad(x, (self.reflect_padding, self.reflect_padding), mode="reflect")
        return F.conv1d(padded, self.weight * self.mask, self.bias, self.stride)


class ReferenceMotionEncoder(nn.Module):
    """
    PAN reference-Mixamo encoder over edge rotations + absolute root position:
    embed (with scale, spatial encoding, dropout) -> prepend part tokens -> post-norm Transformer
    under a directed mask -> two stride-2 temporal convs per part, the first summed with a masked
    stride-2 shortcut from the raw input.
    """

    def __init__(
        self,
        num_parts: int,
        attention_mask: torch.Tensor,
        shortcut_connectivity: torch.Tensor,
        part_connectivity: torch.Tensor,
        token_dim: int = 4,
        embed_dim: int = 32,
        embed_hidden_dim: int = 512,
        num_layers: int = 2,
        num_heads: int = 2,
        feedforward_dim: int = 256,
        dropout: float = 0.2,
        part_token_std: float = 0.1,
        kernel_size: int = 15,
        positional_encoding_basis: float = 10000.0,
        positional_encoding_max_len: int = 5000,
    ):
        super().__init__()

        self.num_parts = num_parts
        self.token_dim = token_dim
        self.embed_dim = embed_dim

        self.register_buffer("attention_mask", attention_mask)

        self.embedding = MLP(
            in_channels=token_dim,
            out_channels=embed_dim,
            hidden_layers=[embed_hidden_dim],
            activation="relu",
        )
        self.positional_encoding = PositionalEncoding(
            d=embed_dim,
            basis=positional_encoding_basis,
            max_len=positional_encoding_max_len,
        )
        self.dropout = nn.Dropout(dropout)

        # [P, 1, D]: sequence-first, like the transformer below. Part tokens get no positional encoding.
        self.part_tokens = nn.Parameter(torch.randn(num_parts, 1, embed_dim) * part_token_std)

        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(
                d_model=embed_dim,
                nhead=num_heads,
                dim_feedforward=feedforward_dim,
                dropout=dropout,
                activation="gelu",
            ),
            num_layers=num_layers,
            enable_nested_tensor=False,
        )

        self.conv_shortcut = MaskedConv1d(shortcut_connectivity, kernel_size, stride=2)
        self.conv1 = MaskedConv1d(part_connectivity, kernel_size, stride=2)
        self.conv2 = MaskedConv1d(part_connectivity, kernel_size, stride=2)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, 4E + 3]; a zero pad completes the root token inside the network.
        returns: [B, T_ds, P, embed_dim]
        """
        B, T, _ = x.shape

        padded = torch.cat([x, x.new_zeros(B, T, 1)], dim=-1)  # [B, T, 4(E + 1)]
        raw = padded.transpose(1, 2)  # [B, 4(E + 1), T]

        tokens = padded.reshape(B, T, -1, self.token_dim)  # [B, T, E + 1, 4]
        tokens = self.embedding(tokens) * math.sqrt(self.embed_dim)
        tokens = self.dropout(self.positional_encoding(tokens))

        tokens = tokens.permute(2, 0, 1, 3).flatten(1, 2)  # [E + 1, B * T, D]
        sequence = torch.cat([self.part_tokens.expand(-1, B * T, -1), tokens], dim=0)
        sequence = self.transformer(sequence, mask=self.attention_mask)

        parts = sequence[:self.num_parts].transpose(0, 1).reshape(B, T, -1).transpose(1, 2)  # [B, P * D, T]

        h = self.activation(self.conv_shortcut(raw) + self.conv1(parts))
        h = self.activation(self.conv2(h))

        return h.transpose(1, 2).reshape(B, -1, self.num_parts, self.embed_dim)
