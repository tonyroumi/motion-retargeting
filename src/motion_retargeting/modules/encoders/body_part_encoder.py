from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn
import torch.nn.functional as F

from ..mlp import MLP


@dataclass
class BodyPartEncoderConfig:
    embed_dim: int = 64
    hidden_dims: List[int] = field(default_factory=lambda: [64, 64])
    activation: str = "relu"


class BodyPartEncoder(nn.Module):
    """
    Each body part has its own independent encoder (3 linear layers by
    default) over the flattened per-joint features of just the joints
    belonging to that body, so bodies with different joint counts each get
    an appropriately-sized input. The per-body embeddings are stacked along
    the body dimension, in body-key order, to produce a [B, N, d] tensor.
    """

    def __init__(
        self,
        input_dim: int,
        embed_dim: int,
        body_joints: Optional[Dict[int, List[int]]],
        hidden_dims: Optional[Sequence[int]] = None,
        activation: str = "relu",
    ):
        super().__init__()

        self.num_bodies = len(body_joints)
        # Index k of the output is body key k, the position build_joint_mask gives that body's
        # token, whatever order the dict iterates in.
        if sorted(body_joints) != list(range(self.num_bodies)):
            raise ValueError(f"Body keys must be 0..{self.num_bodies - 1}, got {sorted(body_joints)}")
        self.body_joints = [body_joints[body] for body in range(self.num_bodies)]

        hidden_dims = hidden_dims if hidden_dims is not None else (embed_dim, embed_dim)

        self.encoders = nn.ModuleList(
            [
                MLP(
                    in_channels=len(joints)*input_dim,
                    out_channels=embed_dim,
                    hidden_layers=list(hidden_dims),
                    activation=activation,
                )
                for joints in self.body_joints
            ]
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, J, input_dim] per-joint features (e.g. offsets)
        returns: [B, N, embed_dim] where N == num_bodies
        """
        batch_size = x.shape[0]

        return torch.stack(
            [
                encoder(x[:, joints, :].reshape(batch_size, -1))
                for joints, encoder in zip(self.body_joints, self.encoders)
            ],
            dim=1,
        )


class MaskedLinear(nn.Linear):
    """ Linear over `connectivity` [out_features, in_features] (bool), a persistent buffer. """

    def __init__(self, connectivity: torch.Tensor):
        out_features, in_features = connectivity.shape
        super().__init__(in_features, out_features)
        self.register_buffer("mask", connectivity.to(self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class ReferenceBodyPartEncoder(nn.Module):
    """
    PAN reference-Mixamo skeleton encoder: three masked linear layers over the flattened offsets of all
    joints, each followed by LeakyReLU(0.2), in per-part blocks. What each part reads is fixed by
    `input_connectivity`; see the reference's mask quirks in `methods/pan/masks.py`.
    """

    def __init__(self, num_parts: int, input_connectivity: torch.Tensor, part_connectivity: torch.Tensor):
        super().__init__()

        self.num_parts = num_parts
        self.linear1 = MaskedLinear(input_connectivity)
        self.linear2 = MaskedLinear(part_connectivity)
        self.linear3 = MaskedLinear(part_connectivity)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        offsets: [K, J, 3], root included
        returns: [K, P, D]
        """
        x = self.activation(self.linear1(offsets.flatten(1)))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        return x.reshape(x.shape[0], self.num_parts, -1)
