from dataclasses import dataclass, field
from typing import Dict, List, Optional, Sequence

import torch
from torch import nn

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
    the body dimension to produce a [B, N, d] tensor.
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
        self.body_joints = list(body_joints.values())

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
