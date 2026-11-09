from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import torch

from motion_retargeting.modules.decoders import MotionDecoderConfig
from motion_retargeting.modules.discriminators import MotionDiscriminatorConfig
from motion_retargeting.modules.encoders import MotionEncoderConfig

from .kinematics import end_effector_ids, path_heights
from .masks import labeled_parts

if TYPE_CHECKING:
    from .data import PANMotionDataset


@dataclass(frozen=True)
class DomainSpec:
    """ One domain: its topology and skeleton banks, validated on construction. """
    parents: torch.Tensor  # [J], root first with parent -1
    offsets: torch.Tensor  # [K, J, 3], metres
    mean: torch.Tensor     # [K, 4J + 4]
    std: torch.Tensor      # [K, 4J + 4]
    body_joints: Dict[int, List[int]]  # the topology's manually labeled body parts

    def __post_init__(self):
        labeled_parts(self.body_joints, self.num_joints)
        expected = (self.num_banks, 4 * self.num_tokens)
        if self.mean.shape != expected or self.std.shape != expected:
            raise ValueError(f"PAN joint statistics must have shape {expected}; rebuild legacy edge data")

    @property
    def parts(self) -> List[List[int]]:
        return labeled_parts(self.body_joints, self.num_joints)

    @property
    def num_parts(self) -> int:
        return len(self.parts)

    @property
    def num_joints(self) -> int:
        return self.parents.shape[0]

    @property
    def num_tokens(self) -> int:
        """ Network tokens: J local joint quaternions plus the root-velocity token. """
        return self.num_joints + 1

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    @property
    def ee_ids(self) -> List[int]:
        return end_effector_ids(self.parts, self.parents.tolist(), self.offsets)

    @property
    def heights(self) -> torch.Tensor:
        """ [K], from the left foot and head end effectors. """
        return path_heights(self.offsets, self.parents.tolist(), self.ee_ids)

    @classmethod
    def from_dataset(cls, dataset: PANMotionDataset) -> DomainSpec:
        return cls(
            parents=dataset.parents,
            offsets=dataset.offsets,
            mean=dataset.mean,
            std=dataset.std,
            body_joints=dataset.body_joints,
        )


@dataclass
class PANConfig:
    # Discriminator replay buffer, holding discriminator-space positions
    replay_buffer_size: int = 50
    swap_probability: float = 0.5

    # Objective: every outer domain / pair aggregation is a sum
    reconstruction_lambda: float = 1.0
    root_lambda: float = 100.0
    position_lambda: float = 100.0
    cycle_lambda: float = 2.5
    adversarial_lambda: float = 1.0
    end_effector_lambda: float = 50.0
    contact_lambda: float = 100.0  # On top of end_effector_lambda
    contact_threshold: float = 0.008  # Source, normalized, scaled end-effector velocity norm

    # Log each loss term's gradient norm every N train steps (0 disables; one extra backward per term)
    loss_grad_norm_interval: int = 0


@dataclass
class PANModelConfig:
    motion_encoder: MotionEncoderConfig = field(default_factory=MotionEncoderConfig)
    motion_decoder: MotionDecoderConfig = field(default_factory=MotionDecoderConfig)
    motion_discriminator: MotionDiscriminatorConfig = field(default_factory=MotionDiscriminatorConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> PANModelConfig:
        """ Build from the `method.model` section of the YAML config. """
        return cls(
            motion_encoder=MotionEncoderConfig(**cfg.get("motion_encoder", {})),
            motion_decoder=MotionDecoderConfig(**cfg.get("motion_decoder", {})),
            motion_discriminator=MotionDiscriminatorConfig(**cfg.get("motion_discriminator", {})),
        )
