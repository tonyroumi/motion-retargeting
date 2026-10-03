"""
PAN configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List

import torch

from motion_retargeting.data.dataset import MotionDataset, FeatureStats
from motion_retargeting.modules.decoders import MotionDecoderConfig
from motion_retargeting.modules.discriminators import MotionDiscriminatorConfig
from motion_retargeting.modules.encoders import BodyPartEncoderConfig, MotionEncoderConfig


@dataclass(frozen=True)
class DomainSpec:
    """ Skeleton-dependent facts needed to size one domain's modules. """
    body_joints: Dict[int, List[int]]
    parents: torch.Tensor
    num_joints: int  # J, excluding the appended global velocity row
    rotation_dim: int = 4
    offset_dim: int = 3

    vel_stats: FeatureStats = field(default_factory=FeatureStats)
    ang_vel_stats: FeatureStats = field(default_factory=FeatureStats)

    @property
    def num_bodies(self) -> int:
        return len(self.body_joints)

    @property
    def motion_dim(self) -> int:
        """ Flattened per-frame motion features: (J + 1) * 4. """
        return (self.num_joints + 1) * self.rotation_dim

    @property
    def root_body(self) -> int:
        return next(
            (body_idx for body_idx, joints in self.body_joints.items() if 0 in joints),
            None,
        )

    @classmethod
    def from_dataset(cls, dataset: MotionDataset) -> DomainSpec:
        # offsets: [N, J, 3]
        return cls(
            body_joints=dataset.body_joints,
            parents=dataset.parents,
            num_joints=dataset.offsets.shape[1],
            vel_stats=dataset.vel_stats,
            ang_vel_stats=dataset.ang_vel_stats,
        )


@dataclass 
class PANConfig:
    # Discriminator replay buffer
    replay_buffer_size: int = 50
    swap_probability: float = 0.5

    # Loss weights
    reconstruction_lambda: int = 1
    cycle_consistency_lambda: int = 2.5
    kinematic_lambda: int = 100
    adversarial_lambda: int = 1

    # Discriminator regularization (0 disables)
    disc_logit_reg: float = 0.0
    disc_grad_penalty: float = 0.0
    disc_weight_decay: float = 0.0
    grad_norm_clip: float = 0.0

    # Log each loss term's gradient norm every N train steps (0 disables; one extra backward per term)
    loss_grad_norm_interval: int = 0


@dataclass
class PANModelConfig:
    motion_encoder: MotionEncoderConfig = field(default_factory=MotionEncoderConfig)
    body_part_encoder: BodyPartEncoderConfig = field(default_factory=BodyPartEncoderConfig)
    motion_decoder: MotionDecoderConfig = field(default_factory=MotionDecoderConfig)
    motion_discriminator: MotionDiscriminatorConfig = field(default_factory=MotionDiscriminatorConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> PANModelConfig:
        """ Build from the `method.model` section of the YAML config. """
        return cls(
            motion_encoder=MotionEncoderConfig(**cfg.get("motion_encoder", {})),
            body_part_encoder=BodyPartEncoderConfig(**cfg.get("body_part_encoder", {})),
            motion_decoder=MotionDecoderConfig(**cfg.get("motion_decoder", {})),
            motion_discriminator=MotionDiscriminatorConfig(**cfg.get("motion_discriminator", {})),
        )
