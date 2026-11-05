"""
PAN configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import torch

from motion_retargeting.data.dataset import MotionDataset, FeatureStats
from motion_retargeting.modules.decoders import MotionDecoderConfig, ReferenceMotionDecoderConfig
from motion_retargeting.modules.discriminators import MotionDiscriminatorConfig, ReferenceMotionDiscriminatorConfig
from motion_retargeting.modules.encoders import (
    BodyPartEncoderConfig,
    MotionEncoderConfig,
    ReferenceMotionEncoderConfig,
)

from .kinematics import end_effector_ids, path_heights
from .masks import humanoid_chains

if TYPE_CHECKING:
    from .data import ReferenceMotionDataset

# Value of `method.params.recipe` selecting the reference-Mixamo computation.
LEGACY_RECIPE = "legacy"
REFERENCE_RECIPE = "mixamo_reference"


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


@dataclass(frozen=True)
class PANReferenceDomainSpec:
    """ One domain of the reference recipe: its topology and skeleton banks, validated on construction. """
    parents: torch.Tensor  # [J], root first with parent -1
    offsets: torch.Tensor  # [K, J, 3], metres
    mean: torch.Tensor     # [K, 4E + 3]
    std: torch.Tensor      # [K, 4E + 3]

    def __post_init__(self):
        humanoid_chains(self.parents)

    @property
    def chains(self) -> List[List[int]]:
        return humanoid_chains(self.parents)

    @property
    def num_parts(self) -> int:
        return len(self.chains)

    @property
    def num_joints(self) -> int:
        return self.parents.shape[0]

    @property
    def num_tokens(self) -> int:
        """ Network tokens: E edges + the root-position token, which is also J. """
        return self.num_joints

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    @property
    def ee_ids(self) -> List[int]:
        return end_effector_ids(self.chains)

    @property
    def heights(self) -> torch.Tensor:
        """ [K], from the validated chains' endpoints. """
        return path_heights(self.offsets, self.parents.tolist(), self.chains)

    @classmethod
    def from_dataset(cls, dataset: ReferenceMotionDataset) -> PANReferenceDomainSpec:
        return cls(parents=dataset.parents, offsets=dataset.offsets, mean=dataset.mean, std=dataset.std)


@dataclass
class PANReferenceConfig:
    recipe: str = REFERENCE_RECIPE

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
    contact_threshold: float = 0.008  # Generated, normalized, scaled end-effector velocity norm

    # Log each loss term's gradient norm every N train steps (0 disables; one extra backward per term)
    loss_grad_norm_interval: int = 0

    def __post_init__(self):
        if self.recipe != REFERENCE_RECIPE:
            raise ValueError(f"PANReferenceConfig only runs recipe '{REFERENCE_RECIPE}', got '{self.recipe}'")


@dataclass
class PANReferenceModelConfig:
    motion_encoder: ReferenceMotionEncoderConfig = field(default_factory=ReferenceMotionEncoderConfig)
    motion_decoder: ReferenceMotionDecoderConfig = field(default_factory=ReferenceMotionDecoderConfig)
    motion_discriminator: ReferenceMotionDiscriminatorConfig = field(default_factory=ReferenceMotionDiscriminatorConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> PANReferenceModelConfig:
        """ Build from the `method.model` section of the YAML config. """
        return cls(
            motion_encoder=ReferenceMotionEncoderConfig(**cfg.get("motion_encoder", {})),
            motion_decoder=ReferenceMotionDecoderConfig(**cfg.get("motion_decoder", {})),
            motion_discriminator=ReferenceMotionDiscriminatorConfig(**cfg.get("motion_discriminator", {})),
        )
