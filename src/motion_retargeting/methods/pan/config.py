"""
PAN configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List

import torch

from motion_retargeting.modules.decoders import MotionDecoderConfig
from motion_retargeting.modules.discriminators import MotionDiscriminatorConfig
from motion_retargeting.modules.encoders import MotionEncoderConfig

from .kinematics import end_effector_ids, path_heights
from .masks import humanoid_chains

if TYPE_CHECKING:
    from .data import PANMotionDataset


@dataclass(frozen=True)
class DomainSpec:
    """ One domain: its topology and skeleton banks, validated on construction. """
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
    def from_dataset(cls, dataset: PANMotionDataset) -> DomainSpec:
        return cls(parents=dataset.parents, offsets=dataset.offsets, mean=dataset.mean, std=dataset.std)


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
    contact_threshold: float = 0.008  # Generated, normalized, scaled end-effector velocity norm

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
