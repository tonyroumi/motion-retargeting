"""
SAN configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from motion_retargeting.modules.decoders import SkeletalDecoderConfig
from motion_retargeting.modules.discriminators import SkeletalDiscriminatorConfig
from motion_retargeting.modules.encoders import SkeletalEncoderBlockConfig, SkeletalEncoderConfig
from motion_retargeting.modules.skeletal import SkeletalConvConfig, build_edge_list

from .kinematics import end_effector_ids, humanoid_chains, skeleton_height

if TYPE_CHECKING:
    from .data import SANMotionDataset


@dataclass(frozen=True)
class DomainSpec:
    """ One domain's topology and skeleton banks, validated on construction. """
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
    def edge_list(self) -> List[List[int]]:
        """ Graph row e is edge e, and the final row is the root-position token. """
        return build_edge_list(self.parents.tolist())

    @property
    def end_effectors(self) -> List[int]:
        """ Left foot tip, right foot tip, head tip, left hand, right hand. """
        return end_effector_ids(self.chains)

    @property
    def heights(self) -> torch.Tensor:
        """ [K], left foot tip to root plus head tip to root. """
        return skeleton_height(self.offsets, self.parents.tolist(), self.chains)

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    @classmethod
    def from_dataset(cls, dataset: SANMotionDataset) -> DomainSpec:
        return cls(parents=dataset.parents, offsets=dataset.offsets, mean=dataset.mean, std=dataset.std)


@dataclass
class SANConfig:
    # Discriminator replay buffer, holding bone-space positions
    replay_buffer_size: int = 50
    swap_probability: float = 0.5

    # Adam betas of both optimizers
    betas: Tuple[float, float] = (0.9, 0.999)

    # Loss weights. The root / joint position terms are also scaled by the reconstruction weight.
    reconstruction_lambda: float = 5.0
    position_lambda: float = 100.0
    root_position_lambda: float = 2.5
    cycle_consistency_lambda: float = 2.5
    end_effector_lambda: float = 50.0
    adversarial_lambda: float = 1.0

    # End-effector velocity loss: generated frames slower than the threshold (scaled skeleton heights
    # per frame) are treated as contacts, and weighted by contact_lambda on top of the base term.
    contact_lambda: float = 100.0
    contact_velocity_threshold: float = 0.008

    # Log each loss term's gradient norm every N train steps (0 disables; one extra backward per term)
    loss_grad_norm_interval: int = 0


def default_offset_encoder() -> SkeletalEncoderConfig:
    """ Static offset branch: per-frame (kernel 1) convolutions over a [B, rows * 3, 1] input. """
    return SkeletalEncoderConfig(
        block1=SkeletalEncoderBlockConfig(
            conv_params=SkeletalConvConfig(
                in_channels_per_joint=3,
                out_channels_per_joint=6,
                kernel_size=1,
                padding=0,
                padding_mode="constant",
            ),
        ),
        block2=SkeletalEncoderBlockConfig(
            conv_params=SkeletalConvConfig(
                in_channels_per_joint=6,
                out_channels_per_joint=12,
                kernel_size=1,
                padding=0,
                padding_mode="constant",
            ),
            pool_features=False,
        ),
    )


@dataclass
class SANModelConfig:
    motion_encoder: SkeletalEncoderConfig = field(default_factory=SkeletalEncoderConfig)
    offset_encoder: SkeletalEncoderConfig = field(default_factory=default_offset_encoder)
    motion_decoder: SkeletalDecoderConfig = field(default_factory=SkeletalDecoderConfig)
    motion_discriminator: SkeletalDiscriminatorConfig = field(default_factory=SkeletalDiscriminatorConfig)

    @classmethod
    def from_dict(cls, cfg: Dict[str, Any]) -> SANModelConfig:
        """ Build from the `method.model` section of the YAML config; missing sections keep their defaults. """
        sections = {
            "motion_encoder": SkeletalEncoderConfig,
            "offset_encoder": SkeletalEncoderConfig,
            "motion_decoder": SkeletalDecoderConfig,
            "motion_discriminator": SkeletalDiscriminatorConfig,
        }
        return cls(**{name: section.from_dict(cfg[name]) for name, section in sections.items() if name in cfg})
