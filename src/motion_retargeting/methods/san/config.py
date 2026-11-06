"""
SAN configuration.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, List, Tuple

import torch

from motion_retargeting.methods.pan.kinematics import end_effector_ids, path_heights
from motion_retargeting.methods.pan.masks import humanoid_chains
from motion_retargeting.modules.decoders import SkeletalDecoderConfig
from motion_retargeting.modules.discriminators import SkeletalDiscriminatorConfig
from motion_retargeting.modules.encoders import SkeletalEncoderBlockConfig, SkeletalEncoderConfig
from motion_retargeting.modules.skeletal import SkeletalConvConfig, build_edge_list, build_reference_edge_list

if TYPE_CHECKING:
    from motion_retargeting.data.dataset import FeatureStats, MotionDataset
    from motion_retargeting.methods.pan.data import ReferenceMotionDataset

# Value of `method.params.recipe` selecting the reference computation; the same strings PAN uses.
LEGACY_RECIPE = "legacy"
REFERENCE_RECIPE = "mixamo_reference"


@dataclass(frozen=True)
class DomainSpec:
    """ Skeleton-dependent facts needed to size one domain's modules and run its kinematics. """
    parents: torch.Tensor
    num_joints: int  # J, excluding the appended global velocity row

    vel_stats: FeatureStats | None = None
    ang_vel_stats: FeatureStats | None = None

    @property
    def edge_list(self) -> List[List[int]]:
        """ Graph edge j is motion row j. Together with the global row, the graph has J + 1 rows. """
        return build_edge_list(self.parents.tolist())

    @property
    def end_effectors(self) -> List[int]:
        """ Leaf joints of the skeleton, in joint order. """
        internal = set(self.parents.tolist())
        return [joint for joint in range(self.num_joints) if joint not in internal]

    @classmethod
    def from_dataset(cls, dataset: MotionDataset) -> DomainSpec:
        # offsets: [N, J, 3]
        return cls(
            parents=dataset.parents,
            num_joints=dataset.offsets.shape[1],
            vel_stats=dataset.vel_stats,
            ang_vel_stats=dataset.ang_vel_stats,
        )


@dataclass
class SANConfig:
    # Discriminator replay buffer
    replay_buffer_size: int = 50
    swap_probability: float = 0.5

    # Adam betas of both optimizers
    betas: Tuple[float, float] = (0.09, 0.99)

    # Loss weights. The root / joint position terms are also scaled by the reconstruction weight.
    reconstruction_lambda: float = 5.0
    position_lambda: float = 100.0
    root_position_lambda: float = 2.5
    cycle_consistency_lambda: float = 2.5
    end_effector_lambda: float = 1.0
    adversarial_lambda: float = 50.0

    # End-effector velocity loss: frames slower than the threshold (skeleton heights per frame)
    # are treated as contacts, and weighted by contact_lambda on top of the base term.
    contact_lambda: float = 100.0
    contact_velocity_threshold: float = 0.008

    # Log each loss term's gradient norm every N train steps (0 disables; one extra backward per term)
    loss_grad_norm_interval: int = 0

    # This class and SANMethod are the legacy recipe; see REFERENCE_RECIPE for the other one.
    recipe: str = LEGACY_RECIPE

    def __post_init__(self):
        if self.recipe != LEGACY_RECIPE:
            raise ValueError(f"Unknown SAN recipe '{self.recipe}'; expected '{LEGACY_RECIPE}' or '{REFERENCE_RECIPE}'")


@dataclass(frozen=True)
class SANReferenceDomainSpec:
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
    def ee_ids(self) -> List[int]:
        return end_effector_ids(self.chains)

    @property
    def heights(self) -> torch.Tensor:
        """ [K], left foot tip to root plus head tip to root. """
        return path_heights(self.offsets, self.parents.tolist(), self.chains)

    @property
    def edge_list(self) -> List[List[int]]:
        """ Graph row e is edge e, and the final row is the root-position token. """
        return build_reference_edge_list(self.parents.tolist())

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    @classmethod
    def from_dataset(cls, dataset: ReferenceMotionDataset) -> SANReferenceDomainSpec:
        return cls(parents=dataset.parents, offsets=dataset.offsets, mean=dataset.mean, std=dataset.std)


@dataclass
class SANReferenceConfig:
    recipe: str = REFERENCE_RECIPE

    # Discriminator replay buffer, holding bone-space positions
    replay_buffer_size: int = 50
    swap_probability: float = 0.5

    # Objective: every outer domain / pair aggregation is a sum
    reconstruction_lambda: float = 5.0
    root_lambda: float = 1250.0
    position_lambda: float = 500.0
    latent_lambda: float = 2.5
    end_effector_lambda: float = 50.0
    contact_lambda: float = 100.0  # On top of end_effector_lambda
    contact_threshold: float = 0.008  # Generated, normalized, scaled end-effector velocity norm
    adversarial_lambda: float = 1.0

    # Log each loss term's gradient norm every N train steps (0 disables; one extra backward per term)
    loss_grad_norm_interval: int = 0

    def __post_init__(self):
        if self.recipe != REFERENCE_RECIPE:
            raise ValueError(f"SANReferenceConfig only runs recipe '{REFERENCE_RECIPE}', got '{self.recipe}'")


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
