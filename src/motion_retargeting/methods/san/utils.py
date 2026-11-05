"""
SAN helpers: generator outputs and kinematic utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, NamedTuple

import torch

from motion_retargeting.kinematics.quaternions import quat_apply, quat_mul

from .config import DomainSpec


@dataclass
class GeneratorOutputs:
    # Main predictions
    group_ab: torch.Tensor
    group_ba: torch.Tensor

    # Reconstructions
    group_aa: torch.Tensor
    group_bb: torch.Tensor

    # Latent representations
    motion_code_a: torch.Tensor
    motion_code_b: torch.Tensor
    motion_code_aa: torch.Tensor  # group_aa re-encoded by group_a's motion encoder
    motion_code_ab: torch.Tensor  # group_ab re-encoded by group_b's motion encoder
    motion_code_ba: torch.Tensor  # group_ba re-encoded by group_a's motion encoder
    motion_code_bb: torch.Tensor  # group_bb re-encoded by group_b's motion encoder


class DomainOutputs(NamedTuple):
    """ The generator outputs one domain's losses are computed from (decoder layout for motions). """
    reconstruction: torch.Tensor    # d -> d
    retargeted_into: torch.Tensor   # other -> d
    motion_code: torch.Tensor       # E_d(d)
    reconstructed_code: torch.Tensor  # E_d(d -> d)
    retargeted_code: torch.Tensor   # E_other(d -> other)


class DomainPositions(NamedTuple):
    """ Global joint positions [B, W, J, 3] on one domain's skeleton. """
    real: torch.Tensor              # d's input motion
    reconstruction: torch.Tensor    # d -> d
    retargeted_into: torch.Tensor   # other -> d, judged by d's discriminator


def split_by_domain(outputs: GeneratorOutputs) -> Dict[str, DomainOutputs]:
    return {
        "group_a": DomainOutputs(
            reconstruction=outputs.group_aa,
            retargeted_into=outputs.group_ba,
            motion_code=outputs.motion_code_a,
            reconstructed_code=outputs.motion_code_aa,
            retargeted_code=outputs.motion_code_ab,
        ),
        "group_b": DomainOutputs(
            reconstruction=outputs.group_bb,
            retargeted_into=outputs.group_ab,
            motion_code=outputs.motion_code_b,
            reconstructed_code=outputs.motion_code_bb,
            retargeted_code=outputs.motion_code_ba,
        ),
    }


def restore_motion(
    rotations: torch.Tensor,
    root_vel: torch.Tensor,
    q_heading: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """ Put the window's heading [B, W, 4] back on the root rotation and root velocity. """
    rotations = rotations.clone()

    rotations[..., 0, :] = quat_mul(q_heading, rotations[..., 0, :])

    return rotations, quat_apply(q_heading, root_vel)


def denormalize(motion: torch.Tensor, spec: DomainSpec) -> torch.Tensor:
    """ Undo the domain's z-score normalization of network-layout motion [..., J+1, 4]. """
    motion[..., -1, :-1] = motion[..., -1, :-1] * spec.vel_stats.std.to(motion) + spec.vel_stats.mean.to(motion)
    motion[..., -1, -1] = motion[..., -1, -1] * spec.ang_vel_stats.std.to(motion) + spec.ang_vel_stats.mean.to(motion)
    return motion


def total(losses: Dict[str, torch.Tensor], total_name: str) -> tuple[torch.Tensor, Dict[str, float]]:
    """ Sum the loss terms; return the total and the scalar metrics (terms + total). """
    loss = sum(losses.values())
    info = {name: value.item() for name, value in losses.items()}
    info[total_name] = loss.item()
    return loss, info


def skeleton_height(offsets: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """ Longest root-to-joint bone chain per skeleton: offsets [B, J, 3] -> [B]. """
    bone_lengths = offsets.norm(dim=-1)

    chains = [bone_lengths[:, 0]]
    for joint in range(1, offsets.shape[1]):
        chains.append(chains[parents[joint].item()] + bone_lengths[:, joint])

    return torch.stack(chains, dim=1).amax(dim=1)


def end_effector_velocity(positions: torch.Tensor, end_effectors: List[int]) -> torch.Tensor:
    """ Per-frame displacement of the end-effectors: positions [B, W, J, 3] -> [B, W-1, E, 3]. """
    return positions[:, 1:, end_effectors] - positions[:, :-1, end_effectors]


def mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """ Mean per-joint position error: mean Euclidean distance over [..., J, 3]. """
    return (predicted - target).norm(dim=-1).mean().item()


def side_by_side(
    left: torch.Tensor,
    left_parents: torch.Tensor,
    right: torch.Tensor,
    right_parents: torch.Tensor,
    gap: float = 0.5,
) -> tuple[torch.Tensor, list[int]]:
    """
    Merge two skeleton animations [T, J1, 3] and [T, J2, 3] into one [T, J1+J2, 3] with a
    combined parent list, shifting `right` along X so it sits `gap` meters beyond `left`.
    """
    shift = left[..., 0].max() - right[..., 0].min() + gap
    right = right + right.new_tensor([shift, 0.0, 0.0])

    num_left = left.shape[-2]
    parents = left_parents.tolist() + [p + num_left if p >= 0 else -1 for p in right_parents.tolist()]
    return torch.cat([left, right], dim=-2), parents
