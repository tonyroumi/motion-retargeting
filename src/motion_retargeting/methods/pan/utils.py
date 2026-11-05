from __future__ import annotations

from typing import Dict, NamedTuple, Tuple
from dataclasses import  dataclass

import torch

from motion_retargeting.kinematics.quaternions import quat_apply, quat_mul

from .config import DomainSpec


@dataclass
class GeneratorOutputs:
    # Main predictions
    group_ab: torch.Tensor
    group_ba: torch.Tensor

    # Latent representations
    motion_code_a: torch.Tensor
    motion_code_b: torch.Tensor
    motion_code_ab: torch.Tensor  # group_ab re-encoded by group_b's motion encoder
    motion_code_ba: torch.Tensor  # group_ba re-encoded by group_a's motion encoder
    skeleton_code_a: torch.Tensor
    skeleton_code_b: torch.Tensor

    # Reconstructions / cycles
    group_aa: torch.Tensor
    group_bb: torch.Tensor
    group_aba: torch.Tensor
    group_bab: torch.Tensor 


class DomainOutputs(NamedTuple):
    """ The generator outputs one domain's losses are computed from (decoder layout for motions). """
    reconstruction: torch.Tensor    # d -> d
    cycle: torch.Tensor             # d -> other -> d
    retargeted_into: torch.Tensor   # other -> d, judged by d's discriminator
    motion_code: torch.Tensor       # E_d(d)
    retargeted_code: torch.Tensor   # E_other(d -> other)


@dataclass
class ReferencePairOutputs:
    """ One source -> destination route of the reference recipe (same-domain routes included). """
    target_ids: torch.Tensor  # [B], the destination bank each sample is retargeted onto
    fake: torch.Tensor        # [B, T, 4E + 3], normalized with the target's statistics
    fake_code: torch.Tensor   # fake re-encoded by the destination's motion encoder
    cycle: torch.Tensor       # [B, T, 4E + 3], decoded back onto the source's own banks


@dataclass
class ReferenceGeneratorOutputs:
    motion_codes: Dict[str, torch.Tensor]
    reconstructions: Dict[str, torch.Tensor]
    pairs: Dict[Tuple[str, str], ReferencePairOutputs]


@dataclass
class ReferenceSourceState:
    """ One domain's source batch and its reconstruction, in the spaces the losses use. """
    bank_ids: torch.Tensor
    motion: torch.Tensor
    denormalized: torch.Tensor
    bones: torch.Tensor  # bone_vectors_with_absolute_root, detached
    reconstruction: torch.Tensor
    reconstruction_denormalized: torch.Tensor
    reconstruction_bones: torch.Tensor


@dataclass
class ReferencePairState:
    target_ids: torch.Tensor
    fake_bones: torch.Tensor   # With the target bank's statistics and offsets
    cycle_bones: torch.Tensor  # With the source bank's statistics and offsets


@dataclass
class ReferenceForwardState:
    """ One generator forward and its kinematics; both optimizer steps of a train step reuse it. """
    outputs: ReferenceGeneratorOutputs
    sources: Dict[str, ReferenceSourceState]
    pairs: Dict[Tuple[str, str], ReferencePairState]


def split_by_domain(outputs: GeneratorOutputs) -> Dict[str, DomainOutputs]:
    return {
        "group_a": DomainOutputs(
            reconstruction=outputs.group_aa,
            cycle=outputs.group_aba,
            retargeted_into=outputs.group_ba,
            motion_code=outputs.motion_code_a,
            retargeted_code=outputs.motion_code_ab,
        ),
        "group_b": DomainOutputs(
            reconstruction=outputs.group_bb,
            cycle=outputs.group_bab,
            retargeted_into=outputs.group_ab,
            motion_code=outputs.motion_code_b,
            retargeted_code=outputs.motion_code_ba,
        ),
    }


def restore_motion(
    rotations,
    root_vel,
    q_heading,
):
    rotations = rotations.clone()

    rotations[..., 0, :] = quat_mul(
        q_heading,
        rotations[..., 0, :],
    )

    root_vel = quat_apply(
        q_heading,
        root_vel,
    )

    return rotations, root_vel



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


def require_finite(losses: Dict[str, torch.Tensor]) -> None:
    """ Stop before an optimizer step would apply a NaN or infinite loss to the weights. """
    bad = [name for name, loss in losses.items() if not torch.isfinite(loss)]
    if bad:
        raise RuntimeError(f"Non-finite loss terms, no optimizer stepped on them: {bad}")


def rest_height(offsets: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """ Rest-pose height (extent along the Z up axis) per skeleton: offsets [B, J, 3] -> [B]. """
    positions = [offsets[:, 0]]
    for joint in range(1, offsets.shape[1]):
        positions.append(positions[parents[joint].item()] + offsets[:, joint])
    z = torch.stack(positions, dim=1)[..., 2]
    return z.amax(dim=1) - z.amin(dim=1)


def mpjpe(predicted: torch.Tensor, target: torch.Tensor) -> float:
    """ Mean per-joint position error: mean Euclidean distance over [..., J, 3]. """
    return (predicted - target).norm(dim=-1).mean().item()


def squared_error_over_height2(predicted: torch.Tensor, target: torch.Tensor, heights: torch.Tensor) -> float:
    """ Mean squared coordinate error / height^2: predicted, target [B, ..., J, 3], heights [B]. Not MPJPE. """
    heights = heights.reshape(-1, *[1] * (predicted.ndim - 1))
    return (((predicted - target) / heights) ** 2).mean().item()


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
