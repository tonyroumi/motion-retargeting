from __future__ import annotations

from typing import Dict, Tuple
from dataclasses import dataclass

import torch


@dataclass
class PairOutputs:
    """ One source -> destination route (same-domain routes included). """
    target_ids: torch.Tensor  # [B], the destination bank each sample is retargeted onto
    fake: torch.Tensor        # [B, T, 4J + 4], normalized with the target's statistics
    fake_code: torch.Tensor   # fake re-encoded by the destination's motion encoder
    cycle: torch.Tensor       # [B, T, 4J + 4], decoded back onto the source's own banks


@dataclass
class GeneratorOutputs:
    motion_codes: Dict[str, torch.Tensor]
    reconstructions: Dict[str, torch.Tensor]
    pairs: Dict[Tuple[str, str], PairOutputs]


@dataclass
class SourceState:
    """ One domain's source batch and its reconstruction, in the spaces the losses use. """
    bank_ids: torch.Tensor
    motion: torch.Tensor
    denormalized: torch.Tensor
    bones: torch.Tensor  # bone_vectors_with_absolute_root, detached
    reconstruction: torch.Tensor
    reconstruction_denormalized: torch.Tensor
    reconstruction_bones: torch.Tensor


@dataclass
class PairState:
    target_ids: torch.Tensor
    fake_bones: torch.Tensor   # With the target bank's statistics and offsets
    cycle_bones: torch.Tensor  # With the source bank's statistics and offsets


@dataclass
class ForwardState:
    """ One generator forward and its kinematics; both optimizer steps of a train step reuse it. """
    outputs: GeneratorOutputs
    sources: Dict[str, SourceState]
    pairs: Dict[Tuple[str, str], PairState]


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
