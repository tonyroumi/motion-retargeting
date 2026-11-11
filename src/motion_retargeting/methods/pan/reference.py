"""Opt-in LAFAN1/Dog archive contract (original BVH frames, centimetres).

No reference repository imports. Archive loading never accesses raw-adapter caches.
Archive dictionaries are pickled NumPy objects: load only trusted reference archives.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch
from scipy.ndimage import gaussian_filter1d

from motion_retargeting.kinematics.quaternions import quat_conjugate, quat_mul

if TYPE_CHECKING:
    from .data import PANMotionDataset


PARTS = {
    "lafan1": [[3, 4, 7, 8, 1, 2, 5, 6], [9, 10, 11], [12, 13]],
    "dog": [[8, 15, 12, 18, 5, 6, 7, 9, 10, 11, 13, 14, 16, 17], [1, 2], [3, 4]],
}
SELECTED_ENDS = {"lafan1": [5, 10, 16], "dog": [10, 15, 19, 23]}


@dataclass(frozen=True)
class ReferenceData:
    """Additional FK/statistics metadata unavailable in the generic motion collection."""
    kind: str
    parents_withend: torch.Tensor
    offsets_withend: torch.Tensor  # [K, J_fk, 3], cm, root zero (published archives: J_fk=27)
    not_end: torch.Tensor  # rotation-bearing joint -> expanded FK joint
    min_velocity: float
    max_velocity: float
    statistics_sha256: str
    statistics_source: str = "archive"
    protocol: str = "lafan_dog"
    body_parts: list[list[int]] | None = None
    selected_positions: list[int] | None = None

    @property
    def parts(self) -> list[list[int]]:
        return PARTS[self.kind] if self.body_parts is None else self.body_parts

    @property
    def rotation_indices(self) -> list[int]:
        return [0] + [joint for part in self.parts for joint in part]

    @property
    def position_indices(self) -> list[int]:
        if self.selected_positions is not None:
            return self.selected_positions
        return self.not_end[self.rotation_indices].tolist() + SELECTED_ENDS[self.kind]


def _rotate(q: torch.Tensor, vector: torch.Tensor) -> torch.Tensor:
    """Reference preprocessing's quaternion-vector product, without renormalization."""
    xyz, vector = torch.broadcast_tensors(q[..., 1:], vector)
    cross = torch.cross(xyz, vector, dim=-1)
    return vector + 2 * (q[..., :1] * cross + torch.cross(xyz, cross, dim=-1))


def reference_channels(q: torch.Tensor, root_velocity: torch.Tensor, heading: torch.Tensor) -> torch.Tensor:
    """Window tensors [N,T,J,4], [N,T,3], [N,T,1,4] -> [N,T,4J+4].

    Q/V already have heading removed. Match datasetserial's positive-w signs and
    float32 yaw computation, including repetition of the final within-window increment.
    Keep Q/V precision until AFTER normalization (tiny saved toe stds matter).
    """
    if q.ndim != 4 or q.shape[-1] != 4 or q.shape[1] < 2 or heading.shape != (*q.shape[:2], 1, 4):
        raise ValueError("Reference channels need windows of at least two frames and heading [N,T,1,4]")
    if root_velocity.shape != (*q.shape[:2], 3):
        raise ValueError("Reference root velocity must be [N,T,3]")
    q = torch.where(q[..., :1] < 0, -q, q)
    previous, following = quat_conjugate(heading[:, :-1]), heading[:, 1:]
    # The oracle qmultipy casts to float32 after dot/cross products.
    scalar = previous[..., :1] * following[..., :1] - (previous[..., 1:] * following[..., 1:]).sum(-1, keepdim=True)
    vector = (previous[..., :1] * following[..., 1:] + following[..., :1] * previous[..., 1:]
              + torch.cross(previous[..., 1:], following[..., 1:], dim=-1))
    delta = torch.cat((scalar, vector), -1).float()
    forward = _rotate(delta, delta.new_tensor([0., 0., 1.]))
    yaw = torch.atan2(forward[..., 0], forward[..., 2])
    yaw = torch.cat((yaw, yaw[:, -1:]), dim=1)
    return torch.cat((q.flatten(2), root_velocity, yaw), dim=-1)


def reference_window_starts(num_frames: int, window: int = 64, stride: int = 20) -> range:
    """Strict reference endpoint: start + window < num_frames - 1."""
    if window < 2 or stride < 1:
        raise ValueError("Reference windows need length >= 2 and positive stride")
    return range(0, num_frames - window - 1, stride)


def preprocess_reference_window(
    rotations: torch.Tensor,
    root_position: torch.Tensor,
    offsets: torch.Tensor,
    parents: torch.Tensor,
    kind: str,
    forward_axis: torch.Tensor | None = None,
    stable_heading: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Canonicalize one raw window PLUS one lookahead frame, already sampled at 30 Hz.

    CPU, original BVH local frames, wxyz, centimetres; NOT rebased generic adapter
    tensors. Returns channels [W,4J+4] and removed heading [W,1,4]. Dog sigma=8
    smoothing is deliberately per-window, nearest-padded, never clip-wide.
    An explicit root-local forward_axis generalizes the human math to separately
    labeled experiments; the published LAFAN1 axis remains +Y by default.
    """
    if kind not in PARTS:
        raise ValueError(f"Unknown reference domain {kind!r}")
    if rotations.device.type != "cpu" or rotations.shape[0] < 3:
        raise ValueError("Reference preprocessing needs CPU tensors and at least three frames")
    q = rotations[:-1]
    global_q, positions = [q[:, 0]], [torch.zeros_like(root_position[:-1])]
    for joint in range(1, len(parents)):
        parent = int(parents[joint])
        positions.append(positions[parent] + _rotate(global_q[parent], offsets[joint]))
        global_q.append(quat_mul(global_q[parent], q[:, joint]))
    global_q = torch.stack(global_q, dim=1)
    if kind == "lafan1":
        axis = q.new_tensor([0., 1., 0.]) if forward_axis is None else forward_axis.to(q)
        forward = _rotate(global_q[:, 0], axis)
        forward = forward * q.new_tensor([1., 0., 1.])
        forward = forward / (forward.norm(dim=-1, keepdim=True) + 1e-8)
    else:
        forward = positions[2] - positions[0]
        forward = forward * q.new_tensor([1., 0., 1.])
        forward = forward / forward.norm(dim=-1, keepdim=True)
        forward = torch.from_numpy(gaussian_filter1d(forward.numpy(), 8, axis=0, mode="nearest"))
        forward = forward / forward.norm(dim=-1, keepdim=True)
    if stable_heading:
        # The human experiment must handle exact 180-degree headings: the
        # reference quaternion-between formula collapses to zero there. Keep the
        # published path unchanged, but define the human experiment by atan2.
        if kind != "lafan1" or (forward.norm(dim=-1) < 1e-8).any():
            raise ValueError("Stable human heading needs a nonvertical projected forward axis")
        half_yaw = torch.atan2(forward[:, 0], forward[:, 2]) / 2
        zeros = torch.zeros_like(half_yaw)
        heading = torch.stack((half_yaw.cos(), zeros, half_yaw.sin(), zeros), -1)
    else:
        heading = torch.stack((forward.norm(dim=-1) + forward[:, 2], -forward[:, 1],
                               forward[:, 0], torch.zeros_like(forward[:, 0])), dim=-1)
        heading = heading / (heading.norm(dim=-1, keepdim=True) + 1e-8)
    inverse = quat_conjugate(heading[:, None])
    canonical_global = quat_mul(inverse, global_q)
    # Reproduce the reference FK -> remove heading -> IK, including its epsilon
    # normalization effects on non-root quaternions.
    canonical = torch.cat((canonical_global[:, :1], quat_mul(
        quat_conjugate(canonical_global[:, parents[1:]]), canonical_global[:, 1:]
    )), dim=1)
    velocity = _rotate(inverse[:, 0], root_position[1:] - root_position[:-1])
    channels = reference_channels(canonical[None], velocity[None], heading[None, :, None])[0]
    return channels, heading[:, None]


def load_reference_splits(kind: str, directory: str | Path) -> tuple[PANMotionDataset, PANMotionDataset]:
    """Read shipped train/test/stats NPZs; share a bank table including held-out subjects.

    Uses the reference train+test statistics AS SAVED, not a train-only estimate.
    Never accesses raw adapters or their caches.
    """
    from .data import PANMotionDataset

    if kind not in PARTS:
        raise ValueError(f"Unknown reference domain {kind!r}; expected lafan1 or dog")
    prefix = "hum" if kind == "lafan1" else "dog"
    directory = Path(directory).expanduser()
    with np.load(directory / f"{prefix}stats.npz", allow_pickle=False) as stats:
        mean = torch.from_numpy(stats["mean"].copy())
        std = torch.from_numpy(stats["std"].copy())
        parents = torch.from_numpy(stats["parents"].copy()).long()
        parents_withend = torch.from_numpy(stats["parents_withend"].copy()).long()
        not_end = torch.from_numpy(stats["not_end"].copy()).long()
        min_velocity, max_velocity = float(stats["min_vel"]), float(stats["max_vel"])
    joints = 22 if kind == "lafan1" else 21
    if parents.shape != (joints,) or parents_withend.shape != (27,) or not_end.shape != (joints,):
        raise ValueError("Expected reference 22/21 rotation joints and 27 FK joints")
    if mean.shape != (4 * joints + 4,) or std.shape != mean.shape or not (std > 0).all():
        raise ValueError("Invalid reference channel statistics")
    if not torch.isfinite(mean).all() or not torch.isfinite(std).all() or not (0 <= min_velocity < max_velocity < float("inf")):
        raise ValueError("Nonfinite reference statistics or invalid speed range")
    if parents[0] != -1 or parents_withend[0] != -1 or any(
        not 0 <= int(p) < j for tree in (parents, parents_withend) for j, p in enumerate(tree[1:], 1)
    ):
        raise ValueError("Reference FK topology must be root-first, parent-before-child")
    if not_end[0] != 0 or not torch.equal(not_end, not_end.unique(sorted=True)) or not_end[-1] >= 27:
        raise ValueError("Invalid reference rotation-to-FK joint mapping")
    if not torch.equal(parents_withend[not_end[1:]], not_end[parents[1:]]):
        raise ValueError("Reference rotation and End Site hierarchies disagree")

    offsets, expanded_offsets, banks, splits = [], [], {}, []
    for split in ("train", "test"):
        with np.load(directory / f"{prefix}{split}.npz", allow_pickle=True) as archive:
            if not np.array_equal(archive["parents"], parents.numpy()):
                raise ValueError("Archive topology differs from reference statistics")
            q = torch.from_numpy(archive["Q"])
            if q.ndim != 4 or q.shape[1:] != (64, joints, 4):
                raise ValueError("Expected reference 64-frame quaternion windows")
            motion = reference_channels(q, torch.from_numpy(archive["V"][:, :, 0]), torch.from_numpy(archive["yrot"]))
            raw_offsets = archive["skel_offsets"].item()
            raw_expanded = archive["skel_offsets_withend"].item()
            names = archive["skel_names"]
            if len(names) != len(motion):
                raise ValueError("Reference skeleton names and window counts disagree")
            ids = []
            for name in names:
                # Keep differing skeletons even if two archives reuse a subject label.
                regular = np.asarray(raw_offsets[name]).reshape(joints - 1, 3)
                expanded = np.asarray(raw_expanded[name]).reshape(26, 3)
                key = (str(name), regular.tobytes(), expanded.tobytes())
                if key not in banks:
                    banks[key] = len(offsets)
                    offsets.append(torch.cat((torch.zeros(1, 3), torch.from_numpy(regular).float())))
                    expanded_offsets.append(torch.cat((torch.zeros(1, 3), torch.from_numpy(expanded).float())))
                    if not torch.isfinite(offsets[-1]).all() or not torch.isfinite(expanded_offsets[-1]).all():
                        raise ValueError("Nonfinite reference skeleton offsets")
                    if not torch.equal(offsets[-1], expanded_offsets[-1][not_end]):
                        raise ValueError("Reference rotation and End Site skeleton offsets disagree")
                ids.append(banks[key])
            if not torch.isfinite(motion).all():
                raise ValueError("Nonfinite reference motion")
            splits.append((motion, torch.tensor(ids, dtype=torch.long)))
    statistics_sha256 = hashlib.sha256((directory / f"{prefix}stats.npz").read_bytes()).hexdigest()
    reference = ReferenceData(kind, parents_withend, torch.stack(expanded_offsets), not_end,
                              min_velocity, max_velocity, statistics_sha256)
    kwargs = dict(offsets=torch.stack(offsets), mean=mean.expand(len(offsets), -1), std=std.expand(len(offsets), -1),
                  parents=parents, body_joints={i: list(part) for i, part in enumerate(PARTS[kind])}, reference=reference)
    return tuple(PANMotionDataset(motion=motion, bank_ids=ids, **kwargs) for motion, ids in splits)
