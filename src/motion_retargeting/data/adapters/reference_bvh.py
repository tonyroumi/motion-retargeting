"""Shared BVH conversion for the PAN reference datasets, not their network tensors."""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import warnings

import torch

from motion_retargeting.data.representations.motion import MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.kinematics.forward import ForwardKinematics
from motion_retargeting.kinematics.quaternions import quat_apply, quat_conjugate, quat_mul

from .base import MotionSourceAdapter
from .bandai import flatten_hierarchy, keyframe_positions, rest_offsets


class ReferenceBVHAdapter(MotionSourceAdapter):
    """Y-up, centimeter BVHs with a fixed named topology and noninteractive body labels.

    Keep joint-local rotations (world-relative at the root), all source frames and
    the file's frame rate. Heading, resampling, velocities and windows belong to
    MotionDatasetBuilder. Anonymous End Sites are not rotation-bearing joints.

    The builder reads heading from the root's local +X, so an identity root must be an
    upright character whose +X stays horizontal. `root_rest` is the fixed rotation that
    moves a file's root frame there; see `rebase_root`.
    """

    joint_names: List[str]
    body_names: Dict[int, List[str]]
    root_rest: Optional[Tuple[float, float, float, float]] = None

    def _files(self) -> List[Path]:
        files = sorted(self.raw_dir.rglob("*.bvh"))
        # The reference LAFAN/dog lists contain stems, one per line. Optional for
        # small demos or trees already containing only the desired training clips.
        manifest = self.raw_dir / "train.txt"
        if manifest.exists():
            wanted = {Path(line.strip()).stem for line in manifest.read_text().splitlines() if line.strip()}
            available = {path.stem for path in files}
            missing = wanted - available
            if missing:
                raise FileNotFoundError(f"{manifest}: missing clips: {sorted(missing)}")
            files = [path for path in files if path.stem in wanted]
        else:
            warnings.warn(f"{self.dataset_name}: no {manifest}; using every BVH, not the reference train split", stacklevel=2)
        return files

    def _extract(self) -> List[MotionData]:
        from tqdm import tqdm

        return [self._read_clip(path) for path in tqdm(self._files(), desc=f"Extracting {self.dataset_name}")]

    def _read_clip(self, path: Path) -> MotionData:
        import bvhio

        data = bvhio.readAsBvh(str(path))
        if data.FrameCount < 1 or not math.isfinite(data.FrameTime) or data.FrameTime <= 0:
            raise ValueError(f"{path}: expected frames and a positive finite Frame Time")
        joints, parents = flatten_hierarchy(data.Root)
        names = [joint.Name.rsplit(":", 1)[-1] for joint in joints]
        if len(set(names)) != len(names):
            raise ValueError(f"{path}: duplicate joint names after removing namespaces")
        missing = set(self.joint_names) - set(names)
        if missing:
            raise ValueError(f"{path}: missing required joints: {sorted(missing)}")

        offsets = rest_offsets(joints)
        rotations = torch.tensor(
            [[list(pose.Rotation) for pose in joint.Keyframes] for joint in joints],
            dtype=torch.float32,
        ).transpose(0, 1)
        # bvhio respects each joint's intrinsic CHANNELS order and iterates wxyz.
        # Position channels replace OFFSET, as in the reference BVH readers; do
        # not add the root OFFSET a second time (the demos have nonzero offsets).
        root_position = keyframe_positions(joints[0]) * 0.01
        for joint in joints[1:]:
            positions = keyframe_positions(joint)
            rest = positions.new_tensor(list(joint.Offset))
            if not torch.allclose(positions, rest.expand_as(positions), atol=1e-4, rtol=1e-5):
                raise ValueError(f"{path}: non-root translation at {joint.Name} cannot use fixed offsets")
        if not all(torch.isfinite(tensor).all() for tensor in (offsets, rotations, root_position)):
            raise ValueError(f"{path}: non-finite motion values")

        selected = [names.index(name) for name in self.joint_names]
        parents, offsets, rotations, root_position = select_joints(
            parents, offsets, rotations, root_position, selected, path
        )
        if self.root_rest is not None:
            rotations, offsets = rebase_root(rotations, offsets, parents, self.root_rest)

        # A real heading axis stays horizontal; a file authored in another frame would feed
        # the builder near-vertical noise here (crawling clips dip low, so this only warns).
        facing = quat_apply(rotations[:, 0], rotations.new_tensor([1.0, 0.0, 0.0]).expand(rotations.shape[0], 3))
        if facing[:, [0, 2]].norm(dim=-1).median() < 0.5:
            warnings.warn(f"{path}: root +X is mostly vertical, so the builder's heading is unreliable", stacklevel=2)

        if self._topology is None:
            self._topology = SkeletonTopology(
                parents=torch.tensor(parents, dtype=torch.long),
                joint_names=list(self.joint_names),
                body_joints={key: [self.joint_names.index(name) for name in part]
                             for key, part in self.body_names.items()},
            )

        return self._construct_motion(
            parents, offsets, rotations, root_position,
            float(round(1.0 / data.FrameTime)),  # FrameTime is rounded: 0.0333 or 0.033333 at 30 Hz
            list(self.joint_names),
        )


def rebase_root(rotations, offsets, parents, rest):
    """
    Move the fixed rotation `rest` out of the root's frame and into its children, which
    leaves every world joint position and every other joint's rotation unchanged:

        q_root' = q_root * rest^-1,  q_child' = rest * q_child,  offset_child' = rest * offset_child
    """
    rotations = rotations.clone()
    offsets = offsets.clone()
    rest = rotations.new_tensor(rest)

    rotations[:, 0] = quat_mul(rotations[:, 0], quat_conjugate(rest))
    for joint, parent in enumerate(parents):
        if parent == 0:
            rotations[:, joint] = quat_mul(rest, rotations[:, joint])
            offsets[joint] = quat_apply(rest, offsets[joint])
    return rotations, offsets


def select_joints(parents, offsets, rotations, root_position, selected, path):
    """Prune details/re-root without turning joint rotations into edge rotations.

    Normal retained edges keep their exact local data. A reversed or collapsed
    edge is representable only if its new parent-relative offset is constant;
    reject moving offsets rather than silently changing the motion's FK.
    """
    neighbors = [[] for _ in parents]
    for joint, parent in enumerate(parents[1:], 1):
        neighbors[parent].append(joint)
        neighbors[joint].append(parent)
    index = {joint: i for i, joint in enumerate(selected)}
    new_parents = [-1] * len(selected)

    def visit(joint, previous, retained_parent):
        if joint in index:
            new_parents[index[joint]] = retained_parent
            retained_parent = index[joint]
        for neighbor in neighbors[joint]:
            if neighbor != previous:
                visit(neighbor, joint, retained_parent)

    visit(selected[0], -1, -1)
    if any(parent < 0 or parent >= i for i, parent in enumerate(new_parents[1:], 1)):
        raise ValueError(f"{path}: selected joint order is not parent-before-child")

    new_offsets = offsets[selected].clone()
    new_offsets[0] = 0
    new_rotations = rotations[:, selected].clone()
    changed = [i for i in range(1, len(selected))
               if parents[selected[i]] != selected[new_parents[i]]]
    if selected[0] != 0 or changed:
        # FK from a zero root: float32 error at large translations must not look like a moving offset.
        batched_offsets = offsets.expand(rotations.shape[0], -1, -1)
        positions, global_q = ForwardKinematics(torch.tensor(parents), batched_offsets)(
            rotations, torch.zeros_like(root_position)
        )
        root_position = root_position + positions[:, selected[0]]
        new_rotations[:, 0] = global_q[:, selected[0]]
        for i in changed:
            joint, parent = selected[i], selected[new_parents[i]]
            inverse = quat_conjugate(global_q[:, parent])
            local_offset = quat_apply(inverse, positions[:, joint] - positions[:, parent])
            if not torch.allclose(local_offset, local_offset[:1].expand_as(local_offset), atol=1e-5, rtol=1e-4):
                raise ValueError(f"{path}: pruning/re-rooting joint {joint} would create a moving offset")
            new_offsets[i] = local_offset[0]
            new_rotations[:, i] = quat_mul(inverse, global_q[:, joint])
    return new_parents, new_offsets, new_rotations, root_position
