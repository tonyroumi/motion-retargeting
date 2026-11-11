"""
Adapter for the BANDAI-Namco motion dataset, stored as BVH files (Y-up, Z-forward,
centimeters, one skeleton per clip with identical joint topology).

Two properties of the raw files need canonicalizing, so that the clips match the
convention of the other adapters (joint 0 is the moving root, and an identity root
rotation is an upright character):
  - The BVH root `joint_Root` is a rotation-free ground anchor. It is dropped; `Hips`
    becomes the root, with its world position as the root position.
  - `Hips` is authored with its local X axis along the spine. That fixed rotation is
    moved out of the root's rotation and into its children, which leaves every world
    joint position unchanged.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import List, Tuple

import torch

from motion_retargeting.data.representations.motion import MotionData
from motion_retargeting.kinematics.quaternions import quat_apply, quat_conjugate, quat_mul

from .base import MotionSourceAdapter
from .cmu import zforward_to_xforward_quaternion, zforward_to_xforward_vector


CENTIMETERS_TO_METERS = 0.01

# Rotation of an upright `Hips` joint (+90 degrees about Z), in (w, x, y, z).
HIPS_UPRIGHT = (math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4))


class BANDAIAdapter(MotionSourceAdapter):
    def __init__(
        self,
        device: str = "cpu",
    ):
        super().__init__("bandai", device=device)

    def _extract(self) -> List[MotionData]:
        files = sorted(p for p in self.raw_dir.rglob("*.bvh"))

        from tqdm import tqdm

        return [self._read_clip(path) for path in tqdm(files, desc="Extracting BANDAI Data")]

    def _read_clip(self, path: Path) -> MotionData:
        import bvhio

        data = bvhio.readAsBvh(str(path))
        joints, parents = flatten_hierarchy(data.Root.Children[0])
        offsets = rest_offsets(joints)
        joint_rotations = torch.as_tensor(
            [[list(pose.Rotation) for pose in joint.Keyframes] for joint in joints],
            dtype=torch.float32,
        ).transpose(0, 1)
        # `joint_Root` never rotates, so its translation adds to the Hips channels.
        root_position = CENTIMETERS_TO_METERS * (
            keyframe_positions(data.Root) + keyframe_positions(joints[0])
        )
        joint_rotations, offsets = uprighten_root(joint_rotations, offsets, parents)
        joint_rotations[:, 0] = zforward_to_xforward_quaternion(joint_rotations[:, 0])
        return self._construct_motion(
            parents=parents, offsets=offsets, joint_rotations=joint_rotations,
            root_position=zforward_to_xforward_vector(root_position),
            fps=float(round(1.0 / data.FrameTime)), joint_names=[joint.Name for joint in joints],
        )


def flatten_hierarchy(root) -> Tuple[list, List[int]]:
    """ The joints below (and including) a bvhio `root`, depth-first, with their parent indices. """
    joints: list = []
    parents: List[int] = []

    def visit(joint, parent: int) -> None:
        joints.append(joint)
        parents.append(parent)

        index = len(joints) - 1
        for child in joint.Children:
            visit(child, index)

    visit(root, -1)

    return joints, parents


def rest_offsets(joints: list) -> torch.Tensor:
    """ Parent-relative rest-pose offsets [J, 3] in meters (root offset zero). """
    offsets = torch.as_tensor([list(joint.Offset) for joint in joints], dtype=torch.float32)
    offsets[0] = 0.0
    return offsets * CENTIMETERS_TO_METERS


def keyframe_positions(joint) -> torch.Tensor:
    """ A joint's position channels, [T, 3] in the file's centimeters. """
    return torch.as_tensor([list(pose.Position) for pose in joint.Keyframes], dtype=torch.float32)


def uprighten_root(
    joint_rotations: torch.Tensor,
    offsets: torch.Tensor,
    parents: List[int],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Move the fixed `HIPS_UPRIGHT` rotation out of the root and into its children, so that an
    identity root rotation is upright. World joint positions are unchanged:

        q_root' = q_root * A^-1,  q_child' = A * q_child,  offset_child' = A * offset_child

    joint_rotations: [T, J, 4]
    offsets:         [J, 3]
    """
    joint_rotations = joint_rotations.clone()
    offsets = offsets.clone()

    upright = joint_rotations.new_tensor(HIPS_UPRIGHT)

    joint_rotations[:, 0] = quat_mul(joint_rotations[:, 0], quat_conjugate(upright))

    for joint, parent in enumerate(parents):
        if parent != 0:
            continue

        joint_rotations[:, joint] = quat_mul(upright, joint_rotations[:, joint])
        offsets[joint] = quat_apply(upright, offsets[joint])

    return joint_rotations, offsets
