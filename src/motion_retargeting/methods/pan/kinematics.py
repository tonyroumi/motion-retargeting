"""
PAN's joint forward kinematics, following the LAFAN1/dog quaternion hierarchy.

The bone ending at joint i is rotated by its parent's accumulated rotation. Each joint has one
local quaternion, shared by all its outgoing bones; a leaf rotation does not move its own position.
Root heading and world-frame displacement retain this implementation's existing conventions.

The losses use three distinct position spaces, named here rather than folded into one "global
positions" helper:
  - bone_vectors_with_absolute_root:  root entry its position, integrated from the root velocity from
                                      the window's first frame; every other entry its own bone vector
  - root_relative_with_absolute_root: root entry absolute; every other entry relative to the root
  - world_positions:                  every entry in world coordinates
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from motion_retargeting.kinematics.quaternions import quat_apply, quat_mul, quat_normalize


END_EFFECTOR_VELOCITY_SCALE = 10.0


class JointForwardKinematics:
    def __init__(self, parents: Sequence[int]):
        self.parents = [int(parent) for parent in parents]

    def bone_vectors_with_absolute_root(self, motion: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        motion:  [B, T, 4J + 4], denormalized; the trailing 4 channels are the root velocity (XYZ, yaw rate)
        offsets: [B, J, 3]
        returns: [B, T, J, 3]
        """
        B, T, _ = motion.shape
        if motion.shape[-1] != 4 * len(self.parents) + 4:
            raise ValueError("PAN joint motion must have 4J + 4 channels; rebuild legacy edge data")
        rotations = quat_normalize(motion[..., :-4].reshape(B, T, len(self.parents), 4))
        offsets = offsets[:, None].expand(-1, T, -1, -1)

        accumulated = [rotations[:, :, 0]]
        # Frame t carries the step t -> t + 1, so the position at t is the sum of the earlier steps; the
        # first frame is the origin. The yaw rate is not needed, the heading being in the root rotation.
        linear_velocity = motion[..., -4:-1]
        vectors = [torch.cumsum(linear_velocity, dim=1) - linear_velocity]
        for joint in range(1, len(self.parents)):
            parent_rotation = accumulated[self.parents[joint]]
            vectors.append(quat_apply(parent_rotation, offsets[:, :, joint]))
            accumulated.append(quat_mul(parent_rotation, rotations[:, :, joint]))
        return torch.stack(vectors, dim=-2)

    def root_relative_with_absolute_root(self, bone_vectors: torch.Tensor) -> torch.Tensor:
        """ [B, T, J, 3] -> [B, T, J, 3]; bones leaving the root stay as they are. """
        positions = [bone_vectors[..., 0, :]]
        for joint in range(1, len(self.parents)):
            parent = self.parents[joint]
            vector = bone_vectors[..., joint, :]
            positions.append(vector if parent == 0 else positions[parent] + vector)
        return torch.stack(positions, dim=-2)

    def world_positions(self, bone_vectors: torch.Tensor) -> torch.Tensor:
        """ [B, T, J, 3] -> [B, T, J, 3] """
        positions = [bone_vectors[..., 0, :]]
        for joint in range(1, len(self.parents)):
            positions.append(positions[self.parents[joint]] + bone_vectors[..., joint, :])
        return torch.stack(positions, dim=-2)


def end_effector_ids(parts: Sequence[Sequence[int]], parents: Sequence[int], offsets: torch.Tensor) -> List[int]:
    """
    Part endpoints, ordered left foot tip, right foot tip, head tip, left hand, right hand.

    Part keys follow the labeler's click order, not a fixed anatomy (the cached AMASS / CMU labels are
    arms, head, trunk, legs; the Mixamo groups legs, spine, head, arms), so roles come from the Y-up rest
    pose of the mean skeleton, offsets [K, J, 3]: the part holding the root is the trunk; of the other five,
    the two lowest endpoints are the feet, the highest the head, the remaining two the hands. Within each
    pair the lower key is the left side, as in every supported labeling and as the key-matched latent
    slots assume.
    """
    mean_offsets = offsets.mean(dim=0)
    rest = [torch.zeros_like(mean_offsets[0])]
    for joint in range(1, len(parents)):
        rest.append(rest[parents[joint]] + mean_offsets[joint])

    limbs = [index for index, part in enumerate(parts) if 0 not in part]
    limbs.sort(key=lambda index: rest[parts[index][-1]][1].item())
    feet, hands, head = sorted(limbs[:2]), sorted(limbs[2:4]), limbs[4]
    return [parts[index][-1] for index in (*feet, head, *hands)]


def end_effector_velocity(world_positions: torch.Tensor, ee_ids: Sequence[int]) -> torch.Tensor:
    """ Scaled world displacement between frames: [B, T, J, 3] -> [B, T - 1, 5, 3]. """
    end_effectors = world_positions[:, :, list(ee_ids)]
    return (end_effectors[:, 1:] - end_effectors[:, :-1]) * END_EFFECTOR_VELOCITY_SCALE


def path_heights(offsets: torch.Tensor, parents: Sequence[int], ee_ids: Sequence[int]) -> torch.Tensor:
    """
    The official code's skeleton height per bank, [K, J, 3] -> [K]: bone lengths from the left foot tip to
    the root plus from the head tip to the root (`end_effector_ids` slots 0 and 2). Not a vertical extent.
    """
    lengths = torch.linalg.vector_norm(offsets, dim=-1)  # [K, J]
    height = torch.zeros(offsets.shape[0], dtype=offsets.dtype, device=offsets.device)
    for end in (ee_ids[0], ee_ids[2]):
        joint = end
        while joint != 0:
            height = height + lengths[:, joint]
            joint = int(parents[joint])
    return height
