"""
Edge forward kinematics of the PAN reference-Mixamo recipe.

Predicted edge rotations are applied independently: copies of a branching joint's rotation on its
outgoing edges may differ, which is why this is not the generic joint `ForwardKinematics`. The bone
ending at joint i is rotated by the accumulated transform including i's own edge rotation.

Reference losses use three distinct position spaces, named here rather than folded into one "global
positions" helper:
  - bone_vectors_with_absolute_root:  root entry absolute; every other entry its own bone vector
  - root_relative_with_absolute_root: root entry absolute; every other entry relative to the root
  - world_positions:                  every entry in world coordinates
"""

from __future__ import annotations

from typing import List, Sequence

import torch

from motion_retargeting.kinematics.quaternions import quat_apply, quat_mul, quat_normalize

# The reference's scale on end-effector frame displacement.
END_EFFECTOR_VELOCITY_SCALE = 10.0


class EdgeForwardKinematics:
    def __init__(self, parents: Sequence[int]):
        self.parents = [int(parent) for parent in parents]

    def bone_vectors_with_absolute_root(self, motion: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        """
        motion:  [B, T, 4E + 3], denormalized
        offsets: [B, J, 3]
        returns: [B, T, J, 3]
        """
        B, T, _ = motion.shape
        rotations = quat_normalize(motion[..., :-3].reshape(B, T, -1, 4))  # [B, T, E, 4]
        offsets = offsets[:, None].expand(-1, T, -1, -1)

        accumulated = [motion.new_tensor([1.0, 0.0, 0.0, 0.0]).expand(B, T, 4)]
        vectors = [motion[..., -3:]]
        for joint in range(1, len(self.parents)):
            rotation = quat_mul(accumulated[self.parents[joint]], rotations[:, :, joint - 1])
            accumulated.append(rotation)
            vectors.append(quat_apply(rotation, offsets[:, :, joint]))
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


def end_effector_ids(chains: Sequence[Sequence[int]]) -> List[int]:
    """ Chain endpoints, ordered left foot tip, right foot tip, head tip, left hand, right hand. """
    return [chains[part][-1] for part in (0, 1, 3, 4, 5)]


def end_effector_velocity(world_positions: torch.Tensor, ee_ids: Sequence[int]) -> torch.Tensor:
    """ Scaled world displacement between frames: [B, T, J, 3] -> [B, T - 1, 5, 3]. """
    end_effectors = world_positions[:, :, list(ee_ids)]
    return (end_effectors[:, 1:] - end_effectors[:, :-1]) * END_EFFECTOR_VELOCITY_SCALE


def path_heights(offsets: torch.Tensor, parents: Sequence[int], chains: Sequence[Sequence[int]]) -> torch.Tensor:
    """
    The reference's skeleton height per bank, [K, J, 3] -> [K]: bone lengths from the left foot tip to
    the root plus from the head tip to the root. Neither a vertical extent nor legacy `rest_height`.
    """
    lengths = torch.linalg.vector_norm(offsets, dim=-1)  # [K, J]
    height = torch.zeros(offsets.shape[0], dtype=offsets.dtype, device=offsets.device)
    for end in (chains[0][-1], chains[3][-1]):
        joint = end
        while joint != 0:
            height = height + lengths[:, joint]
            joint = int(parents[joint])
    return height
