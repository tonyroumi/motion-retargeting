from __future__ import annotations

import torch

from .quaternions import quat_normalize, quat_mul, quat_apply

class ForwardKinematics:
    """
    Batched forward kinematics for a tree-structured skeleton.

    Assumptions
    -----------
    - Quaternion convention is (w, x, y, z).
    - `parents[j]` is the parent index of joint j.
    - The root joint has parent index -1.
    - `offsets[j]` is the rest-pose offset from the parent to joint j,
      expressed in the parent's local frame.
    - `local_rotations[j]` is the rotation of joint j relative to its parent.

    Returns
    -------
    global_positions:
        [..., J, 3]

    global_rotations:
        [..., J, 4]
    """

    def __init__(
        self,
        parents: torch.Tensor,
        offsets: torch.Tensor,
    ) -> None:
        if parents.ndim != 1:
            raise ValueError(f"`parents` must have shape [J], got {parents.shape}")

        if offsets.shape[-2:] != (parents.shape[0], 3):
            raise ValueError(
                f"`offsets` must end in [J, 3]. "
                f"Got {offsets.shape} for J={parents.shape[0]}"
            )

        if parents[0].item() != -1:
            raise ValueError("Expected joint 0 to be the root with parent -1.")

        self.parents = parents.long()
        self.offsets = offsets

        self.validate_skeleton()

    def validate_skeleton(self) -> None:
        """
        Validate that parents appear before their children.
        """
        num_joints = self.parents.shape[0]

        for joint_idx in range(1, num_joints):
            parent_idx = self.parents[joint_idx].item()

            if parent_idx < 0:
                raise ValueError(
                    f"Joint {joint_idx} has invalid parent {parent_idx}. "
                    "Only the root may have parent -1."
                )

            if parent_idx >= joint_idx:
                raise ValueError(
                    f"Parent of joint {joint_idx} is {parent_idx}. "
                    "Parents must appear before children."
                )

    def __call__(
        self,
        local_rotations: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return self.forward(local_rotations, root_positions)

    def forward(
        self,
        local_rotations: torch.Tensor,
        root_positions: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Convert local joint transforms into global joint transforms.
        """
        if local_rotations.shape[-1] != 4:
            raise ValueError(
                "`local_rotations` must contain quaternions with shape [..., J, 4]"
            )

        num_joints = self.parents.shape[0]

        if local_rotations.shape[-2] != num_joints:
            raise ValueError(
                f"Expected {num_joints} joints, "
                f"got {local_rotations.shape[-2]}"
            )

        expected_root_shape = local_rotations.shape[:-2] + (3,)
        if root_positions.shape != expected_root_shape:
            raise ValueError(
                f"`root_positions` must have shape {expected_root_shape}, "
                f"got {root_positions.shape}"
            )

        local_rotations = quat_normalize(local_rotations)

        offsets = self.offsets.to(
            device=local_rotations.device,
            dtype=local_rotations.dtype,
        )

        # Per-joint results are collected in lists and stacked at the end (rather than
        # written in place into a preallocated tensor) so the pass stays differentiable.
        global_positions = [root_positions]
        global_rotations = [local_rotations[..., 0, :]]

        # Remaining joints
        for joint_idx in range(1, num_joints):
            parent_idx = self.parents[joint_idx].item()

            parent_rotation = global_rotations[parent_idx]
            parent_position = global_positions[parent_idx]

            local_rotation = local_rotations[..., joint_idx, :]
            local_offset = offsets[..., joint_idx, :]

            # R_global[j] = R_global[parent] * R_local[j]
            global_rotations.append(quat_mul(
                parent_rotation,
                local_rotation,
            ))

            # p_global[j] =
            #     p_global[parent]
            #     + R_global[parent] * offset[j]
            global_positions.append(
                parent_position
                + quat_apply(parent_rotation, local_offset)
            )

        return torch.stack(global_positions, dim=-2), torch.stack(global_rotations, dim=-2)
