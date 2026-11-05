"""
Motion math shared by the data layer: the PAN attention mask, skeleton height, velocities,
and heading (yaw) canonicalization. Quaternions are (w, x, y, z), Y is up.
"""

from __future__ import annotations

from typing import Dict, List

import torch

from motion_retargeting.kinematics.quaternions import quat_conjugate, quat_mul, quat_apply


def build_joint_mask(
    body_joints: Dict[int, List[int]],
    device: torch.device | str | None = None,
) -> torch.Tensor:
    """
    Build PAN attention mask for tokens ordered as:

        [body_0, ..., body_B-1,
         joint_0, ..., joint_J-1,
         velocity]

    Each body token and its associated joints attend within their body group.

    The velocity token attends to every body token, and every body token
    attends to the velocity token.

    The velocity token does not attend directly to joint tokens.
    """
    num_bodies = len(body_joints)
    num_joints = len([
        joint
        for joints in body_joints.values()
        for joint in joints
    ])

    # +1 for the velocity token.
    num_tokens = num_bodies + num_joints + 1

    velocity_idx = num_tokens - 1

    mask = torch.full(
        (num_tokens, num_tokens),
        float("-inf"),
        device=device,
    )

    for body_idx, joints in body_joints.items():

        # Original joint index -> token index after the body tokens.
        joint_tokens = [
            num_bodies + joint_idx
            for joint_idx in joints
        ]

        # Body token + joints belonging to that body.
        body_tokens = [body_idx] + joint_tokens

        idx = torch.tensor(
            body_tokens,
            device=device,
            dtype=torch.long,
        )

        # Allow attention within each body group.
        mask[idx[:, None], idx[None, :]] = 0.0

    # Allow velocity token <-> every body token.
    body_idx = torch.arange(
        num_bodies,
        device=device,
        dtype=torch.long,
    )

    mask[velocity_idx, body_idx] = 0.0
    mask[body_idx, velocity_idx] = 0.0

    # Allow velocity token to attend to itself.
    mask[velocity_idx, velocity_idx] = 0.0

    return mask


def skeleton_height(
    offsets: torch.Tensor,
    parents: torch.Tensor,
    up_axis: int = 1,
) -> torch.Tensor:
    """
    offsets: [B, J, 3]
    parents: [J]

    returns: [B]
    """

    positions = [offsets[:, 0]]

    for j in range(1, offsets.shape[1]):
        parent = parents[j].item()
        positions.append(
            positions[parent] + offsets[:, j]
        )

    positions = torch.stack(positions, dim=1)

    h = positions[..., up_axis]

    return h.amax(dim=1) - h.amin(dim=1)


def compute_linear_velocity(
    positions: torch.Tensor,
    dt: float,
) -> torch.Tensor:
    """
    positions: [T, 3]
    """
    velocity = torch.zeros_like(positions)

    velocity[:-1] = (
        positions[1:] - positions[:-1]
    ) / dt

    velocity[-1] = velocity[-2]

    return velocity


def compute_angular_velocity(
    q: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    """
    Compute angular velocity from a sequence of root quaternions.

    Args:
        q:
            root quaternion sequence in (w, x, y, z) convention.
            Shape: [T, 4]

        fps:
            Motion sampling frequency in Hz.

    Returns:
        angular_velocity:
            Angular velocity in rad/s.
            Shape: [T, 3]

            Since the coordinate system is +Y up:
                angular_velocity[:, 0] -> rotation about X
                angular_velocity[:, 1] -> rotation about Y (yaw)
                angular_velocity[:, 2] -> rotation about Z
    """
    # Relative rotation:
    #
    # q_delta = q_t * inverse(q_{t-1})
    #
    # This produces angular velocity expressed in world coordinates.
    q_prev = q[:-1]
    q_curr = q[1:]

    q_delta = quat_mul(q_curr, quat_conjugate(q_prev))

    # Ensure shortest rotation.
    q_delta = torch.where(
        q_delta[:, :1] < 0,
        -q_delta,
        q_delta,
    )

    # Quaternion -> rotation vector.
    w = q_delta[:, 0].clamp(-1.0, 1.0)
    xyz = q_delta[:, 1:]

    xyz_norm = xyz.norm(dim=-1, keepdim=True)

    angle = 2.0 * torch.atan2(
        xyz_norm.squeeze(-1),
        w,
    )

    eps = 1e-8

    axis = xyz / xyz_norm.clamp_min(eps)

    rotation_vector = axis * angle.unsqueeze(-1)

    # For very small rotations:
    #
    # sin(theta / 2) ~= theta / 2
    # => rotation_vector ~= 2 * xyz
    small = xyz_norm.squeeze(-1) < eps

    rotation_vector[small] = 2.0 * xyz[small]

    angular_velocity = rotation_vector * fps

    # Match the original T frames.
    # Replicate the first valid angular velocity at frame 0.
    angular_velocity = torch.cat(
        [angular_velocity[:1], angular_velocity],
        dim=0,
    )

    return angular_velocity


def extract_heading_yup(q: torch.Tensor) -> torch.Tensor:
    """
    Extract the Y-axis heading quaternion from root orientations.

    q: [..., 4] quaternion in (w, x, y, z)
    Assumes:
        - Y is up
        - +X is character forward
        - active rotations
    Returns:
        q_heading: [..., 4]
    """
    forward = torch.zeros(
        *q.shape[:-1], 3,
        device=q.device,
        dtype=q.dtype,
    )
    forward[..., 0] = 1.0

    # Global facing direction
    facing = quat_apply(q, forward)

    # Project onto horizontal XZ plane
    facing[..., 1] = 0.0
    facing = facing / torch.linalg.vector_norm(
        facing, dim=-1, keepdim=True
    ).clamp_min(1e-8)

    # For +X-forward:
    # x = cos(yaw), z = -sin(yaw)
    yaw = torch.atan2(-facing[..., 2], facing[..., 0])

    half = 0.5 * yaw

    q_heading = torch.stack([
        torch.cos(half),
        torch.zeros_like(half),
        torch.sin(half),
        torch.zeros_like(half),
    ], dim=-1)

    return q_heading


def canonicalize_root_yup(
    root_q: torch.Tensor,
    root_vel: torch.Tensor,
):
    """
    root_q:   [..., 4]  (w, x, y, z)
    root_vel: [..., 3]  world-frame linear velocity
    """
    q_heading = extract_heading_yup(root_q)

    q_heading_inv = quat_conjugate(q_heading)

    root_q_canonical = quat_mul(
        q_heading_inv,
        root_q,
    )

    root_vel_canonical = quat_apply(
        q_heading_inv,
        root_vel,
    )

    return root_q_canonical, root_vel_canonical, q_heading


def compute_yaw_velocity_from_heading(
    q_heading: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    """
    Compute yaw angular velocity directly from pure heading quaternions.

    Args:
        q_heading:
            Pure Y-axis heading quaternions in (w, x, y, z).
            Shape: [T, 4]

        fps:
            Sampling frequency in Hz.

    Returns:
        yaw_velocity:
            Shape: [T, 1]
            Units: rad/s
            Frame t is the step t -> t+1 (like compute_linear_velocity);
            the last frame repeats the previous step.
    """
    w = q_heading[:, 0]
    y = q_heading[:, 2]

    # Pure Y-axis quaternion:
    # q = [cos(yaw/2), 0, sin(yaw/2), 0]
    yaw = 2.0 * torch.atan2(y, w)

    # Wrapped frame-to-frame difference in [-pi, pi].
    dyaw = yaw[1:] - yaw[:-1]
    dyaw = torch.atan2(
        torch.sin(dyaw),
        torch.cos(dyaw),
    )

    yaw_velocity = dyaw * fps

    # Match T frames.
    yaw_velocity = torch.cat([
        yaw_velocity,
        yaw_velocity[-1:],
    ], dim=0)

    return yaw_velocity.unsqueeze(-1)
