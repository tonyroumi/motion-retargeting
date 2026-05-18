from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Convention
# ---------------------------------------------------------------------------
# All quaternions in this module are SCALAR-FIRST: q = (w, x, y, z).
# All rotation matrices are 3x3, applied as `v_world = R @ v_local`.
# All exp-maps (rotation vectors) are 3-vectors `theta * axis`.
# ---------------------------------------------------------------------------

def normalize_angle(x):
    return np.arctan2(np.sin(x), np.cos(x))


def normalize_quat(x, eps: float = 1e-9):
    norms = np.linalg.norm(x, axis=-1, keepdims=True)
    norms = np.clip(norms, a_min=eps, a_max=None)
    return x / norms


def normalize_exp_map(exp_map):
    """Wrap an exp-map's angle into (-pi, pi] while keeping the same rotation.
    """
    angle = np.linalg.norm(exp_map, axis=-1)
    safe_angle = np.where(angle > 1e-9, angle, 1.0)
    axis = exp_map / safe_angle[..., np.newaxis]

    # Reduce angle into [0, 2pi), then into (-pi, pi] by flipping axis if needed.
    wrapped = np.mod(angle + np.pi, 2.0 * np.pi) - np.pi  # in (-pi, pi]
    # If wrapped came out negative, flip both so axis*|angle| == original direction.
    sign = np.where(wrapped < 0, -1.0, 1.0)
    new_angle = wrapped * sign
    new_axis = axis * sign[..., np.newaxis]

    norm_exp_map = new_axis * new_angle[..., np.newaxis]
    # Zero out where the original was effectively zero.
    mask = (angle > 1e-9)[..., np.newaxis]
    return np.where(mask, norm_exp_map, np.zeros_like(exp_map))


def quat_pos(x):
    """Force scalar part to be non-negative.

    Enforces a canonical sign so the same rotation always has the same
    quaternion. Not ideal for learning, since q and -q are the same rotation.
    """
    q = x.copy()
    z = (q[..., :1] < 0).astype(q.dtype)
    q = (1 - 2 * z) * q
    return q


def quat_mul(q1, q2):
    """Hamilton product of two quaternions (scalar-first)."""
    w1, x1, y1, z1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
    w2, x2, y2, z2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
    return np.stack([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], axis=-1)


def quat_unit(a):
    return normalize_quat(a)


def quat_normalize(q):
    q = quat_unit(quat_pos(q))
    return q


def quat_conjugate(q):
    """Conjugate (inverse for unit quaternions), scalar-first."""
    return np.concatenate([q[..., :1], -q[..., 1:]], axis=-1)


def quat_rotate(q, v):
    """Rotate vector *v* by quaternion *q*. Both broadcastable."""
    q_v = np.concatenate([np.zeros_like(v[..., :1]), v], axis=-1)
    return quat_mul(quat_mul(q, q_v), quat_conjugate(q))[..., 1:]


def quat_to_matrix(q):
    """Convert scalar-first quaternion to 3x3 rotation matrix."""
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    R = np.empty(q.shape[:-1] + (3, 3), dtype=q.dtype)
    R[..., 0, 0] = 1 - 2*(y*y + z*z)
    R[..., 0, 1] = 2*(x*y - z*w)
    R[..., 0, 2] = 2*(x*z + y*w)
    R[..., 1, 0] = 2*(x*y + z*w)
    R[..., 1, 1] = 1 - 2*(x*x + z*z)
    R[..., 1, 2] = 2*(y*z - x*w)
    R[..., 2, 0] = 2*(x*z - y*w)
    R[..., 2, 1] = 2*(y*z + x*w)
    R[..., 2, 2] = 1 - 2*(x*x + y*y)
    return R


def quat_diff(q0, q1):
    """World-frame delta rotation: dq satisfies dq * q0 == q1.

    For a body-frame delta (such that q0 * dq == q1), use
    ``quat_mul(quat_conjugate(q0), q1)`` instead.
    """
    dq = quat_mul(q1, quat_conjugate(q0))
    return dq


def quat_to_axis_angle(q, eps: float = 1e-5):
    """Extract (axis, angle) from a scalar-first quaternion.

    Scalar-first convention: q[..., 0] is w, q[..., 1:4] is (x, y, z).
    Returns axis as a unit vector and angle in radians in [0, pi].
    """
    q = quat_pos(q)
    vec = q[..., 1:4]
    w = q[..., 0]

    length = np.linalg.norm(vec, axis=-1)
    angle = 2.0 * np.arctan2(length, w)

    safe_length = np.where(length > eps, length, 1.0)
    axis = vec / safe_length[..., np.newaxis]

    default_axis = np.zeros_like(axis)
    default_axis[..., -1] = 1
    mask = length > eps

    angle = np.where(mask, angle, np.zeros_like(angle))
    axis = np.where(mask[..., np.newaxis], axis, default_axis)

    return axis, angle


def quat_to_exp_map(q):
    axis, angle = quat_to_axis_angle(q)
    exp_map = axis_angle_to_exp_map(axis, angle)
    return exp_map


def exp_map_to_quat(exp_map):
    axis, angle = exp_map_to_axis_angle(exp_map)
    q = axis_angle_to_quat(axis, angle)
    return q


def exp_map_to_axis_angle(exp_map, min_theta: float = 1e-5):
    """Decompose an exp-map (rotation vector) into (axis, angle).

    Returns axis as a unit vector and angle equal to ||exp_map|| (>= 0).
    The (axis, angle) pair always satisfies axis * angle == exp_map.

    NOTE: the previous implementation called normalize_angle on the magnitude,
    which produced an inconsistent (axis, angle) pair for |exp_map| > pi.
    """
    angle = np.linalg.norm(exp_map, axis=-1)

    safe_angle = np.where(angle > min_theta, angle, 1.0)
    axis = exp_map / safe_angle[..., np.newaxis]

    default_axis = np.zeros_like(exp_map)
    default_axis[..., -1] = 1

    mask = angle > min_theta
    angle = np.where(mask, angle, np.zeros_like(angle))
    axis = np.where(mask[..., np.newaxis], axis, default_axis)

    return axis, angle


def axis_angle_to_exp_map(axis, angle):
    angle_expand = angle[..., np.newaxis]
    exp_map = angle_expand * axis
    return exp_map


def axis_angle_to_quat(axis, angle, eps: float = 1e-12):
    """Build a scalar-first quaternion from (axis, angle).

    A zero-norm axis with zero angle is treated as the identity rotation,
    avoiding divide-by-zero warnings that the previous version produced.
    """
    axis_norm = np.linalg.norm(axis, axis=-1, keepdims=True)
    safe_norm = np.clip(axis_norm, a_min=eps, a_max=None)
    axis = axis / safe_norm

    theta = 0.5 * angle[..., np.newaxis]

    w = np.cos(theta)
    xyz = axis * np.sin(theta)

    return quat_unit(np.concatenate([w, xyz], axis=-1))


def calc_heading(q):
    """Compute the yaw / heading angle of a quaternion."""
    ref_dir = np.zeros_like(q[..., 0:3])
    ref_dir[..., 0] = 1
    rot_dir = quat_rotate(q, ref_dir)

    heading = np.arctan2(rot_dir[..., 1], rot_dir[..., 0])
    return heading


def calc_heading_quat_inv(q):
    """Extract the inverse heading rotation from a quaternion.

    Multiplying ``calc_heading_quat_inv(q) * q`` (world frame) removes the yaw
    component, leaving an orientation invariant to global heading. Two
    identical walking clips facing different directions then produce the same
    orientation features, which is a simpler learning task.
    """
    heading = calc_heading(q)
    axis = np.zeros_like(q[..., 0:3])
    axis[..., 2] = 1

    heading_q = axis_angle_to_quat(axis, -heading)
    return heading_q


def quat_to_tan_norm(q):
    """Represent a quaternion as a 6D continuous rotation (tangent + normal).

    Avoids the q = -q discontinuity that hurts neural-network learning.
    """
    ref_tan = np.zeros_like(q[..., 0:3])
    ref_tan[..., 0] = 1
    tan = quat_rotate(q, ref_tan)

    ref_norm = np.zeros_like(q[..., 0:3])
    ref_norm[..., -1] = 1
    norm = quat_rotate(q, ref_norm)

    norm_tan = np.concatenate([tan, norm], axis=-1)
    return norm_tan


def heading_quat_from_root(root_quat):
    """Extract the yaw-only (heading) quaternion from a full root orientation.

    Projects the root's local x-axis onto the ground plane to determine
    the heading angle, then returns a quaternion that represents *only*
    the rotation about the world z-axis by that angle.

    Parameters
    ----------
    root_quat : ndarray, shape (..., 4)
        Root orientation as a scalar-first quaternion.

    Returns
    -------
    heading_q : ndarray, shape (..., 4)
        Yaw-only quaternion (rotation about z).
    """
    forward_local = np.zeros(root_quat.shape[:-1] + (3,), dtype=root_quat.dtype)
    forward_local[..., 0] = 1.0
    forward_world = quat_rotate(root_quat, forward_local)

    heading_angle = np.arctan2(forward_world[..., 1], forward_world[..., 0])

    half = heading_angle / 2.0
    heading_q = np.zeros_like(root_quat)
    heading_q[..., 0] = np.cos(half)
    heading_q[..., 3] = np.sin(half)
    return heading_q


def slerp(q0, q1, t, eps: float = 1e-6):
    """Spherical linear interpolation between two unit quaternions.

    Blends orientations between two poses so motion can be sampled smoothly
    at any time. Assumes both inputs are unit quaternions (scalar-first).

    Parameters
    ----------
    q0, q1 : ndarray, shape (..., 4)
    t      : float or ndarray (broadcastable to the batch shape of q0/q1)
    """
    t = np.asarray(t, dtype=q0.dtype)

    # Pick the shorter arc by flipping q1 if the dot product is negative.
    dot = np.sum(q0 * q1, axis=-1)
    neg_mask = dot < 0
    q1 = np.where(neg_mask[..., np.newaxis], -q1, q1)
    dot = np.abs(dot)

    # Clamp to valid acos domain, then compute half-angle.
    dot_clamped = np.clip(dot, -1.0, 1.0)
    half_theta = np.arccos(dot_clamped)
    sin_half_theta = np.sqrt(np.maximum(1.0 - dot_clamped * dot_clamped, 0.0))

    # Broadcast scalars to vector shape.
    t = t[..., np.newaxis] if t.ndim > 0 else t
    half_theta = half_theta[..., np.newaxis]
    sin_half_theta = sin_half_theta[..., np.newaxis]

    # Guard division against sin_half_theta == 0 (parallel quats).
    safe_sin = np.where(sin_half_theta > eps, sin_half_theta, 1.0)
    ratioA = np.sin((1 - t) * half_theta) / safe_sin
    ratioB = np.sin(t * half_theta) / safe_sin

    new_q = ratioA * q0 + ratioB * q1

    # When the quats are nearly parallel, fall back to linear interpolation.
    parallel_mask = sin_half_theta <= eps
    lerp_q = (1 - t) * q0 + t * q1
    new_q = np.where(parallel_mask, lerp_q, new_q)

    # Renormalize to handle any drift introduced by lerp fallback.
    new_q = normalize_quat(new_q)
    return new_q


# ---------------------------------------------------------------------------
# Frame transforms
# ---------------------------------------------------------------------------

def compute_root_relative_frame(root_pos, root_quat):
    """Compute the transform that places the origin at the root and aligns
    the x-axis with the root's facing (heading) direction.

    Parameters
    ----------
    root_pos : (..., 3)
    root_quat : (..., 4)  scalar-first

    Returns
    -------
    inv_heading_q : (..., 4)
    origin        : (..., 3)
    """
    heading_q = heading_quat_from_root(root_quat)
    inv_heading_q = quat_conjugate(heading_q)

    origin = np.array(root_pos, copy=True)
    origin[..., 2] = 0.0

    return inv_heading_q, origin


def transform_positions_to_root_frame(positions, root_pos, root_quat):
    """Transform world-frame positions into the root-relative heading frame.

    Parameters
    ----------
    positions : (..., N, 3) or (..., 3)
    root_pos  : (..., 3)
    root_quat : (..., 4)

    Returns
    -------
    local_positions : same shape as *positions*
    """
    inv_heading_q, origin = compute_root_relative_frame(root_pos, root_quat)

    if positions.ndim > root_pos.ndim:
        shifted = positions - np.expand_dims(origin, -2)
    else:
        shifted = positions - origin

    if shifted.ndim > inv_heading_q.ndim:
        q = np.broadcast_to(np.expand_dims(inv_heading_q, -2), shifted.shape[:-1] + (4,))
    else:
        q = inv_heading_q

    return quat_rotate(q, shifted)


def transform_velocities_to_root_frame(velocities, root_quat):
    """Rotate world-frame velocities into the root-relative heading frame.

    Velocities are direction-only, so there is no translation offset.
    """
    inv_heading_q, _ = compute_root_relative_frame(
        np.zeros_like(root_quat[..., :3]), root_quat
    )
    if velocities.ndim > inv_heading_q.ndim:
        q = np.broadcast_to(np.expand_dims(inv_heading_q, -2), velocities.shape[:-1] + (4,))
    else:
        q = inv_heading_q
    return quat_rotate(q, velocities)


def transform_rotations_to_root_frame(quats, root_quat):
    """Express body orientations relative to the root heading frame.

    Parameters
    ----------
    quats     : (..., N, 4) or (..., 4)  body orientations in world frame
    root_quat : (..., 4)

    Returns
    -------
    local_quats : same shape as *quats*
    """
    inv_heading_q, _ = compute_root_relative_frame(
        np.zeros_like(root_quat[..., :3]), root_quat
    )
    if quats.ndim > inv_heading_q.ndim:
        q = np.broadcast_to(np.expand_dims(inv_heading_q, -2), quats.shape[:-1] + (4,))
    else:
        q = inv_heading_q
    return quat_mul(q, quats)