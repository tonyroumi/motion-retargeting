import torch


def quat_normalize(q: torch.Tensor) -> torch.Tensor:
    return q / torch.linalg.vector_norm(q, dim=-1, keepdim=True)


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    """
    Conjugate of a quaternion in (w, x, y, z) convention.
    Equal to the inverse for unit quaternions.
    """
    return q * q.new_tensor([1.0, -1.0, -1.0, -1.0])


def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    """
    Hamilton product for quaternions in (w, x, y, z) convention.
    """
    w1, x1, y1, z1 = q1.unbind(dim=-1)
    w2, x2, y2, z2 = q2.unbind(dim=-1)

    return torch.stack(
        (
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ),
        dim=-1,
    )


def quat_apply(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """
    Rotate 3D vectors by quaternions in (w, x, y, z) convention.

    q: [..., 4]
    v: [..., 3]
    """
    q = quat_normalize(q)

    q_xyz = q[..., 1:]
    q_w = q[..., :1]

    # Equivalent to q * [0, v] * q^-1, but cheaper.
    t = 2.0 * torch.cross(q_xyz, v, dim=-1)

    return v + q_w * t + torch.cross(q_xyz, t, dim=-1)