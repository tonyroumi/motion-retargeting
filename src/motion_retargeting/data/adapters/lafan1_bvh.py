"""Precision-preserving reader for PAN's original LAFAN1/Dog BVHs.

This intentionally narrow reader keeps float64 channels/offsets and End Sites. The
shared bvhio/metre/rebased MotionCollection path cannot preserve tiny toe channels
used by the reference statistics. Unsupported BVH layouts fail rather than being
silently reinterpreted. No dependency on the upstream repository at runtime.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from motion_retargeting.kinematics.quaternions import quat_mul


# Ordered hierarchies and rotation -> expanded FK mapping from the published
# stats. Selected End Site indices in PAN's losses rely on this exact layout.
REFERENCE_HIERARCHIES = {
    "lafan1": (
        [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 11, 14, 15, 16, 11, 18, 19, 20],
        [-1, 0, 1, 2, 3, 4, 0, 6, 7, 8, 9, 0, 11, 12, 13, 14, 15, 13, 17, 18, 19, 20, 13, 22, 23, 24, 25],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 15, 17, 18, 19, 20, 22, 23, 24, 25],
    ),
    "dog": (
        [-1, 0, 1, 2, 3, 2, 5, 6, 7, 2, 9, 10, 11, 0, 13, 14, 0, 16, 17, 0, 19],
        [-1, 0, 1, 2, 3, 4, 2, 6, 7, 8, 9, 2, 11, 12, 13, 14, 0, 16, 17, 18, 0, 20, 21, 22, 0, 24, 25],
        [0, 1, 2, 3, 4, 6, 7, 8, 9, 11, 12, 13, 14, 16, 17, 18, 20, 21, 22, 24, 25],
    ),
}


@dataclass
class LAFANReferenceClip:
    rotations: torch.Tensor  # [T,J,4], float64 wxyz, original local frames
    root_position: torch.Tensor  # [T,3], cm (root OFFSET is NOT added)
    offsets: torch.Tensor  # [J,3], cm, root zero
    parents: torch.Tensor
    offsets_withend: torch.Tensor  # [27,3], cm, root zero
    parents_withend: torch.Tensor
    not_end: torch.Tensor
    names: list[str]
    fps: float


def read_lafan_reference_bvh(path: Path) -> LAFANReferenceClip:
    names, parents, offsets, channels, not_end = [], [], [], [], []
    active = -1
    with Path(path).open() as stream:
        for line in stream:
            tokens = line.split()
            if not tokens or tokens[0] in ("HIERARCHY", "{"):
                continue
            if tokens[0] == "MOTION":
                break
            if tokens[0] in ("ROOT", "JOINT", "End"):
                is_end = tokens[0] == "End"
                names.append("End Site" if is_end else tokens[1])
                parents.append(active)
                offsets.append(None)
                channels.append([])
                active = len(parents) - 1
                if not is_end:
                    not_end.append(active)
            elif tokens[0] == "}":
                active = parents[active]
            elif tokens[0] == "OFFSET":
                offsets[active] = [float(value) for value in tokens[1:]]
            elif tokens[0] == "CHANNELS":
                if int(tokens[1]) != len(tokens[2:]):
                    raise ValueError(f"{path}: invalid CHANNELS declaration")
                channels[active] = tokens[2:]
            else:
                raise ValueError(f"{path}: unsupported BVH hierarchy line: {line.strip()}")
        if not names or active != -1 or any(value is None or len(value) != 3 for value in offsets):
            raise ValueError(f"{path}: incomplete BVH hierarchy")
        frame_line = next(stream).split()
        time_line = next(stream).split()
        if frame_line[:1] != ["Frames:"] or time_line[:2] != ["Frame", "Time:"]:
            raise ValueError(f"{path}: missing BVH frame count/time")
        frames, frame_time = int(frame_line[1]), float(time_line[2])
        if frames < 1 or not np.isfinite(frame_time) or frame_time <= 0:
            raise ValueError(f"{path}: invalid BVH frame count/time")
        values = np.loadtxt(stream, dtype=np.float64, ndmin=2)
    if values.shape != (frames, sum(map(len, channels))) or not np.isfinite(values).all():
        raise ValueError(f"{path}: frame count/channel count mismatch or nonfinite channels")
    root_channels = channels[0]
    if root_channels[:3] != ["Xposition", "Yposition", "Zposition"] or len(root_channels) != 6:
        raise ValueError(f"{path}: reference requires root XYZ translation and three rotations")
    order = root_channels[3:]
    if set(order) != {"Xrotation", "Yrotation", "Zrotation"} or any(channels[i] != order for i in not_end[1:]):
        raise ValueError(f"{path}: reference requires a common rotation order and no non-root translation")
    if any(channels[i] for i, name in enumerate(names) if name == "End Site"):
        raise ValueError(f"{path}: End Sites cannot have channels")
    offsets_np = np.asarray(offsets, dtype=np.float64)
    if not np.isfinite(offsets_np).all():
        raise ValueError(f"{path}: nonfinite offsets")
    offsets_np[0] = 0
    # NumPy trigonometry in double precision, as in the original BVH reader.
    angles = np.radians(values[:, 3:].reshape(frames, len(not_end), 3))
    axis_quats = []
    for axis, channel in enumerate(order):
        q = np.zeros((*angles.shape[:2], 4), dtype=np.float64)
        q[..., 0] = np.cos(angles[..., axis] / 2)
        q[..., "XYZ".index(channel[0]) + 1] = np.sin(angles[..., axis] / 2)
        axis_quats.append(torch.from_numpy(q))
    rotations = quat_mul(axis_quats[0], quat_mul(axis_quats[1], axis_quats[2]))
    # Equivalent to upstream's sequential closest-sign pass, before downsampling.
    flips = (rotations[1:] * rotations[:-1]).sum(-1) < 0
    signs = torch.cat((torch.ones(1, len(not_end)), torch.where(flips, -1., 1.)), 0).cumprod(0)
    rotations *= signs[..., None]
    index = {joint: i for i, joint in enumerate(not_end)}
    regular_parents = [-1] + [index[parents[joint]] for joint in not_end[1:]]
    return LAFANReferenceClip(
        rotations, torch.from_numpy(values[:, :3].copy()), torch.from_numpy(offsets_np[not_end].copy()),
        torch.tensor(regular_parents), torch.from_numpy(offsets_np), torch.tensor(parents),
        torch.tensor(not_end), [names[i] for i in not_end], 1 / frame_time,
    )
