"""Explicit CMU/AMASS experiment borrowing LAFAN1/Dog math, NOT dataset equivalence.

Available human joints replace unavailable reference End Sites. A rest-anatomy
forward axis replaces LAFAN1's authored root +Y axis. Split membership is explicit
and shared across the two CMU-derived domains; no legacy collection caches are used.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from motion_retargeting.data.adapters.amass import AMASSAdapter
from motion_retargeting.data.adapters.cmu import CMUAdapter
from motion_retargeting.data.representations.skeleton import SkeletonTopology

from .reference import ReferenceData, preprocess_reference_window, reference_window_starts

if TYPE_CHECKING:
    from .data import PANMotionDataset


HUMAN_NAMES = {
    "amass": ["pelvis", "left_hip", "right_hip", "spine1", "left_knee", "right_knee", "spine2",
              "left_ankle", "right_ankle", "spine3", "left_foot", "right_foot", "neck",
              "left_collar", "right_collar", "head", "left_shoulder", "right_shoulder",
              "left_elbow", "right_elbow", "left_wrist", "right_wrist"],
    "cmu": ["root", "lfemur", "ltibia", "lfoot", "ltoes", "rfemur", "rtibia", "rfoot", "rtoes",
            "lowerback", "upperback", "thorax", "lowerneck", "upperneck", "head", "head_end",
            "lclavicle", "lhumerus", "lradius", "lwrist", "rclavicle", "rhumerus", "rradius", "rwrist"],
}
HUMAN_PARENTS = {
    "amass": [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19],
    "cmu": [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22],
}
# CMU head_end is a known identity-rotation End Site, not a network output token.
HUMAN_ROTATION_IDS = {"amass": list(range(22)), "cmu": list(range(15)) + list(range(16, 24))}
HUMAN_FK_PARENTS = {"amass": HUMAN_PARENTS["amass"], "cmu": HUMAN_PARENTS["cmu"] + [4, 8, 19, 23]}
HUMAN_SELECTED_ENDS = {"amass": [], "cmu": [15, 24, 25]}  # head and both toe tips; arms omitted from positional loss
HUMAN_PARTS = {
    "amass": [[7, 10, 8, 11, 1, 4, 2, 5], [3, 6, 9], [12, 15]],
    "cmu": [[3, 4, 7, 8, 1, 2, 5, 6], [9, 10, 11], [12, 13, 14]],
}


def human_adapter(source: str, body_models: str | Path | None = None):
    """Known topology/labels avoid interactive labeling during bounded reads or preparation."""
    if source == "amass":
        adapter = AMASSAdapter(body_models=body_models)
    elif source == "cmu":
        if body_models is not None:
            raise ValueError("body_models only applies to AMASS")
        adapter = CMUAdapter()
    else:
        raise ValueError("human_reference_style requires amass or cmu")
    adapter._topology = SkeletonTopology(
        torch.tensor(HUMAN_PARENTS[source]),
        joint_names=HUMAN_NAMES[source] if source == "cmu" else None,
        body_joints=dict(enumerate(HUMAN_PARTS[source])),
    )
    return adapter


def anatomical_forward(offsets: torch.Tensor, parents: torch.Tensor, left: int, right: int, head: int) -> torch.Tensor:
    """Root-local forward = up x right in the rest skeleton (not a guessed root axis)."""
    positions = [torch.zeros_like(offsets[0])]
    for joint, parent in enumerate(parents[1:], 1):
        positions.append(positions[int(parent)] + offsets[joint])
    up = positions[head] - positions[0]
    across = positions[right] - positions[left]
    forward = torch.cross(up, across, dim=-1)
    norm = forward.norm()
    if not torch.isfinite(norm) or norm < 1e-10:
        raise ValueError("Cannot define anatomical forward from degenerate hips/head rest geometry")
    return forward / norm


def human_forward_axis(source: str, offsets: torch.Tensor) -> torch.Tensor:
    left, right, head = (1, 2, 15) if source == "amass" else (1, 5, 14)
    return anatomical_forward(offsets, torch.tensor(HUMAN_PARENTS[source]), left, right, head)


def human_fk_offsets(source: str, offsets: torch.Tensor, asf=None) -> torch.Tensor:
    """Available physical FK endpoints in metres, never invented lengths.

    CMU's parser already supplies head_end. Its ASF additionally specifies real
    toe/wrist terminal bone vectors, so append four identity-rotation End Sites.
    The retained SMPL-H body has no such head/toe tip joints; do not synthesize them.
    """
    if source == "cmu":
        if asf is None:
            raise ValueError("CMU reference-style FK needs original ASF terminal bone vectors")
        tips = torch.stack([torch.as_tensor(asf.bone_vectors[name], dtype=offsets.dtype)
                            for name in ("ltoes", "rtoes", "lwrist", "rwrist")])
        offsets = torch.cat((offsets, tips))
    if not torch.isfinite(offsets).all():
        raise ValueError("Nonfinite human FK skeleton offsets")
    return offsets


def human_rotation_channels(source: str, channels: torch.Tensor) -> torch.Tensor:
    rotations = channels[..., :-4].reshape(*channels.shape[:-1], len(HUMAN_PARENTS[source]), 4)
    return torch.cat((rotations[..., HUMAN_ROTATION_IDS[source], :].flatten(-2), channels[..., -4:]), -1)


def human_files(source: str, directory: Path, test_subjects: list[str]) -> dict[str, list[Path]]:
    """All clips, explicit shared CMU session-ID holdout, no window-level leakage.

    AMASS is restricted to its CMU subset for this experiment. Multi-actor session
    files may share a stem: retain distinct relative paths, but keep all actors and
    sessions 18--23 together (the release's Justin/Rory recordings).
    """
    if source not in HUMAN_PARTS:
        raise ValueError("Expected amass or cmu")
    heldout = {str(int(subject)).zfill(2) for subject in test_subjects}
    linked = {str(i) for i in range(18, 24)}
    if heldout & linked and not linked <= heldout:
        raise ValueError("Known multi-actor CMU sessions 18--23 must all share the same split")
    files = sorted(directory.rglob("*_poses.npz" if source == "amass" else "*.amc"))
    result = {"train": [], "test": []}
    for path in files:
        subject = path.stem.split("_", 1)[0]
        if not subject.isdigit():
            raise ValueError(f"{path}: expected a CMU numeric session prefix")
        result["test" if subject.zfill(2) in heldout else "train"].append(path)
    if not all(result.values()):
        raise ValueError(f"{source}: require nonempty explicit train and test clip splits under {directory}")
    return result


def load_human_reference_splits(
    source: str,
    test_subjects: list[str],
    directory: str | Path | None = None,
    body_models: str | Path | None = None,
) -> tuple[PANMotionDataset, PANMotionDataset]:
    from .data import PANMotionDataset, resample_indices

    adapter = human_adapter(source, body_models)
    directory = (adapter.raw_dir / "CMU" if source == "amass" else adapter.raw_dir) if directory is None else Path(directory).expanduser()
    directory = directory.resolve()
    files = human_files(source, directory, test_subjects)
    parents = torch.tensor(HUMAN_PARENTS[source])
    offsets, fk_offsets, banks, axes, splits, records = [], [], {}, [], [], []
    for split in ("train", "test"):
        windows, ids = [], []
        for path in files[split]:
            clip = adapter._read_clip(path, dtype=torch.float64)
            if clip is None:
                raise ValueError(f"{path}: empty motion")
            if not np.isfinite(clip.fps) or clip.fps < 30:
                raise ValueError(f"{path}: expected finite source rate >= 30 Hz")
            ix = resample_indices(len(clip.joint_rotations), clip.fps, 30.)
            rotations, root = clip.joint_rotations[ix], clip.root_position[ix] * 100
            starts = reference_window_starts(len(ix))
            record = {"path": str(path.relative_to(directory)), "split": split, "fps": clip.fps,
                      "source_frames": len(clip.joint_rotations), "sampled_frames": len(ix), "windows": len(starts)}
            records.append(record)
            if not starts:
                continue
            # AMASS multi-actor directories denote distinct fitted bodies. CMU
            # skeletons are per session; byte-roundoff never splits a subject bank.
            subject = path.stem.split("_", 1)[0]
            identity = f"{subject}:{path.parent.relative_to(directory)}" if source == "amass" else subject
            if identity not in banks:
                banks[identity] = len(offsets)
                offsets.append(clip.offsets * 100)
                asf = adapter._asf_model(path) if source == "cmu" else None
                fk_offsets.append(human_fk_offsets(source, clip.offsets, asf) * 100)
                axes.append(human_forward_axis(source, offsets[-1]))
            bank = banks[identity]
            for start in starts:
                motion, _ = preprocess_reference_window(
                    rotations[start:start+65], root[start:start+65], clip.offsets * 100, parents,
                    "lafan1", forward_axis=axes[bank], stable_heading=True,
                )
                motion = human_rotation_channels(source, motion)
                if not torch.isfinite(motion).all():
                    raise ValueError(f"{path} frame {start}: undefined heading or nonfinite motion")
                windows.append(motion)
                ids.append(bank)
        if not windows:
            raise ValueError(f"{source} {split}: no 64-frame windows")
        splits.append((torch.stack(windows), torch.tensor(ids, dtype=torch.long)))

    # Same intentional train+test leakage and population/zero-only policy as PAN.
    samples = np.concatenate([motion.numpy().reshape(-1, motion.shape[-1]) for motion, _ in splits])
    mean, std = samples.mean(0), samples.std(0)
    std = np.where(std == 0, 1, std)
    speed = np.linalg.norm(samples[:, -4:-1], axis=-1)
    minimum, maximum = float(speed.min()), float(speed.max())
    if not 0 <= minimum < maximum < float("inf"):
        raise ValueError(f"{source}: a finite nonconstant root speed range is required")
    provenance = {"protocol": "human_lafan_style_v1", "source": source, "directory": str(directory),
                  "test_subjects": sorted(test_subjects), "clips": records, "banks": list(banks),
                  "forward_axes": [axis.tolist() for axis in axes], "zero_speed_frames": int((speed == 0).sum())}
    fingerprint = hashlib.sha256(json.dumps(provenance, sort_keys=True).encode() + mean.tobytes() + std.tobytes()
                                 + np.asarray([minimum, maximum]).tobytes()).hexdigest()
    parts = HUMAN_PARTS[source]
    selected = [0] + [joint for part in parts for joint in part] + HUMAN_SELECTED_ENDS[source]
    source_offsets = torch.stack(offsets).float()
    expanded_offsets = torch.stack(fk_offsets).float()
    rotation_ids = torch.tensor(HUMAN_ROTATION_IDS[source])
    index = {joint: i for i, joint in enumerate(HUMAN_ROTATION_IDS[source])}
    rotation_parents = torch.tensor([-1] + [index[int(parents[joint])] for joint in rotation_ids[1:]])
    reference = ReferenceData(source, torch.tensor(HUMAN_FK_PARENTS[source]), expanded_offsets, rotation_ids, minimum, maximum,
                              fingerprint, "human_raw", "human_lafan_style_v1", parts, selected)
    kwargs = dict(offsets=source_offsets[:, rotation_ids], parents=rotation_parents, body_joints=dict(enumerate(parts)), reference=reference,
                  mean=torch.from_numpy(mean).expand(len(offsets), -1), std=torch.from_numpy(std).expand(len(offsets), -1),
                  provenance=provenance)
    return tuple(PANMotionDataset(motion=motion, bank_ids=ids, **kwargs) for motion, ids in splits)
