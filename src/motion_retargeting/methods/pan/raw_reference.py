"""Raw LAFAN1/Dog loading using the reference's domain/split conventions.

No MotionCollection caches are read or written: they have rebased FP32 metre
coordinates and lack End Sites. Use reference_archive for the exact historical
window order, skeleton choices and saved statistics instead of regenerating them.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import TYPE_CHECKING

import numpy as np
import torch

from motion_retargeting.data.adapters.factory import build_adapter
from motion_retargeting.data.adapters.lafan1_bvh import REFERENCE_HIERARCHIES, read_lafan_reference_bvh
from motion_retargeting.data.adapters.lafan1_splits import reference_files

from .reference import PARTS, ReferenceData, preprocess_reference_window, reference_window_starts

if TYPE_CHECKING:
    from .data import PANMotionDataset


def load_raw_reference_splits(kind: str, directory: str | Path | None = None) -> tuple[PANMotionDataset, PANMotionDataset]:
    """64/20 windows at 30 Hz, explicit clip holdout, train+test population stats.

    Input units/local frames are original BVH centimetres. Each split uses its
    first (sorted) clip's offsets per subject, just as upstream uses its first
    filesystem-enumerated clip. No per-clip skeleton/normalization populations.
    """
    from .data import PANMotionDataset

    if kind not in PARTS:
        raise ValueError(f"Unknown reference domain {kind!r}; expected lafan1 or dog")
    adapter = build_adapter(kind)
    directory = adapter.raw_dir if directory is None else Path(directory).expanduser()
    files = {split: reference_files(directory, kind, split) for split in ("train", "test")}
    offsets, expanded_offsets, banks, splits = [], [], {}, []
    topology = tuple(torch.tensor(values) for values in REFERENCE_HIERARCHIES[kind])
    for split in ("train", "test"):
        subjects, windows, bank_ids = {}, [], []
        for path in files[split]:
            clip = read_lafan_reference_bvh(path)
            if clip.names != adapter.joint_names or len(clip.parents_withend) != 27:
                raise ValueError(f"{path}: expected the reference joint order and 27-joint End Site hierarchy")
            current = (clip.parents, clip.parents_withend, clip.not_end)
            if any(not torch.equal(a, b) for a, b in zip(topology, current)):
                raise ValueError(f"{path}: hierarchy/End Site mapping differs from the published reference")
            source_fps, step = (30, 1) if kind == "lafan1" else (60, 2)
            if round(clip.fps) != source_fps:
                raise ValueError(f"{path}: reference {kind} requires {source_fps} Hz, got {clip.fps}")
            rotations = clip.rotations[::step]
            root = clip.root_position[::step]
            subject = path.stem.rsplit("_", 1)[1] if kind == "lafan1" else path.stem.split("_", 1)[0]
            if subject not in subjects:
                key = (subject, clip.offsets.numpy().tobytes(), clip.offsets_withend.numpy().tobytes())
                if key not in banks:
                    banks[key] = len(offsets)
                    offsets.append(clip.offsets.float())
                    expanded_offsets.append(clip.offsets_withend.float())
                subjects[subject] = banks[key]
            bank = subjects[subject]
            for start in reference_window_starts(len(rotations)):
                motion, _ = preprocess_reference_window(
                    rotations[start:start + 65], root[start:start + 65], clip.offsets, clip.parents, kind,
                )
                if not torch.isfinite(motion).all():
                    raise ValueError(f"{path} frame {start}: undefined reference heading/nonfinite channels")
                windows.append(motion)
                bank_ids.append(bank)
        if not windows:
            raise ValueError(f"{kind} {split}: no reference windows")
        splits.append((torch.stack(windows), torch.tensor(bank_ids, dtype=torch.long)))

    # Match extract.getStats: double precision, population std, exact-zero-only
    # replacement, and intentional test-distribution leakage. Do not use the
    # generic builder's per-bank sample std / small-value floor.
    samples = np.concatenate([motion.numpy().reshape(-1, motion.shape[-1]) for motion, _ in splits])
    mean, std = samples.mean(0), samples.std(0)
    std = np.where(std == 0, 1, std)
    speed = np.linalg.norm(samples[:, -4:-1], axis=-1)
    min_speed, max_speed = float(speed.min()), float(speed.max())
    if not 0 <= min_speed < max_speed < float("inf"):
        raise ValueError(f"{kind}: reference speed mapping requires a finite nonconstant speed range")
    fingerprint = hashlib.sha256(b"pan-raw-reference-v1" + mean.tobytes() + std.tobytes()
                                 + np.asarray([min_speed, max_speed]).tobytes()).hexdigest()
    parents, expanded_parents, not_end = topology
    reference = ReferenceData(kind, expanded_parents, torch.stack(expanded_offsets), not_end,
                              min_speed, max_speed, fingerprint, statistics_source="raw")
    kwargs = dict(offsets=torch.stack(offsets), mean=torch.from_numpy(mean).expand(len(offsets), -1),
                  std=torch.from_numpy(std).expand(len(offsets), -1), parents=parents,
                  body_joints=dict(enumerate(PARTS[kind])), reference=reference)
    return tuple(PANMotionDataset(motion=motion, bank_ids=ids, **kwargs) for motion, ids in splits)
