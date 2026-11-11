from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from motion_retargeting.data.adapters.factory import build_adapter
from motion_retargeting.data.representations.motion import MotionCollection
from motion_retargeting.data.utils import (
    compute_linear_velocity,
    compute_yaw_velocity_from_heading,
    extract_heading_yup,
)

from .reference import PARTS, ReferenceData, load_reference_splits
from .raw_reference import load_raw_reference_splits


@dataclass
class PANDataConfig:
    sequence_length: int = 64
    stride: int = 32
    target_fps: float | None = 30.0
    holdout_percent: int = 5
    # Cap on this domain's training windows, drawn after the holdout split (None: use all)
    max_train_windows: int | None = None
    max_train_windows_seed: int = 0


def resample_indices(num_frames: int, fps: float, target_fps: float) -> torch.Tensor:
    """
    Source frame nearest to each target timestamp k / target_fps, anchored at time zero and
    stopping before the clip's duration. An integer rate ratio r gives exactly 0, r, 2r, ...
    """
    ratio = fps / target_fps
    num_resampled = math.ceil(num_frames / ratio - 1e-9)
    indices = torch.floor(torch.arange(num_resampled, dtype=torch.float64) * ratio + 0.5).long()
    return indices.clamp(max=num_frames - 1)


def to_joint_channels(joint_rotations: torch.Tensor, root_position: torch.Tensor) -> torch.Tensor:
    """ All J local quaternions (root heading kept), then world XYZ displacement and yaw per frame. """
    rotations = joint_rotations[:-1]  # [T - 1, J, 4]
    linear_velocity = compute_linear_velocity(root_position, dt=1.0)[:-1]  # [T - 1, 3]
    yaw_velocity = compute_yaw_velocity_from_heading(
        extract_heading_yup(joint_rotations[:, 0]),
        fps=1.0,
    )[:-1]  # [T - 1, 1]
    return torch.cat([rotations.flatten(1), linear_velocity, yaw_velocity], dim=-1)


class PANMotionDataset(Dataset):
    """
    Each sample is a window of one skeleton bank:
      - motion:   [W, 4J + 4], z-score normalized per channel with its bank's statistics
      - bank_id:  scalar long, index into `offsets` / `mean` / `std`

    Samples retain their original temporal order; no data augmentation is applied.
    """

    def __init__(
        self,
        motion: torch.Tensor,
        bank_ids: torch.Tensor,
        offsets: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        parents: torch.Tensor,
        body_joints: Dict[int, List[int]],
        reference: ReferenceData | None = None,
        provenance: Dict[str, Any] | None = None,
    ):
        self.reference = reference
        self.provenance = provenance
        self.motion = motion  # [N, W, 4J + 4], un-normalized
        self.bank_ids = bank_ids  # [N]
        self.offsets = offsets  # [K, J, 3], metres (reference archives: cm)
        self.mean = mean  # [K, 4J + 4]
        self.std = std  # [K, 4J + 4]
        self.parents = parents  # [J]
        self.body_joints = body_joints  # manually labeled body parts: {part: [joints]}

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    def subset(self, indices: torch.Tensor) -> PANMotionDataset:
        """ The windows at `indices`, sharing this dataset's banks and statistics. """
        return PANMotionDataset(
            motion=self.motion[indices],
            bank_ids=self.bank_ids[indices],
            offsets=self.offsets,
            mean=self.mean,
            std=self.std,
            parents=self.parents,
            body_joints=self.body_joints,
            reference=self.reference,
            provenance=self.provenance,
        )

    def normalize(self, motion: torch.Tensor, bank_id: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean[bank_id]) / self.std[bank_id]

    def __len__(self) -> int:
        return self.motion.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bank_id = self.bank_ids[idx]
        motion = self.motion[idx]
        motion = self.normalize(motion, bank_id)
        if self.reference is not None:
            motion = motion.float()  # reference normalizes in archive precision before FP32
        return {"motion": motion, "bank_id": bank_id}


def fit_statistics(motion: torch.Tensor, bank_ids: torch.Tensor, num_banks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Per-bank, per-channel mean / sample std over all of the bank's windows and frames. """
    mean = torch.zeros(num_banks, motion.shape[-1], dtype=motion.dtype)
    std = torch.ones(num_banks, motion.shape[-1], dtype=motion.dtype)
    for bank in range(num_banks):
        windows = motion[bank_ids == bank]
        mean[bank] = windows.mean(dim=(0, 1))
        bank_std = windows.var(dim=(0, 1)).sqrt()
        std[bank] = torch.where(bank_std < 1e-5, torch.ones_like(bank_std), bank_std)
    return mean, std


def build_from_collections(collections: List[MotionCollection], config: PANDataConfig) -> PANMotionDataset:
    """ Windows, skeleton banks and statistics of one domain; every collection shares one topology. """
    parents = collections[0].topology.parents
    body_joints = collections[0].topology.body_joints
    for collection in collections[1:]:
        if not torch.equal(collection.topology.parents, parents):
            raise ValueError("All sources in a group must share one skeleton topology")
        if collection.topology.body_joints != body_joints:
            raise ValueError("All sources in a group must share one set of labeled body parts")

    windows: List[torch.Tensor] = []
    bank_ids: List[int] = []
    banks: Dict[bytes, int] = {}
    offsets: List[torch.Tensor] = []

    for collection in collections:
        for clip in collection.motions:
            rotations, root_position = clip.joint_rotations, clip.root_position
            if config.target_fps is not None and clip.fps != config.target_fps:
                indices = resample_indices(rotations.shape[0], clip.fps, config.target_fps)
                rotations, root_position = rotations[indices], root_position[indices]
            if rotations.shape[0] < 2:
                continue  # A velocity needs two frames.

            channels = to_joint_channels(rotations, root_position)
            starts = range(0, channels.shape[0] - config.sequence_length + 1, config.stride)
            if len(starts) == 0:
                continue  # A bank exists only with windows to fit its statistics on.

            key = (clip.offsets + 0.0).numpy().tobytes()
            bank = banks.setdefault(key, len(banks))
            if bank == len(offsets):
                offsets.append(clip.offsets)

            windows.extend(channels[start:start + config.sequence_length] for start in starts)
            bank_ids.extend([bank] * len(starts))

    if not windows:
        raise ValueError(f"No clip is long enough for a window of {config.sequence_length} frames")

    motion = torch.stack(windows)
    bank_id_tensor = torch.tensor(bank_ids, dtype=torch.long)
    mean, std = fit_statistics(motion, bank_id_tensor, len(offsets))

    return PANMotionDataset(
        motion=motion,
        bank_ids=bank_id_tensor,
        offsets=torch.stack(offsets),
        mean=mean,
        std=std,
        parents=parents,
        body_joints=body_joints,
    )


def split_holdout(dataset: PANMotionDataset, holdout_percent: int = 5) -> Tuple[PANMotionDataset, PANMotionDataset]:
    """ (train, holdout): each bank's first windows, in order, train; neither split is augmented. """
    train_indices, holdout_indices = [], []
    for bank in range(dataset.num_banks):
        indices = (dataset.bank_ids == bank).nonzero().flatten()
        num_train = indices.shape[0] * (100 - holdout_percent) // 100
        train_indices.append(indices[:num_train])
        holdout_indices.append(indices[num_train:])

    return (
        dataset.subset(torch.cat(train_indices)),
        dataset.subset(torch.cat(holdout_indices)),
    )


def limit_windows(dataset: PANMotionDataset, max_windows: int | None, seed: int = 0) -> PANMotionDataset:
    """ A seeded random subset of at most `max_windows` windows, kept in temporal order. """
    if max_windows is None:
        return dataset
    if isinstance(max_windows, bool) or not isinstance(max_windows, int) or max_windows < 1:
        raise ValueError(f"max_train_windows must be a positive integer or None, got {max_windows}")
    if max_windows >= len(dataset):
        return dataset

    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:max_windows]
    return dataset.subset(indices.sort().values)


def build_pan_splits(dataset_cfg: Dict[str, Any]) -> Tuple[Dict[str, PANMotionDataset], Dict[str, PANMotionDataset]]:
    train_datasets, holdout_datasets = {}, {}
    if any("human_reference_style" in group for group in dataset_cfg.values()):
        from .human_reference import load_human_reference_splits

        if len(dataset_cfg) != 2 or any(set(group) != {"human_reference_style"} for group in dataset_cfg.values()):
            raise ValueError("human_reference_style requires two explicit domains without generic/archive mixing")
        profiles = [group["human_reference_style"] for group in dataset_cfg.values()]
        if sorted(profile["source"] for profile in profiles) != ["amass", "cmu"]:
            raise ValueError("human_reference_style requires one AMASS and one CMU source")
        if set(profiles[0]["test_subjects"]) != set(profiles[1]["test_subjects"]):
            raise ValueError("CMU-derived domains must use the same held-out subjects to avoid cross-domain leakage")
        for name, group in dataset_cfg.items():
            train_datasets[name], holdout_datasets[name] = load_human_reference_splits(**group["human_reference_style"])
        return train_datasets, holdout_datasets
    dataset_cfg = dict(dataset_cfg)
    for name, group in dataset_cfg.items():
        sources = group.get("sources", [])
        if any(source in PARTS for source in sources):
            if len(sources) != 1 or set(group) - {"sources", "params"}:
                raise ValueError("LAFAN1/Dog sources cannot be combined with other sources or reference options")
            fixed = {"sequence_length": 64, "stride": 20, "target_fps": 30}
            if any(key not in fixed or value != fixed[key] for key, value in (group.get("params") or {}).items()):
                raise ValueError("LAFAN1/Dog use fixed 64/20/30 preprocessing and explicit reference splits; "
                                 "remove holdout/cap params or use reference_archive")
            dataset_cfg[name] = {"reference_raw": {"kind": sources[0]}}
    reference_groups = []
    for group in dataset_cfg.values():
        keys = set(group) & {"reference_archive", "reference_raw"}
        if keys:
            if len(keys) != 1 or set(group) != keys:
                raise ValueError("reference_archive/reference_raw cannot be combined with raw sources/params")
            reference_groups.append(group[next(iter(keys))])
    if reference_groups and (len(dataset_cfg) != 2 or len(reference_groups) != 2 or
                              sorted(group["kind"] for group in reference_groups) != ["dog", "lafan1"]):
        raise ValueError("Reference PAN requires exactly one lafan1 and one dog domain; no raw mixing with generic sources")
    for name, group_cfg in dataset_cfg.items():
        if "reference_archive" in group_cfg or "reference_raw" in group_cfg:
            key = "reference_archive" if "reference_archive" in group_cfg else "reference_raw"
            loader = load_reference_splits if key == "reference_archive" else load_raw_reference_splits
            train_datasets[name], holdout_datasets[name] = loader(**group_cfg[key])
            continue
        config = PANDataConfig(**(group_cfg.get("params") or {}))
        collections = [build_adapter(source).load() for source in group_cfg["sources"]]

        dataset = build_from_collections(collections, config)
        print(f"[PANData] {name}: {len(dataset)} windows, {dataset.num_banks} skeleton banks")

        train, holdout_datasets[name] = split_holdout(dataset, config.holdout_percent)
        train_datasets[name] = limit_windows(train, config.max_train_windows, config.max_train_windows_seed)
        if config.max_train_windows is not None:
            print(f"[PANData] {name}: training on {len(train_datasets[name])} of {len(train)} training windows")
    return train_datasets, holdout_datasets
