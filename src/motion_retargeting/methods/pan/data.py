"""
Edge / absolute-root data view of the PAN reference-Mixamo recipe.

Built directly from the adapters' MotionCollections, before the shared window construction of
`MotionDatasetBuilder` (whose windows have already discarded the absolute root and cannot express
edge rotations). Legacy PAN and legacy SAN never see this module; SAN's reference recipe reads it too.

Known deviations from the original Mixamo assets, on purpose:
  - Sources stay AMASS / CMU. Their clips are resampled on a timestamp grid (see
    `resample_indices`), rather than Mixamo's literal `::2`.
  - A "skeleton bank" stands in for a Mixamo character: clips share a bank only when their offset
    tensors are exactly equal. That says nothing about the identity of the captured subject.
  - Like the reference, statistics are fit before the holdout, so the holdout includes windows the
    statistics saw. It is a reproduction diagnostic, not clean evidence of generalization.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

import torch
from torch.utils.data import Dataset

from motion_retargeting.data.adapters.factory import build_adapter
from motion_retargeting.data.representations.motion import MotionCollection

# A channel whose std falls below this is left unscaled (std 1), as in the reference.
MIN_STD = 1e-5


@dataclass
class ReferenceDataConfig:
    sequence_length: int = 64
    stride: int = 32
    target_fps: float | None = 30.0
    holdout_percent: int = 5


def resample_indices(num_frames: int, fps: float, target_fps: float) -> torch.Tensor:
    """
    Source frame nearest to each target timestamp k / target_fps, anchored at time zero and
    stopping before the clip's duration. An integer rate ratio r gives exactly 0, r, 2r, ...
    """
    ratio = fps / target_fps
    # The small slack keeps float error from adding a timestamp at exactly the clip's duration.
    num_resampled = math.ceil(num_frames / ratio - 1e-9)
    indices = torch.floor(torch.arange(num_resampled, dtype=torch.float64) * ratio + 0.5).long()
    return indices.clamp(max=num_frames - 1)


def to_edge_channels(joint_rotations: torch.Tensor, root_position: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    """
    joint_rotations [T, J, 4] + root_position [T, 3] -> physical channels [T, 4E + 3].

    Edge e = (parent, child), in ascending child index, carries the parent's rotation: a branching
    joint's rotation repeats on each outgoing edge and a leaf's rotation disappears. Heading, root
    height and quaternion signs are kept as they are.
    """
    edge_rotations = joint_rotations[:, parents[1:]]  # [T, E, 4]
    return torch.cat([edge_rotations.flatten(1), root_position], dim=-1)


class ReferenceMotionDataset(Dataset):
    """
    Each sample is a window of one skeleton bank:
      - motion:   [W, 4E + 3], z-score normalized per channel with its bank's statistics
      - bank_id:  scalar long, index into `offsets` / `mean` / `std`

    With `augment`, a sample is reversed in time with probability 0.5.
    """

    def __init__(
        self,
        motion: torch.Tensor,
        bank_ids: torch.Tensor,
        offsets: torch.Tensor,
        mean: torch.Tensor,
        std: torch.Tensor,
        parents: torch.Tensor,
        augment: bool = False,
    ):
        self.motion = motion  # [N, W, 4E + 3], un-normalized
        self.bank_ids = bank_ids  # [N]
        self.offsets = offsets  # [K, J, 3], metres
        self.mean = mean  # [K, 4E + 3]
        self.std = std  # [K, 4E + 3]
        self.parents = parents  # [J]
        self.augment = augment

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    def subset(self, indices: torch.Tensor, augment: bool) -> ReferenceMotionDataset:
        """ The windows at `indices`, sharing this dataset's banks and statistics. """
        return ReferenceMotionDataset(
            motion=self.motion[indices],
            bank_ids=self.bank_ids[indices],
            offsets=self.offsets,
            mean=self.mean,
            std=self.std,
            parents=self.parents,
            augment=augment,
        )

    def normalize(self, motion: torch.Tensor, bank_id: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean[bank_id]) / self.std[bank_id]

    def __len__(self) -> int:
        return self.motion.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        bank_id = self.bank_ids[idx]
        motion = self.normalize(self.motion[idx], bank_id)
        if self.augment and torch.rand(1).item() < 0.5:
            motion = motion.flip(0)
        return {"motion": motion, "bank_id": bank_id}


def fit_statistics(motion: torch.Tensor, bank_ids: torch.Tensor, num_banks: int) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Per-bank, per-channel mean / sample std over all of the bank's windows and frames. """
    mean = torch.zeros(num_banks, motion.shape[-1], dtype=motion.dtype)
    std = torch.ones(num_banks, motion.shape[-1], dtype=motion.dtype)
    for bank in range(num_banks):
        windows = motion[bank_ids == bank]
        mean[bank] = windows.mean(dim=(0, 1))
        bank_std = windows.var(dim=(0, 1)).sqrt()
        std[bank] = torch.where(bank_std < MIN_STD, torch.ones_like(bank_std), bank_std)
    return mean, std


def build_from_collections(collections: List[MotionCollection], config: ReferenceDataConfig) -> ReferenceMotionDataset:
    """ Windows, skeleton banks and statistics of one domain; every collection shares one topology. """
    parents = collections[0].topology.parents
    for collection in collections[1:]:
        if not torch.equal(collection.topology.parents, parents):
            raise ValueError("All sources in a group must share one skeleton topology")

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

            channels = to_edge_channels(rotations, root_position, parents)
            starts = range(0, channels.shape[0] - config.sequence_length + 1, config.stride)
            if len(starts) == 0:
                continue  # A bank exists only with windows to fit its statistics on.

            # Exact equality, no rounding; adding 0.0 folds -0.0 into +0.0 before comparing bytes.
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

    return ReferenceMotionDataset(
        motion=motion,
        bank_ids=bank_id_tensor,
        offsets=torch.stack(offsets),
        mean=mean,
        std=std,
        parents=parents,
    )


def split_holdout(dataset: ReferenceMotionDataset, holdout_percent: int = 5) -> Tuple[ReferenceMotionDataset, ReferenceMotionDataset]:
    """ (train, holdout): each bank's first windows, in order, train; only training samples are augmented. """
    train_indices, holdout_indices = [], []
    for bank in range(dataset.num_banks):
        indices = (dataset.bank_ids == bank).nonzero().flatten()
        num_train = indices.shape[0] * (100 - holdout_percent) // 100
        train_indices.append(indices[:num_train])
        holdout_indices.append(indices[num_train:])

    return (
        dataset.subset(torch.cat(train_indices), augment=True),
        dataset.subset(torch.cat(holdout_indices), augment=False),
    )


def build_reference_splits(dataset_cfg: Dict[str, Any]) -> Tuple[Dict[str, ReferenceMotionDataset], Dict[str, ReferenceMotionDataset]]:
    """ {domain: train}, {domain: holdout} from the `dataset` section of the config. """
    train_datasets, holdout_datasets = {}, {}
    for name, group_cfg in dataset_cfg.items():
        config = ReferenceDataConfig(**(group_cfg.get("params") or {}))
        collections = [build_adapter(source).load() for source in group_cfg["sources"]]

        dataset = build_from_collections(collections, config)
        print(f"[ReferenceData] {name}: {len(dataset)} windows, {dataset.num_banks} skeleton banks")

        train_datasets[name], holdout_datasets[name] = split_holdout(dataset, config.holdout_percent)
    return train_datasets, holdout_datasets
