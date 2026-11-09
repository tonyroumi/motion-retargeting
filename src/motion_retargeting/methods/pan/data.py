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


@dataclass
class PANDataConfig:
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
    num_resampled = math.ceil(num_frames / ratio - 1e-9)
    indices = torch.floor(torch.arange(num_resampled, dtype=torch.float64) * ratio + 0.5).long()
    return indices.clamp(max=num_frames - 1)


def to_edge_channels(joint_rotations: torch.Tensor, root_position: torch.Tensor, parents: torch.Tensor) -> torch.Tensor:
    edge_rotations = joint_rotations[:-1, parents[1:]]  # [T - 1, E, 4]
    linear_velocity = compute_linear_velocity(root_position, dt=1.0)[:-1]  # [T - 1, 3]
    yaw_velocity = compute_yaw_velocity_from_heading(
        extract_heading_yup(joint_rotations[:, 0]),
        fps=1.0,
    )[:-1]  # [T - 1, 1]
    return torch.cat([edge_rotations.flatten(1), linear_velocity, yaw_velocity], dim=-1)


class PANMotionDataset(Dataset):
    """
    Each sample is a window of one skeleton bank:
      - motion:   [W, 4E + 4], z-score normalized per channel with its bank's statistics
      - bank_id:  scalar long, index into `offsets` / `mean` / `std`
      - index:    scalar long, the sample's index in this dataset
      - reversed: scalar bool, whether the window was reversed in time
      - window:   [3] long, (source, clip, start frame) the window was cut from; -1 when unknown

    With `augment`, a sample is reversed in time with probability 0.5
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
        augment: bool = False,
        windows: torch.Tensor | None = None,
    ):
        self.motion = motion  # [N, W, 4E + 4], un-normalized
        self.bank_ids = bank_ids  # [N]
        self.offsets = offsets  # [K, J, 3], metres
        self.mean = mean  # [K, 4E + 4]
        self.std = std  # [K, 4E + 4]
        self.parents = parents  # [J]
        self.body_joints = body_joints  # manually labeled body parts: {part: [joints]}
        self.augment = augment
        # [N, 3]: source index in the group's `sources`, clip index in that source's collection, and
        # start frame after resampling
        self.windows = windows if windows is not None else torch.full((motion.shape[0], 3), -1, dtype=torch.long)

    @property
    def num_banks(self) -> int:
        return self.offsets.shape[0]

    def subset(self, indices: torch.Tensor, augment: bool) -> PANMotionDataset:
        """ The windows at `indices`, sharing this dataset's banks and statistics. """
        return PANMotionDataset(
            motion=self.motion[indices],
            bank_ids=self.bank_ids[indices],
            offsets=self.offsets,
            mean=self.mean,
            std=self.std,
            parents=self.parents,
            body_joints=self.body_joints,
            augment=augment,
            windows=self.windows[indices],
        )

    def normalize(self, motion: torch.Tensor, bank_id: torch.Tensor) -> torch.Tensor:
        return (motion - self.mean[bank_id]) / self.std[bank_id]

    def __len__(self) -> int:
        return self.motion.shape[0]

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        reverse = self.augment and torch.rand(1).item() < 0.5
        return self.sample(idx, reverse)

    def sample(self, idx: int, reverse: bool) -> Dict[str, torch.Tensor]:
        """ Window `idx`, reversed in time or not, normalized with its bank's statistics. """
        bank_id = self.bank_ids[idx]
        motion = self.motion[idx]
        if reverse:
            motion = motion.flip(0)
            root_velocity = -motion[:, -4:]
            root_velocity = torch.cat([root_velocity[1:], root_velocity[-1:]], dim=0)
            motion = torch.cat([motion[:, :-4], root_velocity], dim=-1)
        motion = self.normalize(motion, bank_id)
        return {
            "motion": motion,
            "bank_id": bank_id,
            "index": torch.tensor(idx),
            "reversed": torch.tensor(reverse),
            "window": self.windows[idx],
        }


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
    window_sources: List[Tuple[int, int, int]] = []
    bank_ids: List[int] = []
    banks: Dict[bytes, int] = {}
    offsets: List[torch.Tensor] = []

    for source, collection in enumerate(collections):
        for clip_index, clip in enumerate(collection.motions):
            rotations, root_position = clip.joint_rotations, clip.root_position
            if config.target_fps is not None and clip.fps != config.target_fps:
                indices = resample_indices(rotations.shape[0], clip.fps, config.target_fps)
                rotations, root_position = rotations[indices], root_position[indices]
            if rotations.shape[0] < 2:
                continue  # A velocity needs two frames.

            channels = to_edge_channels(rotations, root_position, parents)
            starts = range(0, channels.shape[0] - config.sequence_length + 1, config.stride)
            if len(starts) == 0:
                continue  # A bank exists only with windows to fit its statistics on.

            key = (clip.offsets + 0.0).numpy().tobytes()
            bank = banks.setdefault(key, len(banks))
            if bank == len(offsets):
                offsets.append(clip.offsets)

            windows.extend(channels[start:start + config.sequence_length] for start in starts)
            window_sources.extend((source, clip_index, start) for start in starts)
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
        windows=torch.tensor(window_sources, dtype=torch.long),
    )


def split_holdout(dataset: PANMotionDataset, holdout_percent: int = 5) -> Tuple[PANMotionDataset, PANMotionDataset]:
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


def build_pan_splits(dataset_cfg: Dict[str, Any]) -> Tuple[Dict[str, PANMotionDataset], Dict[str, PANMotionDataset]]:
    train_datasets, holdout_datasets = {}, {}
    for name, group_cfg in dataset_cfg.items():
        config = PANDataConfig(**(group_cfg.get("params") or {}))
        collections = [build_adapter(source).load() for source in group_cfg["sources"]]

        dataset = build_from_collections(collections, config)
        print(f"[PANData] {name}: {len(dataset)} windows, {dataset.num_banks} skeleton banks")

        train_datasets[name], holdout_datasets[name] = split_holdout(dataset, config.holdout_percent)
    return train_datasets, holdout_datasets


def batch_checksum(motion: torch.Tensor) -> torch.Tensor:
    """ [B], a per-sample fingerprint of normalized motion [B, T, C], to check a rebuilt batch against. """
    return motion.detach().double().sum(dim=(1, 2)).cpu()


def rebuild_batch(
    datasets: Dict[str, PANMotionDataset],
    provenance: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, torch.Tensor]]:
    """
    A captured batch from its per-domain provenance ({"index", "reversed", "checksum"}, or "motion"
    when the capture stored it), drawing the windows from `datasets` and checking their checksums.
    """
    batch = {}
    for name, item in provenance.items():
        if "motion" in item:
            batch[name] = {key: value for key, value in item.items() if key != "checksum"}
            continue

        dataset = datasets[name]
        samples = [
            dataset.sample(int(idx), bool(reverse))
            for idx, reverse in zip(item["index"].tolist(), item["reversed"].tolist())
        ]
        batch[name] = {key: torch.stack([sample[key] for sample in samples]) for key in samples[0]}

        if not torch.allclose(batch_checksum(batch[name]["motion"]), item["checksum"].cpu(), rtol=1e-9, atol=1e-6):
            raise ValueError(f"Rebuilt {name} windows do not match the capture: the dataset differs from training")
    return batch
