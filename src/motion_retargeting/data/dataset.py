# """
# Torch dataset holding preprocessed, windowed motion tensors, z-score normalized
# on access once `set_normalization` has been called.
# """

# from __future__ import annotations

# from dataclasses import dataclass
# from typing import Dict, List, Optional

# from torch.utils.data import Dataset
# import torch


# @dataclass
# class FeatureStats:
#     """ Mean/std used to z-score normalize (and invert) one feature of the motion's final row. """
#     mean: torch.Tensor
#     std: torch.Tensor


# class MotionDataset(Dataset):
#     """
#     Each sample is a fixed-length window:
#       - motion:                 [W, J+1, 4]  (per-joint rotations, with the
#                                  concatenation of global_velocity and
#                                  global_angular_velocity appended as the
#                                  final "joint" row). That row is z-score
#                                  normalized with `vel_stats` / `ang_vel_stats`
#                                  if set; the rotations are always left raw
#       - offsets:                [J, 3]  (rest-pose offsets of the skeleton
#                                  the window was taken from)
#       - q_heading:              [W, 4]  (the absolute root heading removed
#                                  from the rotations and velocity)
#     """

#     def __init__(
#         self,
#         motion: torch.Tensor,
#         q_heading: torch.Tensor,
#         offsets: torch.Tensor,
#         body_joints: Dict[int, List[int]],
#         parents: torch.Tensor,
#         vel_stats: Optional[FeatureStats] = None,
#         ang_vel_stats: Optional[FeatureStats] = None,
#     ):
#         self.motion = motion  # [N, W, J+1, 4], un-normalized
#         self.vel_stats = vel_stats
#         self.ang_vel_stats = ang_vel_stats

#         self.q_heading = q_heading
#         self.offsets = offsets  # [N, J, 3], one per window
#         self.body_joints = body_joints
#         self.parents = parents

#     def set_normalization(self, vel_mean: torch.Tensor, vel_std: torch.Tensor, ang_mean: torch.Tensor, ang_std: torch.Tensor) -> None:
#         """ Normalize the final row of `motion` with these stats (velocity [3], yaw rate scalar) from now on. """
#         self.vel_stats = FeatureStats(mean=vel_mean, std=vel_std)
#         self.ang_vel_stats = FeatureStats(mean=ang_mean, std=ang_std)

#     def subset(self, indices: torch.Tensor) -> "MotionDataset":
#         """ A new MotionDataset holding only the windows at `indices`. """
#         return MotionDataset(
#             motion=self.motion[indices],
#             q_heading=self.q_heading[indices],
#             offsets=self.offsets[indices],
#             body_joints=self.body_joints,
#             parents=self.parents,
#             vel_stats=self.vel_stats,
#             ang_vel_stats=self.ang_vel_stats,
#         )

#     def __len__(self) -> int:
#         return self.motion.shape[0]

#     def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
#         """ Returns {"motion": [W, J+1, 4], "offsets": [J, 3], "q_heading": [W, 4]} for window `idx`. """
#         # Copy: indexing returns a view, and normalizing it in place would rewrite the stored data.
#         motion = self.motion[idx].clone()
#         if self.vel_stats is not None:
#             motion[:, -1, :-1] = (motion[:, -1, :-1] - self.vel_stats.mean) / self.vel_stats.std
#         if self.ang_vel_stats is not None:
#             motion[:,-1, -1] = (motion[:, -1, -1] - self.ang_vel_stats.mean) / self.ang_vel_stats.std
#         return {"motion": motion, "offsets": self.offsets[idx], "q_heading": self.q_heading[idx]}


# def split_dataset(
#     dataset: MotionDataset,
#     val_fraction: float = 0.1,
#     test_fraction: float = 0.1,
#     seed: int = 0,
# ) -> tuple[MotionDataset, MotionDataset, MotionDataset]:
#     """ Randomly split the windows into (train, val, test) datasets. """
#     generator = torch.Generator().manual_seed(seed)
#     indices = torch.randperm(len(dataset), generator=generator)

#     num_val = int(len(dataset) * val_fraction)
#     num_test = int(len(dataset) * test_fraction)

#     val_indices = indices[:num_val]
#     test_indices = indices[num_val : num_val + num_test]
#     train_indices = indices[num_val + num_test :]

#     return dataset.subset(train_indices), dataset.subset(val_indices), dataset.subset(test_indices)


# def compute_stats(dataset: MotionDataset, eps: float = 1e-6) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
#     """ Mean / std of the raw velocity ([3]) and yaw rate (scalar), pooled over windows and frames. """
#     vel_mean = dataset.motion[..., -1, :-1].mean(dim=(0, 1))
#     vel_std = dataset.motion[..., -1, :-1].std(dim=(0, 1))

#     vel_std = torch.where(vel_std < eps, torch.ones_like(vel_std), vel_std)

#     ang_mean = dataset.motion[...,-1, -1].mean(dim=(0, 1))
#     ang_std = dataset.motion[..., -1, -1].std(dim=(0, 1))

#     ang_std = torch.where(ang_std < eps, torch.ones_like(ang_std), ang_std)

#     return vel_mean, vel_std, ang_mean, ang_std
