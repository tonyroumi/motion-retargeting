"""MotionDatasetBuilder: turns the MotionCollections of one or more adapters into a
windowed MotionDataset of (motion, offsets) tensors, where `motion` is the
network input (local rotations with the global velocity / angular velocity
appended as a final "joint" row). The motion is left un-normalized; see
`compute_stats` / `MotionDataset.set_normalization`.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from motion_retargeting.data.adapters import MotionSourceAdapter
from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.kinematics.quaternions import quat_normalize

from .dataset import MotionDataset
from .utils import compute_linear_velocity, canonicalize_root_yup, compute_yaw_velocity_from_heading, extract_heading_yup


@dataclass
class MotionDatasetBuilderConfig:
    sequence_length: int
    stride: int = 1
    target_fps: float | None = None
    
    max_samples: int | None = None

    seed: int = 0


class MotionDatasetBuilder:
    def __init__(self, config: MotionDatasetBuilderConfig):
        self.config = config

    def build(self, sources: List[MotionSourceAdapter]) -> MotionDataset:
        collections = self.load_collections(sources)
        topology = self.shared_topology(collections, sources)

        # Each window keeps the rest-pose offsets of the clip (subject) it came from.
        rotations, velocity, angular_velocity, q_heading, offsets = [], [], [], [], []
        samples_per_source = {source.dataset_name: 0 for source in sources}
        for source, collection in zip(sources, collections):
            for motion in collection.motions:
                processed = self.process_clip(motion)
                if processed is None:
                    continue

                clip_rotations, clip_velocity, clip_angular_velocity, clip_q_heading = processed

                windows = self.window(clip_rotations, clip_velocity, clip_angular_velocity, clip_q_heading)
                samples_per_source[source.dataset_name] += len(windows)

                for w_rotations, w_velocity, w_angular_velocity, w_q_heading in windows:
                    rotations.append(w_rotations)
                    velocity.append(w_velocity)
                    angular_velocity.append(w_angular_velocity)
                    q_heading.append(w_q_heading)
                    offsets.append(motion.offsets)

        for source_name, num_samples in samples_per_source.items():
            print(f"[MotionDatasetBuilder] {source_name}: {num_samples} samples")

        rotations = torch.stack(rotations)
        velocity = torch.stack(velocity)
        angular_velocity = torch.stack(angular_velocity)
        q_heading = torch.stack(q_heading)
        offsets = torch.stack(offsets)

        rotations, velocity, angular_velocity, q_heading, offsets = self.subsample(
            rotations, velocity, angular_velocity, q_heading, offsets
        )
        print(f"[MotionDatasetBuilder] total: {rotations.shape[0]} samples (after max_samples subsampling)")

        motion = self.to_input(rotations, velocity, angular_velocity)

        return MotionDataset(
            motion=motion,
            q_heading=q_heading,
            offsets=offsets,
            body_joints=topology.body_joints,
            parents=topology.parents,
        )

    def load_collections(self, sources: List[MotionSourceAdapter]) -> List[MotionCollection]:
        return [source.load() for source in sources]

    def shared_topology(
        self,
        collections: List[MotionCollection],
        sources: List[MotionSourceAdapter],
    ) -> SkeletonTopology:
        """ The joint topology every source shares; a group's windows must all index the same joints. """
        topology = collections[0].topology
        for source, collection in zip(sources[1:], collections[1:]):
            if not torch.equal(collection.topology.parents, topology.parents):
                raise ValueError(
                    f"All sources in a group must share one skeleton topology; "
                    f"{source.dataset_name} differs from {sources[0].dataset_name}"
                )
        return topology

    def process_clip(self, motion: MotionData) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Resample to the target fps (if configured) and derive, per frame:
          - local_rotations:         [T, J, 4]
          - global_velocity:         [T, 3]   (root displacement per frame)
          - global_angular_velocity: [T, 1]   (root yaw per frame, about the up axis, axis*angle)
          - q_heading:               [T, 4]   (the absolute heading removed from the root)
        """
        rotations = quat_normalize(motion.joint_rotations)
        root_positions = motion.root_position

        if self.config.target_fps is not None and motion.fps != self.config.target_fps:
            rotations, root_positions = self.resample(rotations, root_positions, motion.fps)

        if rotations.shape[0] < 2:
            return None

        # Compute displacement/velocity for intervals [t, t+1]
        root_vel = compute_linear_velocity(root_positions, dt= 1.0)[:-1]

        # Heading rate for the same intervals [t, t+1], from the headings of all T frames
        # (taken before the root rotations are canonicalized in place below).
        yaw_velocity = compute_yaw_velocity_from_heading(
            extract_heading_yup(rotations[:, 0]),
            fps=1.0,
        )[:-1]

        # Use the corresponding T-1 root rotations
        root_q = rotations[:-1, 0]

        # Remove absolute heading
        root_q_canonical, root_vel_canonical, q_heading = (
            canonicalize_root_yup(root_q, root_vel)
        )

        rotations = rotations[:-1]
        rotations[:, 0] = root_q_canonical

        return rotations, root_vel_canonical, yaw_velocity, q_heading

    def resample(
        self,
        rotations: torch.Tensor,
        root_positions: torch.Tensor,
        fps: float,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """ Nearest-frame resampling to `target_fps`. """
        num_frames = rotations.shape[0]
        num_resampled = max(2, round(num_frames * self.config.target_fps / fps))

        indices = torch.linspace(0, num_frames - 1, num_resampled).round().long()

        return rotations[indices], root_positions[indices]

    def window(self, rotations, velocity, angular_velocity, q_heading):
        """ Slice the aligned per-frame tensors into fixed-length, overlapping windows. """
        length = self.config.sequence_length
        stride = self.config.stride

        num_frames = velocity.shape[0]
        windows = []
        for start in range(0, num_frames - length + 1, stride):
            end = start + length
            windows.append((rotations[start:end], velocity[start:end], angular_velocity[start:end], q_heading[start:end]))

        return windows

    def subsample(self, rotations, velocity, angular_velocity, q_heading, offsets):
        if self.config.max_samples is None or rotations.shape[0] <= self.config.max_samples:
            return rotations, velocity, angular_velocity, q_heading, offsets

        generator = torch.Generator().manual_seed(self.config.seed)
        indices = torch.randperm(rotations.shape[0], generator=generator)[: self.config.max_samples]

        return rotations[indices], velocity[indices], angular_velocity[indices], q_heading[indices], offsets[indices]

    def to_input(self, rotations, velocity, angular_velocity):
        """
        Concatenate into the network input [N, W, J+1, 4]: rotations [N, W, J, 4] with
        (global_velocity [N, W, 3], global_angular_velocity [N, W, 1]) appended as the final "joint" row.
        """
        velocities = torch.cat([velocity, angular_velocity], dim=-1)
        return torch.cat([rotations, velocities.unsqueeze(-2)], dim=-2)
