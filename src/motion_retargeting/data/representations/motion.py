"""
Per-clip motion containers shared across adapters, the builder, and kinematics.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import torch

from .skeleton import SkeletonTopology


@dataclass
class MotionData:
    """ One clip: its subject's skeleton proportions plus its motion. """
    offsets: torch.Tensor              # [J, 3], parent-relative rest-pose offsets in meters (root zero)
    joint_rotations: torch.Tensor      # [T, J, 4], (w,x,y,z); joint 0 is the root's global rotation
    fps: float

    root_position: Optional[torch.Tensor] = None      # [T, 3]

    height: Optional[torch.Tensor] = None   # scalar, of this clip's subject


@dataclass
class MotionCollection:
    """ Every clip of one dataset, sharing a single joint topology. """
    topology: SkeletonTopology
    motions: List[MotionData]

    def __len__(self) -> int:
        return len(self.motions)

    def __getitem__(self, idx: int) -> MotionData:
        return self.motions[idx]
