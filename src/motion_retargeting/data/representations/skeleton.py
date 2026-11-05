"""
Joint topology shared by every clip of a dataset.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import torch


@dataclass
class SkeletonTopology:
    """ Joint structure shared by every clip of a dataset (per-clip offsets live in MotionData). """
    parents: torch.Tensor          # [J], long, root first with parent -1
    joint_names: Optional[List[str]] = None
    body_joints: Optional[Dict[int, List[int]]] = None

    @property
    def num_joints(self) -> int:
        return self.parents.shape[0]
