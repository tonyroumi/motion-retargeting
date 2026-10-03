"""
Abstract base class for processing different motion capture data sources
"""

from abc import ABC, abstractmethod
from typing import List, Optional

import torch

from motion_retargeting.config.paths import PATHS
from motion_retargeting.data.adapters.utils import label_joint_bodies
from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology


class MotionSourceAdapter(ABC):
    def __init__(self, dataset_name: str, device: str = "cpu"):
        """ Setup data directories """
        data_dir = PATHS.data_root

        self.dataset_name = dataset_name
        self.device = device

        # Subdirectories inside data
        self.raw_dir = data_dir / "raw" / dataset_name
        self.cache_dir = data_dir / "processed" 

        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Where a cached skeleton / the cached motion list are read from and written to.
        self.motion_cache_path = self.cache_dir / f"{dataset_name}.pt"

        # Joint structure shared by every clip
        self._topology: Optional[SkeletonTopology] = None

    def load(self) -> MotionCollection:
        """
        Load and save motion sequences from the raw data source.
        """
        if self.motion_cache_path.exists():
            return torch.load(self.motion_cache_path, weights_only=False)

        motions = self._extract()
        data = MotionCollection(topology=self._topology, motions=motions)
        torch.save(data, self.motion_cache_path)
        return data

    @abstractmethod
    def _extract(self) -> List[MotionData]:
        """ Every clip's MotionData, each built with `_construct_motion`. """
        raise NotImplementedError

    def _construct_motion(
        self,
        parents,
        offsets,
        joint_rotations,
        root_position,
        fps: float,
        joint_names: Optional[List[str]] = None,
    ) -> MotionData:
        """
        Build one clip's MotionData from its canonical (Y-up) data.

        parents:         [J], root first with parent -1
        offsets:         [J, 3] parent-relative rest-pose offsets in meters (root offset zero)
        joint_rotations: [T, J, 4] (w, x, y, z); joint 0 is the root's global rotation
        root_position:   [T, 3] in meters

        Every clip of a dataset shares one joint topology: the first clip sets it (and its body
        parts are labeled interactively, once); later clips must have the same parents / names.
        """
        parents = torch.as_tensor(parents, dtype=torch.long, device=self.device)
        offsets = torch.as_tensor(offsets, dtype=torch.float32, device=self.device)
        joint_rotations = torch.as_tensor(joint_rotations, dtype=torch.float32, device=self.device)
        root_position = torch.as_tensor(root_position, dtype=torch.float32, device=self.device)

        if self._topology is None:
            self._topology = SkeletonTopology(
                parents=parents.detach().cpu(),
                joint_names=joint_names,
            )
            self._topology.body_joints = label_joint_bodies(self._topology, offsets)
        elif not torch.equal(parents.detach().cpu(), self._topology.parents) or joint_names != self._topology.joint_names:
            raise ValueError(f"{self.dataset_name}: clip joint topology differs from the dataset's first clip")

        return MotionData(
            offsets=offsets.detach().cpu(),
            joint_rotations=joint_rotations.detach().cpu(),
            fps=fps,
            root_position=root_position.detach().cpu(),
        )
