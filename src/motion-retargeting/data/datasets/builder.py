"""
MotionDataset builder to construct MotionDataset for a single motion domain.
"""

from omegaconf import DictConfig
from typing import Any, Dict, Tuple, List
import numpy as np
import torch

from src.data.adapters import BaseAdapter, list_characters, get_adapter_for_character
from src.data.metadata import SkeletonMetadata, MotionSequence
from src.utils import ArrayUtils, SkeletonUtils

class MotionDatasetBuilder:
    def __init__(self, character: str, device: str):
        self.character = character
        self.adapter : BaseAdapter = get_adapter_for_character(character, device)
        self.device = self.adapter.device
    
    def get_characters(self):
        """
        Returns all of the characters with a matching topology to the source character.
        """
        source_skel = self._get_char(self.character)

        same_topologies = []
        for char in list_characters(self.adapter):
            char_skel = self._get_char(char)

            if source_skel == char_skel:
                same_topologies.append(char)

        return same_topologies

    def get_or_process(self, character: str):
        """ 
        Load or generate all skeleton metadata and motion data for a given character. 
        
        Raises:
            FileNotFoundError: If no raw files exist for provided character.
        """ 
        cache_path = (
            self.adapter.cache_dir / character
        )
        processed_files = list(cache_path.glob("*"))

        raw_path = (
            self.adapter.raw_dir / character
        )
        raw_files = list(raw_path.glob("*"))

        if not raw_files:
            raise FileNotFoundError(f"No raw files for {character}")

        skeleton = self._get_char(character)       
        offsets = skeleton.offsets
        edge_topology = skeleton.edge_topology
        ee_ids = skeleton.ee_ids
        height = skeleton.height

        # If there are no processed files, proccess from raw dir. 
        if not processed_files:
            motions = self.adapter.extract_motion(character)
        else:
            motions = []
            for processed_motion in processed_files:
                motions.append(MotionSequence.load(processed_motion))

        all_windowed_motion, all_windowed_pos, total_length = self._process_motion(motions)
        ee_vels = SkeletonUtils.get_ee_velocity(all_windowed_pos, skeleton) / height

        print(f"{character} contains {total_length} frames after downsampling.")

        return (ArrayUtils.to_torch(all_windowed_motion, self.device),
                ArrayUtils.to_torch(all_windowed_pos, self.device),
                ArrayUtils.to_torch(offsets, self.device).reshape(-1),  # (3*edges)
                ArrayUtils.to_torch(edge_topology, self.device, torch.int),
                ArrayUtils.to_torch(ee_ids, self.device, torch.int),
                ArrayUtils.to_torch(ee_vels, self.device),
                ArrayUtils.to_torch(height, self.device))
    
    def _get_char(self, character: str) -> SkeletonMetadata:
        skel_path = (
            self.adapter.skeleton_dir / f"{character}.npz"
        )
        if not skel_path.exists():
            return self.adapter.extract_skeleton(character)
        else:
            return SkeletonMetadata.load(skel_path)

    def _process_motion(self, motions: List[MotionSequence]) -> Tuple[torch.Tensor, torch.Tensor, int]:
        """ Concatentate, downsample and break up motion data into fixed size windows """  

        motion_chunks, position_chunks = [], []
        total_length = 0
        for m in motions:
            rotations = m.rotations.reshape(m.rotations.shape[0], -1)
            root_pos = m.positions[:, 0]

            full_motion = np.hstack([rotations, root_pos])

            motion_ds = self._downsample(full_motion, fps_in=m.fps)
            positions_ds = self._downsample(m.positions, fps_in=m.fps)

            motion_chunks.append(motion_ds)
            position_chunks.append(positions_ds)

            total_length += motion_ds.shape[0]

        full_motion = np.concatenate(motion_chunks, axis=0)
        full_position = np.concatenate(position_chunks, axis=0)

        # window into [num_windows, 64, feature_dim]
        windowed_motion = self._get_windows(full_motion)
        windowed_position = self._get_windows(full_position)

        windowed_motion = np.transpose(windowed_motion, axes=(0, 2, 1))
        return windowed_motion, windowed_position, total_length

    def _downsample(self, data: np.ndarray, fps_in: float):
        """ Downsample by uniform subsampling. """
        stride = int(round(fps_in / 30))
        return data[::stride]

    def _get_windows(self, data: np.ndarray):
        """ 
        Slice a motion array of shape [T, J*4+3] into windows of shape [64, J*4+3].

        Returns:
          [num_windows, 64, J*4+7].
        """
        T = data.shape[0]
        step = 64 // 2

        n_window = T // step - 1
        windows = []

        for i in range(n_window):
            start = i * step
            end = start + 64
            if end > T:
                break
            windows.append(data[start:end])

        return np.stack(windows)
