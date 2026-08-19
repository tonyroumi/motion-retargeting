"""
Normalization and torch dataset classes for single and cross motion domains.
"""

from .builder import MotionDatasetBuilder

from torch.utils.data import Dataset
from typing import List
import torch

from src.core.normalization import NormalizationStats
from src.core.types import PairedSample, SkeletonTopology
from src.utils import SkeletonUtils

class MotionDataset(Dataset):
    """
    Represents all motion for a motion domain where all characters share the same topology
    and end-effectors.

    __getitem__ returns:
        rotations, normed_rotations, offsets, height, gt_positions, gt_ee_vel
    """
    def __init__(self, character: str, device: str):
        builder = MotionDatasetBuilder(character, device)
        
        self.characters = builder.get_characters()
        self.char_index = []         # maps each sample → char_id
        self.char_meta = {}

        shared_topology = None
        shared_ee_ids = None

        motion_seqs, pos_seqs, ee_vels = [], [], []
        for char_id, char in enumerate(self.characters):
            (
                motion_windows,
                position_windows,
                offsets,
                edge_topology,
                ee_ids,
                ee_vel,
                height,
            ) = builder.get_or_process(char)

            if shared_topology is None:
                shared_topology = edge_topology
                shared_ee_ids = ee_ids
            else:
                if not torch.equal(edge_topology, shared_topology):
                    raise ValueError(f"Topology mismatch for character {char}")
                if not torch.equal(ee_ids, shared_ee_ids):
                    raise ValueError(f"EE ids mismatch for character {char}")

            self.char_meta[char] = {
                "character": char,
                "offsets": offsets,
                "height": height,
                "id": char_id,
            }
            motion_seqs.append(motion_windows)
            pos_seqs.append(position_windows)
            ee_vels.append(ee_vel)
            self.char_index.extend([char_id] * motion_windows.shape[0])

        self.rotations = torch.cat(motion_seqs, dim=0)
        self.gt_positions = torch.cat(pos_seqs, dim=0)
        self.gt_ee_vels = torch.cat(ee_vels, dim=0)
        self.char_index = torch.tensor(self.char_index, dtype=torch.long)

        shared_adjacency = SkeletonUtils.construct_adj(shared_topology)

        self.topology = SkeletonTopology(edge_topology=shared_topology, edge_adjacency=shared_adjacency, ee_ids=shared_ee_ids)

        self.norm_stats = self._compute_normalization_stats()
        self.motion_samples = self.norm_stats.norm(self.rotations)
    
    def _compute_normalization_stats(self):
        # mean/var over batch and time
        mean = torch.mean(self.rotations, dim=(0, 2), keepdim=True)
        var  = torch.var(self.rotations, dim=(0, 2), keepdim=True) ** (1/2)
        return NormalizationStats(mean, var)

    def __len__(self):
        return self.motion_samples.shape[0]

    def __getitem__(self, idx):
        char_id = self.char_index[idx]

        # find which character this corresponds to
        char_name = list(self.char_meta.keys())[char_id]
        meta = self.char_meta[char_name]

        rotations = self.rotations[idx]
        motion_samples = self.motion_samples[idx]
        gt_positions = self.gt_positions[idx]
        gt_ee_vels = self.gt_ee_vels[idx]

        if torch.rand(1) < 0.5:
            # Reverse along the time dimension
            rotations = torch.flip(rotations, dims=[1])
            motion_samples = torch.flip(motion_samples, dims=[1])
            gt_positions = torch.flip(gt_positions, dims=[0])
            gt_ee_vels = torch.flip(gt_ee_vels, dims=[0])

        return rotations, \
               motion_samples, \
               meta["offsets"], \
               meta["height"], \
               gt_positions, \
               gt_ee_vels

class CrossDomainMotionDataset(Dataset):
    """ 
    Combines two MotionDataset instances for two different motion domains.
    
    __getitem__ returns:
        PairedSample(...)
    """
    def __init__(
        self,
        domain_A: MotionDataset,
        domain_B: MotionDataset
    ):
        self.domain_A = domain_A
        self.domain_B = domain_B
        self.domains = [self.domain_A, self.domain_B]
        self.topologies = [self.domain_A.topology, self.domain_B.topology]

    def __len__(self):
        return max(len(self.domain_A), len(self.domain_B))

    def __getitem__(self, idx):
        idx_A = idx % len(self.domain_A)
        idx_B = idx % len(self.domain_B)

        domain_A = self.domain_A[idx_A]
        domain_B = self.domain_B[idx_B]

        return PairedSample(
            rotations=(domain_A[0], domain_B[0]),
            motions=(domain_A[1], domain_B[1]),
            offsets=(domain_A[2], domain_B[2]),
            heights=(domain_A[3], domain_B[3]),
            gt_positions=(domain_A[4], domain_B[4]),
            gt_ee_vels=(domain_A[5], domain_B[5])
        )
