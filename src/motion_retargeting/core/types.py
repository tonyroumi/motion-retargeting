from dataclasses import dataclass
from typing import Tuple
import torch

@dataclass
class SkeletonTopology:
    edge_topology: torch.Tensor = None
    edge_adjacency: torch.Tensor = None
    ee_ids: torch.Tensor = None

@dataclass
class MotionOutput:
    latents: torch.Tensor
    motion: torch.Tensor          # normalized rotations
    rotations: torch.Tensor       # denormalized rotations
    positions: torch.Tensor       # FK world positions
    ee_vels: torch.Tensor = None  # end-effector velocities

@dataclass
class PairedSample:
    rotations: Tuple[torch.Tensor, torch.Tensor]
    motions: Tuple[torch.Tensor, torch.Tensor]
    offsets: Tuple[torch.Tensor, torch.Tensor]
    heights: Tuple[torch.Tensor, torch.Tensor]
    gt_positions: Tuple[torch.Tensor, torch.Tensor]
    gt_ee_vels: Tuple[torch.Tensor, torch.Tensor]

    def to(self, device: torch.device):
        return PairedSample(
            rotations=tuple(r.to(device) for r in self.rotations),
            motions=tuple(m.to(device) for m in self.motions),
            offsets=tuple(o.to(device) for o in self.offsets),
            heights=tuple(h.to(device) for h in self.heights),
            gt_positions=tuple(p.to(device) for p in self.gt_positions),
            gt_ee_vels=tuple(ee.to(device) for ee in self.gt_ee_vels),
        )
