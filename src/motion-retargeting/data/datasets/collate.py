import torch

from src.core.types import PairedSample

def paired_collate(batch):
    rotations = tuple(
        torch.stack([b.rotations[i] for b in batch])
        for i in range(2)
    )

    motions = tuple(
        torch.stack([b.motions[i] for b in batch])
        for i in range(2)
    )

    offsets = tuple(
        torch.stack([b.offsets[i] for b in batch])
        for i in range(2)
    )

    heights = tuple(
        torch.stack([b.heights[i] for b in batch])
        for i in range(2)
    )

    gt_positions = tuple(
        torch.stack([b.gt_positions[i] for b in batch])
        for i in range(2)
    )

    gt_ee_vels = tuple(
        torch.stack([b.gt_ee_vels[i] for b in batch])
        for i in range(2)
    )

    return PairedSample(
        rotations=rotations,
        motions=motions,
        offsets=offsets,
        heights=heights,
        gt_positions=gt_positions,
        gt_ee_vels=gt_ee_vels)

