"""
Quick smoke test for MotionDatasetBuilder.build().
"""

import torch

from motion_retargeting.data.builder import MotionDatasetBuilder, MotionDatasetBuilderConfig
from motion_retargeting.data.adapters import AMASSAdapter


def test_builder_build_produces_normalized_dataset():
    torch.manual_seed(0)

    source = AMASSAdapter(device="cuda")
    config = MotionDatasetBuilderConfig(sequence_length=16, stride=8, target_fps=30)

    dataset = MotionDatasetBuilder(config).build([source])


if __name__ == "__main__":
    test_builder_build_produces_normalized_dataset()
