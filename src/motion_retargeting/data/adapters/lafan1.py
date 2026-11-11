"""Generic metre MotionCollections from LAFAN1/Dog BVHs.

PAN selects the separate original-frame/cm raw loader, not these rebased tensors.
"""

from __future__ import annotations

import math
from pathlib import Path

from .lafan1_splits import reference_files
from .reference_bvh import ReferenceBVHAdapter


class LAFAN1Adapter(ReferenceBVHAdapter):
    cache_version = "reference-split-metres-v2"

    # LAFAN authors its Hips with local +X up the spine and +Y forward, so the builder's
    # heading axis would point at the ceiling. This half turn about (1, 1, 0) swaps them to
    # +X forward, +Y up, +Z right, so that a character walking forward moves along canonical +X.
    root_rest = (0.0, math.sqrt(0.5), math.sqrt(0.5), 0.0)
    joint_names = [
        "Hips", "LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe",
        "RightUpLeg", "RightLeg", "RightFoot", "RightToe",
        "Spine", "Spine1", "Spine2", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
    ]
    # The generic collection retains a complete, disjoint joint partition with
    # matching human/dog keys: limbs, trunk, head. PAN's raw/reference loaders use
    # their own partial correspondence masks instead of these generic labels.
    body_names = {
        0: joint_names[1:9] + joint_names[14:],
        1: ["Hips", "Spine", "Spine1", "Spine2"],
        2: ["Neck", "Head"],
    }

    def __init__(self, device: str = "cpu"):
        super().__init__("lafan1", device=device)
        if not self.raw_dir.exists() and (self.raw_dir.parent / "LAFAN1").exists():
            self.raw_dir = self.raw_dir.parent / "LAFAN1"

    def _files(self) -> list[Path]:
        return reference_files(self.raw_dir, "lafan1", "train")


class DogAdapter(ReferenceBVHAdapter):
    cache_version = "reference-split-metres-v2"

    # The dog's Hips has +Y up but +X pointing at its tail, with the half turn undone by `Spine`.
    # This half turn about Y gives the same frame as the human, so both domains' canonical root
    # velocities agree instead of having opposite signs.
    root_rest = (0.0, 0.0, 1.0, 0.0)
    joint_names = [
        "Hips", "Spine", "Spine1", "Neck", "Head",
        "LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand",
        "RightShoulder", "RightArm", "RightForeArm", "RightHand",
        "LeftUpLeg", "LeftLeg", "LeftFoot", "RightUpLeg", "RightLeg", "RightFoot",
        "Tail", "Tail1",
    ]
    body_names = {
        0: joint_names[5:19],
        1: ["Hips", "Spine", "Spine1", "Tail", "Tail1"],
        2: ["Neck", "Head"],
    }

    def __init__(self, device: str = "cpu"):
        super().__init__("dog", device=device)
        if not self.raw_dir.exists() and (self.raw_dir.parent / "Dog").exists():
            self.raw_dir = self.raw_dir.parent / "Dog"

    def _files(self) -> list[Path]:
        return reference_files(self.raw_dir, "dog", "train")
