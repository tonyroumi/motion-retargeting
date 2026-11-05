"""The reference Mixamo character groups, from centimeter, Y-up BVHs.

Group B consumes the authors' already split *_m BVHs, not their .npy edge
rotation exports. No split-joint synthesis or heading removal is done here.
"""

from __future__ import annotations

import warnings
from pathlib import Path
from typing import List

from .reference_bvh import ReferenceBVHAdapter


GROUP_A_CHARACTERS = ("Aj", "BigVegas", "Kaya", "SportyGranny")
GROUP_B_CHARACTERS = (
    "Malcolm_m", "Remy_m", "Maria_m", "Jasper_m", "Knight_m",
    "Liam_m", "ParasiteLStarkie_m", "Pearl_m", "Michelle_m", "LolaB_m",
    "Pumpkinhulk_m", "Ortiz_m", "Paladin_m", "James_m", "Joe_m",
    "Olivia_m", "Yaku_m", "Timmy_m", "Racer_m", "Abe_m",
)

# Reference corps_name_1 and corps_name_mixamo2_m, as JOINTS, not edges.
GROUP_A_PARTS = {
    0: ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase"],
    1: ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase"],
    2: ["Pelvis", "Hips", "Spine", "Spine1", "Spine2"],
    3: ["Neck", "Head"],
    4: ["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"],
    5: ["RightShoulder", "RightArm", "RightForeArm", "RightHand"],
}
GROUP_B_PARTS = {
    0: ["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToeBase", "LeftToe_End"],
    1: ["RightUpLeg", "RightLeg", "RightFoot", "RightToeBase", "RightToe_End"],
    2: ["Hips", "Spine", "Spine1", "Spine1_split", "Spine2"],
    3: ["Neck", "Head", "HeadTop_End"],
    4: ["LeftShoulder", "LeftShoulder_split", "LeftArm", "LeftForeArm", "LeftHand"],
    5: ["RightShoulder", "RightShoulder_split", "RightArm", "RightForeArm", "RightHand"],
}


class MixamoAdapter(ReferenceBVHAdapter):
    def __init__(self, group: str = "a", device: str = "cpu"):
        if group not in ("a", "b"):
            raise ValueError("Mixamo group must be 'a' or 'b'")
        super().__init__(f"mixamo_{group}", device=device)
        # Both adapters read the same character tree, but have distinct caches.
        self.raw_dir = self.raw_dir.parent / "mixamo"
        self.characters = GROUP_A_CHARACTERS if group == "a" else GROUP_B_CHARACTERS
        self.body_names = GROUP_A_PARTS if group == "a" else GROUP_B_PARTS
        spine = self.body_names[2]
        self.joint_names = [spine[0]] + self.body_names[0] + self.body_names[1] + spine[1:]
        self.joint_names += self.body_names[3] + self.body_names[4] + self.body_names[5]

    def _files(self) -> List[Path]:
        files = []
        missing = []
        for character in self.characters:
            clips = sorted((self.raw_dir / character).rglob("*.bvh"))
            if not clips:
                missing.append(character)
            files.extend(clips)
        if missing:
            warnings.warn(f"{self.dataset_name}: no BVHs for {', '.join(missing)}; loaded group is incomplete", stacklevel=2)
        return files


class MixamoGroupBAdapter(MixamoAdapter):
    def __init__(self, device: str = "cpu"):
        super().__init__(group="b", device=device)
