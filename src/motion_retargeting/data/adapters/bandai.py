# """
# Adapter for the BANDAI-Namco motion dataset, stored as BVH files.
# """

# from .base import MotionSourceAdapter

# from tqdm import tqdm
# from typing import List, Optional, Tuple
# import bvhio
# import torch

# from motion_retargeting.data.representations.motion import Motion, Motions
# from motion_retargeting.data.representations.skeleton import Skeleton

# CENTIMETERS_TO_METERS = 0.01


# class BANDAIAdapter(MotionSourceAdapter):
#     UP_AXIS = "y"

#     def __init__(self, cache: bool = True, device: str = "cpu"):
#         super().__init__("bandai", device=device, file_extension="bvh")

#         self._skeleton, motions = self.extract(cache=cache)
#         self._motions = [Motions(motions, self._skeleton)]
#         self.to_z_up()

#     def extract(self, cache: bool) -> Tuple[Skeleton, List[Motion]]:
#         if cache:
#             cached = self.load_cache()
#             if cached is not None:
#                 return cached

#         skeleton: Optional[Skeleton] = None
#         motions: List[Motion] = []

#         for bvh_file in tqdm(self.files, desc="Extracting Bandai Data"):
#             data = bvhio.readAsBvh(bvh_file)

#             # Every BANDAI clip shares one skeleton: build it from the first file.
#             if skeleton is None:
#                 skeleton = self.assign_joint_bodies(bvh_skeleton(data.Root, self.device))

#             joints = [joint for joint, _, _ in data.Root.layout()]

#             num_frames = data.FrameCount

#             # BVH positions are in centimeters; convert to meters.
#             root_positions = torch.as_tensor(
#                 [list(pose.Position) for pose in joints[0].Keyframes],
#                 dtype=torch.float32,
#                 device=self.device,
#             ) * CENTIMETERS_TO_METERS

#             # bvhio quaternions iterate as (w, x, y, z), matching the convention
#             # used by `motion_retargeting.kinematics`.
#             local_rotations = torch.as_tensor(
#                 [
#                     [list(joint.Keyframes[t].Rotation) for joint in joints]
#                     for t in range(num_frames)
#                 ],
#                 dtype=torch.float32,
#                 device=self.device,
#             )

#             motion = Motion(
#                 root_positions=root_positions,
#                 # BVH root joint (joint_Root) rotation, matching root_positions.
#                 root_rotations=local_rotations[:, 0].clone(),
#                 local_rotations=local_rotations,
#                 fps=1.0 / data.FrameTime,
#                 metadata={"name": bvh_file.name},
#             )

#             motions.append(motion)

#         if cache:
#             self.save_cache(skeleton, motions)

#         return skeleton, motions


# def bvh_skeleton(root, device: str) -> Skeleton:
#     """ Skeleton (offsets in meters) from a bvhio root joint's hierarchy. """
#     joint_names: List[str] = []
#     parents: List[int] = []
#     raw_offsets: List[list] = []

#     def visit(joint, parent_idx: int) -> None:
#         joint_names.append(joint.Name)
#         parents.append(parent_idx)
#         raw_offsets.append(list(joint.Offset))

#         my_idx = len(joint_names) - 1
#         for child in joint.Children:
#             visit(child, my_idx)

#     visit(root, -1)

#     # BVH offsets are in centimeters; convert to meters.
#     offsets = torch.as_tensor(raw_offsets, dtype=torch.float32, device=device) * CENTIMETERS_TO_METERS
#     offsets[0] = torch.zeros(3, device=device)  # root has no parent offset

#     return Skeleton(
#         joint_names=joint_names,
#         parents=torch.as_tensor(parents, dtype=torch.long, device=device),
#         offsets=offsets,
#     )
