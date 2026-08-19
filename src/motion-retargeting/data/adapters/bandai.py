from utils import skeleton
from .base import BaseAdapter
from ..metadata import SkeletonMetadata, MotionSequence

from tqdm import tqdm
from typing import List, Tuple
import bvhio
import numpy as np
import os
import torch

from src.utils import ArrayUtils, RotationUtils, SkeletonUtils, ForwardKinematics, SkeletonVisualizer

class BANDAIAdapter(BaseAdapter):
    DATASET_NAME = "bandai"

    def __init__(self, device: torch.device):
        super().__init__(self.DATASET_NAME, device)

    def _post_init(self, character: str):
        self.character = character
        self.character_dir = self.raw_dir / character
        
        cache_dir = self.cache_dir / character
        cache_dir.mkdir(parents=True, exist_ok=True)

        self.motion_seqs = sorted([str(f) for f in self.character_dir.glob("*.bvh")])

        self.data = bvhio.readAsBvh(self.motion_seqs[0])

        self.joint_names = []
        self.parents = []

        self.build_kintree(self.data.Root, parent_idx=-1)

        self.kintree = np.array(self.parents, dtype=np.int32)
        self.num_joints = len(self.kintree)

    def extract_skeleton(self, character: str) -> SkeletonMetadata:
        self._post_init(character)

        offsets = ArrayUtils.to_numpy([
            joint.Offset
            for (joint, _, _) in self.data.Root.layout()
        ])
        offsets[0] = np.zeros(3) 
        offsets *= 100 # convert to cm

        edge_topology = SkeletonUtils.construct_edge_topology(self.kintree)
        ee_ids = SkeletonUtils.find_ee(self.kintree)

        height = SkeletonUtils.compute_height(self.kintree, offsets, ee_ids)

        skeleton = SkeletonMetadata(
            edge_topology=ArrayUtils.to_numpy(edge_topology),
            offsets=ArrayUtils.to_numpy(offsets),
            ee_ids=ArrayUtils.to_numpy(ee_ids),
            height=height,
            kintree=ArrayUtils.to_numpy(self.kintree),
        )

        skeleton.save(self.skeleton_dir / f"{character}.npz")
        return skeleton

    def extract_motion(self, character: str) -> List[MotionSequence]:
        self._post_init(character)

        skeleton = SkeletonMetadata.load(self.data_dir / "skeletons" / (str(character) + ".npz"))

        sequences: List[MotionSequence] = []

        for bvh_file in tqdm(self.motion_seqs, desc="Extracting motion sequences"):
            fname = os.path.basename(bvh_file)

            data = bvhio.readAsBvh(bvh_file)

            T = data.FrameCount
            frame_time = self.data.FrameTime
            fps = int(round(1.0 / frame_time))

            positions = torch.zeros((T, self.num_joints, 3))
            rotations = torch.zeros((T, self.num_joints, 4)) 

            for t in range(T):
                positions[t] = ArrayUtils.to_torch([joint.Keyframes[t].Position for (joint, _, _) in data.Root.layout()])
                rotations[t] = ArrayUtils.to_torch([joint.Keyframes[t].Rotation for (joint, _, _) in data.Root.layout()])
            
            rotations = RotationUtils.wxyz_to_xyzw(rotations, return_torch=True)
            rotations = rotations[:, 1:] 
        
            fk_positions = ForwardKinematics.forward(
                quaternions=rotations,
                offsets=skeleton.offsets,
                root_pos=positions[:,0],
                topology=skeleton
            )

            motion = MotionSequence(
                name=fname,
                positions=ArrayUtils.to_numpy(fk_positions), 
                rotations=ArrayUtils.to_numpy(rotations),
                fps=fps
            )

            out = self.cache_dir / character / f"{fname.split('.')[0]}.npz"
            motion.save(out)
            sequences.append(motion)

        return sequences

    def build_kintree(self, node, parent_idx: int):
        my_idx = len(self.joint_names)

        self.joint_names.append(node.Name)
        self.parents.append(parent_idx)

        for child in node.Children:
            self.build_kintree(child, my_idx)

    def _extract_offsets(self) -> np.ndarray:
        offsets = []
        def traverse(node):
            offsets.append(node.Offset)

            for child in node.Children:
                traverse(child)
        
        traverse(self.data.Root)

        return np.stack(offsets, axis=0)
