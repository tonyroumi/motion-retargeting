"""
Skeleton metadata and motion sequence dataclasses.
"""
from dataclasses import dataclass
import numpy as np

@dataclass
class SkeletonMetadata:
    edge_topology: np.ndarray
    offsets: np.ndarray # (J-1, 3) # cm
    ee_ids: np.ndarray
    height: np.ndarray # cm
    kintree: np.ndarray

    def save(self, path: str):
        np.savez_compressed(
            path,
            edge_topology=self.edge_topology,
            offsets=self.offsets,
            ee_ids=self.ee_ids,
            height=self.height,
            kintree=self.kintree,
        )
        print(f"Saved skeleton: {path}")

    @classmethod
    def load(cls, path: str) -> "SkeletonMetadata":
        d = np.load(path, allow_pickle=False)

        edge_topology = d["edge_topology"]
        offsets = d["offsets"]
        ee_ids = d["ee_ids"]
        height = d["height"]
        kintree = d["kintree"]
        return cls(
            edge_topology=edge_topology,
            offsets=offsets,
            ee_ids=ee_ids,
            height=height,
            kintree=kintree,
        )
    
    def __eq__(self, other) -> bool:
        if not isinstance(other, SkeletonMetadata):
            return NotImplemented

        return (
            np.array_equal(self.edge_topology, other.edge_topology) and
            np.array_equal(self.ee_ids, other.ee_ids) and
            np.array_equal(self.kintree, other.kintree)
        )

@dataclass
class MotionSequence:
    name: str
    positions: np.ndarray # (T, J, 3) , cm
    rotations: np.ndarray # (T, J, 4) , (x, y, z, w) radians
    fps: float = 60

    def save(self, path: str):
        np.savez_compressed(
            path,
            name=self.name,
            positions=self.positions,
            rotations=self.rotations,
            fps=self.fps,
        )
        print(f"Saved motion sequence: {path}")

    @classmethod
    def load(cls, path: str) -> "MotionSequence":
        d = np.load(path, allow_pickle=False)

        name = d["name"]
        positions = d["positions"]
        rotations = d["rotations"]
        fps = d["fps"]

        return cls(
            name=name,
            positions=positions,
            rotations=rotations,
            fps=fps,
        )
