from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import numpy as np

@dataclass
class MotionClip:
    """A single motion sequence with pre-computed kinematic state."""

    name: str
    fps: float
    dt: float

    # Raw qpos (N, nq) — root free-joint + actuated DOFs
    frames: np.ndarray

    root_pos: np.ndarray       # (N, 3)
    root_rot: np.ndarray       # (N, 4)  quaternion (w, x, y, z)
    root_vel: np.ndarray       # (N, 3)
    root_ang_vel: np.ndarray   # (N, 3)
    body_pos: np.ndarray       # (N, nbody, 3)
    joint_pos: np.ndarray      # (N, njoint, 3)
    dof_vel: np.ndarray        # (N, nv)

    @property
    def num_frames(self) -> int:
        return int(self.frames.shape[0])

    @property
    def num_dof(self) -> int:
        return int(self.frames.shape[1])

    @property
    def duration(self) -> float:
        return self.num_frames * self.dt

    def get_frame(self, idx: int) -> np.ndarray:
        return self.frames[idx % self.num_frames]

    def get_frame_pair(self, idx: int) -> Tuple[np.ndarray, np.ndarray]:
        return self.get_frame(idx), self.get_frame(idx + 1)

    def save(self, path: str) -> None:
        np.savez_compressed(
            path,
            name=np.array(self.name),
            fps=np.float32(self.fps),
            dt=np.float32(self.dt),
            frames=self.frames.astype(np.float32),
            root_pos=self.root_pos.astype(np.float32),
            root_rot=self.root_rot.astype(np.float32),
            root_vel=self.root_vel.astype(np.float32),
            root_ang_vel=self.root_ang_vel.astype(np.float32),
            body_pos=self.body_pos.astype(np.float32),
            joint_pos=self.joint_pos.astype(np.float32),
            dof_vel=self.dof_vel.astype(np.float32),
        )

    @classmethod
    def load(cls, path: str) -> MotionClip:
        d = np.load(path, allow_pickle=False)
        return cls(
            name=str(d["name"]),
            fps=float(d["fps"]),
            dt=float(d["dt"]),
            frames=d["dof_pos"].astype(np.float32),
            root_pos=d["root_pos"].astype(np.float32),
            root_rot=d["root_rot"].astype(np.float32),
            root_vel=d["root_vel"].astype(np.float32),
            root_ang_vel=d["root_ang_vel"].astype(np.float32),
            body_pos=d["body_pos"].astype(np.float32),
            joint_pos=d["joint_pos"].astype(np.float32),
            dof_vel=d["dof_vel"].astype(np.float32),
        )
