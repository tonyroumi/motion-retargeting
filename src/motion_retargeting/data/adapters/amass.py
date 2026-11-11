"""
Adapter for the AMASS dataset (SMPL-H `.npz` clips), via the `human_body_prior` BodyModel.

Every AMASS clip shares the same joint topology but has its own betas, so each clip's
MotionData carries its subject's offsets.
"""

from __future__ import annotations

from pathlib import Path
from typing import List

import numpy as np
import torch

from motion_retargeting.data.representations.motion import MotionData

from motion_retargeting.config.paths import PATHS
from motion_retargeting.data.adapters.base import MotionSourceAdapter
from motion_retargeting.kinematics.quaternions import quat_mul

class AMASSAdapter(MotionSourceAdapter):
    cache_version = "metres-v2"
    NUM_JOINTS = 22  # Number of body joints (excluding fingers)

    def __init__(
        self,
        model_type: str = "smplh",
        num_betas: int = 16,
        device: str = "cpu",
        body_models: str | Path | None = None,
    ):
        super().__init__("amass", device=device)

        self.model_type = model_type
        self.num_betas = num_betas
        self.body_models = PATHS.body_models if body_models is None else Path(body_models).expanduser()

        self._bms = {}
        
    def _extract(self) -> List[MotionData]:
        files = sorted(p for p in self.raw_dir.rglob(f"*.npz") 
                       if p.name != "shape.npz")

        from tqdm import tqdm

        return [self._read_clip(path) for path in tqdm(files, desc="Extracting AMASS Data")]

    def _read_clip(self, path: Path, dtype: torch.dtype = torch.float32) -> MotionData:
        """Read one clip without touching caches; optional FP64 preserves source motion precision."""
        with np.load(path) as data:
            bm = self._body_model(np.asarray(data["gender"]).item())
            fps = float(data.get("mocap_framerate", 30.0))
            poses = torch.as_tensor(data["poses"], dtype=dtype, device=self.device)
            trans = torch.as_tensor(data["trans"], dtype=dtype, device=self.device)
            betas = torch.as_tensor(data["betas"], dtype=torch.float32, device=self.device)

        parents = bm.kintree_table[0].long()[:self.NUM_JOINTS]
        # The installed body model is FP32; motion channels need not be quantized to it.
        with torch.no_grad():
            J0 = bm(betas=betas.unsqueeze(0)).Jtr[0, :self.NUM_JOINTS].to(dtype)
        offsets = torch.zeros_like(J0)
        offsets[1:] = J0[1:] - J0[parents[1:]]
        quat_rotations = axis_angle_to_quaternion(poses.reshape(len(poses), -1, 3))
        joint_rotations = quat_rotations[:, :self.NUM_JOINTS].clone()
        joint_rotations[:, 0] = zup_to_yup_quaternion(quat_rotations[:, 0])
        return self._construct_motion(
            parents=parents, offsets=offsets, joint_rotations=joint_rotations,
            root_position=zup_to_yup_vector(J0[0] + trans), fps=fps, dtype=dtype,
        )

    def _body_model(self, gender: str):
        if gender not in self._bms:
            from human_body_prior.body_model.body_model import BodyModel
            bm_path = str(self.body_models / self.model_type / gender / "model.npz")
            self._bms[gender] = BodyModel(
                bm_path,
                model_type=self.model_type,
                num_betas=self.num_betas
            ).to(self.device)
        return self._bms[gender]


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    """
    Convert axis-angle rotations [..., 3] to quaternions [..., 4] in (w, x, y, z) convention.
    """
    angle = torch.linalg.vector_norm(axis_angle, dim=-1, keepdim=True)
    half_angle = angle * 0.5

    # sinc(x) = sin(pi*x)/(pi*x), so sin(half_angle)/angle = 0.5 * sinc(half_angle/pi).
    # Using sinc keeps the axis*angle -> 0 limit well-defined without an epsilon guard.
    axis_scale = 0.5 * torch.sinc(half_angle / torch.pi)

    w = torch.cos(half_angle)
    xyz = axis_angle * axis_scale
    return torch.cat([w, xyz], dim=-1)


def zup_to_yup_quaternion(q: torch.Tensor) -> torch.Tensor:
    """
    Rotate a quaternion for -90 degrees around the x-axis
    """
    import math
    q_convert = torch.tensor([
        math.cos(-math.pi / 4),  # w
        math.sin(-math.pi / 4),  # x
        0.0,
        0.0,
    ], dtype=q.dtype, device=q.device)
    return quat_mul(q_convert, q)


def zup_to_yup_vector(x: torch.Tensor) -> torch.Tensor:
    """
    x: [..., 3]
    """
    return torch.stack([
        x[..., 0],
        x[..., 2],
        -x[..., 1],
    ], dim=-1)
