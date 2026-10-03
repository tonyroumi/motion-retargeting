from typing import List

import numpy as np
import torch

from motion_retargeting.data.representations.motion import MotionData

from motion_retargeting.config.paths import PATHS
from motion_retargeting.data.adapters.base import MotionSourceAdapter
from motion_retargeting.kinematics.quaternions import quat_mul

class AMASSAdapter(MotionSourceAdapter):
    NUM_JOINTS = 22  # Number of body joints (excluding fingers)

    def __init__(
        self,
        model_type: str = "smplh",
        num_betas: int = 16,
        device: str = "cpu"
    ):
        super().__init__("amass", device=device)

        self.model_type = model_type
        self.num_betas = num_betas

        self._bms = {}
        
    def _extract(self) -> List[MotionData]:
        files = sorted(p for p in self.raw_dir.rglob(f"*.npz") 
                       if p.name != "shape.npz")

        from tqdm import tqdm

        motions: List[MotionData] = []

        for npz_file in tqdm(files, desc="Extracting AMASS Data"):
            data = np.load(npz_file)
            
            bm = self._body_model(np.asarray(data["gender"]).item())

            fps = float(data.get("mocap_framerate", 30.0))
            poses = torch.as_tensor(data["poses"], dtype=torch.float32, device=self.device)
            trans = torch.as_tensor(data["trans"], dtype=torch.float32, device=self.device)
            betas = torch.as_tensor(data["betas"], dtype=torch.float32, device=self.device)

            num_frames = poses.shape[0]

            parents = bm.kintree_table[0].long()[:self.NUM_JOINTS] # Clip fingers

            J0 = bm(betas=betas.unsqueeze(0)).Jtr[0, :self.NUM_JOINTS]
            offsets = torch.zeros_like(J0)
            offsets[1:] = J0[1:] - J0[parents[1:]]  # Parent-relative offsets (root offset zero)

            rots = poses.reshape(num_frames, -1, 3)

            quat_rotations = axis_angle_to_quaternion(rots)

            root_positions = J0[0] + trans
            root_rotation = quat_rotations[:, 0]

            # Z-up -> canonical Y-up
            root_positions_yup = zup_to_yup_vector(root_positions)
            root_rotation_yup = zup_to_yup_quaternion(root_rotation)
            joint_rotations = quat_rotations[:, :self.NUM_JOINTS] # Remove the root and clip fingers
            joint_rotations[:, 0] = root_rotation_yup

            motion = self._construct_motion(
                parents=parents,
                offsets=offsets,
                joint_rotations=joint_rotations,
                root_position=root_positions_yup,
                fps=fps,
            )

            motions.append(motion)

        return motions

    def _body_model(self, gender: str):
        if gender not in self._bms:
            from human_body_prior.body_model.body_model import BodyModel
            bm_path = str(PATHS.body_models / self.model_type / gender / "model.npz")
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
    ])
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
