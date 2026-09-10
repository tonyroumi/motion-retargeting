from .array import ArrayLike, ArrayUtils

from scipy.spatial.transform import Rotation as R
import numpy as np
import torch

class RotationUtils:
    """Rotation representation conversion utilities for numpy arrays and tensors."""

    @staticmethod
    def aa_to_quat(
        axis_angles: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert axis-angle vectors to quaternions [x, y, z, w]."""
        out = R.from_rotvec(ArrayUtils.to_numpy(axis_angles)).as_quat()
        return ArrayUtils.to_torch(out, device) if return_torch else out

    @staticmethod
    def aa_to_rotmat(
        axis_angles: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert axis-angle vectors to rotation matrices."""
        out = R.from_rotvec(ArrayUtils.to_numpy(axis_angles)).as_matrix()
        return ArrayUtils.to_torch(out, device) if return_torch else out

    @staticmethod
    def matrix_to_quat(
        matrices: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert rotation matrices to quaternions [x, y, z, w]."""
        out = R.from_matrix(ArrayUtils.to_numpy(matrices)).as_quat()
        return ArrayUtils.to_torch(out, device) if return_torch else out

    @staticmethod
    def euler_to_quat(
        euler_angles: ArrayLike,
        axis_order: str = 'xyz',
        return_torch: bool = False,
        device: torch.device = None,
    ):
        """Convert euler angles to quaternions [x, y, z, w]."""
        out = R.from_euler(axis_order, ArrayUtils.to_numpy(euler_angles), degrees=False).as_quat()
        return ArrayUtils.to_torch(out, device) if return_torch else out

    @staticmethod
    def euler_to_matrix(
        euler_angles: ArrayLike,
        axis_order: str = 'xyz',
        return_torch: bool = False,
        device: torch.device = None
    ):
        """Convert euler angles to quaternions [x, y, z, w]."""
        out = R.from_euler(axis_order, ArrayUtils.to_numpy(euler_angles), degrees=False).as_matrix()
        return ArrayUtils.to_torch(out, device) if return_torch else out
    
    @staticmethod
    def wxyz_to_xyzw(
        quat: ArrayLike,
        return_torch: bool = False,
        device: torch.device = None
    ):
        q = ArrayUtils.to_numpy(quat)
        out = q[..., [1, 2, 3, 0]]  
        return ArrayUtils.to_torch(out, device) if return_torch else out