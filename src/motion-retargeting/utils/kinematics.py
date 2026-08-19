from data.metadata import SkeletonMetadata
import torch

from src.core.types import SkeletonTopology

class ForwardKinematics:
    """Forward kinematics utilities."""

    @staticmethod
    def forward(
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        topology: SkeletonTopology | SkeletonMetadata,
        world: bool = True
    ) -> torch.Tensor:
        """
        Compute joint positions for unbatched motion.
        
        NOTE: This assume that the root quat is not included.
        """
        T, J, _ = quaternions.shape

        P = torch.zeros(T, J+1, 3, device=quaternions.device)
        R = torch.zeros(T, J+1, 3, 3, device=quaternions.device)

        rotmats = ForwardKinematics.quat_to_rotmat(quaternions)

        P[:, 0, :] = root_pos
        R[:, 0, :, :] = torch.eye(3, device=R.device, dtype=R.dtype)
        
        for (parent, child) in topology.edge_topology:
            R[:, child, :, :] = R[:, parent, :, :].clone() @ rotmats[:, child-1, :, :].clone()
            P[:, child, :] = ( R[:, parent, :, :] @ offsets[child, :, None] ).squeeze(-1) 

            if world:
                P[:, child, :] += P[:, parent, :]
        
        return P

    @staticmethod
    def forward_batched(
        quaternions: torch.Tensor,
        offsets: torch.Tensor,
        root_pos: torch.Tensor,
        topology: SkeletonTopology,
        world: bool = True,
    ) -> torch.Tensor:
        """
        Compute joint positions for batched motion.

        NOTE: This assume that the root quat is not included.
        """
        B, T, J, _ = quaternions.shape

        P = torch.zeros(B, T, J+1, 3, device=quaternions.device) 
        R = torch.zeros(B, T, J+1, 3, 3, device=quaternions.device) 
        
        rotmats = ForwardKinematics.quat_to_rotmat(quaternions) 
        
        P[:, :, 0, :] = root_pos.transpose(1,2) 
        R[:, :, 0, :, :] = torch.eye(3, device=R.device, dtype=R.dtype)
        
        for (parent, child) in topology.edge_topology:
            R[:, :, child, :, :] = R[:, :, parent, :, :].clone() @ rotmats[:, :, child-1, :, :].clone()
            P[:, :, child, :] = ( R[:, :, parent, :, :] @ offsets[:, None, child, :, None]).squeeze(-1) 

            if world:
                P[:, :, child, :] += P[:, :, parent, :]
            

        return P
    
    @staticmethod
    def local_to_world(positions: torch.Tensor, topology: torch.Tensor):
        """ Convert from local positions to world """
        positions = positions.clone()
        for parent, child in topology:
            positions[:, child, :] += positions[:, parent, :]

        return positions

    @staticmethod
    def quat_to_rotmat(quaternions: torch.Tensor,) -> torch.Tensor:
        """Convert unit quaternions to rotation matrices. Axis order is fixed x, y, z, w"""
        q = quaternions / quaternions.norm(dim=-1, keepdim=True)
        x, y, z, w = q.unbind(-1)

        R = torch.empty(q.shape[:-1] + (3, 3), device=q.device)

        R[..., 0, 0] = 1 - 2*(y*y + z*z)
        R[..., 0, 1] = 2*(x*y - w*z)
        R[..., 0, 2] = 2*(x*z + w*y)
        R[..., 1, 0] = 2*(x*y + w*z)
        R[..., 1, 1] = 1 - 2*(x*x + z*z)
        R[..., 1, 2] = 2*(y*z - w*x)
        R[..., 2, 0] = 2*(x*z - w*y)
        R[..., 2, 1] = 2*(y*z + w*x)
        R[..., 2, 2] = 1 - 2*(x*x + y*y)

        return R
