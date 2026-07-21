"""
Skeleton wrapper around a MuJoCo model.
"""

from __future__ import annotations

import enum
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np
import mujoco
import moveitmoveit.src.transforms as transforms

class JointType(enum.Enum):
    """Semantic joint classification used by the framework."""
    HINGE = "hinge"
    SLIDE = "slide"
    BALL = "ball"
    FREE = "free"
    SPHERICAL = "spherical" 

@dataclass
class Joint:
    """Metadata for a single MuJoCo joint."""
    name: str
    id: int                         # joint id in mj_model
    semantic_type: JointType
    dof_idx: int                    # index into qvel / ctrl
    qpos_offset: int                # index into qpos
    n_dof: int
    axis: Optional[np.ndarray]      # (3,) joint axis in body frame
    range: np.ndarray               # (2,) joint limits [lo, hi]
    has_limits: bool

    def dof_to_quat(self, dof):
        if self.semantic_type == JointType.HINGE:
            axis = self.axis
            quat = transforms.axis_angle_to_quat(axis, dof[:, 0])
        elif self.semantic_type == JointType.SPHERICAL:
            q0 = transforms.axis_angle_to_quat(self.axis[0], dof[:, 0])
            q1 = transforms.axis_angle_to_quat(self.axis[1], dof[:, 1])
            q2 = transforms.axis_angle_to_quat(self.axis[2], dof[:, 2])
            quat = transforms.quat_mul(transforms.quat_mul(q0, q1), q2)
        return quat

@dataclass
class Body:
    """Metadata for a single MuJoCo body."""
    name: str
    id: int
    parent_id: int
    total_dof: int = 0
    position: Optional[np.ndarray] = None
    rotation: Optional[np.ndarray] = None
    joints: List[Joint] =  field(default_factory=list)

class Skeleton:
    """
    High-level wrapper around ``mujoco.MjModel`` that exposes joint / body
    topology, action-bound computation, and rotation encoding helpers.
    """

    def __init__(self, model: mujoco.MjModel) -> None:
        self._model = model

        self._joints: List[Joint] = []
        self._bodies: List[Body] = []
        self._ee_ids = self._find_ee_bodies()

        self.default_pose = model.key_qpos[0].copy()

        self._parse()

    def _parse(self) -> None:
        """Walk the model and build joint / body metadata."""
        m = self._model

        for bid in range(m.nbody):
            bname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid)
            if bname == "world":
                continue
            parent = m.body_parentid[bid]
            local_position = m.body_pos[bid].copy()
            local_rotation = m.body_quat[bid].copy()
            dof_num = m.body_dofnum[bid]
            start_jnt_addr = m.body_jntadr[bid]
            num_joints = m.body_jntnum[bid]

            jnts = []
            for j in range(num_joints):    
                jid = start_jnt_addr + j           
                jname = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, jid)
                jtype = m.jnt_type[jid]
                bid = m.jnt_bodyid[jid]

                n_dof = {
                    mujoco.mjtJoint.mjJNT_FREE: 6,
                    mujoco.mjtJoint.mjJNT_BALL: 3,
                    mujoco.mjtJoint.mjJNT_HINGE: 1,
                    mujoco.mjtJoint.mjJNT_SLIDE: 1,
                }[jtype]

                sem_type = {
                    mujoco.mjtJoint.mjJNT_FREE: JointType.FREE,
                    mujoco.mjtJoint.mjJNT_BALL: JointType.BALL,
                    mujoco.mjtJoint.mjJNT_HINGE: JointType.HINGE,
                    mujoco.mjtJoint.mjJNT_SLIDE: JointType.SLIDE,
                }[jtype]

                has_limits = bool(m.jnt_limited[jid])
                jnt_range = m.jnt_range[jid].copy() if has_limits else np.array([-np.pi, np.pi])

                joint = Joint(
                    name=jname,
                    id=jid,
                    semantic_type=sem_type,
                    dof_idx=m.jnt_dofadr[jid],
                    qpos_offset=m.jnt_qposadr[jid],
                    n_dof=n_dof,
                    axis=m.jnt_axis[jid].copy(),
                    range=jnt_range,
                    has_limits=has_limits,
                )

                jnts.append(joint)
               
            
            if len(jnts) == 3:
                spherical = self._parse_spherical(jnts)
                if spherical is not None:
                    jnts = [spherical]

            body = Body(
                name=bname,
                id=bid, 
                parent_id=parent, 
                total_dof=dof_num,
                position=local_position, 
                rotation=local_rotation, 
                joints=jnts
            )

            self._joints.extend(jnts)
            self._bodies.append(body)

    def _parse_spherical(self, joints: List[Joint]) -> Optional[Joint]:
        """
        A *spherical joint* is a body that owns exactly 3 hinge joints whose
        axes span three distinct directions.
        """
        if not all(j.semantic_type == JointType.HINGE for j in joints):
            return  

        spherical = Joint(
            name=joints[0].name.rsplit("_", 1)[0],
            id=joints[0].id,
            semantic_type=JointType.SPHERICAL,
            dof_idx=joints[0].dof_idx,
            qpos_offset=joints[0].qpos_offset,
            n_dof=3,
            axis=np.stack([j.axis for j in joints]),
            range=np.stack([j.range for j in joints]),
            has_limits=True,
        )

        return spherical

    def _find_ee_bodies(self) -> List[int]:
        """
        Finds end-effector body IDs in a MuJoCo model.
        End effectors = leaf bodies in the kinematic tree.
        """
        parent = self.model.body_parentid  # shape (nbodies,)

        children = {i: [] for i in range(self.model.nbody)}
        for body_id, p in enumerate(parent):
            if p >= 0:
                children[p].append(body_id)

        # leaf bodies (ignore world body = 0)
        leaves = [b for b, c in children.items() if len(c) == 0 and b != 0]
        return leaves    

    @property
    def model(self) -> mujoco.MjModel:
        return self._model

    @property
    def num_bodies(self) -> int:
        return self._model.nbody

    @property
    def num_joints(self) -> int:
        return len(self._joints)

    @property
    def num_dof(self) -> int:
        return self._model.nv
    
    @property
    def num_actuated_dof(self) -> int:
        return sum([j.n_dof for j in self.get_actuated_joints()])

    @property
    def joints(self) -> List[Joint]:
        return self._joints

    @property
    def bodies(self) -> List[Body]:
        return self._bodies

    @property
    def ee_ids(self) -> List[int]:
        return self._ee_ids

    def get_joint(self, name: str) -> Joint:
        for j in self._joints:
            if j.name == name:
                return j
        raise KeyError(f"Joint '{name}' not found")

    def get_body(self, name: str) -> Body:
        for b in self._bodies:
            if b.name == name:
                return b
        raise KeyError(f"Body '{name}' not found")

    def get_actuated_joints(self) -> List[Joint]:
        """Return joints that are NOT free joints (i.e. actuatable DOFs)."""
        return [j for j in self._joints if j.semantic_type != JointType.FREE]

    def get_dof_offsets(self) -> np.ndarray:
        """Ordered array of dof addresses for all actuated joints."""
        return np.array([j.dof_idx for j in self.get_actuated_joints()])
    
    def dof_to_rot(self, joint_dof) -> np.ndarray:
        """ Convert MuJoCo's qpos representation into quaternion rotations"""
        N = joint_dof.shape[0]
        n_actuated = len(self.get_actuated_joints())
        joint_rot = np.zeros([N, n_actuated, 4], dtype=joint_dof.dtype)

        start = 0
        for i, jnt in enumerate(self.get_actuated_joints()):
            end = start + jnt.n_dof
            j_dof = joint_dof[..., start:end]
            j_rot = jnt.dof_to_quat(j_dof) #TODO[visualize]: visualize the joint rotations
            joint_rot[:, i] = j_rot
            start += jnt.n_dof
        return joint_rot

    def compute_dof_vel(self, joint_rot, dt: float) -> np.ndarray:
        joint_rot0 = joint_rot[:-1, :, :]
        joint_rot1 = joint_rot[1:, :, :]

        dof_dim = list(joint_rot0.shape[:-2]) + [self.num_actuated_dof]
        dof_vel = np.zeros(dof_dim, dtype=joint_rot0.dtype)

        drot = transforms.quat_mul(transforms.quat_conjugate(joint_rot0), joint_rot1)
        drot = transforms.normalize_quat(drot)

        end_idx = 0
        for i, jnt in enumerate(self.joints[1:]):
            j_drot = drot[:, i, :]

            match jnt.semantic_type:
                case JointType.HINGE:
                    j_axis = jnt.axis
                    j_dof_vel = transforms.quat_to_exp_map(j_drot) / dt
                    j_dof_vel = np.sum(j_axis * j_dof_vel, axis=-1, keepdims=True)
                case JointType.SPHERICAL:
                    j_dof_vel = transforms.quat_to_exp_map(j_drot) / dt
                case _:
                    continue

            start_idx = 0 if not end_idx else end_idx
            end_idx = start_idx + jnt.n_dof
            dof_vel[:, start_idx:end_idx] = j_dof_vel

        return dof_vel


    def compute_action_bounds(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute per-DOF action bounds centred on the action-point centre.

        For spherical joints the bounds are computed differently from
        simple hinge / slide joints.

        Returns
        -------
        low : np.ndarray, shape (n_act_dof,)
        high : np.ndarray, shape (n_act_dof,)
        """
        actuated = self.get_actuated_joints()
        n = sum(j.n_dof for j in actuated)
        low = np.full((n,), -1.0, dtype=np.float32)
        high = np.full((n,), 1.0, dtype=np.float32)

        idx = 0

        for joint in actuated:
            dof = joint.n_dof
            if joint.semantic_type == JointType.SPHERICAL:
                lo, hi = self._spherical_action_bounds(joint)
            else:
                lo, hi = self._standard_action_bounds(joint)
            
            lo = np.asarray(lo, dtype=np.float32)
            hi = np.asarray(hi, dtype=np.float32)

            low[idx: idx + dof] = lo
            high[idx: idx + dof] = hi
            idx += dof

        return low, high

    @staticmethod
    def _standard_action_bounds(joint: Joint) -> Tuple[np.ndarray, np.ndarray]:
        """
        Centered, symmetric action bounds with some cushion around the mean.
        """
        j_low, j_high = joint.range
        mid = 0.5 * (j_high + j_low)

        diff_high = np.abs(j_high - mid)
        diff_low = np.abs(j_low - mid)
        scale = np.maximum(diff_high, diff_low)
        scale = scale * 1.4

        low = mid - scale
        high = mid + scale
        return low, high

    @staticmethod
    def _spherical_action_bounds(joint: Joint) -> Tuple[np.ndarray, np.ndarray]:
        """
        """
        low = np.max(np.abs(joint.range[:, 0]))
        high = np.max(np.abs(joint.range[:, 1]))
        scale = np.maximum(low, high)
        scale = scale * 1.2
        return -scale, scale
