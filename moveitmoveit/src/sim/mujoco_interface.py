from __future__ import annotations

from typing import Dict, Optional

import mujoco
import numpy as np

from .skeleton import Skeleton
from .sim_interface import SimInterface, SimParams

class MujocoInterface(SimInterface):
    """
    MuJoCo C-backend simulation interface.
    """

    def __init__(
        self,
        skeleton: Skeleton,
        data: mujoco.MjData,
        params: SimParams,
    ) -> None:
        super().__init__(skeleton, params)
        self._model = skeleton.model
        self._data = data
        self._model.opt.timestep = 1.0 / params.sim_freq

    def _set_timestep(self, dt: float) -> None:
        self._model.opt.timestep = dt

    @property
    def timestep(self) -> float:
        return self._model.opt.timestep

    @property
    def root_pos(self) -> np.ndarray:
        """Root position (3,)."""
        return self._data.qpos[:3].copy()

    @root_pos.setter
    def root_pos(self, value: np.ndarray) -> None:
        self._data.qpos[:3] = value

    @property
    def root_quat(self) -> np.ndarray:
        """Root orientation as a scalar-first quaternion (4,)."""
        return self._data.qpos[3:7].copy()

    @root_quat.setter
    def root_quat(self, value: np.ndarray) -> None:
        self._data.qpos[3:7] = value

    @property
    def root_vel(self) -> np.ndarray:
        """Root linear velocity (3,) in world frame."""
        return self._data.qvel[:3].copy()

    @root_vel.setter
    def root_vel(self, value: np.ndarray) -> None:
        self._data.qvel[:3] = value

    @property
    def root_ang_vel(self) -> np.ndarray:
        """Root angular velocity (3,) in world frame."""
        return self._data.qvel[3:6].copy()

    @root_ang_vel.setter
    def root_ang_vel(self, value: np.ndarray) -> None:
        self._data.qvel[3:6] = value

    @property
    def dof_pos(self) -> np.ndarray:
        """Joint positions excluding the root free-joint (nq - 7,)."""
        return self._data.qpos[7:].copy()

    @dof_pos.setter
    def dof_pos(self, value: np.ndarray) -> None:
        self._data.qpos[7:] = value

    @property
    def dof_vel(self) -> np.ndarray:
        """Joint velocities excluding the root free-joint (nv - 6,)."""
        return self._data.qvel[6:].copy()

    @dof_vel.setter
    def dof_vel(self, value: np.ndarray) -> None:
        self._data.qvel[6:] = value

    @property
    def body_pos(self) -> np.ndarray:
        """All body positions (nbody, 3) in world frame."""
        return self._data.xpos.copy()

    @body_pos.setter
    def body_pos(self, value: np.ndarray) -> None:
        self._data.xpos = value

    @property
    def dof_pos(self) -> np.ndarray:
        """All joints positions in world frame (qpos)."""
        return self._data.qpos[7:].copy()

    @dof_pos.setter
    def dof_pos(self, value: np.ndarray) -> None:
        self._data.qpos[7:] = value

    @property
    def body_quat(self) -> np.ndarray:
        """All body orientations (nbody, 4) in world frame (xquat)."""
        return self._data.xquat.copy()

    @property
    def ee_positions(self) -> np.ndarray:
        """End-effector positions (n_ee, 3) in world frame."""
        return self._data.xpos[self._ee_ids].copy()

    def get_ee_position(self, name: str) -> np.ndarray:
        bid = self._skel.get_body(name).mj_id
        return self._data.xpos[bid].copy()

    def get_state(self) -> Dict:
        return {
            "qpos": self._data.qpos.copy(),
            "qvel": self._data.qvel.copy(),
            "time": float(self._data.time),
        }

    def set_state(self, state: Dict) -> None:
        self._data.qpos[:] = state["qpos"]
        self._data.qvel[:] = state["qvel"]
        self._data.time = state["time"]
        mujoco.mj_forward(self._model, self._data)


    def init_from_reference_motion(
        self,
        ref_qpos: np.ndarray,
        ref_qvel: Optional[np.ndarray] = None,
    ) -> None:
        """
        Init the simulation state from a reference-motion frame.
        """
        self._data.qpos[:] = ref_qpos
        if ref_qvel is not None:
            self._data.qvel[:] = ref_qvel
        else:
            self._data.qvel[:] = 0.0
        self._data.time = 0.0
        mujoco.mj_forward(self._model, self._data)

    def init_from_keyframe(self, keyframe_id: int) -> None:
        """Initialize simulation from a keyframe."""
        self._data.qpos[:] = self._model.key_qpos[keyframe_id]
        self._data.qvel[:] = self._model.key_qvel[keyframe_id]
        self._data.time = 0.0
        mujoco.mj_forward(self._model, self._data)

    def step(self, ctrl: Optional[np.ndarray] = None) -> None:
        for _ in range(self._sim_steps_per_ctrl):
            if ctrl is not None:
                self._data.ctrl[:] = ctrl
            mujoco.mj_step(self._model, self._data)
