"""
Abstract base class for physics simulation backends.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, Optional

from moveitmoveit.src.types import BaseParams

from .skeleton import Skeleton

@dataclass(frozen=True)
class SimParams(BaseParams):
    sim_freq: float = 120.0
    control_freq: float = 30.0

class SimInterface(ABC):
    """ Abstract simulation interface. """

    def __init__(
        self,
        skeleton: Skeleton,
        params: SimParams,
    ) -> None:
        self._skel = skeleton
        self._ee_ids = skeleton.ee_ids
        self._sim_freq = params.sim_freq
        self._sim_steps_per_ctrl = int(params.sim_freq / params.control_freq)

    @property
    def sim_freq(self) -> float:
        return self._sim_freq

    @sim_freq.setter
    def sim_freq(self, freq: float) -> None:
        self._sim_freq = freq
        self._set_timestep(1.0 / freq)

    @property
    def dt(self) -> float:
        return 1.0 / self._sim_freq

    @abstractmethod
    def _set_timestep(self, dt: float) -> None:
        """Push the new timestep into the underlying simulator model."""
        ...

    @property
    @abstractmethod
    def root_pos(self):
        """Root position (3,)."""
        ...

    @root_pos.setter
    @abstractmethod
    def root_pos(self, value) -> None: ...

    @property
    @abstractmethod
    def root_quat(self):
        """Root orientation as a scalar-first quaternion (4,)."""
        ...

    @root_quat.setter
    @abstractmethod
    def root_quat(self, value) -> None: ...

    @property
    @abstractmethod
    def root_vel(self):
        """Root linear velocity (3,) in world frame."""
        ...

    @root_vel.setter
    @abstractmethod
    def root_vel(self, value) -> None: ...

    @property
    @abstractmethod
    def root_ang_vel(self):
        """Root angular velocity (3,) in world frame."""
        ...

    @root_ang_vel.setter
    @abstractmethod
    def root_ang_vel(self, value) -> None: ...


    @property
    @abstractmethod
    def dof_pos(self):
        """Joint positions excluding root free-joint (nq - 7,)."""
        ...

    @dof_pos.setter
    @abstractmethod
    def dof_pos(self, value) -> None: ...

    @property
    @abstractmethod
    def dof_vel(self):
        """Joint velocities excluding root free-joint (nv - 6,)."""
        ...

    @dof_vel.setter
    @abstractmethod
    def dof_vel(self, value) -> None: ...

    @property
    @abstractmethod
    def body_pos(self):
        """All body positions (nbody, 3) in world frame."""
        ...

    @body_pos.setter
    @abstractmethod
    def body_pos(self, value) -> None: ...

    @property
    @abstractmethod
    def dof_pos(self):
        """Full qpos vector."""
        ...

    @dof_pos.setter
    @abstractmethod
    def dof_pos(self, value) -> None: ...

    @property
    @abstractmethod
    def body_quat(self):
        """All body orientations (nbody, 4) in world frame (xquat)."""
        ...

    @property
    @abstractmethod
    def ee_positions(self):
        """End-effector positions (n_ee, 3) in world frame."""
        ...

    @abstractmethod
    def get_ee_position(self, name: str):
        """Position of a single end-effector by body name."""
        ...


    @abstractmethod
    def get_state(self) -> Dict:
        """Return a serialisable snapshot of the current simulation state."""
        ...

    @abstractmethod
    def set_state(self, state: Dict) -> None:
        """Restore simulation state from a snapshot."""
        ...

    @abstractmethod
    def init_from_reference_motion(
        self,
        ref_qpos,
        ref_qvel=None,
    ) -> None:
        """Initialize simulation from a reference-motion frame."""
        ...

    @abstractmethod
    def init_from_keyframe(self, keyframe_id: int) -> None:
        """Initialize simulation from a keyframe."""
        ...

    @abstractmethod
    def step(self, ctrl=None) -> None:
        """Advance the simulation by one timestep."""
        ...
