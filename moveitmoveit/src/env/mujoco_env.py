from __future__ import annotations

from abc import abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

import mujoco
import numpy as np
import gymnasium as gym
from gymnasium import spaces

from moveitmoveit.src.sim import Skeleton, MujocoInterface, SimParams
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer

@dataclass(frozen=True)
class MujocoEnvParams(SimParams):
    max_episode_time: float = 10.0  # seconds

class MujocoEnv(gym.Env):
    """Base MuJoCo environment with a gym-compatible interface."""

    metadata = {
        "render_modes": ["human", "rgb_array", "depth_array"],
        "render_fps": 30,
    }

    def __init__(self, model_path: str, params: MujocoEnvParams, render_mode: str = None) -> None:
        super().__init__()
        self._mj_model = mujoco.MjModel.from_xml_path(model_path)
        self._mj_data = mujoco.MjData(self._mj_model)

        self.skeleton = Skeleton(self._mj_model)
        self.sim = MujocoInterface(self.skeleton, self._mj_data, params=params)
        self.params = params

        self._init_pose = self.skeleton.default_pose

        self._episode_step = 0

        a_low, a_high = self.skeleton.compute_action_bounds()
        self.action_space = spaces.Box(low=a_low, high=a_high, dtype=np.float32)

        obs = self._get_obs()
        self.observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=obs.shape, dtype=np.float32,
        )

        self.render_mode = render_mode
        self.mujoco_renderer = MujocoRenderer(
            self._mj_model,
            self._mj_data,
        )

    @property
    def episode_time(self) -> float:
        """Elapsed time in seconds for the current episode."""
        return self._episode_step * self.sim.timestep

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        self.sim.init_from_keyframe(0)
        self._episode_step = 0
        obs = self._get_obs()
        return obs, {}

    def step(
        self, action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        self.sim.step(self._action_to_ctrl(action))
        self._episode_step += 1

        obs = self._get_obs()
        reward = self._compute_reward()
        terminated = self._check_termination()
        truncated = self.episode_time >= self.params.max_episode_time

        return obs, reward, terminated, truncated, {}

    def _action_to_ctrl(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64)
        low = self.action_space.low
        high = self.action_space.high
        return np.minimum(np.maximum(a, low), high)

    def render(self):
        return self.mujoco_renderer.render(self.render_mode)

    def close(self):
        if hasattr(self, "mujoco_renderer"):
            self.mujoco_renderer.close()

    @abstractmethod
    def _get_obs(self) -> np.ndarray:
        ...

    @abstractmethod
    def _compute_reward(self) -> float:
        ...

    @abstractmethod
    def _check_termination(self) -> bool:
        ...
