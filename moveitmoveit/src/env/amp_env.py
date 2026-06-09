from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple, List

import numpy as np
from gymnasium import spaces

from moveitmoveit.src.motion import MotionLibrary
import moveitmoveit.src.transforms as transforms

from .mujoco_env import MujocoEnv, MujocoEnvParams

@dataclass(frozen=True)
class AmpEnvParams(MujocoEnvParams):
    num_disc_obs_steps: int = 10

    # Purely Tracking rewards

    tracking_sigma: float = 1.0
    tracking_root_pos_weight: float = 0.15
    tracking_root_rot_weight: float = 0.15
    tracking_root_vel_weight: float = 0.10
    tracking_joint_rot_weight: float = 0.20
    tracking_dof_vel_weight: float = 0.05
    tracking_ee_pos_weight: float = 0.10

class AMPEnv(MujocoEnv):
    """Combined goal + style RL environment."""

    def __init__(
        self,
        model_path: str,
        motion_clips: List[str],
        params: AmpEnvParams,
        render_mode: str = None
    ) -> None:
        super().__init__(model_path, params, render_mode)

        self.motion_lib = MotionLibrary(srcs=motion_clips, skeleton=self.skeleton)

        disc_obs = self._get_disc_obs()
        self.disc_observation_space = spaces.Box(
            low=-np.inf, high=np.inf, shape=disc_obs.shape, dtype=np.float32,
        )

        self._ref_disc_obs = np.zeros((self.params.num_disc_obs_steps, disc_obs.shape[0]), dtype=np.float32)

        self._motion_clip_id = 0
        self._motion_frame_id = 0

        self._timestep_buf = 0
        self._time_buf = 0

    def clone(self):
        return AMPEnv(
            model_path=self.model_path,
            motion_clips=self.motion_clips,
            params=self.params,
        )

    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> Tuple[np.ndarray, dict]:
        super().reset(seed=seed, options=options)
        
        self._timestep_buf = 0 

        qpos, qvel, frame_info = self.motion_lib.sample_start_state(rng=self.np_random)
        self._motion_clip_id = int(np.asarray(frame_info["clip_id"]).item())
        self._motion_frame_id = int(np.asarray(frame_info["motion_frames"]).item())

        self.sim.init_from_reference_motion(
            qpos, qvel,
        )

        obs = self._get_obs()

        # Discriminator observations are mixed with simulator state and motion lib window... TODO(better notes)
        ref_disc_obs = self._fetch_ref_disc_obs(frame_info["clip_id"], frame_info["motion_frames"])
        self._ref_disc_obs = ref_disc_obs.reshape(self.params.num_disc_obs_steps, -1)

        motion_ids = self.motion_lib.sample_motions(1, rng=self.np_random)
        motion_frames = self.motion_lib.sample_frames(motion_ids, rng=self.np_random)
        ref_disc_obs = self._fetch_ref_disc_obs(motion_ids, motion_frames)

        info = {
            "disc_obs": self._ref_disc_obs,
            "ref_disc_obs": ref_disc_obs,
            "clip_id": self._motion_clip_id,
            "frame_id": self._motion_frame_id,
        }
        return obs, info

    def _fetch_ref_disc_obs(self, clip_id: np.ndarray, motion_frames: np.ndarray) -> np.ndarray:
        """WHY DO WE DO THIS CLARIFYYYYYY"""
        # # Provided the timestep, obtain each state and next state for the observation window
        curr_frame = np.arange(0, self.params.num_disc_obs_steps)

        motion_frames = motion_frames + curr_frame

        root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos = (
            self.motion_lib.get_frame_data(clip_id, motion_frames)
        )

        root_rot = transforms.quat_pos(root_rot)
        joint_rot = transforms.quat_pos(joint_rot)

        root_rot_norm = transforms.quat_to_tan_norm(root_rot)
        joint_rot_norm = transforms.quat_to_tan_norm(joint_rot)
        ee_rel_pos = body_pos[:, self.skeleton.ee_ids, :] - root_pos[:, np.newaxis, :]
        root_height = root_pos[:, -1]

        disc_obs = np.concatenate([
            root_rot_norm.ravel(),
            root_vel.ravel(),
            root_ang_vel.ravel(),
            joint_rot_norm.ravel(),
            dof_vel.ravel(),
            ee_rel_pos.ravel(),
            root_height.ravel(),
        ])
        return disc_obs.astype(np.float32)

    def step(
        self,
        action: np.ndarray,
    ) -> Tuple[np.ndarray, float, bool, bool, dict]:
        obs, reward, terminated, truncated, _ = super().step(action)

        self._timestep_buf += 1
        self._time_buf = self.sim.timestep * self._timestep_buf

        # Circular buffer
        curr_disc_obs = self._get_disc_obs()
        self._ref_disc_obs = np.roll(
            self._ref_disc_obs, shift=-1, axis=0
        )
        self._ref_disc_obs[-1] = curr_disc_obs
        disc_obs = self._ref_disc_obs

        motion_ids = self.motion_lib.sample_motions(1, rng=self.np_random)
        motion_frames = self.motion_lib.sample_frames(motion_ids, rng=self.np_random)
        ref_disc_obs = self._fetch_ref_disc_obs(motion_ids, motion_frames)

        info = {
            "disc_obs": disc_obs,
            "ref_disc_obs": ref_disc_obs,
            "motion_frame": self._motion_frame_id,
        }
        return obs, reward, terminated, truncated, info

    def _get_obs(self) -> np.ndarray:
        root_pos = self.sim.root_pos
        root_rot = self.sim.root_quat
        root_vel = self.sim.root_vel
        root_ang_vel = self.sim.root_ang_vel
        dof_pos = self.sim.dof_pos
        dof_vel = self.sim.dof_vel
        ee_pos = self.sim.ee_positions

        joint_rot = self.skeleton.dof_to_rot(dof_pos[np.newaxis])

        root_rot = transforms.quat_pos(root_rot)
        joint_rot = transforms.quat_pos(joint_rot)

        root_rot_norm = transforms.quat_to_tan_norm(root_rot)
        joint_rot_norm = transforms.quat_to_tan_norm(joint_rot)
        ee_rel_pos = ee_pos - root_pos[np.newaxis]
        root_height = root_pos[-1]

        obs = np.concatenate([
            root_rot_norm.ravel(),
            root_vel.ravel(),
            root_ang_vel.ravel(),
            joint_rot_norm.ravel(),
            dof_vel.ravel(),
            ee_rel_pos.ravel(),
            np.atleast_1d(root_height),
        ])
        return obs.astype(np.float32)

    def _get_disc_obs(self) -> np.ndarray:
        """Discriminator observation vector."""
        return self._get_obs()

    def _compute_tracking_cost(self) -> float:
        """Squared tracking error vs the current reference motion frame."""
        current_frame = self._motion_frame_id + self._timestep_buf 
        
        ref = self.motion_lib.get_frame_state(self._motion_clip_id, current_frame)

        # ref_root_pos = ref["root_pos"]
        # sim_root_pos = self.sim.root_pos

        # root_pos_err = sim_root_pos - ref_root_pos
        root_rot_err = self.sim.root_quat - ref["root_rot"]

        ref_joint_rot = transforms.quat_pos(ref["joint_rot"])
        sim_joint_rot = self.skeleton.dof_to_rot(self.sim.dof_pos[np.newaxis])[0]#transforms.quat_pos(
           # self.skeleton.dof_to_rot(self.sim.dof_pos[np.newaxis])[0],
        #)
        joint_rot_err =  sim_joint_rot - ref["joint_rot"]

        dof_vel_err = self.sim.dof_vel - ref["dof_vel"]

        # ee_pos_err = self.sim.ee_positions - ref["body_pos"][self.skeleton.ee_ids]

        cost = (
            # self.params.tracking_root_pos_weight * np.mean(np.square(root_pos_err))
            + self.params.tracking_root_rot_weight * np.mean(np.square(root_rot_err))
            + self.params.tracking_joint_rot_weight * np.mean(np.square(joint_rot_err))
            + self.params.tracking_dof_vel_weight * np.mean(np.square(dof_vel_err))
            # + self.params.tracking_ee_pos_weight * np.mean(np.square(ee_pos_err))
        )
        return float(cost)

    def _compute_reward(self) -> float:
        """Task reward: exponential decay of motion-tracking cost."""
        cost = self._compute_tracking_cost()
        sigma_sq = self.params.tracking_sigma ** 2
        return float(np.exp(-cost / sigma_sq))

    def _check_termination(self) -> bool:
        """Early termination conditions (e.g. root height, deviation)."""
        if self.sim.body_pos[1][-1] < 0.6:
            return True
        return False