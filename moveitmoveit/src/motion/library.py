from __future__ import annotations

from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np

from moveitmoveit.src.sim import Skeleton
import moveitmoveit.src.transforms as transforms

from .clip import MotionClip

class MotionLibrary:
    """ Collection of motion clips with weighted random sampling. """

    def __init__(
        self,
        srcs: List[str],
        skeleton: Skeleton
    ) -> None:
        self._clips: List[MotionClip] = [MotionClip.load(src) for src in srcs]

        # Weight proportional to duration. why? TODO
        weights = [c.duration for c in self._clips]
        w = np.array(weights, dtype=np.float64)
        self._weights = w / w.sum()

        self._skeleton = skeleton

        self._load()

    def _load(self):
        """
        This function depends on the notion that joint posiitons and rotations are on a per joint basis. 
        It assumes that joints are quats and are in the form w,x,y,z.
        """
        motion_lengths = []
        motion_root_pos_delta = []
        motion_num_frames = []

        frame_root_pos = []
        frame_root_rot = []
        frame_root_vel = []
        frame_root_ang_vel = []
        frame_joint_dof = []
        frame_joint_rot = []
        frame_body_pos = []
        frame_dof_vel = []

        total_length = 0 
        for clip in self._clips:
            num_frames = clip.num_frames
            curr_len = clip.duration
            total_length += curr_len

            root_pos = clip.root_pos
            root_rot = clip.root_rot
            joint_dof = clip.frames
            joint_rot = self._skeleton.dof_to_rot(clip.frames)
            body_pos = clip.body_pos

            root_pos_delta = root_pos[-1, :] - root_pos[0, :]
            root_pos_delta[..., -1] = 0.0
                        
            root_vel = clip.root_vel
            root_ang_vel = clip.root_ang_vel

            dof_vel = clip.dof_vel

            motion_lengths.append([curr_len])
            motion_root_pos_delta.append([root_pos_delta])
            motion_num_frames.append(np.atleast_1d(num_frames))
            
            frame_root_pos.append(root_pos)
            frame_root_rot.append(root_rot)
            frame_root_vel.append(root_vel)
            frame_root_ang_vel.append(root_ang_vel)
            frame_joint_dof.append(joint_dof)
            frame_joint_rot.append(joint_rot)
            frame_body_pos.append(body_pos)
            frame_dof_vel.append(dof_vel)

        self._motion_lengths = np.concatenate(motion_lengths, axis=0)
        self._motion_root_pos_delta = np.concatenate(motion_root_pos_delta, axis=0)
        self._motion_num_frames = np.concatenate(motion_num_frames, axis=0)

        self._frame_root_pos = np.concatenate(frame_root_pos, axis=0)
        self._frame_root_rot = np.concatenate(frame_root_rot, axis=0)
        self._frame_root_vel = np.concatenate(frame_root_vel, axis=0)
        self._frame_root_ang_vel = np.concatenate(frame_root_ang_vel, axis=0)
        self._frame_joint_rot = np.concatenate(frame_joint_rot, axis=0)
        self._frame_joint_dof = np.concatenate(frame_joint_dof, axis=0)
        self._frame_body_pos = np.concatenate(frame_body_pos, axis=0)
        self._frame_dof_vel = np.concatenate(frame_dof_vel, axis=0)

        self._motion_ids = np.arange(self.num_clips)

    @property
    def num_clips(self) -> int:
        return len(self._clips)

    @property
    def total_length(self) -> int:
        return np.sum(self._motion_lengths)

    def get_clip(self, idx: int) -> MotionClip:
        return self._clips[idx]
    
    def _resolve_rng(
        self,
        rng: np.random.Generator | int | np.ndarray | None,
    ) -> np.random.Generator:
        if rng is None:
            return np.random.default_rng()
        if isinstance(rng, np.random.Generator):
            return rng
        return np.random.default_rng(rng)
    
    def _compute_frame_from_time(
        self,
        motion_ids: List[int],
        motion_times: List[float],
    ) -> np.ndarray:
        number_frames = self._motion_num_frames[motion_ids]
        motion_lengths = self._motion_lengths[motion_ids]

        phase = motion_times / motion_lengths
        phase = phase - np.floor(phase)
        phase = np.clip(phase, 0.0, 1.0)

        frame_idx = np.long((phase * (number_frames - 1)))
        return frame_idx

    def sample_motions(
        self,
        n: int,
        rng: np.random.Generator | int | np.ndarray | None = None,
    ) -> np.ndarray:
        gen = self._resolve_rng(rng)
        motion_ids = gen.choice(len(self._clips), size=n, p=self._weights)
        return motion_ids

    def sample_frames(
        self,
        motion_ids: List[int],
        rng: np.random.Generator | int | np.ndarray | None = None,
    ) -> np.ndarray:
        """
        Uniformly sample *n* (clip_index, time) pairs,
        weighted by clip duration.
        """
        number_frames = self._motion_num_frames[motion_ids]
        gen = self._resolve_rng(rng)
        return gen.integers(low=0, high=number_frames)

    def sample_start_state(
        self,
        rng: np.random.Generator | int | np.ndarray | None = None,
    ) -> Tuple[np.ndarray, np.ndarray, dict]:
        """ Sample a single random starting state for episode initialisation. """
        gen = self._resolve_rng(rng)
        motion_ids = self.sample_motions(1, rng=gen)
        motion_frames = self.sample_frames(motion_ids, rng=gen)

        frame_info = {
            "clip_id": motion_ids,
            "motion_frames": motion_frames,
        }

        qpos = np.concatenate([
            self._frame_root_pos[motion_frames], 
            self._frame_root_rot[motion_frames], 
            self._frame_joint_dof[motion_frames],
        ], axis=-1)
        qvel = np.concatenate([
            self._frame_root_vel[motion_frames], 
            self._frame_root_ang_vel[motion_frames], 
            self._frame_dof_vel[motion_frames],
        ], axis=-1)

        return qpos, qvel, frame_info

    def get_frame_data(self, clip_ids, motion_frames):
        num_frames = self._motion_num_frames[clip_ids]

        wrapped_frames = motion_frames % num_frames
        num_wraps = motion_frames // num_frames

        root_pos = self._frame_root_pos[wrapped_frames].copy()
        root_rot = self._frame_root_rot[wrapped_frames]
        root_vel = self._frame_root_vel[wrapped_frames]
        root_ang_vel = self._frame_root_ang_vel[wrapped_frames]
        joint_rot = self._frame_joint_rot[wrapped_frames]
        dof_vel = self._frame_dof_vel[wrapped_frames]
        body_pos = self._frame_body_pos[wrapped_frames].copy()

        root_pos += num_wraps[:, np.newaxis] * self._motion_root_pos_delta[clip_ids]

        body_pos += num_wraps[:, np.newaxis, np.newaxis] * self._motion_root_pos_delta[clip_ids]

        return root_pos, root_rot, root_vel, root_ang_vel, joint_rot, dof_vel, body_pos

    def get_frame_state(self, clip_id: int, frame_id: int) -> dict:
        """Kinematic state for a single frame within a clip."""
        clip = self._clips[int(clip_id)]
        frame_id = int(np.clip(frame_id, 0, clip.num_frames - 1))
        joint_dof = clip.frames[frame_id : frame_id + 1]
        return {
            "root_pos": clip.root_pos[frame_id],
            "root_rot": clip.root_rot[frame_id],
            "root_vel": clip.root_vel[frame_id],
            "root_ang_vel": clip.root_ang_vel[frame_id],
            "joint_dof": clip.frames[frame_id],
            "joint_rot": self._skeleton.dof_to_rot(joint_dof)[0],
            "dof_vel": clip.dof_vel[frame_id],
            "body_pos": clip.body_pos[frame_id],
        }

