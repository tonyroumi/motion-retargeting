"""
Visualize motion clips using Gymnasium's MujocoRenderer.

Replays each clip by setting qpos/qvel at every frame and calling mj_forward.
Playback is paced to real-time by default; use --speed to change rate.

Controls (in the viewer window):
  SPACE        pause / unpause
  RIGHT arrow  advance one step while paused
  TAB          cycle cameras
  ESC          quit

Clip navigation (terminal):
  n   next clip
  p   previous clip
  r   restart current clip
  q   quit

Usage:
    # single clip
    python visualize_motion.py path/to/clip.npz path/to/model.xml

    # whole directory (cycles through all clips)
    python visualize_motion.py data/humanoid_rollout data/humanoid/humanoid.xml

    # half speed
    python visualize_motion.py data/humanoid_rollout data/humanoid/humanoid.xml --speed 0.5
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import mujoco
import numpy as np
from gymnasium.envs.mujoco.mujoco_rendering import MujocoRenderer


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_clips(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    clips = sorted(path.glob("*.npz"))
    if not clips:
        raise FileNotFoundError(f"No .npz files found in {path}")
    return [Path("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/better_bro/humanoid_jog.npz")]  # clips


def load_clip(path: Path) -> dict:
    d = np.load(str(path), allow_pickle=False)
    return {
        "name": str(d["name"]),
        "fps": float(d["fps"]),
        "dt": float(d["dt"]),
        "root_pos": d["root_pos"].astype(np.float64),
        "root_rot": d["root_rot"].astype(np.float64),
        "root_vel": d["root_vel"].astype(np.float64),
        "root_ang_vel": d["root_ang_vel"].astype(np.float64),
        "dof_pos": d["dof_pos"].astype(np.float64),
        "dof_vel": d["dof_vel"].astype(np.float64),
        "ctrl": d["ctrl"].astype(np.float64)
    }


def clip_root_pos_offset(clip: dict) -> np.ndarray:
    """Per-loop root translation so periodic clips advance in world x/y."""
    delta = clip["root_pos"][-1] - clip["root_pos"][0]
    delta[2] = 0.0
    return delta


def set_frame_qpos(
    data: mujoco.MjData,
    model: mujoco.MjModel,
    clip: dict,
    frame_idx: int,
    root_pos_offset: np.ndarray,
) -> None:
    """Write clip frame into MjData and call mj_forward."""
    qpos = np.concatenate([
        clip["root_pos"][frame_idx] + root_pos_offset,
        clip["root_rot"][frame_idx],
        clip["dof_pos"][frame_idx],
    ])
    qvel = np.concatenate([
        clip["root_vel"][frame_idx],
        clip["root_ang_vel"][frame_idx],
        clip["dof_vel"][frame_idx],
    ])
    data.qpos[:] = qpos
    data.qvel[:] = qvel
    mujoco.mj_forward(model, data)

def set_frame(data: mujoco.MjData, model: mujoco.MjModel, clip: dict, frame_idx: int, root_pos_offset) -> None:
    """Write clip frame into MjData and call mj_forward."""
    for _ in range(4):
        # Let actuators drive joints via ctrl
        data.ctrl[:] = clip["ctrl"][frame_idx]
        
        # But pin the root to mocap (common during reference validation)
        data.qpos[:7]  = np.concatenate([clip["root_pos"][frame_idx], clip["root_rot"][frame_idx]])
        data.qvel[:6]  = np.concatenate([clip["root_vel"][frame_idx], clip["root_ang_vel"][frame_idx]])
        
        mujoco.mj_step(model, data)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(clips: list[Path], xml_path: Path, speed: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    model.opt.timestep = 1.0 / 120.0
    model.opt.gravity = [0,0,0]
    data = mujoco.MjData(model)

    renderer = MujocoRenderer(model, data)

    clip_idx = 0
    clip = load_clip(clips[clip_idx])
    n_frames = clip["dof_pos"].shape[0]
    frame_idx = 0
    root_pos_offset = np.zeros(3, dtype=np.float64)

    print(f"\nPlaying: {clips[clip_idx].name}  ({n_frames} frames @ {clip['fps']:.0f} fps)")
    print("Viewer window: SPACE=pause  RIGHT=step  TAB=camera  ESC=quit")
    print("Terminal:      n=next clip  p=prev clip  r=restart  q=quit\n")

    last_frame_time = time.perf_counter()

    try:
        while True:
            frame_dt = clip["dt"] / speed
            now = time.perf_counter()

            if (now - last_frame_time) >= frame_dt:
                last_frame_time += frame_dt

                set_frame(data, model, clip, frame_idx, root_pos_offset)
                renderer.render("human")

                next_frame_idx = (frame_idx + 1) % n_frames
                if next_frame_idx == 0 and frame_idx == n_frames - 1:
                    root_pos_offset += clip_root_pos_offset(clip)
                frame_idx = next_frame_idx

            # Non-blocking terminal input for clip navigation
            import select
            if select.select([sys.stdin], [], [], 0)[0]:
                key = sys.stdin.read(1)
                if key == "n":
                    clip_idx = (clip_idx + 1) % len(clips)
                    clip = load_clip(clips[clip_idx])
                    n_frames = clip["dof_pos"].shape[0]
                    frame_idx = 0
                    root_pos_offset[:] = 0.0
                    last_frame_time = time.perf_counter()
                    print(f"Playing: {clips[clip_idx].name}  ({n_frames} frames @ {clip['fps']:.0f} fps)")
                elif key == "p":
                    clip_idx = (clip_idx - 1) % len(clips)
                    clip = load_clip(clips[clip_idx])
                    n_frames = clip["dof_pos"].shape[0]
                    frame_idx = 0
                    root_pos_offset[:] = 0.0
                    last_frame_time = time.perf_counter()
                    print(f"Playing: {clips[clip_idx].name}  ({n_frames} frames @ {clip['fps']:.0f} fps)")
                elif key == "r":
                    frame_idx = 0
                    root_pos_offset[:] = 0.0
                    last_frame_time = time.perf_counter()
                    print(f"Restarting: {clips[clip_idx].name}")
                elif key in ("q", "\x03"):
                    break

    finally:
        renderer.close()


def main() -> None:
    src = Path("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/better_bro")
    xml_path = Path("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/humanoid/humanoid.xml")
    speed = 1.0

    clips = find_clips(src)

    if not xml_path.is_file():
        print(f"Error: model file not found: {xml_path}")
        sys.exit(1)

    run(clips, xml_path, speed)


if __name__ == "__main__":
    main()

# ****** ANTHONY TODO
# Another thing to try is to collect position inputs ("What the policy is meant to be outputting")
# and see if it works with this model -- it's new position servos and torques, etc. 
# is it within the realm of possiblility for our policy. ***** 

# decimation meaning my model outputs this value and we step the sim with the same output. 
# that is about right.

# what does that translate to. 
# 
# I want to see what it takes for the position servos to produce the motion.
# NEED TO LOOK IN PRE PROCESS..  
# 

# TODO list:
#2.) Verify I can rollout the motion, stepping 4 times in the sim.
