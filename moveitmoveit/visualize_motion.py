"""
Visualize motion clips in the MuJoCo viewer.

Replays each clip by setting qpos/qvel at every frame and calling mj_forward.
Playback is paced to real-time by default; use --speed to change rate.

Controls (focus the terminal, not the viewer window):
  SPACE   pause / unpause
  n       next clip
  p       previous clip
  r       restart current clip
  q       quit

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
import threading
import time
from pathlib import Path

import mujoco
import mujoco.viewer
import numpy as np


# ---------------------------------------------------------------------------
# Non-blocking keyboard input (Linux/macOS)
# ---------------------------------------------------------------------------

import select
import termios
import tty


class KeyReader:
    """Reads single keypresses from stdin without blocking the main loop."""

    def __init__(self) -> None:
        self._fd = sys.stdin.fileno()
        self._old = termios.tcgetattr(self._fd)
        tty.setraw(self._fd)

    def read(self) -> str | None:
        if select.select([sys.stdin], [], [], 0)[0]:
            return sys.stdin.read(1)
        return None

    def close(self) -> None:
        termios.tcsetattr(self._fd, termios.TCSADRAIN, self._old)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def find_clips(path: Path) -> list[Path]:
    if path.is_file():
        return [path]
    clips = sorted(path.glob("*.npz"))
    if not clips:
        raise FileNotFoundError(f"No .npz files found in {path}")
    return clips


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
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def run(clips: list[Path], xml_path: Path, speed: float) -> None:
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)

    clip_idx = 0
    frame_idx = 0
    paused = False
    advance = 0       # +1 next, -1 prev
    restart = False

    clip = load_clip(clips[clip_idx])

    keys = KeyReader()
    print(f"\nPlaying: {clips[clip_idx].name}  ({clip['dof_pos'].shape[0]} frames @ {clip['fps']:.0f}fps)")
    print("SPACE=pause  n=next  p=prev  r=restart  q=quit\n")

    with mujoco.viewer.launch_passive(model, data) as viewer:
        last_frame_time = time.perf_counter()

        while viewer.is_running():
            # --- keyboard ---
            key = keys.read()
            if key == " ":
                paused = not paused
                print("paused" if paused else "resumed")
            elif key == "n":
                advance = 1
            elif key == "p":
                advance = -1
            elif key == "r":
                restart = True
            elif key in ("q", "\x03"):   # q or ctrl-c
                break

            # --- clip switching / restart ---
            if advance != 0:
                clip_idx = (clip_idx + advance) % len(clips)
                clip = load_clip(clips[clip_idx])
                frame_idx = 0
                advance = 0
                last_frame_time = time.perf_counter()
                print(f"Playing: {clips[clip_idx].name}  ({clip['dof_pos'].shape[0]} frames @ {clip['fps']:.0f}fps)")

            if restart:
                frame_idx = 0
                restart = False
                last_frame_time = time.perf_counter()
                print(f"Restarting: {clips[clip_idx].name}")

            # --- frame advance ---
            frame_dt = clip["dt"] / speed
            now = time.perf_counter()

            if not paused and (now - last_frame_time) >= frame_dt:
                last_frame_time += frame_dt

                n_frames = clip["dof_pos"].shape[0]
                frame_idx = frame_idx % n_frames

                qpos = np.concatenate([clip["root_pos"][frame_idx], clip["root_rot"][frame_idx], clip["dof_pos"][frame_idx]])
                qvel = np.concatenate([clip["root_vel"][frame_idx], clip["root_ang_vel"][frame_idx], clip["dof_vel"][frame_idx]])

                data.qpos[:] = qpos
                data.qvel[:] = qvel
                mujoco.mj_forward(model, data)
                viewer.sync()

                frame_idx += 1
                if frame_idx >= n_frames:
                    frame_idx = 0  # loop

            # else:
                # Sync at ~200 Hz when paused or waiting for next frame
                # viewer.sync()
                # time.sleep(0.005)

    keys.close()


def main() -> None:
    src = Path("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/better_humanoid")
    xml_path = Path("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/humanoid/humanoid.xml")
    speed = 1.0

    clips = find_clips(src)

    if not xml_path.is_file():
        print(f"Error: model file not found: {xml_path}")
        sys.exit(1)

    run(clips, xml_path, speed)


if __name__ == "__main__":
    main()
