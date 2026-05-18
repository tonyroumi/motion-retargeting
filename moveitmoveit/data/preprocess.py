import pickle 
import numpy as np
from scipy.spatial.transform import Rotation
import os 
from pathlib import Path 
from dataclasses import dataclass

def frame_to_qpos(frame, nq):
    """Convert a 34-value frame [pos(3), rot_vec(3), joints(28)] to MuJoCo qpos (35)."""
    frame = np.array(frame, dtype=float)
    np.nan_to_num(frame, copy=False, nan=0.0)

    pos = frame[:3]
    rot_vec = frame[3:6]
    joints = frame[6:]

    quat_xyzw = Rotation.from_rotvec(rot_vec).as_quat()  # [x, y, z, w]
    quat_wxyz = quat_xyzw[[3, 0, 1, 2]]                  # MuJoCo: [w, x, y, z]

    qpos = np.empty(nq)
    qpos[:3] = pos
    qpos[3:7] = quat_wxyz
    qpos[7:] = joints
    return qpos


def load_motion(pkl_path):
    with open(pkl_path, "rb") as f:
        data = pickle.load(f)
    return data["frames"], data["fps"]

@dataclass
class MotionClip:
    """ A single motion sequence for learning. """

    name: str
    fps: float
    frames: np.ndarray  # (ndof) (w, x, y, z)

    def save(self, path: str | Path) -> None:
        np.savez_compressed(
            str(path),
            name=np.array(self.name),
            fps=np.array(self.fps, dtype=np.float32),
            frames=self.frames.astype(np.float32),
        )


if __name__ == "__main__":
    files  = os.listdir("/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/humanoid")
    for fname in files:
        if "xml" in fname:
            continue
        file = f"/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/humanoid/{fname}"
        frames, fps = load_motion(file)

        motion = np.zeros([len(frames), 35])

        for i, frame in enumerate(frames):
            qpos = frame_to_qpos(frame, 35)
            motion[i] = qpos
        
        clip = MotionClip(fname, fps=fps, frames=motion)
        fpath = f"/home/tonyroumi/Desktop/move-it-move-it/moveitmoveit/data/humanoid_Better/{fname[:-4]}.npz"
        clip.save(fpath)



