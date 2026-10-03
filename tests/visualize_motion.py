"""
Renders AMASS and CMU motion clips through ForwardKinematics + visualize_motion.
"""

from pathlib import Path

from motion_retargeting.data.adapters import AMASSAdapter, CMUAdapter
from motion_retargeting.kinematics.forward import ForwardKinematics
from motion_retargeting.data.visualization.visualize import visualize_motion

TARGET_FPS = 30.0
NUM_SECONDS = 4


def global_positions(motion, topology):
    # Subsample to TARGET_FPS and keep the first few seconds to keep rendering fast.
    step = max(1, round(motion.fps / TARGET_FPS))
    num_frames = int(NUM_SECONDS * TARGET_FPS)
    local_rotations = motion.joint_rotations[::step][:num_frames]
    root_positions = motion.root_position[::step][:num_frames]

    # quat_apply doesn't broadcast [J, 3] offsets against [T, J, 4] rotations, so expand per frame.
    offsets = motion.offsets.expand(local_rotations.shape[0], -1, -1)
    fk = ForwardKinematics(topology.parents, offsets)
    global_positions, _ = fk(local_rotations, root_positions)
    return global_positions


def amass_global_positions():
    adapter = AMASSAdapter(device="cuda")
    # One shared topology; each clip carries its subject's offsets (from its betas).
    collection = adapter.load()
    return global_positions(collection[1], collection.topology), collection.topology


def cmu_global_positions():
    adapter = CMUAdapter(device="cuda")
    # One shared topology; each clip carries its subject's offsets (from its ASF file).
    collection = adapter.load()
    return global_positions(collection[1], collection.topology), collection.topology


def visualize(global_positions, topology, save_path=None, show=False):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    return visualize_motion(
        global_positions,
        parent_indices=topology.parents.tolist(),
        joint_names=topology.joint_names,
        fps=TARGET_FPS,
        save_path=save_path,
        show=show,
    )


class TestVisualizeMotion:
    def test_amass_motion(self, tmp_path):
        save_path = Path(tmp_path) / "amass_motion.gif"
        visualize(*amass_global_positions(), save_path=save_path)
        assert save_path.exists()

    def test_cmu_motion(self, tmp_path):
        save_path = Path(tmp_path) / "cmu_motion.gif"
        visualize(*cmu_global_positions(), save_path=save_path)
        assert save_path.exists()


if __name__ == "__main__":
    test = TestVisualizeMotion()
    test.test_amass_motion(Path("visuals"))
    test.test_cmu_motion(Path("visuals"))
