"""
Renders the rest-pose offsets of the AMASS and CMU skeletons through visualize_offsets.
"""

from pathlib import Path

from motion_retargeting.data.adapters import AMASSAdapter, CMUAdapter
from motion_retargeting.data.visualization.visualize import visualize_offsets


def amass_skeleton():
    # One shared topology; each clip carries its subject's offsets (from its betas).
    collection = AMASSAdapter().load()
    return collection.topology, collection[0].offsets


def cmu_skeleton():
    # One shared topology; each clip carries its subject's offsets (from its ASF file).
    collection = CMUAdapter().load()
    return collection.topology, collection[0].offsets


def visualize(topology, offsets, save_path=None, show=False):
    if save_path is not None:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    return visualize_offsets(
        offsets.detach().cpu().numpy(),
        parent_indices=topology.parents.tolist(),
        joint_names=topology.joint_names,
        save_path=save_path,
        show=show,
    )


class TestVisualizeOffsets:
    def test_amass_offsets(self, tmp_path):
        save_path = Path(tmp_path) / "amass_offsets.png"
        visualize(*amass_skeleton(), save_path=save_path)
        assert save_path.exists()

    def test_cmu_offsets(self, tmp_path):
        save_path = Path(tmp_path) / "cmu_offsets.png"
        visualize(*cmu_skeleton(), save_path=save_path)
        assert save_path.exists()


if __name__ == "__main__":
    test = TestVisualizeOffsets()
    test.test_amass_offsets(Path("visuals"))
    test.test_cmu_offsets(Path("visuals"))
