"""
Data layer tests on synthetic data: no raw datasets or body models needed.
"""

from __future__ import annotations

import math
from types import SimpleNamespace

import pytest
import torch

from motion_retargeting.data.adapters import BANDAIAdapter, CMUAdapter, MotionSourceAdapter, base
from motion_retargeting.data.adapters.amass import axis_angle_to_quaternion, zup_to_yup_quaternion, zup_to_yup_vector
from motion_retargeting.data.adapters.factory import ADAPTERS, build_adapter
from motion_retargeting.data.builder import MotionDatasetBuilder, MotionDatasetBuilderConfig
from motion_retargeting.data.dataset import MotionDataset, compute_stats, split_dataset
from motion_retargeting.data.factory import build_datasets
from motion_retargeting.data.utils import skeleton_height
from motion_retargeting.kinematics.forward import ForwardKinematics
from motion_retargeting.kinematics.quaternions import quat_apply
from motion_retargeting.methods.san.utils import restore_motion

YAW_PER_FRAME = 0.01
VELOCITY_PER_FRAME = (0.02, 0.0, 0.01)


def yaw_quaternion(yaw: torch.Tensor) -> torch.Tensor:
    """ Rotation about +Y: yaw [...] -> [..., 4] in (w, x, y, z). """
    half = 0.5 * yaw
    return torch.stack([half.cos(), torch.zeros_like(half), half.sin(), torch.zeros_like(half)], dim=-1)


def synthetic_clip(num_joints: int, num_frames: int) -> dict:
    """ A chain skeleton whose root yaws and translates at a constant rate. """
    frames = torch.arange(num_frames, dtype=torch.float32)

    joint_rotations = torch.zeros(num_frames, num_joints, 4)
    joint_rotations[..., 0] = 1.0
    joint_rotations[:, 0] = yaw_quaternion(YAW_PER_FRAME * frames)

    offsets = torch.zeros(num_joints, 3)
    offsets[1:, 1] = 0.1

    return dict(
        parents=[-1] + list(range(num_joints - 1)),
        offsets=offsets,
        joint_rotations=joint_rotations,
        root_position=frames[:, None] * torch.tensor(VELOCITY_PER_FRAME),
    )


class SyntheticAdapter(MotionSourceAdapter):
    def __init__(self, name: str = "synthetic", num_joints: int = 4, num_frames: int = 80, fps: float = 30.0, num_clips: int = 2):
        super().__init__(name)
        self.num_joints = num_joints
        self.num_frames = num_frames
        self.fps = fps
        self.num_clips = num_clips
        self.num_extractions = 0

    def _extract(self):
        self.num_extractions += 1
        return [
            self._construct_motion(fps=self.fps, **synthetic_clip(self.num_joints, self.num_frames))
            for _ in range(self.num_clips)
        ]


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    """ Point adapters at a temporary data root and label bodies without the interactive plot. """
    monkeypatch.setattr(base, "PATHS", SimpleNamespace(data_root=tmp_path))

    def two_bodies(topology, offsets):
        joints = list(range(topology.num_joints))
        return {0: joints[:2], 1: joints[2:]}

    monkeypatch.setattr(base, "label_joint_bodies", two_bodies)
    return tmp_path


def build(sequence_length: int = 16, stride: int = 8, **adapter_kwargs) -> MotionDataset:
    config = MotionDatasetBuilderConfig(sequence_length=sequence_length, stride=stride)
    return MotionDatasetBuilder(config).build([SyntheticAdapter(**adapter_kwargs)])


class TestAdapters:
    def test_factory_exposes_configured_sources(self):
        assert {"amass", "cmu", "bandai"} <= set(ADAPTERS)
        assert isinstance(build_adapter("cmu"), CMUAdapter)
        with pytest.raises(KeyError):
            build_adapter("nope")

    def test_load_caches_extraction(self):
        adapter = SyntheticAdapter()
        collection = adapter.load()

        assert len(collection) == 2
        assert collection.topology.num_joints == 4
        assert collection.topology.body_joints == {0: [0, 1], 1: [2, 3]}
        assert adapter.motion_cache_path.exists()

        cached = adapter.load()
        assert adapter.num_extractions == 1
        assert torch.equal(cached[0].joint_rotations, collection[0].joint_rotations)

    def test_missing_raw_data_does_not_poison_the_cache(self):
        adapter = SyntheticAdapter(num_clips=0)
        with pytest.raises(FileNotFoundError):
            adapter.load()
        assert not adapter.motion_cache_path.exists()

    def test_clip_with_different_topology_is_rejected(self):
        adapter = SyntheticAdapter()
        adapter._construct_motion(fps=30.0, **synthetic_clip(4, 8))
        with pytest.raises(ValueError):
            adapter._construct_motion(fps=30.0, **synthetic_clip(5, 8))

    def test_amass_axis_conversions(self):
        angle = torch.tensor([[0.0, 0.0, math.pi / 2]])
        q = axis_angle_to_quaternion(angle)
        assert torch.allclose(q, torch.tensor([[math.cos(math.pi / 4), 0.0, 0.0, math.sin(math.pi / 4)]]))

        # The vector and quaternion conversions describe one rotation of the world (-90 degrees about X).
        v = torch.tensor([[0.3, -0.2, 0.9]])
        identity = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        assert torch.allclose(quat_apply(zup_to_yup_quaternion(identity), v), zup_to_yup_vector(v))


ASF = """\
:version 1.10
:units
  length 0.45
  angle deg
:root
  order TX TY TZ RX RY RZ
  axis XYZ
  position 0 0 0
  orientation 0 0 0
:bonedata
  begin
    id 1
    name lfemur
    direction 0 -1 0
    length 9
    axis 0 0 0 XYZ
    dof rx ry rz
    limits (-160 20)
           (-70 70)
           (-60 70)
  end
:hierarchy
  begin
    root lfemur
  end
"""

AMC = """\
:FULLY-SPECIFIED
:DEGREES
1
root 1 2 3 0 0 0
lfemur 0 0 0
2
root 2 2 3 0 0 0
lfemur 90 0 0
"""


class TestCMUAdapter:
    def test_extract(self, isolated_data_root):
        raw_dir = isolated_data_root / "raw" / "cmu" / "01"
        raw_dir.mkdir(parents=True)
        (raw_dir / "01.asf").write_text(ASF)
        (raw_dir / "01_01.amc").write_text(AMC)

        collection = CMUAdapter().load()

        assert collection.topology.joint_names == ["root", "lfemur", "lfemur_end"]
        assert collection.topology.parents.tolist() == [-1, 0, 1]

        scale = (1.0 / 0.45) * (2.54 / 100.0)
        motion = collection[0]
        assert motion.fps == 120.0
        assert motion.joint_rotations.shape == (2, 3, 4)
        assert torch.allclose(motion.offsets[2], torch.tensor([0.0, -9.0 * scale, 0.0]))

        # Z-forward -> X-forward: (x, y, z) becomes (z, y, -x).
        assert torch.allclose(motion.root_position[0], torch.tensor([3.0, 2.0, -1.0]) * scale)


BVH = """\
HIERARCHY
ROOT joint_Root
{
  OFFSET 0 0 0
  CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
  JOINT Hips
  {
    OFFSET 0 90 0
    CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
    JOINT Spine
    {
      OFFSET 20 0 0
      CHANNELS 6 Xposition Yposition Zposition Zrotation Xrotation Yrotation
      End Site
      {
        OFFSET 0 0 0
      }
    }
  }
}
MOTION
Frames: 2
Frame Time: 0.0333333
10 0 0 0 0 0 1 90 5 90 0 0 20 0 0 0 0 0
10 0 0 0 0 0 1 90 6 90 0 0 20 0 0 0 0 0
"""


class TestBANDAIAdapter:
    def test_extract_canonicalizes_root(self, isolated_data_root):
        pytest.importorskip("bvhio")

        raw_dir = isolated_data_root / "raw" / "bandai"
        raw_dir.mkdir(parents=True)
        (raw_dir / "clip.bvh").write_text(BVH)

        collection = BANDAIAdapter().load()
        motion = collection[0]

        # `joint_Root` is dropped, and `Hips` becomes the root.
        assert collection.topology.joint_names == ["Hips", "Spine"]
        assert collection.topology.parents.tolist() == [-1, 0]
        assert motion.fps == 30.0

        # Root position composes both channels: raw (11, 0.9, 0.05) m, then (x, y, z) -> (z, y, -x).
        assert torch.allclose(motion.root_position[0], torch.tensor([0.05, 0.9, -0.11]), atol=1e-5)

        # The hips were upright in the file (+90 degrees about Z), so the root only keeps the
        # Z-forward -> X-forward world rotation (+90 degrees about Y).
        half = math.sin(math.pi / 4)
        assert torch.allclose(motion.joint_rotations[0, 0], torch.tensor([half, 0.0, half, 0.0]), atol=1e-5)

        # An upright character: the spine sits straight above the root, 20 cm up.
        fk = ForwardKinematics(collection.topology.parents, motion.offsets.expand(2, -1, -1))
        positions, _ = fk(motion.joint_rotations, motion.root_position)
        expected = motion.root_position[0] + torch.tensor([0.0, 0.2, 0.0])
        assert torch.allclose(positions[0, 1], expected, atol=1e-5)


class TestBuilder:
    def test_sample_contract(self):
        dataset = build(sequence_length=16, stride=8, num_joints=4, num_frames=40, num_clips=1)

        # 39 frames remain after velocities: windows start at frames 0, 8, 16.
        assert len(dataset) == 3

        sample = dataset[0]
        assert sample["motion"].shape == (16, 5, 4)
        assert sample["offsets"].shape == (4, 3)
        assert sample["q_heading"].shape == (16, 4)

        assert dataset.parents.tolist() == [-1, 0, 1, 2]
        assert dataset.body_joints == {0: [0, 1], 1: [2, 3]}

    def test_final_row_holds_velocity_and_yaw_rate(self):
        dataset = build()
        velocity, yaw_rate = dataset.motion[..., -1, :3], dataset.motion[..., -1, 3]

        assert torch.allclose(yaw_rate, torch.full_like(yaw_rate, YAW_PER_FRAME), atol=1e-5)

        # Removing the heading is undone by `restore_motion`, recovering the world-frame velocity.
        rotations = dataset.motion[..., :-1, :]
        _, world_velocity = restore_motion(rotations, velocity, dataset.q_heading)
        assert torch.allclose(world_velocity, torch.tensor(VELOCITY_PER_FRAME).expand_as(world_velocity), atol=1e-4)

        # The root's rotation is entirely heading, so canonicalization leaves it at identity.
        root = rotations[..., 0, :]
        assert torch.allclose(root, torch.tensor([1.0, 0.0, 0.0, 0.0]).expand_as(root), atol=1e-4)

    def test_frame_velocity_is_the_step_to_the_next_frame(self):
        """ SAN rebuilds root positions as cumsum(v) - v, which relies on this alignment. """
        clip = synthetic_clip(num_joints=4, num_frames=20)
        clip["joint_rotations"][:, 0] = torch.tensor([1.0, 0.0, 0.0, 0.0])
        clip["root_position"] = torch.arange(20.0)[:, None] ** 2 * torch.tensor([0.01, 0.0, 0.0])

        motion = SyntheticAdapter()._construct_motion(fps=30.0, **clip)
        config = MotionDatasetBuilderConfig(sequence_length=8)
        _, velocity, _, _ = MotionDatasetBuilder(config).process_clip(motion)

        rebuilt = torch.cumsum(velocity, dim=0) - velocity
        assert torch.allclose(rebuilt, clip["root_position"][:-1], atol=1e-5)

    def test_resamples_to_target_fps(self):
        config = MotionDatasetBuilderConfig(sequence_length=16, stride=8, target_fps=30)
        dataset = MotionDatasetBuilder(config).build([SyntheticAdapter(fps=60.0, num_frames=40, num_clips=1)])

        assert len(dataset) == 1
        velocity = dataset.motion[..., -1, :3].norm(dim=-1)
        assert velocity.mean() > 1.5 * torch.tensor(VELOCITY_PER_FRAME).norm()

    def test_max_samples_is_deterministic(self):
        config = MotionDatasetBuilderConfig(sequence_length=16, stride=8, max_samples=3, seed=1)
        adapter = SyntheticAdapter()
        first = MotionDatasetBuilder(config).build([adapter])
        second = MotionDatasetBuilder(config).build([adapter])

        assert len(first) == 3
        assert torch.equal(first.motion, second.motion)

    def test_sources_in_a_group_must_share_topology(self):
        config = MotionDatasetBuilderConfig(sequence_length=16, stride=8)
        sources = [SyntheticAdapter("a", num_joints=4), SyntheticAdapter("b", num_joints=5)]
        with pytest.raises(ValueError):
            MotionDatasetBuilder(config).build(sources)

    def test_build_datasets_from_config(self, monkeypatch):
        adapters = {"a": SyntheticAdapter("a"), "b": SyntheticAdapter("b", num_joints=6)}
        monkeypatch.setattr("motion_retargeting.data.factory.build_adapter", adapters.__getitem__)

        datasets = build_datasets({
            "group_a": {"sources": ["a"], "params": {"sequence_length": 16, "stride": 8}},
            "group_b": {"sources": ["b"], "params": {"sequence_length": 16, "stride": 8}},
        })

        assert set(datasets) == {"group_a", "group_b"}
        assert datasets["group_a"][0]["motion"].shape == (16, 5, 4)
        assert datasets["group_b"][0]["motion"].shape == (16, 7, 4)


class TestDataset:
    def test_split_partitions_windows(self):
        dataset = build(num_clips=4)
        dataset.motion[:, 0, 0, 0] = torch.arange(len(dataset), dtype=torch.float32)

        train, val, test = split_dataset(dataset, val_fraction=0.25, test_fraction=0.25)

        assert len(val) == len(test) == len(dataset) // 4
        assert len(train) + len(val) + len(test) == len(dataset)

        ids = torch.cat([split.motion[:, 0, 0, 0] for split in (train, val, test)])
        assert ids.sort().values.tolist() == list(range(len(dataset)))

    def test_split_is_seeded(self):
        dataset = build(num_clips=4)
        first, _, _ = split_dataset(dataset, seed=3)
        second, _, _ = split_dataset(dataset, seed=3)
        assert torch.equal(first.motion, second.motion)

    def test_normalization_only_touches_the_final_row(self):
        dataset = build(num_clips=4)
        dataset.motion[..., -1, :] += torch.randn_like(dataset.motion[..., -1, :])
        raw = dataset.motion.clone()

        vel_mean, vel_std, ang_mean, ang_std = compute_stats(dataset)
        assert vel_mean.shape == vel_std.shape == (3,)
        assert ang_mean.ndim == ang_std.ndim == 0

        dataset.set_normalization(vel_mean, vel_std, ang_mean, ang_std)
        normalized = torch.stack([dataset[i]["motion"] for i in range(len(dataset))])

        assert torch.equal(dataset.motion, raw)
        assert torch.equal(normalized[..., :-1, :], raw[..., :-1, :])

        final_row = normalized[..., -1, :].reshape(-1, 4)
        assert torch.allclose(final_row.mean(dim=0), torch.zeros(4), atol=1e-4)
        assert torch.allclose(final_row.std(dim=0), torch.ones(4), atol=1e-4)

    def test_constant_features_keep_unit_std(self):
        dataset = build()
        dataset.motion[..., -1, :] = 2.0

        _, vel_std, _, ang_std = compute_stats(dataset)
        assert torch.equal(vel_std, torch.ones(3))
        assert ang_std == 1.0

    def test_subset_carries_stats_and_skeleton(self):
        dataset = build()
        dataset.set_normalization(*compute_stats(dataset))

        subset = dataset.subset(torch.tensor([0, 1]))

        assert len(subset) == 2
        assert subset.vel_stats is dataset.vel_stats
        assert subset.ang_vel_stats is dataset.ang_vel_stats
        assert subset.body_joints == dataset.body_joints
        assert torch.equal(subset.parents, dataset.parents)


class TestUtils:
    def test_skeleton_height(self):
        offsets = torch.tensor([[[0.0, 0.0, 0.0], [0.0, 0.5, 0.0], [0.0, -1.0, 0.0]]])
        assert skeleton_height(offsets, torch.tensor([-1, 0, 0])).item() == pytest.approx(1.5)
