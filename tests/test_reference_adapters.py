"""
Reference dataset adapters on synthetic BVHs, through the existing builder and PAN method.

Set PAN_REFERENCE_ROOT to a checkout of the PAN reference to also run on its demo clips.
"""

from __future__ import annotations

import math
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")

import pytest
import torch
import yaml
from scipy.spatial.transform import Rotation

from motion_retargeting.data.adapters import DogAdapter, LAFAN1Adapter, MixamoAdapter, MixamoGroupBAdapter, base
from motion_retargeting.data.adapters.factory import build_adapter
from motion_retargeting.data.builder import MotionDatasetBuilder, MotionDatasetBuilderConfig
from motion_retargeting.data.dataset import compute_stats
from motion_retargeting.data.factory import build_datasets
from motion_retargeting.kinematics.forward import ForwardKinematics
from motion_retargeting.kinematics.quaternions import quat_apply
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.method import PANMethod

pytest.importorskip("bvhio")

CONFIGS = Path(__file__).resolve().parents[1] / "configs" / "methods"
REFERENCE = os.environ.get("PAN_REFERENCE_ROOT")

NUM_FRAMES = 100
FRAME_TIME = "0.033333"


@pytest.fixture(autouse=True)
def isolated_data_root(tmp_path, monkeypatch):
    """ Point adapters at a temporary data root; these adapters must never open the labeler. """
    monkeypatch.setattr(base, "PATHS", SimpleNamespace(data_root=tmp_path))

    def no_labeling(topology, offsets):
        raise AssertionError("reference adapters assign body parts by name")

    monkeypatch.setattr(base, "label_joint_bodies", no_labeling)
    return tmp_path


def chain(names, step=(10.0, 0.0, 0.0), children=()):
    """ Nested (name, offset, children) for a serial chain, with `children` under its last joint. """
    node = children
    for name in reversed(names):
        node = [(name, step, node)]
    return node


def lafan_tree():
    return ("Hips", (5.0, 90.0, -50.0), [
        *chain(["LeftUpLeg", "LeftLeg", "LeftFoot", "LeftToe"]),
        *chain(["RightUpLeg", "RightLeg", "RightFoot", "RightToe"]),
        *chain(["Spine", "Spine1", "Spine2"], children=[
            *chain(["Neck", "Head"]),
            *chain(["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]),
            *chain(["RightShoulder", "RightArm", "RightForeArm", "RightHand"]),
        ]),
    ])


def dog_tree():
    return ("Hips", (0.0, 40.0, 0.0), [
        *chain(["Spine", "Spine1"], children=[
            *chain(["Neck", "Head"]),
            *chain(["LeftShoulder", "LeftArm", "LeftForeArm", "LeftHand"]),
            *chain(["RightShoulder", "RightArm", "RightForeArm", "RightHand"]),
        ]),
        *chain(["LeftUpLeg", "LeftLeg", "LeftFoot"]),
        *chain(["RightUpLeg", "RightLeg", "RightFoot"]),
        *chain(["Tail", "Tail1"]),
    ])


def arm(side, split=False, fingers=False):
    hand = [(f"mixamorig:{side}Hand", (10.0, 0.0, 0.0), chain([f"mixamorig:{side}HandIndex1"]) if fingers else [])]
    joints = [f"mixamorig:{side}Shoulder"] + ([f"mixamorig:{side}Shoulder_split"] if split else [])
    return chain(joints + [f"mixamorig:{side}Arm", f"mixamorig:{side}ForeArm"], children=hand)


def mixamo_a_tree(fingers=True):
    """ A file rooted above `Pelvis`, as the reference's re-rooted group A characters are. """
    prefix = "mixamorig:"
    legs = [
        joint
        for side in ("Left", "Right")
        for joint in chain([f"{prefix}{side}{part}" for part in ("UpLeg", "Leg", "Foot", "ToeBase")])
    ]
    spine = chain([f"{prefix}Hips", f"{prefix}Spine", f"{prefix}Spine1", f"{prefix}Spine2"], children=[
        *chain([f"{prefix}Neck", f"{prefix}Head"]),
        *arm("Left", fingers=fingers),
        *arm("Right", fingers=fingers),
    ])
    return (f"{prefix}Reference", (0.0, 0.0, 0.0), [(f"{prefix}Pelvis", (0.0, 90.0, 0.0), legs + spine)])


def mixamo_a_hips_rooted_tree():
    """ The other shape `set_new_root(1)` allows: `Hips` is the file root, with `Pelvis` as its first child. """
    prefix = "mixamorig:"
    legs = [
        joint
        for side in ("Left", "Right")
        for joint in chain([f"{prefix}{side}{part}" for part in ("UpLeg", "Leg", "Foot", "ToeBase")])
    ]
    spine = chain([f"{prefix}Spine", f"{prefix}Spine1", f"{prefix}Spine2"], children=[
        *chain([f"{prefix}Neck", f"{prefix}Head"]),
        *arm("Left"),
        *arm("Right"),
    ])
    return (f"{prefix}Hips", (0.0, 90.0, 0.0), [(f"{prefix}Pelvis", (0.0, -5.0, 2.0), legs), *spine])


def mixamo_b_tree():
    prefix = "mixamorig:"
    legs = [
        joint
        for side in ("Left", "Right")
        for joint in chain([f"{prefix}{side}{part}" for part in ("UpLeg", "Leg", "Foot", "ToeBase", "Toe_End")])
    ]
    spine = chain([f"{prefix}Spine", f"{prefix}Spine1", f"{prefix}Spine1_split", f"{prefix}Spine2"], children=[
        *chain([f"{prefix}Neck", f"{prefix}Head", f"{prefix}HeadTop_End"]),
        *arm("Left", split=True),
        *arm("Right", split=True),
    ])
    return (f"{prefix}Hips", (0.0, 90.0, 0.0), legs + spine)


def walk(tree):
    """ Depth-first (name, parent index, offset) of a nested tree. """
    joints = []

    def visit(node, parent):
        name, offset, children = node
        joints.append((name, parent, offset))
        index = len(joints) - 1
        for child in children:
            visit(child, index)

    visit(tree, -1)
    return joints


def hierarchy(node, order, depth=0, root=True):
    name, offset, children = node
    pad = "  " * depth
    channels = f"6 Xposition Yposition Zposition {order}" if root else f"3 {order}"
    lines = [f"{pad}{'ROOT' if root else 'JOINT'} {name}", f"{pad}{{", f"{pad}  OFFSET {offset[0]} {offset[1]} {offset[2]}",
             f"{pad}  CHANNELS {channels}"]
    if children:
        for child in children:
            lines += hierarchy(child, order, depth + 1, root=False)
    else:
        lines += [f"{pad}  End Site", f"{pad}  {{", f"{pad}    OFFSET 3 0 0", f"{pad}  }}"]
    lines.append(f"{pad}}}")
    return lines


def write_bvh(path, tree, order="Zrotation Yrotation Xrotation", frame_time=FRAME_TIME, num_frames=NUM_FRAMES, seed=0,
              still=(), start=(0.0, 0.0, 0.0)):
    """
    Write a clip of small random joint angles on a moving root; returns [T, 3 + 3J] of channel values.
    Joints named in `still` hold their first frame's angles; `start` is added to the root's path.
    """
    joints = walk(tree)
    generator = torch.Generator().manual_seed(seed)

    angles = 20.0 * torch.sin(
        torch.arange(num_frames, dtype=torch.float64)[:, None, None] * 0.05
        + 6.0 * torch.rand(1, len(joints), 3, generator=generator, dtype=torch.float64)
    )
    position = torch.arange(num_frames, dtype=torch.float64)[:, None] * torch.tensor([[1.0, 0.0, 0.5]], dtype=torch.float64)
    position = position + torch.tensor(joints[0][2], dtype=torch.float64) + 3.0 + torch.tensor(start, dtype=torch.float64)
    for index, (name, _, _) in enumerate(joints):
        if name in still:
            angles[:, index] = angles[0, index]
    frames = torch.cat([position, angles.flatten(1)], dim=-1)

    text = ["HIERARCHY", *hierarchy(tree, order), "MOTION", f"Frames: {num_frames}", f"Frame Time: {frame_time}"]
    text += [" ".join(f"{value:.6f}" for value in row) for row in frames.tolist()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(text) + "\n")
    return frames


def reference_fk(tree, frames, order):
    """ Independent centimeter FK in the file's own frame, from SciPy intrinsic Eulers. Returns {name: [T, 3]} in meters. """
    joints = walk(tree)
    axes = "".join(channel[0] for channel in order.split())
    rotations, positions = [], []
    for index, (_, parent, offset) in enumerate(joints):
        local = Rotation.from_euler(axes.upper(), frames[:, 3 + 3 * index:6 + 3 * index].numpy(), degrees=True)
        if parent < 0:
            rotations.append(local)
            positions.append(frames[:, :3].numpy())
        else:
            rotations.append(rotations[parent] * local)
            positions.append(positions[parent] + rotations[parent].apply(offset))
    return {name.rsplit(":", 1)[-1]: torch.as_tensor(positions[i] * 0.01, dtype=torch.float32)
            for i, (name, _, _) in enumerate(joints)}


def adapter_fk(collection, clip=0):
    motion = collection[clip]
    offsets = motion.offsets.expand(motion.joint_rotations.shape[0], -1, -1)
    positions, _ = ForwardKinematics(collection.topology.parents, offsets)(motion.joint_rotations, motion.root_position)
    return {name: positions[:, i] for i, name in enumerate(collection.topology.joint_names)}


def assert_partition(collection):
    joints = sorted(joint for part in collection.topology.body_joints.values() for joint in part)
    assert joints == list(range(collection.topology.num_joints))


def windows_of(source, stride=20, target_fps=30):
    config = MotionDatasetBuilderConfig(sequence_length=64, stride=stride, target_fps=target_fps)
    return MotionDatasetBuilder(config).build([source])


def train_step(datasets, config_name):
    """ One real PAN train step on these windows, with the config's model. """
    specs, batches = {}, {}
    for name, dataset in datasets.items():
        dataset.set_normalization(*compute_stats(dataset))
        specs[name] = DomainSpec.from_dataset(dataset)
        batches[name] = next(iter(torch.utils.data.DataLoader(dataset, batch_size=2)))

    with (CONFIGS / config_name).open() as f:
        method_cfg = yaml.safe_load(f)["method"]

    method = PANMethod(
        model_cfg=PANModelConfig.from_dict(method_cfg["model"]),
        domain_specs=specs,
        device="cpu",
        cfg=PANConfig(**method_cfg["params"]),
    )
    metrics = method.train_step(batches)
    assert math.isfinite(metrics["total/generator"])
    assert math.isfinite(metrics["total/discriminator"])


class TestConversion:
    @pytest.mark.parametrize("order", ["Zrotation Yrotation Xrotation", "Zrotation Xrotation Yrotation"])
    def test_dog_matches_independent_fk(self, isolated_data_root, order):
        tree = dog_tree()
        frames = write_bvh(isolated_data_root / "raw" / "dog" / "clip.bvh", tree, order=order)
        with pytest.warns(UserWarning, match="train split"):
            collection = DogAdapter().load()

        expected = reference_fk(tree, frames, order)
        for name, positions in adapter_fk(collection).items():
            assert torch.allclose(positions, expected[name], atol=1e-4), name

        motion = collection[0]
        assert collection.topology.joint_names == DogAdapter.joint_names
        assert motion.joint_rotations.shape == (NUM_FRAMES, 21, 4)
        assert motion.fps == 30.0
        assert torch.equal(motion.offsets[0], torch.zeros(3))
        assert motion.offsets[2].tolist() == pytest.approx([0.1, 0.0, 0.0])
        assert_partition(collection)

    def test_root_position_channels_replace_the_root_offset(self, isolated_data_root):
        frames = write_bvh(isolated_data_root / "raw" / "dog" / "clip.bvh", dog_tree())
        with pytest.warns(UserWarning, match="train split"):
            motion = DogAdapter().load()[0]
        assert torch.allclose(motion.root_position, (frames[:, :3] * 0.01).float(), atol=1e-5)

    def test_lafan_rebase_keeps_world_positions(self, isolated_data_root):
        tree = lafan_tree()
        order = "Zrotation Yrotation Xrotation"
        frames = write_bvh(isolated_data_root / "raw" / "lafan1" / "clip.bvh", tree, order=order)
        with pytest.warns(UserWarning):
            collection = LAFAN1Adapter().load()

        expected = reference_fk(tree, frames, order)
        for name, positions in adapter_fk(collection).items():
            assert torch.allclose(positions, expected[name], atol=1e-4), name
        assert collection.topology.joint_names == LAFAN1Adapter.joint_names
        assert_partition(collection)

    def test_lafan_root_axes_match_the_builder_contract(self, isolated_data_root):
        """ In the file +X is up and +Y forward; the adapter must hand the builder +X forward and +Y up. """
        half = math.sqrt(0.5)
        # Euler ZYX for "local +X is world up, local +Y is world +Z": the LAFAN standing pose.
        euler = Rotation.from_matrix([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]).as_euler("ZYX", degrees=True)
        path = isolated_data_root / "raw" / "lafan1" / "clip.bvh"
        frames = write_bvh(path, lafan_tree())
        frames[:, 3:6] = torch.as_tensor(euler, dtype=frames.dtype)
        lines = path.read_text().splitlines()
        lines[-NUM_FRAMES:] = [" ".join(f"{value:.6f}" for value in row) for row in frames.tolist()]
        path.write_text("\n".join(lines) + "\n")

        with pytest.warns(UserWarning, match="train split"):
            root = LAFAN1Adapter().load()[0].joint_rotations[:, 0]

        x_axis = quat_apply(root, torch.tensor([1.0, 0.0, 0.0]).expand(NUM_FRAMES, 3))
        y_axis = quat_apply(root, torch.tensor([0.0, 1.0, 0.0]).expand(NUM_FRAMES, 3))
        assert torch.allclose(x_axis, torch.tensor([0.0, 0.0, 1.0]).expand(NUM_FRAMES, 3), atol=1e-4)
        assert torch.allclose(y_axis, torch.tensor([0.0, 1.0, 0.0]).expand(NUM_FRAMES, 3), atol=1e-4)
        assert half == pytest.approx(LAFAN1Adapter.root_rest[1])

    def test_vertical_heading_axis_warns(self, isolated_data_root):
        """ A dog file's root is taken as authored, so one lying on its +X axis must not pass silently. """
        path = isolated_data_root / "raw" / "dog" / "clip.bvh"
        frames = write_bvh(path, dog_tree())
        frames[:, 3:6] = torch.tensor([90.0, 0.0, 0.0], dtype=frames.dtype)  # Zrotation: +X -> +Y
        lines = path.read_text().splitlines()
        lines[-NUM_FRAMES:] = [" ".join(f"{value:.6f}" for value in row) for row in frames.tolist()]
        path.write_text("\n".join(lines) + "\n")
        (path.parent / "train.txt").write_text("clip\n")

        with pytest.warns(UserWarning, match="mostly vertical"):
            DogAdapter().load()

    def test_train_manifest_selects_clips(self, isolated_data_root):
        raw = isolated_data_root / "raw" / "dog"
        write_bvh(raw / "D1_train.bvh", dog_tree())
        write_bvh(raw / "D1_test.bvh", dog_tree(), seed=1)
        (raw / "train.txt").write_text("D1_train\n")
        assert len(DogAdapter().load()) == 1

        (raw / "train.txt").write_text("D1_missing\n")
        DogAdapter().motion_cache_path.unlink()
        with pytest.raises(FileNotFoundError, match="D1_missing"):
            DogAdapter().load()

    def test_missing_required_joint_is_rejected(self, isolated_data_root):
        write_bvh(isolated_data_root / "raw" / "dog" / "clip.bvh", ("Hips", (0.0, 0.0, 0.0), chain(["Spine"])))
        with pytest.warns(UserWarning, match="train split"), pytest.raises(ValueError, match="missing required joints"):
            DogAdapter().load()

    def test_rounded_frame_times_do_not_force_resampling(self, isolated_data_root):
        raw = isolated_data_root / "raw" / "dog"
        write_bvh(raw / "clip.bvh", dog_tree(), frame_time="0.016667")
        with pytest.warns(UserWarning, match="train split"):
            assert DogAdapter().load()[0].fps == 60.0

    def test_three_digit_frame_time_is_30_hz(self, isolated_data_root):
        """ The LAFAN demo's header says 0.0333, which is 30.03 Hz before rounding. """
        write_bvh(isolated_data_root / "raw" / "lafan1" / "clip.bvh", lafan_tree(), frame_time="0.0333")
        with pytest.warns(UserWarning):
            assert LAFAN1Adapter().load()[0].fps == 30.0

    def test_dog_root_x_is_the_opposite_of_the_authored_x(self, isolated_data_root):
        """ The file's root +X points at the tail; the builder's heading axis must point at the head. """
        frames = write_bvh(isolated_data_root / "raw" / "dog" / "clip.bvh", dog_tree())
        frames[:, 3:6] = 0.0
        path = isolated_data_root / "raw" / "dog" / "clip.bvh"
        lines = path.read_text().splitlines()
        lines[-NUM_FRAMES:] = [" ".join(f"{value:.6f}" for value in row) for row in frames.tolist()]
        path.write_text("\n".join(lines) + "\n")

        with pytest.warns(UserWarning, match="train split"):
            root = DogAdapter().load()[0].joint_rotations[:, 0]

        expected = torch.tensor(DogAdapter.root_rest).expand(NUM_FRAMES, 4)
        assert torch.allclose(root.abs(), expected.abs(), atol=1e-6)
        x_axis = quat_apply(root, torch.tensor([1.0, 0.0, 0.0]).expand(NUM_FRAMES, 3))
        assert torch.allclose(x_axis, torch.tensor([-1.0, 0.0, 0.0]).expand(NUM_FRAMES, 3), atol=1e-6)


class TestMixamo:
    def test_group_a_reroots_at_pelvis_and_drops_details(self, isolated_data_root):
        tree = mixamo_a_tree()
        order = "Xrotation Yrotation Zrotation"
        frames = write_bvh(isolated_data_root / "raw" / "mixamo" / "Aj" / "clip.bvh", tree, order=order)
        with pytest.warns(UserWarning, match="incomplete"):
            collection = MixamoAdapter().load()

        assert collection.topology.num_joints == 23
        assert collection.topology.joint_names[0] == "Pelvis"
        assert "LeftHandIndex1" not in collection.topology.joint_names
        assert_partition(collection)

        expected = reference_fk(tree, frames, order)
        for name, positions in adapter_fk(collection).items():
            assert torch.allclose(positions, expected[name], atol=1e-4), name

    def test_group_a_reroots_a_hips_rooted_file(self, isolated_data_root):
        """ Re-rooting reverses the Hips -> Pelvis edge, which keeps world positions if Pelvis holds still. """
        tree = mixamo_a_hips_rooted_tree()
        order = "Xrotation Yrotation Zrotation"
        frames = write_bvh(
            isolated_data_root / "raw" / "mixamo" / "Aj" / "clip.bvh", tree, order=order,
            still=("mixamorig:Pelvis",), start=(2000.0, 0.0, -3000.0),
        )
        with pytest.warns(UserWarning, match="incomplete"):
            collection = MixamoAdapter().load()

        names = collection.topology.joint_names
        assert names[0] == "Pelvis"
        assert collection.topology.parents[names.index("Hips")] == 0
        expected = reference_fk(tree, frames, order)
        for name, positions in adapter_fk(collection).items():
            assert torch.allclose(positions, expected[name], atol=2e-3), name

    def test_reversed_edge_with_a_moving_offset_is_rejected(self, isolated_data_root):
        """ If Pelvis rotates under Hips, the reversed edge has no constant offset. """
        write_bvh(isolated_data_root / "raw" / "mixamo" / "Aj" / "clip.bvh", mixamo_a_hips_rooted_tree())
        with pytest.warns(UserWarning, match="incomplete"), pytest.raises(ValueError, match="moving offset"):
            MixamoAdapter().load()

    def test_group_b_reads_split_joints(self, isolated_data_root):
        tree = mixamo_b_tree()
        order = "Xrotation Yrotation Zrotation"
        frames = write_bvh(isolated_data_root / "raw" / "mixamo" / "Malcolm_m" / "clip.bvh", tree, order=order)
        with pytest.warns(UserWarning, match="incomplete"):
            collection = MixamoGroupBAdapter().load()

        assert collection.topology.num_joints == 28
        assert {"Spine1_split", "LeftShoulder_split", "RightShoulder_split"} <= set(collection.topology.joint_names)
        assert_partition(collection)

        expected = reference_fk(tree, frames, order)
        for name, positions in adapter_fk(collection).items():
            assert torch.allclose(positions, expected[name], atol=1e-4), name

    def test_groups_read_one_tree_with_separate_caches(self, isolated_data_root):
        write_bvh(isolated_data_root / "raw" / "mixamo" / "Aj" / "clip.bvh", mixamo_a_tree())
        write_bvh(isolated_data_root / "raw" / "mixamo" / "Malcolm_m" / "clip.bvh", mixamo_b_tree())
        a, b = build_adapter("mixamo_a"), build_adapter("mixamo_b")
        assert a.raw_dir == b.raw_dir
        assert a.motion_cache_path != b.motion_cache_path

    def test_moving_collapsed_offset_is_rejected(self, isolated_data_root):
        """ A dropped joint between two kept joints makes the kept offset move; never hide that. """
        tree = mixamo_a_tree()
        name, offset, children = tree
        pelvis, pelvis_offset, pelvis_children = children[0]
        extra = ("mixamorig:Dropped", (5.0, 5.0, 0.0), pelvis_children)
        write_bvh(
            isolated_data_root / "raw" / "mixamo" / "Aj" / "clip.bvh",
            (name, offset, [(pelvis, pelvis_offset, [extra])]),
        )
        with pytest.warns(UserWarning, match="incomplete"), pytest.raises(ValueError, match="moving offset"):
            MixamoAdapter().load()


class TestBuilderRegression:
    """ Loaded clips survive the existing builder as [N, 64, J + 1, 4] and train through PAN. """

    def test_mixamo_config(self, isolated_data_root):
        for character, tree in (("Aj", mixamo_a_tree()), ("Malcolm_m", mixamo_b_tree())):
            write_bvh(isolated_data_root / "raw" / "mixamo" / character / "clip.bvh", tree, num_frames=200)

        with (CONFIGS / "pan_mixamo.yaml").open() as f:
            cfg = yaml.safe_load(f)
        with pytest.warns(UserWarning, match="incomplete"):
            datasets = build_datasets(cfg["dataset"])

        # (200 - 1 - 64) // 32 + 1 = 5 windows.
        assert datasets["group_a"].motion.shape == (5, 64, 24, 4)
        assert datasets["group_b"].motion.shape == (5, 64, 29, 4)
        train_step(datasets, "pan_mixamo.yaml")

    def test_lafan_dog_config(self, isolated_data_root):
        write_bvh(isolated_data_root / "raw" / "lafan1" / "clip.bvh", lafan_tree(), num_frames=200)
        write_bvh(isolated_data_root / "raw" / "dog" / "clip.bvh", dog_tree(), num_frames=200)

        with (CONFIGS / "pan_lafan1_dog.yaml").open() as f:
            cfg = yaml.safe_load(f)
        with pytest.warns(UserWarning):
            datasets = build_datasets(cfg["dataset"])

        # (200 - 1 - 64) // 20 + 1 = 7 windows.
        assert datasets["group_a"].motion.shape == (7, 64, 23, 4)
        assert datasets["group_b"].motion.shape == (7, 64, 22, 4)
        assert sorted(datasets["group_a"].body_joints) == sorted(datasets["group_b"].body_joints)
        train_step(datasets, "pan_lafan1_dog.yaml")

    @pytest.mark.parametrize("config_name", ["pan_mixamo.yaml", "pan_lafan1_dog.yaml"])
    def test_configs_keep_the_pan_model_and_window(self, config_name):
        with (CONFIGS / config_name).open() as f:
            cfg = yaml.safe_load(f)
        with (CONFIGS / "pan.yaml").open() as f:
            pan = yaml.safe_load(f)

        assert cfg["method"] == pan["method"]
        assert cfg["trainer"] == pan["trainer"]
        for group in cfg["dataset"].values():
            assert group["params"]["sequence_length"] == pan["dataset"]["group_a"]["params"]["sequence_length"] == 64


@pytest.mark.skipif(REFERENCE is None, reason="PAN_REFERENCE_ROOT is not set")
class TestReferenceDemoClips:
    """ The real demo clips: 30 Hz, no resampling, heading well-conditioned in both domains. """

    @pytest.fixture
    def demo_sources(self, isolated_data_root):
        demo = Path(REFERENCE) / "demo_dir"
        for name, clip in (("lafan1", "Lafan1/Lafan1_example.bvh"), ("dog", "Dog/Dog_example.bvh")):
            raw = isolated_data_root / "raw" / name
            raw.mkdir(parents=True)
            shutil.copy(demo / clip, raw)
            (raw / "train.txt").write_text(Path(clip).stem + "\n")
        return LAFAN1Adapter(), DogAdapter()

    def test_headers_read_as_30_hz(self, demo_sources):
        for source in demo_sources:
            assert source.load()[0].fps == 30.0, source.dataset_name

    def test_shapes_through_the_builder(self, demo_sources):
        lafan, dog = demo_sources
        human_windows, dog_windows = windows_of(lafan), windows_of(dog)
        # 391 and 494 frames: (T - 1 - 64) // 20 + 1 windows.
        assert human_windows.motion.shape == (17, 64, 23, 4)
        assert dog_windows.motion.shape == (22, 64, 22, 4)
        train_step({"group_a": human_windows, "group_b": dog_windows}, "pan_lafan1_dog.yaml")

    def test_heading_axis_is_horizontal_and_root_is_upright(self, demo_sources):
        for source in demo_sources:
            root = source.load()[0].joint_rotations[:, 0]
            count = root.shape[0]
            x_axis = quat_apply(root, torch.tensor([1.0, 0.0, 0.0]).expand(count, 3))
            y_axis = quat_apply(root, torch.tensor([0.0, 1.0, 0.0]).expand(count, 3))
            assert x_axis[:, [0, 2]].norm(dim=-1).min() > 0.9, source.dataset_name
            assert y_axis[:, 1].mean() > 0.9, source.dataset_name

    def test_canonical_forward_travel_agrees_across_domains(self, demo_sources):
        """ Both demos walk forward, so their canonical velocity must be +X, not opposite signs. """
        for source in demo_sources:
            velocity = windows_of(source).motion[:, :, -1, :3].mean(dim=(0, 1))
            assert velocity[0] > 0.01, source.dataset_name
            assert velocity[0] > 5 * velocity[[1, 2]].abs().max(), source.dataset_name

    def test_canonical_root_pelvis_axis_points_up(self, demo_sources):
        """ Heading removal leaves only tilt: canonical roots stay near one rest orientation. """
        for source in demo_sources:
            canonical = windows_of(source).motion[:, :, 0]
            mean = canonical.mean(dim=(0, 1))
            mean = mean / mean.norm()
            assert ((canonical * mean).sum(dim=-1).abs() > 0.9).all(), source.dataset_name
