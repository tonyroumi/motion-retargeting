"""Joint-token consumers, checkpoint boundaries and optional LAFAN1/dog differential checks."""

import importlib.util
import math
import os
from pathlib import Path
import sys
from types import ModuleType

import pytest
import torch

from motion_retargeting.data.representations.motion import MotionCollection, MotionData
from motion_retargeting.data.representations.skeleton import SkeletonTopology
from motion_retargeting.methods.pan.config import DomainSpec, PANModelConfig
from motion_retargeting.methods.pan.data import PANDataConfig, build_from_collections, to_joint_channels
from motion_retargeting.methods.pan.kinematics import JointForwardKinematics
from motion_retargeting.methods.pan.masks import attention_mask, skeleton_offset_mask, temporal_conv_mask
from motion_retargeting.methods.pan.method import PANMethod

PARENTS = [-1, 0, 0, 0, 3, 3, 3]
PARTS = {0: [0, 1], 1: [2], 2: [3], 3: [4], 4: [5], 5: [6]}
OFFSETS = torch.tensor([
    [0., 0., 0.], [.1, -.5, 0.], [-.1, -.5, 0.], [0., .5, 0.],
    [0., .3, 0.], [.3, .2, 0.], [-.3, .2, 0.],
])


def dataset(extra_leaf=False):
    torch.manual_seed(14)
    parents = PARENTS + ([6] if extra_leaf else [])
    parts = {key: list(value) for key, value in PARTS.items()}
    offsets = OFFSETS
    if extra_leaf:
        parts[5].append(7)
        offsets = torch.cat([offsets, torch.tensor([[-.1, 0., 0.]])])
    rotations = torch.nn.functional.normalize(torch.randn(21, len(parents), 4), dim=-1)
    root = torch.randn(21, 3).cumsum(0) * .01
    collection = MotionCollection(
        SkeletonTopology(torch.tensor(parents), body_joints=parts),
        [MotionData(offsets, rotations, 30., root)],
    )
    result = build_from_collections([collection], PANDataConfig(sequence_length=16, stride=4))
    expected = to_joint_channels(rotations, root)
    assert torch.equal(result.motion[0], expected[:16])
    assert torch.equal(result.motion[1], expected[4:20])
    assert result.mean.shape == (1, 4 * len(parents) + 4)
    return result


def test_masks_use_joint_indices_and_share_root_and_velocity():
    parts = list(PARTS.values())
    conv = temporal_conv_mask(parts, 8)
    attn = attention_mask(parts, 8) == 0
    offsets = skeleton_offset_mask(parts, 7)
    for i, part in enumerate(parts):
        expected = set(part) | {0, 7}
        assert set(conv[i].nonzero().flatten().tolist()) == expected
        assert set(attn[i].nonzero().flatten().tolist()) == {i} | {6 + j for j in expected}
        assert set(offsets[i].nonzero().flatten().tolist()) == set(part) - {0}
    assert attn[:, 6].all() and attn[:, -1].all()
    assert not offsets[:, 0].any()
    assert offsets[:, -1].sum() == 1  # last joint is not a global offset
    assert parts == list(PARTS.values())  # masks must not mutate labels


def test_joint_pipeline_train_validation_and_checkpoint():
    """Small CPU optimizer smoke, not a convergence or dataset-quality experiment."""
    old_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    try:
        datasets = {"a": dataset(), "b": dataset(extra_leaf=True)}
        specs = {name: DomainSpec.from_dataset(data) for name, data in datasets.items()}
        config = PANModelConfig.from_dict({
            "motion_encoder": {"embed_dim": 8, "embed_hidden_dim": 16, "num_layers": 1,
                               "num_heads": 2, "feedforward_dim": 16, "kernel_size": 3, "dropout": 0.0},
            "motion_decoder": {"kernel_size": 3},
            "motion_discriminator": {"kernel_size": 3, "hidden_dim": 16},
        })
        method = PANMethod(config, specs)
        batch = {name: {"motion": torch.stack([data[i]["motion"] for i in range(2)]),
                        "bank_id": data.bank_ids} for name, data in datasets.items()}
        outputs = method.model.forward_generator(batch)
        for name, spec in specs.items():
            assert spec.num_tokens == spec.num_joints + 1
            assert outputs.reconstructions[name].shape == batch[name]["motion"].shape
            for source in specs:
                pair = outputs.pairs[(source, name)]
                assert pair.fake.shape[-1] == 4 * spec.num_joints + 4
        for _ in range(2):
            metrics = method.train_step(batch)
            assert all(math.isfinite(value) for value in metrics.values())
            assert all(torch.isfinite(p).all() for p in method.model.parameters())
        method.eval()
        assert all(math.isfinite(value) for value in method.validation_step(batch).values())
        checkpoint = method.state_dict()
        method.load_state_dict(checkpoint)
        checkpoint["metadata"]["domains"]["a"].pop("skeleton_banks_sha256")
        with pytest.raises(ValueError, match="domains"):
            method.load_state_dict(checkpoint)  # legacy cm banks cannot masquerade as metres
        checkpoint["metadata"] = method.checkpoint_metadata()
        checkpoint["metadata"]["domains"]["a"]["statistics_sha256"] = "different units/statistics"
        with pytest.raises(ValueError, match="domains"):
            method.load_state_dict(checkpoint)
        checkpoint["metadata"] = method.checkpoint_metadata()
        checkpoint["metadata"] = dict(checkpoint["metadata"], representation="edge_parent_quaternions_root_velocity_xyz_yaw")
        with pytest.raises(ValueError, match="representation"):
            method.load_state_dict(checkpoint)
        spec = specs["a"]
        with pytest.raises(ValueError, match="rebuild legacy edge data"):
            DomainSpec(spec.parents, spec.offsets, spec.mean[:, :-4], spec.std[:, :-4], spec.body_joints)
    finally:
        torch.set_num_threads(old_threads)


@pytest.fixture
def reference(monkeypatch):
    root = os.environ.get("PAN_REFERENCE_ROOT")
    if not root:
        pytest.skip("Set PAN_REFERENCE_ROOT to run against the LAFAN1/dog source")

    def load(name, path):
        spec = importlib.util.spec_from_file_location(name, Path(root) / path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module

    rotation = load("pan_reference_rotation", "utils/rotation.py")
    package = ModuleType("utils")
    package.rotation = rotation
    monkeypatch.setitem(sys.modules, "utils", package)
    return load("pan_reference_utils", "utils/utils.py")


def test_lafan1dog_reference_fk(reference):
    torch.manual_seed(4)
    motion = torch.randn(2, 9, 4 * len(PARENTS) + 4)
    offsets = OFFSETS[None].expand(2, -1, -1)
    local_positions = offsets[:, None].expand(-1, 9, -1, -1)
    reference_world, reference_relative = reference.ForwardKinematics(PARENTS, len(PARENTS)).forward(motion, local_positions)
    fk = JointForwardKinematics(PARENTS)
    bones = fk.bone_vectors_with_absolute_root(motion, offsets)
    world = fk.world_positions(bones)
    # Reference includes frame t's displacement in position t; we deliberately retain the exclusive sum.
    torch.testing.assert_close(world, reference_world - motion[..., -4:-1].unsqueeze(-2))
    torch.testing.assert_close(world - world[..., :1, :], reference_relative)


def test_lafan1dog_reference_joint_masks(reference):
    parts = list(PARTS.values())
    torch.testing.assert_close(temporal_conv_mask(parts, 8), reference.get_part_matrix(parts, 8).bool())
    assert torch.equal(attention_mask(parts, 8) == 0, reference.get_transformer_matrix(parts, 8) == 0)
    # Reference physically omits root offset; this implementation retains its storage column, masked out.
    torch.testing.assert_close(skeleton_offset_mask(parts, 7)[:, 1:], reference.get_offset_part_matrix(parts, 6).bool())
