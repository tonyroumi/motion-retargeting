"""
PAN joint channels and parent-rotation FK on a branching skeleton, with no augmentation.
"""

from __future__ import annotations

import pytest
import torch

from motion_retargeting.kinematics import ForwardKinematics
from motion_retargeting.kinematics.quaternions import quat_apply
from motion_retargeting.methods.pan.data import PANMotionDataset, split_holdout, to_joint_channels
from motion_retargeting.methods.pan.kinematics import JointForwardKinematics, end_effector_velocity

# Joint 1 branches into 2 and 4, and the root into 1 and 5. Leaves: 3, 4, 6.
PARENTS = [-1, 0, 1, 2, 1, 0, 5]
LEAVES = [3, 4, 6]
NUM_JOINTS = len(PARENTS)
OFFSETS = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.5, 1.5, 0.0], [0.0, -1.0, 0.0], [2.0, 0.0, 1.0]]
)
NUM_FRAMES = 9
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def random_motion(frames: int = NUM_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    """ [T, J, 4] rotations, nonidentity everywhere with the root both turned and tilted, and [T, 3] root positions. """
    rotations = torch.nn.functional.normalize(torch.randn(frames, NUM_JOINTS, 4), dim=-1)
    return rotations, torch.cumsum(torch.randn(frames, 3), dim=0)


def joint_fk(rotations: torch.Tensor, root_position: torch.Tensor) -> torch.Tensor:
    """ The generic FK, with the root starting at the origin. """
    positions, _ = ForwardKinematics(torch.tensor(PARENTS), OFFSETS.expand(rotations.shape[0], -1, -1))(
        rotations, root_position - root_position[0]
    )
    return positions


def channels_of(rotations: torch.Tensor, root_position: torch.Tensor) -> torch.Tensor:
    return to_joint_channels(rotations, root_position)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


class TestAgainstJointFK:
    def setup_method(self):
        self.rotations, self.root = random_motion()
        self.fk = JointForwardKinematics(PARENTS)
        self.bones = self.fk.bone_vectors_with_absolute_root(channels_of(self.rotations, self.root)[None], OFFSETS[None])
        self.expected = joint_fk(self.rotations[:-1], self.root[:-1])

    def test_world_positions(self):
        assert torch.allclose(self.fk.world_positions(self.bones)[0], self.expected, atol=1e-5)

    def test_root_relative_positions(self):
        relative = self.fk.root_relative_with_absolute_root(self.bones)[0]
        assert torch.allclose(relative[:, 0], self.expected[:, 0], atol=1e-5)
        assert torch.allclose(relative[:, 1:], self.expected[:, 1:] - self.expected[:, :1], atol=1e-5)

    def test_bone_vectors(self):
        assert torch.allclose(self.bones[0, :, 0], self.expected[:, 0], atol=1e-5)
        for joint in range(1, NUM_JOINTS):
            expected = self.expected[:, joint] - self.expected[:, PARENTS[joint]]
            assert torch.allclose(self.bones[0, :, joint], expected, atol=1e-5)

    def test_end_effector_velocity(self):
        ee_ids = LEAVES
        expected = (self.expected[1:, ee_ids] - self.expected[:-1, ee_ids]) * 10.0
        velocity = end_effector_velocity(self.fk.world_positions(self.bones), ee_ids)[0]
        assert torch.allclose(velocity, expected, atol=1e-4)

    def test_a_joint_planted_by_the_root_stays_still(self):
        """ A rigid body turning and tilting about a planted foot: root motion and bones share a frame. """
        rotations = torch.tensor(IDENTITY).repeat(NUM_FRAMES, NUM_JOINTS, 1)
        rotations[:, 0] = torch.nn.functional.normalize(torch.randn(NUM_FRAMES, 4), dim=-1)
        root_to_foot = (OFFSETS[5] + OFFSETS[6]).expand(NUM_FRAMES, 3)
        root = torch.tensor([3.0, 0.5, -2.0]) - quat_apply(rotations[:, 0], root_to_foot)

        bones = self.fk.bone_vectors_with_absolute_root(channels_of(rotations, root)[None], OFFSETS[None])
        assert end_effector_velocity(self.fk.world_positions(bones), [6]).abs().max() < 1e-4


def test_channels_keep_every_joint_including_root_and_leaves():
    rotations, root = random_motion()
    channels = channels_of(rotations, root)
    assert channels.shape == (NUM_FRAMES - 1, 4 * NUM_JOINTS + 4)
    assert torch.equal(channels[:, :-4].reshape(-1, NUM_JOINTS, 4), rotations[:-1])
    assert torch.allclose(channels[:, -4:-1], root[1:] - root[:-1])


def test_yaw_and_velocity_retain_world_frame_conventions():
    angles = torch.arange(NUM_FRAMES) * 0.1
    rotations = torch.tensor(IDENTITY).repeat(NUM_FRAMES, NUM_JOINTS, 1)
    rotations[:, 0, 0] = torch.cos(angles / 2)
    rotations[:, 0, 2] = torch.sin(angles / 2)
    root = torch.stack([angles, angles * 2, angles * 3], dim=-1)
    channels = channels_of(rotations, root)
    assert torch.allclose(channels[:, -1], torch.full((NUM_FRAMES - 1,), 0.1), atol=1e-6)
    assert torch.allclose(channels[:, -4:-1], torch.tensor([0.1, 0.2, 0.3]).expand(NUM_FRAMES - 1, 3))


def test_joint_rotation_moves_descendants_not_its_own_position():
    rotations = torch.tensor(IDENTITY).repeat(NUM_FRAMES, NUM_JOINTS, 1)
    root = torch.zeros(NUM_FRAMES, 3)
    fk = JointForwardKinematics(PARENTS)

    def positions(q):
        return fk.world_positions(fk.bone_vectors_with_absolute_root(channels_of(q, root)[None], OFFSETS[None]))

    baseline = positions(rotations)
    rotations[:, 1] = torch.tensor([0.0, 0.0, 0.0, 1.0])
    changed = positions(rotations)
    assert torch.equal(changed[:, :, [0, 1, 5, 6]], baseline[:, :, [0, 1, 5, 6]])
    assert not torch.allclose(changed[:, :, 2], baseline[:, :, 2])
    assert not torch.allclose(changed[:, :, 4], baseline[:, :, 4])
    rotations[:, LEAVES] = torch.tensor([0.0, 1.0, 0.0, 0.0])
    assert torch.equal(positions(rotations), changed)


def test_train_holdout_and_subset_never_augment():
    rotations, root = random_motion()
    channels = channels_of(rotations, root)
    motion = torch.stack([channels + i for i in range(4)])
    dataset = PANMotionDataset(
        motion=motion,
        bank_ids=torch.zeros(4, dtype=torch.long),
        offsets=OFFSETS[None],
        mean=torch.full((1, channels.shape[-1]), 0.5),
        std=torch.full((1, channels.shape[-1]), 2.0),
        parents=torch.tensor(PARENTS),
        body_joints={},
    )
    train, holdout = split_holdout(dataset, 50)
    for split, indices in ((train, [0, 1]), (holdout, [2, 3]), (dataset.subset(torch.tensor([3, 1])), [3, 1])):
        assert not hasattr(split, "augment")
        for _ in range(5):
            before = torch.get_rng_state()
            for sample, original in enumerate(indices):
                assert torch.equal(split[sample]["motion"], (motion[original] - 0.5) / 2)
            assert torch.equal(before, torch.get_rng_state())
    assert torch.equal(dataset.motion, motion)


def test_fk_gradients_are_finite():
    rotations, root = random_motion()
    motion = channels_of(rotations, root)[None].requires_grad_(True)
    fk = JointForwardKinematics(PARENTS)
    fk.world_positions(fk.bone_vectors_with_absolute_root(motion, OFFSETS[None])).square().sum().backward()
    assert torch.isfinite(motion.grad).all()
    assert motion.grad[..., :4].abs().sum() > 0
    assert motion.grad[..., 4:8].abs().sum() > 0


def test_legacy_edge_channels_are_rejected():
    with pytest.raises(ValueError, match="4J"):
        JointForwardKinematics(PARENTS).bone_vectors_with_absolute_root(torch.zeros(1, 8, 4 * NUM_JOINTS), OFFSETS[None])
