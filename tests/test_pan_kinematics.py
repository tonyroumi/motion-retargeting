"""
PAN's edge channels through its FK against the generic joint `ForwardKinematics`, on a branching skeleton
with nonidentity root and joint rotations, in order and reversed in time.
"""

from __future__ import annotations

import pytest
import torch

from motion_retargeting.kinematics import ForwardKinematics
from motion_retargeting.kinematics.quaternions import quat_apply
from motion_retargeting.methods.pan.data import PANMotionDataset, to_edge_channels
from motion_retargeting.methods.pan.kinematics import EdgeForwardKinematics, end_effector_velocity

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
    return to_edge_channels(rotations, root_position, torch.tensor(PARENTS))


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


class TestAgainstJointFK:
    def setup_method(self):
        self.rotations, self.root = random_motion()
        self.fk = EdgeForwardKinematics(PARENTS)
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


class TestReversedWindow:
    """ Reversing a window must give what the same poses and root positions, played backwards, give. """

    def setup_method(self):
        self.rotations, self.root = random_motion()
        channels = channels_of(self.rotations, self.root)
        dataset = PANMotionDataset(
            motion=channels[None],
            bank_ids=torch.zeros(1, dtype=torch.long),
            offsets=OFFSETS[None],
            mean=torch.zeros(1, channels.shape[-1]),
            std=torch.ones(1, channels.shape[-1]),
            parents=torch.tensor(PARENTS),
            body_joints={},
            augment=True,
        )
        with pytest.MonkeyPatch.context() as patch:
            patch.setattr(torch, "rand", lambda *_: torch.zeros(1))
            self.sample = dataset[0]["motion"]
        # A window's frames hold the poses and root positions of all but the clip's last frame.
        self.backwards_rotations = self.rotations[:-1].flip(0)
        self.backwards_root = self.root[:-1].flip(0)

    def test_fk_matches_joint_fk_of_the_motion_played_backwards(self):
        fk = EdgeForwardKinematics(PARENTS)
        bones = fk.bone_vectors_with_absolute_root(self.sample[None], OFFSETS[None])
        expected = joint_fk(self.backwards_rotations, self.backwards_root)
        assert torch.allclose(fk.world_positions(bones)[0], expected, atol=1e-5)

    def test_channels_match_the_motion_played_backwards(self):
        """ Every frame with a known next pose, so the yaw rate too, which FK does not read. """
        expected = channels_of(self.backwards_rotations, self.backwards_root)
        assert torch.allclose(self.sample[:-1], expected, atol=1e-5)
