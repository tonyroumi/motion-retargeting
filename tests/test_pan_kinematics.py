"""
PAN's edge channels and FK against the generic joint `ForwardKinematics`, on a branching skeleton with
nonidentity root and joint rotations, and through the position spaces the losses compare.
"""

from __future__ import annotations

import math

import pytest
import torch

from motion_retargeting.kinematics import ForwardKinematics
from motion_retargeting.kinematics.quaternions import quat_apply
from motion_retargeting.methods.pan.config import DomainSpec, PANConfig, PANModelConfig
from motion_retargeting.methods.pan.data import to_edge_channels
from motion_retargeting.methods.pan.kinematics import EdgeForwardKinematics, end_effector_velocity
from motion_retargeting.methods.pan.method import PANMethod

# Joint 1 branches into 2 and 4, and the root into 1 and 5. Leaves: 3, 4, 6.
PARENTS = [-1, 0, 1, 2, 1, 0, 5]
LEAVES = [3, 4, 6]
NUM_JOINTS = len(PARENTS)
OFFSETS = torch.tensor(
    [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0], [0.5, 1.5, 0.0], [0.0, -1.0, 0.0], [2.0, 0.0, 1.0]]
)
NUM_FRAMES = 9
IDENTITY = [1.0, 0.0, 0.0, 0.0]


def about_y(angle: float) -> torch.Tensor:
    return torch.tensor([math.cos(angle / 2), 0.0, math.sin(angle / 2), 0.0])


def random_motion(frames: int = NUM_FRAMES) -> tuple[torch.Tensor, torch.Tensor]:
    """ [T, J, 4] rotations, nonidentity everywhere with the root both turned and tilted, and [T, 3] root positions. """
    rotations = torch.nn.functional.normalize(torch.randn(frames, NUM_JOINTS, 4), dim=-1)
    return rotations, torch.cumsum(torch.randn(frames, 3), dim=0)


def joint_fk(rotations: torch.Tensor, root_position: torch.Tensor) -> torch.Tensor:
    """ The generic FK over all but the last frame, as channels are, with the root starting at the origin. """
    frames = rotations.shape[0] - 1
    positions, _ = ForwardKinematics(torch.tensor(PARENTS), OFFSETS.expand(frames, -1, -1))(
        rotations[:-1], root_position[:-1] - root_position[0]
    )
    return positions


def channels_of(rotations: torch.Tensor, root_position: torch.Tensor) -> torch.Tensor:
    return to_edge_channels(rotations, root_position, torch.tensor(PARENTS))


def edge_rotations(channels: torch.Tensor) -> torch.Tensor:
    """ [T, 4E + 4] -> [T, E, 4] """
    return channels[:, :-4].reshape(channels.shape[0], NUM_JOINTS - 1, 4)


@pytest.fixture(autouse=True)
def seeded():
    torch.manual_seed(0)


class TestEdgeChannels:
    def test_each_edge_carries_its_parents_rotation(self):
        rotations, root = random_motion()
        edges = edge_rotations(channels_of(rotations, root))
        for child in range(1, NUM_JOINTS):
            assert torch.equal(edges[:, child - 1], rotations[:-1, PARENTS[child]])

    def test_a_branching_joint_repeats_and_a_leaf_disappears(self):
        rotations, root = random_motion()
        edges = edge_rotations(channels_of(rotations, root))

        assert torch.equal(edges[:, 0], edges[:, 4])  # The root's copies, on edges to 1 and 5
        assert torch.equal(edges[:, 1], edges[:, 3])  # Joint 1's copies, on edges to 2 and 4

        changed = rotations.clone()
        changed[:, LEAVES] = torch.tensor(IDENTITY)
        assert torch.equal(channels_of(changed, root), channels_of(rotations, root))

    def test_root_channels_are_world_velocity_and_yaw_rate(self):
        rotations, root = random_motion()
        rotations[:, 0] = torch.stack([about_y(0.1 * frame * frame) for frame in range(NUM_FRAMES)])
        channels = channels_of(rotations, root)

        assert channels.shape == (NUM_FRAMES - 1, 4 * (NUM_JOINTS - 1) + 4)
        assert torch.allclose(channels[:, -4:-1], root[1:] - root[:-1])
        steps = torch.tensor([0.1 * (2 * frame + 1) for frame in range(NUM_FRAMES - 1)])
        assert torch.allclose(channels[:, -1], steps, atol=1e-5)


class TestAgainstJointFK:
    def setup_method(self):
        self.rotations, self.root = random_motion()
        self.fk = EdgeForwardKinematics(PARENTS)
        self.bones = self.fk.bone_vectors_with_absolute_root(channels_of(self.rotations, self.root)[None], OFFSETS[None])
        self.expected = joint_fk(self.rotations, self.root)

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


class TestPredictedEdges:
    """ A prediction's copies of one joint's rotation may differ, and each stays on its own edge. """

    def setup_method(self):
        self.fk = EdgeForwardKinematics(PARENTS)

    def bones_with(self, edges: dict[int, torch.Tensor]) -> torch.Tensor:
        rotations = torch.tensor(IDENTITY).repeat(NUM_JOINTS - 1, 1)
        for child, rotation in edges.items():
            rotations[child - 1] = rotation
        motion = torch.cat([rotations.flatten(), torch.zeros(4)])[None, None]
        return self.fk.bone_vectors_with_absolute_root(motion, OFFSETS[None])[0, 0]

    def test_an_edge_turns_its_own_bone_and_those_below_it(self):
        rotation = about_y(math.pi / 2)
        bones = self.bones_with({1: rotation})

        for joint in (1, 2, 3, 4):
            assert torch.allclose(bones[joint], quat_apply(rotation, OFFSETS[joint]), atol=1e-6)
        for joint in (5, 6):  # The other copy of the root's rotation
            assert torch.allclose(bones[joint], OFFSETS[joint], atol=1e-6)

    def test_sibling_edges_are_independent(self):
        rotation = about_y(0.7)
        bones = self.bones_with({2: rotation})

        assert torch.allclose(bones[2], quat_apply(rotation, OFFSETS[2]), atol=1e-6)
        assert torch.allclose(bones[4], OFFSETS[4], atol=1e-6)

    def test_rotations_are_normalized(self):
        rotation = about_y(0.7)
        assert torch.allclose(self.bones_with({2: rotation * 3.0}), self.bones_with({2: rotation}), atol=1e-6)


class TestDifferentiability:
    @pytest.mark.parametrize("space", ["bone_vectors", "root_relative_with_absolute_root", "world_positions"])
    def test_gradients_reach_every_channel_positions_depend_on(self, space):
        fk = EdgeForwardKinematics(PARENTS)
        motion = torch.randn(1, NUM_FRAMES, 4 * (NUM_JOINTS - 1) + 4, requires_grad=True)
        bones = fk.bone_vectors_with_absolute_root(motion, OFFSETS[None])
        positions = bones if space == "bone_vectors" else getattr(fk, space)(bones)
        positions.pow(2).sum().backward()

        gradient = motion.grad[0]
        assert edge_rotations(gradient).abs().sum(dim=(0, 2)).min() > 0
        # Each frame's velocity shapes later positions, so the last frame's shapes none.
        assert gradient[:-1, -4:-1].abs().sum(-1).min() > 0
        # The heading comes from the root's edge rotations, so FK does not read the yaw rate.
        assert gradient[:, -1].abs().sum() == 0


def domain_spec(channels: torch.Tensor) -> DomainSpec:
    return DomainSpec(
        parents=torch.tensor(PARENTS),
        offsets=OFFSETS[None],
        mean=channels.mean(dim=(0, 1))[None],
        std=channels.std(dim=(0, 1))[None],
        body_joints={0: [1], 1: [2], 2: [0], 3: [3], 4: [4, 5], 5: [6]},
    )


class TestLossComparisons:
    def setup_method(self):
        self.rotations, self.root = random_motion(17)
        channels = channels_of(self.rotations, self.root)[None]
        spec = domain_spec(channels)
        self.method = PANMethod(PANModelConfig(), {"a": spec, "b": spec}, cfg=PANConfig(contact_threshold=1e6))
        self.domain = self.method.model.domains["a"]
        self.bank = torch.zeros(1, dtype=torch.long)
        self.normalized = (channels - spec.mean[0]) / spec.std[0]

    def positions(self, motion: torch.Tensor) -> torch.Tensor:
        return self.method.normalized_positions("a", self.domain.bones(motion, self.bank), self.bank)

    def test_position_loss_target_is_joint_fk_over_height(self):
        """ Normalized channels through denormalization and FK, to what the position loss compares. """
        expected = joint_fk(self.rotations, self.root)
        expected[:, 1:] -= expected[:, :1].clone()
        assert torch.allclose(self.positions(self.normalized)[0], expected / self.domain.heights[0], atol=1e-4)

    def test_end_effector_target_is_joint_fk_over_height(self):
        bones = self.domain.bones(self.normalized, self.bank)
        velocity = self.method.normalized_end_effector_velocity("a", bones, self.bank)[0]

        expected = joint_fk(self.rotations, self.root)[:, self.domain.ee_ids]
        expected = (expected[1:] - expected[:-1]) * 10.0 / self.domain.heights[0]
        assert torch.allclose(velocity, expected, atol=1e-4)

    def test_position_comparison_sees_a_wrong_prediction(self):
        prediction = self.normalized.clone().requires_grad_()
        target = self.positions(self.normalized).detach()
        assert torch.nn.functional.mse_loss(self.positions(prediction), target).item() < 1e-8

        with torch.no_grad():
            prediction[..., 0] += 1.0  # The root's rotation, on its edge to joint 1
        loss = torch.nn.functional.mse_loss(self.positions(prediction), target)
        assert loss.item() > 1e-4
        loss.backward()
        assert prediction.grad[..., 0].abs().sum() > 0

    def test_train_step_trains_the_generator_through_fk(self):
        batch = {
            name: {"motion": self.normalized.expand(2, -1, -1), "bank_id": self.bank.expand(2)}
            for name in ("a", "b")
        }
        metrics = self.method.train_step(batch)

        for name in ("a/position", "a_to_b/end_effector", "a_to_b/position_cycle", "a_to_b/adversarial"):
            assert math.isfinite(metrics[name]) and metrics[name] > 0
        assert metrics["grad/a/decoder"] > 0
