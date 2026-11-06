"""
PAN's edge FK spaces, against closed-form fixtures and (with PAN_REFERENCE_ROOT
set) the pinned reference ForwardKinematics and get_ee.
"""

from __future__ import annotations

import math

import pytest
import torch

from motion_retargeting.methods.pan.kinematics import (
    EdgeForwardKinematics,
    end_effector_ids,
    end_effector_velocity,
    path_heights,
)
from motion_retargeting.methods.pan.masks import humanoid_chains
from pan_reference_support import ARGS, FIXTURE_23, FIXTURE_28, load_reference, local_parents, reference_edges, requires_reference

# Root branching into 1 and 3; joint 1 continues to 2.
PARENTS = [-1, 0, 1, 0]
OFFSETS = torch.tensor([[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0]]])
ROOT = torch.tensor([5.0, 6.0, 7.0])


def rotation_z(angle: float) -> list:
    return [math.cos(angle / 2), 0.0, 0.0, math.sin(angle / 2)]


def motion_of(edge_rotations, root=ROOT) -> torch.Tensor:
    """ [1, 1, 4E + 3] from E per-edge quaternions. """
    return torch.cat([torch.tensor(edge_rotations).flatten(), root])[None, None]


class TestClosedForm:
    def setup_method(self):
        self.fk = EdgeForwardKinematics(PARENTS)
        # Edge 0 -> 1 turns 90 degrees about Z; its continuation is identity; the other root copy differs.
        identity = rotation_z(0.0)
        self.motion = motion_of([rotation_z(math.pi / 2), identity, identity])

    def test_bone_vector_space(self):
        bones = self.fk.bone_vectors_with_absolute_root(self.motion, OFFSETS)[0, 0]
        assert torch.allclose(bones[0], ROOT)
        # The bone ending at i is rotated by its own edge rotation.
        assert torch.allclose(bones[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
        assert torch.allclose(bones[2], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
        # A differing predicted copy of the root's rotation is applied on its own edge alone.
        assert torch.allclose(bones[3], torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)

    def test_root_relative_space(self):
        bones = self.fk.bone_vectors_with_absolute_root(self.motion, OFFSETS)
        positions = self.fk.root_relative_with_absolute_root(bones)[0, 0]
        assert torch.allclose(positions[0], ROOT)
        assert torch.allclose(positions[1], torch.tensor([0.0, 1.0, 0.0]), atol=1e-6)
        assert torch.allclose(positions[2], torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)

    def test_world_space(self):
        bones = self.fk.bone_vectors_with_absolute_root(self.motion, OFFSETS)
        positions = self.fk.world_positions(bones)[0, 0]
        assert torch.allclose(positions[0], ROOT)
        assert torch.allclose(positions[2], ROOT + torch.tensor([0.0, 2.0, 0.0]), atol=1e-6)

    def test_rotations_are_normalized_before_use(self):
        scaled = self.motion.clone()
        scaled[..., :-3] *= 3.0
        assert torch.allclose(
            self.fk.bone_vectors_with_absolute_root(scaled, OFFSETS),
            self.fk.bone_vectors_with_absolute_root(self.motion, OFFSETS),
            atol=1e-6,
        )

    def test_matches_joint_fk_on_real_edge_copies(self):
        """ With real data every outgoing edge copies its parent's rotation: this is ordinary FK. """
        from motion_retargeting.kinematics import ForwardKinematics
        from motion_retargeting.methods.pan.data import to_edge_channels

        torch.manual_seed(0)
        parents = torch.tensor(PARENTS)
        rotations = torch.nn.functional.normalize(torch.randn(8, 4, 4), dim=-1)
        root = torch.randn(8, 3)

        channels = to_edge_channels(rotations, root, parents)[None]
        bones = self.fk.bone_vectors_with_absolute_root(channels, OFFSETS)

        expected, _ = ForwardKinematics(parents, OFFSETS.expand(8, -1, -1))(rotations, root)
        assert torch.allclose(self.fk.world_positions(bones)[0], expected, atol=1e-5)


class TestEndEffectors:
    def test_order_and_velocity(self):
        chains = humanoid_chains(local_parents(FIXTURE_23))
        assert end_effector_ids(chains) == [4, 8, 14, 18, 22]

        world = torch.zeros(1, 3, 23, 3)
        world[:, 1, 4, 0] = 2.0
        velocity = end_effector_velocity(world, [4, 8, 14, 18, 22])
        assert velocity.shape == (1, 2, 5, 3)
        assert velocity[0, 0, 0, 0] == 20.0 and velocity[0, 1, 0, 0] == -20.0

    def test_configured_sources(self):
        amass = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
        assert end_effector_ids(humanoid_chains(amass)) == [10, 11, 15, 20, 21]


class TestHeights:
    def test_path_sum_not_bounding_box(self):
        parents = local_parents(FIXTURE_23).tolist()
        chains = humanoid_chains(parents)

        # Unit bones all along +X: the vertical extent is zero and so is any Y bounding box.
        offsets = torch.zeros(1, 23, 3)
        offsets[:, 1:, 0] = 1.0
        assert path_heights(offsets, parents, chains).item() == pytest.approx(4 + 6)

    def test_scales_with_geometry_per_bank(self):
        parents = local_parents(FIXTURE_23).tolist()
        chains = humanoid_chains(parents)
        offsets = torch.randn(1, 23, 3)
        heights = path_heights(torch.cat([offsets, 2 * offsets]), parents, chains)
        assert heights[1] == pytest.approx(2 * heights[0].item())


@requires_reference
class TestAgainstReference:
    @pytest.mark.parametrize("topology", [FIXTURE_23, FIXTURE_28], ids=["23", "28"])
    def test_all_spaces_match(self, topology):
        reference = load_reference()
        reference_fk = reference.ForwardKinematics(ARGS, reference_edges(reference, topology))

        torch.manual_seed(0)
        num_joints = len(topology)
        # Every predicted edge rotation differs, including root and branch copies, and the root moves.
        motion = torch.randn(2, 8, 4 * (num_joints - 1) + 3)
        motion[..., -3:] = motion[..., -3:] * 2 + torch.tensor([1.0, 3.0, -2.0])
        offsets = torch.randn(2, num_joints, 3)
        offsets[:, 0] = 0.0

        fk = EdgeForwardKinematics(local_parents(topology).tolist())
        bones = fk.bone_vectors_with_absolute_root(motion, offsets)

        remote_bones = reference_fk.forward_from_raw(motion.transpose(1, 2), offsets)
        assert torch.allclose(bones, remote_bones, atol=1e-5, rtol=1e-4)

        assert torch.allclose(
            fk.root_relative_with_absolute_root(bones),
            reference_fk.from_local_to_world(remote_bones),
            atol=1e-5,
            rtol=1e-4,
        )

        chains = humanoid_chains(local_parents(topology))
        ee_ids = end_effector_ids(chains)

        positions = reference.get_ee(remote_bones, list(topology), list(range(num_joints)), velo=False, from_root=True)
        assert torch.allclose(fk.world_positions(bones), positions, atol=1e-5, rtol=1e-4)

        remote_velocity = reference.get_ee(remote_bones, list(topology), ee_ids, velo=True, from_root=True)
        assert torch.allclose(end_effector_velocity(fk.world_positions(bones), ee_ids), remote_velocity, atol=1e-4, rtol=1e-4)
