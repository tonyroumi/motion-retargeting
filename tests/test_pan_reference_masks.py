"""
Reference-Mixamo body-part masks: exact snapshots of pinned reference HEAD, its directional rules, and
(with PAN_REFERENCE_ROOT set) a differential check against the reference's own mask code.
"""

from __future__ import annotations

import copy

import pytest
import torch

from motion_retargeting.methods.pan.masks import (
    BLOCKED,
    attention_mask,
    double_converted_parts,
    fresh_parts,
    humanoid_chains,
    parts_to_parts,
    parts_to_tokens,
    skeleton_offset_mask,
    temporal_conv_mask,
    tokens_to_parts,
    topology_chains,
)
from pan_reference_support import FIXTURE_23, FIXTURE_28, load_reference, local_parents, reference_edges, requires_reference

PARENTS_23 = local_parents(FIXTURE_23)
PARENTS_28 = local_parents(FIXTURE_28)

# The AMASS and CMU skeletons of configs/methods/pan_reference.yaml.
AMASS_PARENTS = [-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19]
CMU_PARENTS = [-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22]

# Conv rows measured from reference HEAD, always-on root column included.
DOUBLE_CONVERTED_23 = [
    [0, 1, 2, 22],
    [3, 4, 5, 6, 22],
    [7, 8, 9, 10, 22],
    [10, 11, 12, 22],
    [10, 13, 14, 15, 16, 22],
    [10, 17, 18, 19, 20, 22],
]
FRESH_23 = [
    [0, 1, 2, 3, 22],
    [4, 5, 6, 7, 22],
    [8, 9, 10, 11, 22],
    [11, 12, 13, 22],
    [11, 14, 15, 16, 17, 22],
    [11, 18, 19, 20, 21, 22],
]


def rows(mask: torch.Tensor) -> list:
    return [row.nonzero().flatten().tolist() for row in mask]


class TestChains:
    def test_reference_order_on_the_23_joint_fixture(self):
        assert topology_chains(PARENTS_23) == [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11, 12],
            [12, 13, 14],
            [12, 15, 16, 17, 18],
            [12, 19, 20, 21, 22],
        ]

    def test_reference_order_on_the_28_joint_fixture(self):
        chains = humanoid_chains(PARENTS_28)
        assert chains[2] == [0, 11, 12, 13, 14]
        assert chains[3] == [14, 15, 16, 17]

    def test_configured_sources_are_six_part_humanoids(self):
        assert humanoid_chains(AMASS_PARENTS) == [
            [0, 1, 4, 7, 10],
            [0, 2, 5, 8, 11],
            [0, 3, 6, 9],
            [9, 12, 15],
            [9, 13, 16, 18, 20],
            [9, 14, 17, 19, 21],
        ]
        assert humanoid_chains(CMU_PARENTS) == [
            [0, 1, 2, 3, 4],
            [0, 5, 6, 7, 8],
            [0, 9, 10, 11],
            [11, 12, 13, 14, 15],
            [11, 16, 17, 18, 19],
            [11, 20, 21, 22, 23],
        ]

    @pytest.mark.parametrize("parents", [
        [-1, 0, 1, 2],  # one chain
        [-1, 0, 0, 1, 2],  # two chains
        [-1, 0, 0, 0, 3, 3],  # five chains: a two-limbed upper body
        [-1, 0, 0, 0, 0, 0, 0],  # six chains, all leaving the root
        [-1, 0, 1, 1, 1, 0, 0, 6, 6],  # six chains with a branching first limb
        [0, 0, 0, 0, 3, 3, 3],  # reference root convention, not the local one
    ])
    def test_arbitrary_topologies_are_rejected(self, parents):
        with pytest.raises(ValueError):
            humanoid_chains(parents)


class TestSnapshots:
    def test_conv_rows_equal_reference_section_6(self):
        chains = humanoid_chains(PARENTS_23)
        assert rows(temporal_conv_mask(chains, 23)) == DOUBLE_CONVERTED_23

    def test_skeleton_rows_are_a_single_conversion(self):
        chains = humanoid_chains(PARENTS_23)
        assert rows(skeleton_offset_mask(chains, 23)) == FRESH_23

    def test_skeleton_mask_reads_the_last_joint_not_the_root(self):
        mask = skeleton_offset_mask(humanoid_chains(PARENTS_23), 23)
        assert mask[:, 22].all()
        assert mask[0, 0] and not mask[1:, 0].any()

    def test_caller_lists_are_not_mutated(self):
        chains = humanoid_chains(PARENTS_23)
        before = copy.deepcopy(chains)
        fresh_parts(chains)
        double_converted_parts(chains)
        attention_mask(chains, 23)
        temporal_conv_mask(chains, 23)
        skeleton_offset_mask(chains, 23)
        assert chains == before


class TestAttentionRules:
    def setup_method(self):
        self.chains = humanoid_chains(PARENTS_23)
        self.mask = attention_mask(self.chains, 23)
        self.allowed = self.mask == 0
        self.num_parts = 6

    def test_blocked_weight(self):
        assert self.mask.shape == (29, 29)
        assert torch.equal(self.mask.unique(), torch.tensor([BLOCKED, 0.0]))

    def test_is_directed(self):
        assert not torch.equal(self.allowed, self.allowed.t())

    def test_members_cannot_read_part_tokens(self):
        assert not self.allowed[self.num_parts:, :self.num_parts].any()

    def test_part_tokens_read_their_members_and_the_root_only(self):
        assert self.allowed[0].nonzero().flatten().tolist() == [0, 6, 7, 8, 9, 28]

    def test_every_query_reads_the_root_token_which_reads_only_itself(self):
        assert self.allowed[:, -1].all()
        assert self.allowed[-1].nonzero().flatten().tolist() == [28]

    def test_shared_branch_token_relays_over_two_layers(self):
        branch_edge = 6 + 11
        left_arm_edge = 6 + 14
        head_part = 3

        # In one layer the head part cannot see the left arm, but the branch token can.
        assert not self.allowed[head_part, left_arm_edge]
        assert self.allowed[branch_edge, left_arm_edge]
        assert self.allowed[head_part, branch_edge]

        two_layers = (self.allowed.float() @ self.allowed.float()) > 0
        assert two_layers[head_part, left_arm_edge]

        # Legs share no token with the upper body, so that relay does not reach them.
        assert not two_layers[0, left_arm_edge]


class TestConnectivity:
    def setup_method(self):
        self.mask = temporal_conv_mask(humanoid_chains(PARENTS_23), 23)

    def test_tokens_to_parts(self):
        connectivity = tokens_to_parts(self.mask, 4, 32)
        assert connectivity.shape == (192, 92)
        assert connectivity[0, 0] and connectivity[31, 3] and connectivity[0, 91]
        assert not connectivity[0, 3 * 4]
        assert connectivity[32, 3 * 4] and not connectivity[32, 2 * 4]

    def test_parts_to_parts_stays_within_part(self):
        connectivity = parts_to_parts(6, 32, 32)
        assert connectivity.shape == (192, 192)
        assert connectivity[:32, :32].all() and not connectivity[:32, 32:].any()

    def test_uncovered_output_falls_back_to_all_inputs(self):
        connectivity = parts_to_tokens(self.mask, 32, 4)
        assert connectivity.shape == (92, 192)
        # Edge 21 belongs to no part after the double conversion; it reads every part.
        assert connectivity[21 * 4:22 * 4].all()

    def test_multiply_owned_outputs_read_every_owner(self):
        connectivity = parts_to_tokens(self.mask, 32, 4)
        # Edge 10 is shared by the spine, head and both arms.
        owners = connectivity[10 * 4, ::32].nonzero().flatten().tolist()
        assert owners == [2, 3, 4, 5]

    def test_root_token_reads_all_parts(self):
        assert parts_to_tokens(self.mask, 32, 4)[-1].all()


class TestDevices:
    def test_masks_survive_a_module_round_trip(self):
        module = torch.nn.Module()
        mask = temporal_conv_mask(humanoid_chains(PARENTS_23), 23)
        module.register_buffer("mask", mask.clone())
        module.to("cpu")
        assert torch.equal(module.mask, mask)

        restored = torch.nn.Module()
        restored.register_buffer("mask", torch.zeros_like(mask))
        restored.load_state_dict(module.state_dict())
        assert torch.equal(restored.mask, mask)


@requires_reference
class TestAgainstReference:
    """ Each reference call receives its own copies: its conversions mutate the lists they are given. """

    @pytest.mark.parametrize("topology", [FIXTURE_23, FIXTURE_28], ids=["23", "28"])
    def test_masks_equal_reference_head(self, topology):
        reference = load_reference()
        edges = reference_edges(reference, topology)
        num_tokens = len(topology)
        chains = humanoid_chains(local_parents(topology))

        parts = reference.getbodyparts(edges)
        assert [list(part) for part in parts] == chains

        # The encoder's two successive calls, in the same order, over the same lists.
        reference_attention = reference.calselfmask(parts, num_tokens, edges, is_conv=False)
        reference_conv = reference.calselfmask(parts, num_tokens, edges, is_conv=True)
        reference_skeleton = reference.calselfmask(reference.getbodyparts(edges), num_tokens, edges, is_conv=True)

        assert torch.equal(attention_mask(chains, num_tokens), reference_attention)
        assert torch.equal(temporal_conv_mask(chains, num_tokens), reference_conv.bool())
        assert torch.equal(skeleton_offset_mask(chains, num_tokens), reference_skeleton.bool())

    @pytest.mark.parametrize("topology", [FIXTURE_23, FIXTURE_28], ids=["23", "28"])
    def test_channel_connectivity_equals_reference_layers(self, topology):
        reference = load_reference()
        edges = reference_edges(reference, topology)
        num_tokens = len(topology)
        chains = humanoid_chains(local_parents(topology))
        conv = temporal_conv_mask(chains, num_tokens)
        reference_mask = conv.float()

        shortcut = reference.BodyPartConv(6, reference_mask, num_tokens * 4, 192, 15, num_tokens, first_layer=True)
        hidden = reference.BodyPartConv(6, reference_mask, 192, 192, 15, num_tokens)
        last = reference.BodyPartConv(6, reference_mask, 192, num_tokens * 4, 15, num_tokens, last_layer=True)

        assert torch.equal(tokens_to_parts(conv, 4, 32), shortcut.mask[..., 0].bool())
        assert torch.equal(parts_to_parts(6, 32, 32), hidden.mask[..., 0].bool())
        assert torch.equal(parts_to_tokens(conv, 32, 4), last.mask[..., 0].bool())
        assert len(edges) == num_tokens - 1
