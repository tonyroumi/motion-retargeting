"""
PAN body-part masks from the topology's manually labeled body parts: exact snapshots of pinned
reference HEAD, its directional rules, and (with PAN_REFERENCE_ROOT set) a differential check against
the reference's own mask code.
"""

from __future__ import annotations

import copy

import pytest
import torch

from motion_retargeting.methods.pan.masks import (
    BLOCKED,
    attention_mask,
    fresh_parts,
    labeled_parts,
    parts_to_parts,
    parts_to_tokens,
    skeleton_offset_mask,
    temporal_conv_mask,
    tokens_to_parts,
)
from pan_reference_support import (
    AMASS_BODY_JOINTS,
    BODY_JOINTS_23,
    BODY_JOINTS_28,
    CMU_BODY_JOINTS,
    FIXTURE_23,
    FIXTURE_28,
    labeled,
    load_reference,
    reference_edges,
    requires_reference,
)

PARTS_23 = labeled(FIXTURE_23)

# Rows of the labeled parts as edge indices (one conversion), always-on root column included.
FRESH_23 = [
    [0, 1, 2, 3, 22],
    [4, 5, 6, 7, 22],
    [8, 9, 10, 11, 22],
    [12, 13, 22],
    [14, 15, 16, 17, 22],
    [18, 19, 20, 21, 22],
]


def rows(mask: torch.Tensor) -> list:
    return [row.nonzero().flatten().tolist() for row in mask]


class TestLabeledParts:
    def test_parts_are_the_labels_in_key_order(self):
        assert labeled_parts(BODY_JOINTS_23, 23) == PARTS_23
        assert labeled_parts(BODY_JOINTS_28, 28)[2] == [0, 11, 12, 13, 14]

        shuffled = {key: BODY_JOINTS_23[key] for key in (3, 4, 5, 2, 0, 1)}
        assert labeled_parts(shuffled, 23) == PARTS_23

    def test_configured_sources_are_six_part_humanoids(self):
        assert labeled_parts(AMASS_BODY_JOINTS, 22) == list(AMASS_BODY_JOINTS.values())
        assert labeled_parts(CMU_BODY_JOINTS, 24) == list(CMU_BODY_JOINTS.values())

    def test_the_labels_do_not_come_from_the_skeleton_walk(self):
        """ Arms and head are disjoint from the spine: a walk would repeat the spine's last joint in them. """
        parts = labeled_parts(BODY_JOINTS_23, 23)
        assert sorted(joint for part in parts for joint in part) == list(range(23))

    @pytest.mark.parametrize("body_joints", [
        None,
        {},
        {0: [0, 1, 2, 3]},  # one part
        {key: BODY_JOINTS_23[key] for key in range(5)},  # five parts
        {**BODY_JOINTS_23, 6: []},  # seven parts
        {key + 1: part for key, part in BODY_JOINTS_23.items()},  # keys 1 to 6
        {**BODY_JOINTS_23, 3: [13, 14, 15]},  # joint 15 in two parts
        {**BODY_JOINTS_23, 3: [13]},  # joint 14 in no part
        {**BODY_JOINTS_23, 2: BODY_JOINTS_23[2] + BODY_JOINTS_23[3], 3: []},  # an empty part
    ])
    def test_arbitrary_labels_are_rejected(self, body_joints):
        with pytest.raises(ValueError):
            labeled_parts(body_joints, 23)


class TestSnapshots:
    def test_conv_rows_are_a_single_conversion(self):
        assert rows(temporal_conv_mask(PARTS_23, 23)) == FRESH_23

    def test_skeleton_rows_are_a_single_conversion(self):
        assert rows(skeleton_offset_mask(PARTS_23, 23)) == FRESH_23

    def test_skeleton_mask_reads_the_last_joint_not_the_root(self):
        mask = skeleton_offset_mask(PARTS_23, 23)
        assert mask[:, 22].all()
        assert mask[0, 0] and not mask[1:, 0].any()

    def test_caller_lists_are_not_mutated(self):
        parts = labeled(FIXTURE_23)
        before = copy.deepcopy(parts)
        fresh_parts(parts)
        attention_mask(parts, 23)
        temporal_conv_mask(parts, 23)
        skeleton_offset_mask(parts, 23)
        assert parts == before


class TestSourceNumbering:
    def test_cmu_depth_first_numbering_gives_contiguous_edges(self):
        rows_cmu = rows(temporal_conv_mask(labeled_parts(CMU_BODY_JOINTS, 24), 24))
        assert rows_cmu[0] == [0, 1, 2, 3, 23]  # Left leg joints 1-4, edges 0-3.

    def test_amass_numbering_keeps_each_part_on_its_own_chain(self):
        rows_amass = rows(temporal_conv_mask(labeled_parts(AMASS_BODY_JOINTS, 22), 22))
        # The left leg's joints 1, 4, 7, 10 own edges 0, 3, 6, 9: no spine edge is read.
        assert rows_amass[0] == [0, 3, 6, 9, 21]


class TestAttentionRules:
    def setup_method(self):
        self.mask = attention_mask(PARTS_23, 23)
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

    def test_labeled_parts_share_no_token_so_nothing_relays_between_them(self):
        left_arm_edge = 6 + 14
        head_part = 3
        two_layers = (self.allowed.float() @ self.allowed.float()) > 0

        assert not self.allowed[head_part, left_arm_edge]
        assert not two_layers[head_part, left_arm_edge]
        assert not two_layers[0, left_arm_edge]

    def test_a_token_shared_by_several_parts_relays_over_two_layers(self):
        """ The mask rule itself, on explicit parts that overlap: the two parts share edge 1. """
        allowed = attention_mask([[0, 1, 2], [0, 2, 3]], 4) == 0
        shared, only_second, first_part = 2 + 1, 2 + 2, 0

        assert not allowed[first_part, only_second]
        assert allowed[shared, only_second]
        assert allowed[first_part, shared]
        assert ((allowed.float() @ allowed.float()) > 0)[first_part, only_second]


class TestConnectivity:
    def setup_method(self):
        self.mask = temporal_conv_mask(PARTS_23, 23)

    def test_tokens_to_parts(self):
        connectivity = tokens_to_parts(self.mask, 4, 32)
        assert connectivity.shape == (192, 92)
        assert connectivity[0, 0] and connectivity[31, 3] and connectivity[0, 91]
        assert not connectivity[0, 4 * 4]
        assert connectivity[32, 4 * 4] and not connectivity[32, 3 * 4]

    def test_parts_to_parts_stays_within_part(self):
        connectivity = parts_to_parts(6, 32, 32)
        assert connectivity.shape == (192, 192)
        assert connectivity[:32, :32].all() and not connectivity[:32, 32:].any()

    def test_output_shape(self):
        assert parts_to_tokens(self.mask, 32, 4).shape == (92, 192)

    def test_uncovered_output_falls_back_to_all_inputs(self):
        # Edge 3 belongs to neither explicit part; it reads every part.
        connectivity = parts_to_tokens(temporal_conv_mask([[0, 1, 2], [0, 2, 3]], 5), 32, 4)
        assert connectivity[3 * 4:4 * 4].all()

    def test_labeled_outputs_read_their_one_owner(self):
        connectivity = parts_to_tokens(self.mask, 32, 4)
        # Edge 10 belongs to the spine alone.
        owners = connectivity[10 * 4, ::32].nonzero().flatten().tolist()
        assert owners == [2]

    def test_multiply_owned_outputs_read_every_owner(self):
        connectivity = parts_to_tokens(temporal_conv_mask([[0, 1, 2], [0, 2, 3]], 4), 32, 4)
        # Converted, the parts are edges [0, 1] and [1, 2]: edge 1 belongs to both.
        owners = connectivity[1 * 4, ::32].nonzero().flatten().tolist()
        assert owners == [0, 1]

    def test_root_token_reads_all_parts(self):
        assert parts_to_tokens(self.mask, 32, 4)[-1].all()


class TestDevices:
    def test_masks_survive_a_module_round_trip(self):
        module = torch.nn.Module()
        mask = temporal_conv_mask(PARTS_23, 23)
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
        parts = labeled(topology)

        # The reference's temporal conv masks convert twice and are intentionally not reproduced; its
        # single conversion is the skeleton offset mask.
        reference_attention = reference.calselfmask(parts, num_tokens, edges, is_conv=False)
        reference_skeleton = reference.calselfmask(labeled(topology), num_tokens, edges, is_conv=True)

        assert torch.equal(attention_mask(labeled(topology), num_tokens), reference_attention)
        assert torch.equal(temporal_conv_mask(labeled(topology), num_tokens), reference_skeleton.bool())
        assert torch.equal(skeleton_offset_mask(labeled(topology), num_tokens), reference_skeleton.bool())

    @pytest.mark.parametrize("topology", [FIXTURE_23, FIXTURE_28], ids=["23", "28"])
    def test_channel_connectivity_equals_reference_layers(self, topology):
        reference = load_reference()
        edges = reference_edges(reference, topology)
        num_tokens = len(topology)
        conv = temporal_conv_mask(labeled(topology), num_tokens)
        reference_mask = conv.float()

        shortcut = reference.BodyPartConv(6, reference_mask, num_tokens * 4, 192, 15, num_tokens, first_layer=True)
        hidden = reference.BodyPartConv(6, reference_mask, 192, 192, 15, num_tokens)
        last = reference.BodyPartConv(6, reference_mask, 192, num_tokens * 4, 15, num_tokens, last_layer=True)

        assert torch.equal(tokens_to_parts(conv, 4, 32), shortcut.mask[..., 0].bool())
        assert torch.equal(parts_to_parts(6, 32, 32), hidden.mask[..., 0].bool())
        assert torch.equal(parts_to_tokens(conv, 32, 4), last.mask[..., 0].bool())
        assert len(edges) == num_tokens - 1
