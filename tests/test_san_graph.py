"""
SAN's token graph: snapshot tests, plus comparisons with the official code when SAN_REFERENCE_ROOT is set.
"""

from __future__ import annotations

import pytest
import torch

from motion_retargeting.methods.san.config import DomainSpec
from motion_retargeting.modules.skeletal import SkeletalPooling, build_edge_list, construct_adj
from san_reference_support import FIXTURE_23, FIXTURE_28, local_parents, load_reference, reference_edges, requires_reference

AMASS_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)
CMU_PARENTS = (-1, 0, 1, 2, 3, 0, 5, 6, 7, 0, 9, 10, 11, 12, 13, 14, 11, 16, 17, 18, 11, 20, 21, 22)

SKELETONS = {
    "reference_a": tuple(local_parents(FIXTURE_23).tolist()),
    "reference_b": tuple(local_parents(FIXTURE_28).tolist()),
    "amass": AMASS_PARENTS,
    "cmu": CMU_PARENTS,
}

# skeleton: tokens per encoder level, the global token's row, discriminator tokens
SNAPSHOT = {
    "reference_a": ((23, 12, 7), [0, 1, 2, 4, 5, 8, 9], 7),
    "reference_b": ((28, 17, 7), [0, 1, 2, 5, 6, 10, 11], 11),
    "amass": ((22, 12, 7), [0, 1, 2, 3, 4, 5, 6], 7),
    "cmu": ((24, 13, 7), [0, 1, 2, 4, 5, 8, 9], 7),
}


def encoder_levels(parents):
    """ The graph at each level of the motion encoder, as (edges, adjacency, pooled groups). """
    edges = build_edge_list(parents)
    first = SkeletalPooling(edges, 8).pooled_info
    second = SkeletalPooling(first.edge_list, 16, last_pool=True).pooled_info
    return edges, first, second


class TestSnapshots:
    def test_edge_list_uses_joint_ids(self):
        assert build_edge_list(AMASS_PARENTS)[:4] == [[0, 1], [0, 2], [0, 3], [1, 4]]
        assert len(build_edge_list(AMASS_PARENTS)) == len(AMASS_PARENTS) - 1

    @pytest.mark.parametrize("name", SKELETONS)
    def test_tokens_per_level(self, name):
        parents = SKELETONS[name]
        edges, first, second = encoder_levels(parents)
        tokens, _, _ = SNAPSHOT[name]

        assert (len(construct_adj(edges)), len(first.adj_list), len(second.pooled_edges)) == tokens

    @pytest.mark.parametrize("name", SKELETONS)
    def test_global_token_reads_edge_0s_neighbors_and_never_itself(self, name):
        adjacency = construct_adj(build_edge_list(SKELETONS[name]))
        _, global_row, _ = SNAPSHOT[name]

        assert adjacency[-1] == global_row
        assert len(adjacency) - 1 not in adjacency[-1]

    @pytest.mark.parametrize("name", SKELETONS)
    def test_discriminator_tokens(self, name):
        # The discriminator never takes the whole-chain last pool.
        _, first, _ = encoder_levels(SKELETONS[name])
        _, _, tokens = SNAPSHOT[name]
        assert len(SkeletalPooling(first.edge_list, 12).pooled_info.pooled_edges) == tokens

    def test_every_level_has_the_same_primal_skeleton_for_the_real_topologies(self):
        assert len(encoder_levels(AMASS_PARENTS)[2].pooled_edges) == len(encoder_levels(CMU_PARENTS)[2].pooled_edges)

    def test_spec_exposes_the_edge_list(self):
        num_joints = len(CMU_PARENTS)
        spec = DomainSpec(
            parents=torch.tensor(CMU_PARENTS),
            offsets=torch.zeros(1, num_joints, 3),
            mean=torch.zeros(1, 4 * (num_joints - 1) + 3),
            std=torch.ones(1, 4 * (num_joints - 1) + 3),
        )
        assert spec.edge_list == build_edge_list(CMU_PARENTS)


@requires_reference
class TestAgainstReference:
    @pytest.mark.parametrize("name", SKELETONS)
    def test_neighbor_and_pooling_lists_equal_the_references(self, name):
        reference = load_reference()
        parents = SKELETONS[name]
        topology = (0, *parents[1:])

        edges = reference_edges(reference, topology)
        ours, first, second = encoder_levels(parents)

        assert construct_adj(ours) == reference.find_neighbor(edges, 2)

        pool = reference.SkeletonPool(edges, "mean", 8)
        assert first.pooled_edges == pool.pooling_list
        # An unpooled reference edge keeps its (parent, child, offset) tuple.
        assert first.edge_list == [list(edge[:2]) for edge in pool.new_edges]
        assert first.adj_list == reference.find_neighbor(pool.new_edges, 2)

        assert second.pooled_edges == reference.SkeletonPool(pool.new_edges, "mean", 16, last_pool=True).pooling_list
