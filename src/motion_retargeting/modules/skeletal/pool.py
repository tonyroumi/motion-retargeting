"""
Skeletal pooling: merges pairs of edges that meet at a (joint) node of degree 2.
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch
from torch import nn

from .graph import construct_adj


@dataclass
class PoolingInfo:
    """ The graph at one level of the pooling hierarchy. """
    adj_list: List[List[int]]
    edge_list: List[List[int]]
    pooled_edges: Optional[List[List[int]]] = None  # Edges of the previous level merged into each row


class SkeletalPooling(nn.Module):
    def __init__(
        self,
        edge_list: List[List[int]],
        channels_per_edge: int,
        last_pool: bool = False,
    ):
        super().__init__()

        self.edge_list = edge_list
        self.num_rows = len(edge_list) + 1  # + the global row
        self.channels_per_edge = channels_per_edge

        pooled_edges, new_edges, new_adj_list = self.compute_pooling(edge_list, last_pool)
        self.pooled_info = PoolingInfo(new_adj_list, new_edges, pooled_edges)

        # Block-diagonal averaging matrix grouping edges into pooled regions:
        # [pooled rows * channels, rows * channels]
        weight = torch.zeros(len(pooled_edges) * channels_per_edge, self.num_rows * channels_per_edge)
        idx = torch.arange(channels_per_edge)
        for i, group in enumerate(pooled_edges):
            for j in group:
                weight[i * channels_per_edge + idx, j * channels_per_edge + idx] = 1.0 / len(group)

        self.register_buffer("weight", weight, persistent=False)

    def compute_pooling(
        self,
        edge_list: List[List[int]],
        last_pool: bool = False,
    ) -> Tuple[List[List[int]], List[List[int]], List[List[int]]]:
        """
        Pool every chain of edges between branching joints and leaves: pairwise (keeping the
        chain's first edge alone if its length is odd), or as a whole for `last_pool`, which
        reduces the skeleton to its chains. The global row is always kept on its own.
        """
        degree = defaultdict(int)
        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            degree[int(u)] += 1
            degree[int(v)] += 1
            adj[int(u)].append((int(v), idx))

        chains = []
        visited = set()

        # DFS collecting the chains of edges.
        def dfs(node: int, chain: List[int]) -> None:
            if node in visited:
                return
            visited.add(node)

            # End the chain at a branching joint or a leaf.
            if degree[node] > 2 and node != 0:
                chains.append(chain)
                chain = []
            elif degree[node] == 1 and chain:
                chains.append(chain)
                return

            for child, edge in adj[node]:
                if edge not in {i for c in chains for i in c}:  # prevent reuse
                    dfs(child, chain + [edge])

        dfs(0, [])

        pooled_edges, new_edges = [], []
        for chain in chains:
            if last_pool:
                pooled_edges.append(chain)
                new_edges = edge_list
                continue

            if len(chain) % 2 == 1:
                pooled_edges.append([chain[0]])
                new_edges.append(edge_list[chain[0]])
                chain = chain[1:]

            for i in range(0, len(chain), 2):
                pooled_edges.append([chain[i], chain[i + 1]])
                new_edges.append([edge_list[chain[i]][0], edge_list[chain[i + 1]][1]])

        pooled_edges.append([self.num_rows - 1])

        return pooled_edges, new_edges, construct_adj(new_edges)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """ x: [B, rows * channels, T] -> [B, pooled rows * channels, T] """
        return self.weight @ x
