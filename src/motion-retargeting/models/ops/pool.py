from dataclasses import dataclass, astuple
from typing import List, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.utils import SkeletonUtils

@dataclass
class PoolingInfo:
    """Contains all information needed for pooling/unpooling operations"""
    adj_list: List[List[int]]
    edge_list: List[List[int]]
    pooled_edges: List[List[int]] = None
    
    def to_dict(self):
        return {
            'adj_list': self.adj_list,
            'edge_list': self.edge_list,
            'pooled_edges': self.pooled_edges
        }
    
    def __iter__(self):
        return iter(astuple(self))

class SkeletalPooling(nn.Module):
    """
    Skeletal pooling layer. 

    Pooling is applied to pairs of edges that connect to (joint) nodes of degree 2. 
    """
    def __init__(
        self, 
        edge_list: List[Tuple[int, int]],
        channels_per_edge: int,
        last_pool: bool = False
    ):
        super().__init__()
        self.edge_list = edge_list
        self.J = len(self.edge_list) + 1
        self.channels_per_edge = channels_per_edge

        # Precompute pooling regions and new adjacency
        pooling_list, new_edges, new_adj_list = self._compute_pooling(edge_list, last_pool)
        self.pooled_info = PoolingInfo(new_adj_list, new_edges, pooling_list)

        self._init_net()
    
    def _init_net(self):
        pooled_edges = self.pooled_info.pooled_edges
        rows = len(pooled_edges) * self.channels_per_edge
        cols = self.J * self.channels_per_edge

        # Pooling operation
        # Block-diagonal averaging matrix that groups edges into pooled regions:
        # weight: (region-to-pool-features, edge-features)
        weight = torch.zeros(rows, cols)
        for i, group in enumerate(pooled_edges):
            scale = 1.0 / len(group)
            for j in group:
                idx = torch.arange(self.channels_per_edge)
                weight[i * self.channels_per_edge + idx, j * self.channels_per_edge + idx] = scale

        self.weight = nn.Parameter(weight, requires_grad=False)

    def _compute_pooling(self, edge_list: List[List[int]], last_pool: bool = False):
        from collections import defaultdict

        # Build degree and adjacency
        degree = defaultdict(int)
        adj = defaultdict(list)
        for idx, (u, v) in enumerate(edge_list):
            degree[int(u)] += 1
            degree[int(v)] += 1
            adj[int(u)].append((int(v), idx))

        seq_list = []
        visited = set()

        # DFS to collect edge sequences
        def dfs(node, seq):
            if node in visited:
                return
            visited.add(node)

            # Stop sequence at branching or leaf
            if degree[node] > 2 and node != 0:
                seq_list.append(seq)
                seq = []
            elif degree[node] == 1 and seq:
                seq_list.append(seq)
                return

            for nxt, e_idx in adj[node]:
                if e_idx not in {i for s in seq_list for i in s}:  # prevent reuse
                    dfs(nxt, seq + [e_idx])

        dfs(0, [])

        pooling_list, new_edges = [], []
        for seq in seq_list:
            if last_pool:
                pooling_list.append(seq)
                new_edges = edge_list
                continue

            # Handle odd-length path
            if len(seq) % 2 == 1:
                pooling_list.append([seq[0]])
                new_edges.append(edge_list[seq[0]])
                seq = seq[1:]

            # Pairwise pooling
            for i in range(0, len(seq), 2):
                pooling_list.append([seq[i], seq[i + 1]])
                new_edges.append([edge_list[seq[i]][0], edge_list[seq[i + 1]][1]])

        # Add global position pooling
        pooling_list.append([self.J - 1])

        new_adj_list = SkeletonUtils.construct_adj(new_edges)

        return pooling_list, new_edges, new_adj_list
    
    def forward(self, x: torch.Tensor):
        return self.weight @ x
