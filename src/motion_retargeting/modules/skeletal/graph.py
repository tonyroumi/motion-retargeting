"""
Skeleton graph helpers for skeleton-aware layers: edge lists and edge adjacency.
"""

import math
from typing import List, Sequence


def build_edge_list(parents: Sequence[int]) -> List[List[int]]:
    """
    One (parent, child) edge per non-root joint, in child order, as the official SAN code
    builds it. Node ids are joint ids and the root is node 0, so E edges plus the global
    row make J graph rows.
    """
    return [[int(parents[joint]), joint] for joint in range(1, len(parents))]


def edge_distance_matrix(edge_list: Sequence[Sequence[int]]) -> List[List[float]]:
    """ (E, E) matrix of shortest edge-to-edge distances; edges sharing a joint are at distance 1. """
    num_edges = len(edge_list)

    dist = [[math.inf] * num_edges for _ in range(num_edges)]
    for i in range(num_edges):
        dist[i][i] = 0

    for i, (a0, a1) in enumerate(edge_list):
        for j, (b0, b1) in enumerate(edge_list):
            if i != j and (a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1):
                dist[i][j] = 1

    # Floyd-Warshall (E is small for skeletons)
    for k in range(num_edges):
        for i in range(num_edges):
            for j in range(num_edges):
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]

    return dist


def construct_adj(
    edge_list: Sequence[Sequence[int]],
    d: int = 2,
    global_edges: bool = True,
) -> List[List[int]]:
    """
    Adjacency list of the edges within distance <= d of each edge. With `global_edges`, a
    final row is appended for the global motion features, connected to the neighbors of edge 0.
    """
    dist = edge_distance_matrix(edge_list)
    num_edges = len(edge_list)

    neighbors = [
        [j for j in range(num_edges) if dist[i][j] <= d]
        for i in range(num_edges)
    ]

    if global_edges:
        global_neighbors = neighbors[0].copy()
        neighbors.append(global_neighbors)
        for i in global_neighbors:
            neighbors[i].append(num_edges)

    return neighbors
