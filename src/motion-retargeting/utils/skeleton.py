from typing import Tuple, Union, List
import math
import numpy as np
import torch

from src.utils import ArrayLike, ArrayUtils
from src.core.types import SkeletonTopology

class SkeletonUtils:
    """ Skeleton utiities """

    @staticmethod
    def prune_joints(data: ArrayLike, cutoff: int, discard_root: int = False) -> ArrayLike:
        """ Prune joints based on an index cutoff. """
        if ArrayUtils.is_tensor(data):
            idx = torch.arange(cutoff, device=data.device)
        else:
            idx = np.arange(cutoff)

        idx = idx[discard_root:]
        # Joint-major: (J, ...)
        if data.ndim == 2:
            return data[idx]

        # Batch-major: (N, J, ...)
        elif data.ndim == 3:
            return data[:, idx]
    
    @staticmethod
    def construct_edge_topology(parent_kintree: List[int]) -> List[Tuple[int]]:
        """ Constructs the edge topology for a skeleton as (parent, child) joint tuples. """
        topology = []
        for child_idx in range(1, len(parent_kintree)):  # Skip root
            parent_idx = parent_kintree[child_idx]
            topology.append([int(parent_idx), int(child_idx)])
        
        return topology
    
    @staticmethod
    def find_ee(parent_kintree: List[int]):
        """ Finds the skeleton's end effectors by traversing the tree for leaf joints. """
        children = {i: [] for i in range(len(parent_kintree))}
        for j, p in enumerate(parent_kintree):
            if p >= 0:
                children[p].append(j)

        leaves = [j for j, c in children.items() if len(c) == 0]
        return leaves
    
    @staticmethod
    def compute_height(parent_kintree: List[int], offsets: List[List[int]], ee_ids: List[int]) -> float:
        """ Computes the height by summing the size of each offset vector from the head to the feet. """
        def get_chain_height(joint_idx):
            """Calculate cumulative offset magnitude from root to joint"""
            height = 0.0
            current = joint_idx
            while current > 0:
                parent = parent_kintree[current]
                # Use the offset at current joint (which is offset from parent)
                offset_idx = current - 1  # Since offsets array excludes root
                offset_magnitude = np.linalg.norm(offsets[offset_idx])
                height += offset_magnitude
                current = parent
            return height
        
        # Find maximum height among all end effectors
        heights = [get_chain_height(ee_id) for ee_id in ee_ids]
        return np.max(heights)

    @staticmethod
    def construct_adj(
        edge_list: ArrayLike,
        d: int = 2,
        global_edges: bool = True
    ):
        """
        Returns adjacency list of edges within distance <= d
        """
        def add_global_edge(neighbors):
            E = len(neighbors)
            global_idx = E

            
            global_neighbors = neighbors[0].copy()

            neighbors.append(global_neighbors)
            for i in global_neighbors:
                neighbors[i].append(global_idx)

            return neighbors

        dist = SkeletonUtils.calc_edge_distance_matrix(edge_list)
        E = len(edge_list)

        neighbors = []
        for i in range(E):
            nbrs = [j for j in range(E) if dist[i][j] <= d]
            neighbors.append(nbrs)
        
        if global_edges:
            neighbors = add_global_edge(neighbors)

        return neighbors
    
    @staticmethod
    def calc_edge_distance_matrix(edges):
        """
        edges: List[Tuple[int, int]]
        returns: (E, E) matrix of shortest edge–edge distances
        """
        E = len(edges)
        INF = math.inf

        dist = [[INF] * E for _ in range(E)]
        for i in range(E):
            dist[i][i] = 0

        # distance 1 if edges share a joint
        for i, (a0, a1) in enumerate(edges):
            for j, (b0, b1) in enumerate(edges):
                if i == j:
                    continue
                if a0 == b0 or a0 == b1 or a1 == b0 or a1 == b1:
                    dist[i][j] = 1

        # Floyd–Warshall (E is small for skeletons)
        for k in range(E):
            for i in range(E):
                for j in range(E):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]

        return dist

    @staticmethod
    def get_ee_velocity(
        positions: ArrayLike,
        topology: Union[SkeletonTopology] 
    ) -> Tuple[ArrayLike, ArrayLike]:
        """ Compute the velocity of end effectors """
        vel = positions[:, 1:] - positions[:, :-1]  
        ee_vel = vel[:, :, topology.ee_ids, :]  
        return ee_vel