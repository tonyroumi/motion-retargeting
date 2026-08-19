from typing import List
import torch
import torch.nn as nn

class SkeletalUnpool(nn.Module):
    """
    Skeletal unpooling layer.
    """
    def __init__(
        self,
        pooled_edges: List[List[int]],
        channels_per_edge: int,
    ):
        super().__init__()
        self.pooled_edges = pooled_edges
        self.input_edge_num = len(pooled_edges)
        self.output_edge_num = sum(len(t) for t in pooled_edges)
        self.channels_per_edge = channels_per_edge

        self._init_net()

    def _init_net(self):
        rows = self.output_edge_num * self.channels_per_edge
        cols = self.input_edge_num * self.channels_per_edge
        weight = torch.zeros(rows, cols)

        for i, group in enumerate(self.pooled_edges):
            idx = torch.arange(self.channels_per_edge)
            for j in group:
                weight[j * self.channels_per_edge + idx, i * self.channels_per_edge + idx] = 1

        self.weight = nn.Parameter(weight, requires_grad=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.matmul(self.weight, x)