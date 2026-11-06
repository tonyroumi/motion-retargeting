import torch
from torch import nn
import torch.nn.functional as F


class MaskedLinear(nn.Linear):
    """ Linear over `connectivity` [out_features, in_features] (bool), a persistent buffer. """

    def __init__(self, connectivity: torch.Tensor):
        out_features, in_features = connectivity.shape
        super().__init__(in_features, out_features)
        self.register_buffer("mask", connectivity.to(self.weight.dtype))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight * self.mask, self.bias)


class BodyPartEncoder(nn.Module):
    """
    PAN skeleton encoder: three masked linear layers over the flattened offsets of all
    joints, each followed by LeakyReLU(0.2), in per-part blocks. What each part reads is fixed by
    `input_connectivity`; see the official code's mask quirks in `methods/pan/masks.py`.
    """

    def __init__(self, num_parts: int, input_connectivity: torch.Tensor, part_connectivity: torch.Tensor):
        super().__init__()

        self.num_parts = num_parts
        self.linear1 = MaskedLinear(input_connectivity)
        self.linear2 = MaskedLinear(part_connectivity)
        self.linear3 = MaskedLinear(part_connectivity)
        self.activation = nn.LeakyReLU(negative_slope=0.2)

    def forward(self, offsets: torch.Tensor) -> torch.Tensor:
        """
        offsets: [K, J, 3], root included
        returns: [K, P, D]
        """
        x = self.activation(self.linear1(offsets.flatten(1)))
        x = self.activation(self.linear2(x))
        x = self.activation(self.linear3(x))
        return x.reshape(x.shape[0], self.num_parts, -1)
