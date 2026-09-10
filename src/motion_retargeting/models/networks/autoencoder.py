"""
Skeletal AutoEncoder to learn how to reconstruct original motions and a primal skeleton in latent space.
"""

from .encoder import SkeletalEncoder
from .decoder import SkeletalDecoder

from omegaconf import DictConfig
from typing import Dict, List, Optional, Any
import torch
import torch.nn as nn
import torch.nn.functional as F

class SkeletalAutoEncoder(nn.Module):
    def __init__(
        self, 
        adj_init: List[List[int]], 
        edge_init: List[List[int]], 
        params: DictConfig
    ):
        super().__init__()
        self.encoder = SkeletalEncoder(adj_init, edge_init, params=params["encoder"])
        self.decoder = SkeletalDecoder(self.encoder.pooling_hierarchy, 
                                       params=params["decoder"])
    
    def forward(self, x: torch.Tensor, offset: torch.Tensor):
        latent = self.encoder(x, offset, pad_global=True)
        result = self.decoder(latent, offset)
        return latent, result
