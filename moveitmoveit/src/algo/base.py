from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np
import torch
import torch.optim

from moveitmoveit.src.models import NetworkContainer
from moveitmoveit.src.types import BaseParams

from utils import Logger

class BaseAlgo(ABC):
    """ Abstract base class for on-policy algorithms. """

    def __init__(
        self,
        networks: NetworkContainer,
        params: BaseParams,
        logger: Logger,
    ):
        self.networks = networks
        self.params = params
        self.logger = logger

    def to_device(self, device: str):
        self.networks.to_device(device)

    @abstractmethod
    def init_storage(
        self,
        num_envs: int,
        num_transitions: int,
        obs_dim: int,
        action_dim: int,
    ) -> None:
        pass

    @abstractmethod
    def process_reset(self, infos: dict, env_ids: torch.Tensor = None) -> None:
        """Called after env.reset(). Override in subclasses for reset-specific logic."""
        pass

    @abstractmethod
    def act(self, observations: torch.Tensor, deterministic: bool = False) -> torch.Tensor:
        """Sample actions for the current observations. """
        pass

    @abstractmethod
    def process_env_step(
        self,
        rewards: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
        infos: dict | None = None,
    ) -> None:
        """Record reward, done flags, and optional step info, then flush the
        current transition into the rollout buffer. """
        pass

    @abstractmethod
    def compute_returns(self, last_values: torch.Tensor) -> None:
        """Estimate returns and advantages over the collected rollout."""
        pass

    @abstractmethod
    def get_value(self, observations: torch.Tensor) -> torch.Tensor:
        """Compute the value estimate for the given observations. """
        pass

    @abstractmethod
    def update(self, optimizer: torch.optim.Optimizer) -> None:
        """Run gradient updates using the stored rollout data."""
        self._update_count += 1
        self.networks.update_normalizers()

    def train(self) -> None:
        """Set the networks within the algorithm to training mode."""
        self.networks.train()

    def eval(self) -> None:
        """Set the networks within the algorithm to evaluation mode."""
        self.networks.eval()