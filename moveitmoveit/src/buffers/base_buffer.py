from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generator

import torch

class BaseBuffer(ABC):
    """Abstract base class for all experience buffers.
    """

    def __init__(self, capacity: int, device: torch.device) -> None:
        self._capacity = capacity
        self.device = device
        self.step = 0

    @property
    def capacity(self) -> int:
        """Maximum number of entries the buffer can hold."""
        return self._capacity

    @property
    def size(self) -> int:
        """Number of valid entries currently stored."""
        return self.step

    @property
    def full(self) -> bool:
        return self.step >= self._capacity

    @abstractmethod
    def add(self, data: Any) -> None:
        """Insert one timestep / transition into the buffer."""

    @abstractmethod
    def clear(self) -> None:
        """Reset the buffer so it can be filled again."""

    @abstractmethod
    def mini_batch_generator(
        self,
        num_mini_batches: int,
        num_epochs: int = 1,
    ) -> Generator:
        """Yield mini-batches for gradient-based optimisation."""