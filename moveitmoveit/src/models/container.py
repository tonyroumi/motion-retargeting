from __future__ import annotations

from typing import Iterator

import torch
import torch.nn as nn

from abc import ABC, abstractmethod

class NetworkContainer(ABC):
    """Base class for algorithm-specific network containers."""

    @abstractmethod
    def to_device(self, device: str) -> None:
        """Move the network to the given device."""
        raise NotImplementedError

    @property
    def device(self) -> torch.device:
        return next(iter(self.parameters())).device

    @abstractmethod
    def parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def trainable_parameters(self) -> Iterator[nn.Parameter]:
        raise NotImplementedError

    @abstractmethod
    def state_dict(self) -> dict:
        raise NotImplementedError

    @abstractmethod
    def load_state_dict(self, state: dict) -> None:
        raise NotImplementedError

    @abstractmethod
    def eval(self) -> None:
        """Set the network to evaluation mode."""
        raise NotImplementedError

    @abstractmethod
    def train(self) -> None:
        """Set the network to training mode."""
        raise NotImplementedError

    @abstractmethod
    def update_normalizers(self) -> None:
        """Update the normalizers."""
        raise NotImplementedError