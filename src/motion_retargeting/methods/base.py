from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path


class RetargetingMethod(ABC):
    """
    Base class for all motion retargeting methods.
    """

    # Where a method may save diagnostic captures of a training step; the Trainer sets it per run.
    capture_dir: Path | None = None

    def __init__(self) -> None:
        super().__init__()

    @abstractmethod
    def train(self):
        pass

    @abstractmethod
    def eval(self):
        pass

    @abstractmethod
    def train_step(self, batch):
        pass

    @abstractmethod
    def validation_step(self, batch):
        pass

    def visualize(self, batch, save_dir):
        """ Optionally save visualizations of `batch` into `save_dir`; no-op by default. """
        pass

    @abstractmethod
    def state_dict(self):
        pass

    @abstractmethod
    def load_state_dict(self, state_dict):
        pass
