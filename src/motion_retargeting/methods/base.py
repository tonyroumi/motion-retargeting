from __future__ import annotations

from abc import ABC, abstractmethod


class RetargetingMethod(ABC):
    """
    Base class for all motion retargeting methods.
    """

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
