"""
Training helpers: moving batches to a device and iterating several DataLoaders in lockstep.
"""

from typing import Any, Dict, Iterator

import torch
from torch.utils.data import DataLoader


def move_to_device(item: Any, device: torch.device) -> Any:
    """
    Recursively move tensors to `device`.

    Supports common DataLoader batch structures:
        Tensor
        dict
        list
        tuple
    """

    if isinstance(item, torch.Tensor):
        return item.to(
            device,
            non_blocking=True,
        )

    if isinstance(item, dict):
        return {
            key: move_to_device(value, device)
            for key, value in item.items()
        }

    if isinstance(item, list):
        return [
            move_to_device(value, device)
            for value in item
        ]

    if isinstance(item, tuple):
        return tuple(
            move_to_device(value, device)
            for value in item
        )

    return item


class ZippedLoaders:
    """
    Wraps {name: DataLoader} and yields {name: batch} dicts. Like `zip`, an
    epoch ends when the shortest loader is exhausted.
    """

    def __init__(self, loaders: Dict[str, DataLoader]):
        self.loaders = loaders

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        names = list(self.loaders)
        for batches in zip(*self.loaders.values()):
            yield dict(zip(names, batches))

    def __len__(self) -> int:
        return min(len(loader) for loader in self.loaders.values())
