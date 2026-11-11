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


class PrimaryRecycledLoaders:
    """Primary-driven full-batch epochs, reshuffling secondary loaders on exhaustion.

    Secondary iterators survive epoch boundaries, as in reference human/dog
    training. Never cache/replay batches with itertools.cycle. This intentionally
    does not reproduce the upstream extra-primary-batch epoch-boundary quirk.
    """

    def __init__(self, loaders: Dict[str, DataLoader], primary: str):
        if primary not in loaders:
            raise ValueError(f"Unknown primary domain {primary!r}")
        if any(not loader.drop_last or len(loader) == 0 for loader in loaders.values()):
            raise ValueError("Primary recycling requires nonempty drop_last loaders; reduce batch_size if needed")
        if len({loader.batch_size for loader in loaders.values()}) != 1:
            raise ValueError("Primary recycling requires equal batch sizes")
        self.loaders, self.primary = loaders, primary
        self._secondary_iters = {}

    def __iter__(self) -> Iterator[Dict[str, Any]]:
        for primary_batch in self.loaders[self.primary]:
            result = {self.primary: primary_batch}
            for name, loader in self.loaders.items():
                if name == self.primary:
                    continue
                if name not in self._secondary_iters:
                    self._secondary_iters[name] = iter(loader)
                try:
                    result[name] = next(self._secondary_iters[name])
                except StopIteration:
                    self._secondary_iters[name] = iter(loader)
                    result[name] = next(self._secondary_iters[name])
            yield result

    def __len__(self) -> int:
        return len(self.loaders[self.primary])
