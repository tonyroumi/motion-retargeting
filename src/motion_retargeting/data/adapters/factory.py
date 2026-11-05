"""
Maps motion source names (as used in the dataset config) to adapter classes.
"""

from __future__ import annotations

from typing import Any, Dict, Type

from .amass import AMASSAdapter
from .bandai import BANDAIAdapter
from .base import MotionSourceAdapter
from .cmu import CMUAdapter

ADAPTERS: Dict[str, Type[MotionSourceAdapter]] = {
    "amass": AMASSAdapter,
    "bandai": BANDAIAdapter,
    "cmu": CMUAdapter,
}


def build_adapter(name: str, **kwargs: Any) -> MotionSourceAdapter:
    if name not in ADAPTERS:
        raise KeyError(f"Unknown motion source '{name}'. Available: {sorted(ADAPTERS)}")
    return ADAPTERS[name](**kwargs)
