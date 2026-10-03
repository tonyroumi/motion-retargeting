"""
Data adapters for different motion capture datasets.

This module provides adapters for loading and processing motion data from different sources.
"""

from .amass import AMASSAdapter
from .cmu import CMUAdapter
# from .bandai import BANDAIAdapter
from .base import MotionSourceAdapter

__all__ = [
    'AMASSAdapter',
    'CMUAdapter',
    'BANDAIAdapter',
    'MotionSourceAdapter'
]
