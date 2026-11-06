"""
Data adapters for different motion capture datasets.

This module provides adapters for loading and processing motion data from different sources.
"""

from .amass import AMASSAdapter
from .cmu import CMUAdapter
from .bandai import BANDAIAdapter
from .base import MotionSourceAdapter
from .lafan1 import DogAdapter, LAFAN1Adapter
from .mixamo import MixamoAdapter, MixamoGroupBAdapter

__all__ = [
    'AMASSAdapter',
    'CMUAdapter',
    'BANDAIAdapter',
    'MixamoAdapter',
    'MixamoGroupBAdapter',
    'LAFAN1Adapter',
    'DogAdapter',
    'MotionSourceAdapter'
]
