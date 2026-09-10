"""
Data adapters for different motion capture datasets.

This module provides adapters for loading and processing motion data from different sources.
"""

from typing import Type, Optional
from .base import BaseAdapter
from .amass import AMASSAdapter
from .bandai import BANDAIAdapter

__all__ = ['AMASSAdapter', 'BANDAIAdapter', 'AdapterRegistry', 'get_adapter_for_character']

class AdapterRegistry:   
    CHARACTER_ADAPTER_MAP = {
        BANDAIAdapter : ['character1', 'character2'],
        AMASSAdapter : ['Karim','Medhi']#,'Carine', 'Aude', ] 
    }
    
    @classmethod
    def get_adapter(cls, character_name: str, device: str):
        for adapter, chars in cls.CHARACTER_ADAPTER_MAP.items():
            if character_name in chars:
                return adapter(device)
    
    @classmethod
    def list_characters(cls, adapter: BaseAdapter):
        return cls.CHARACTER_ADAPTER_MAP[type(adapter)]

def get_adapter_for_character(character_name: str, device: str) -> BaseAdapter:
    return AdapterRegistry.get_adapter(character_name, device)

def list_characters(adapter: BaseAdapter) -> str:
    return AdapterRegistry.list_characters(adapter)