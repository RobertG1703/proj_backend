"""
Models - Classes de dados e sinais do Control Room
"""

from .base import BaseSignal, SignalMetrics
from .dataPoint import SignalPoint, DataBuffer
from .signals import CardiacSignal , EEGSignal
# Falta depois adicionar os outros sinais 

__all__ = [
    "BaseSignal",
    "SignalMetrics", 
    "SignalPoint",
    "DataBuffer",
    "CardiacSignal",
    "EEGSignal"
]