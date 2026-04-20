"""
Signal implementations
"""

from .cardiacSignal import CardiacSignal
from .eegSignal import EEGSignal
from .sensorSignal import SensorsSignal
from .cameraSignal import CameraSignal

__all__ = ["CardiacSignal", "EEGSignal", "SensorsSignal", "CameraSignal"]