"""
Mock ZeroMQ System - Simulação do fluxo real de dados

Este módulo fornece uma simulação completa do sistema ZeroMQ real, incluindo:
- ZeroMQPublisher: simula o PC Publisher real
- ZeroMQFormatter: converte dados mock para formato ZeroMQ correto  
- Geradores específicos por tópico para dados realistas
- MockZeroMQController: coordena todo o sistema mock
"""

from .zeroMQPublisher import zeroMQPublisher, ZeroMQPublisher
from .zeroMQFormatter import zeroMQFormatter, ZeroMQFormatter
from .mockZeroMQController import mockZeroMQController, MockZeroMQController
from .generators import (
    cardioWheelEcgGenerator, CardioWheelEcgGenerator,
    cardioWheelAccGenerator, CardioWheelAccGenerator,
    cardioWheelGyrGenerator, CardioWheelGyrGenerator,
    polarPpiGenerator, PolarPpiGenerator,
    brainAccessEegGenerator, BrainAccessEegGenerator,
    cameraFaceLandmarksGenerator, CameraFaceLandmarksGenerator
)

__all__ = [
    "mockZeroMQController", "MockZeroMQController",
    "zeroMQPublisher", "ZeroMQPublisher",
    "zeroMQFormatter", "ZeroMQFormatter",
    "cardioWheelEcgGenerator", "CardioWheelEcgGenerator",
    "cardioWheelAccGenerator", "CardioWheelAccGenerator",
    "cardioWheelGyrGenerator", "CardioWheelGyrGenerator",
    "polarPpiGenerator", "PolarPpiGenerator",
    "brainAccessEegGenerator", "BrainAccessEegGenerator",
    "cameraFaceLandmarksGenerator", "CameraFaceLandmarksGenerator"
]