"""
Mock Data Generators - Geradores específicos por tópico ZeroMQ

Cada gerador produz dados realistas para um tópico específico:
- CardioWheelEcgGenerator: dados ECG com conversão ADC realista
- CardioWheelAccGenerator: dados acelerómetro 3-axis com padrões de condução  
- CardioWheelGyrGenerator: dados giroscópio 3-axis com rotações realistas
- PolarPpiGenerator: eventos PPI do Polar ARM Band com fisiologia

Todos os geradores seguem as configurações centralizadas e podem
injetar anomalias baseadas nos thresholds configurados.
"""

from .cardioWheelEcgGenerator import cardioWheelEcgGenerator, CardioWheelEcgGenerator
from .cardioWheelAccGenerator import cardioWheelAccGenerator, CardioWheelAccGenerator  
from .cardioWheelGyrGenerator import cardioWheelGyrGenerator, CardioWheelGyrGenerator
from .polarPpiGenerator import polarPpiGenerator, PolarPpiGenerator
from .brainAccessEegGenerator import brainAccessEegGenerator, BrainAccessEegGenerator
from .cameraFaceLandMarksGenerator import cameraFaceLandmarksGenerator, CameraFaceLandmarksGenerator
from .unityAlcoholGenerator import unityAlcoholGenerator, UnityAlcoholGenerator
from .unityCarInfoGenerator import unityCarInfoGenerator, UnityCarInfoGenerator

__all__ = [
    "cardioWheelEcgGenerator", "CardioWheelEcgGenerator",
    "cardioWheelAccGenerator", "CardioWheelAccGenerator", 
    "cardioWheelGyrGenerator", "CardioWheelGyrGenerator",
    "polarPpiGenerator", "PolarPpiGenerator",
    "brainAccessEegGenerator", "BrainAccessEegGenerator",
    "cameraFaceLandmarksGenerator", "CameraFaceLandmarksGenerator",
    "unityAlcoholGenerator", "UnityAlcoholGenerator",
    "unityCarInfoGenerator", "UnityCarInfoGenerator"
]