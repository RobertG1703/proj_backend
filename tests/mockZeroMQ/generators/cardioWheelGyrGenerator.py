"""
CardioWheelGyrGenerator - Gerador de dados de giroscópio realistas

Resumo:
Gera dados de giroscópio simulando o CardioWheel real com:
- Valores ADC 16-bit realistas com baselines zero (X~0, Y~0, Z~0)
- Chunks de 10 samples (100ms @ 100Hz real)
- Simulação de rotações de condução: curvas, volante, instabilidade
- Anomalias específicas: rotação rápida, spin, instabilidade, deriva
- Coordenação com padrões de condução do acelerómetro
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from app.core import settings

class GyrAnomalyType(Enum):
    """Tipos de anomalias de giroscópio"""
    NORMAL = "normal"
    RAPID_ROTATION = "rapid_rotation"         # Rotação rápida
    SPIN_SLIP = "spin_slip"                   # Derrapagem
    INSTABILITY = "instability"               # Instabilidade do veículo
    EXCESSIVE_YAW = "excessive_yaw"           # Yaw excessivo
    SENSOR_DRIFT = "sensor_drift"             # Deriva do sensor
    HIGH_FREQUENCY_NOISE = "high_freq_noise"  # Ruído alta frequência
    SENSOR_STUCK = "sensor_stuck"             # Sensor travado
    CALIBRATION_ERROR = "calibration_error"   # Erro de calibração

class TurningPattern(Enum):
    """Padrões de rotação simulados"""
    STRAIGHT = "straight"                     # Em linha reta
    GENTLE_LEFT = "gentle_left"               # Curva suave esquerda
    GENTLE_RIGHT = "gentle_right"             # Curva suave direita
    SHARP_LEFT = "sharp_left"                 # Curva apertada esquerda
    SHARP_RIGHT = "sharp_right"               # Curva apertada direita
    LANE_CHANGE_LEFT = "lane_change_left"     # Mudança faixa esquerda
    LANE_CHANGE_RIGHT = "lane_change_right"   # Mudança faixa direita
    ROUNDABOUT = "roundabout"                 # Rotunda
    PARKING_MANEUVER = "parking_maneuver"     # Manobra estacionamento
    CORRECTIVE_STEERING = "corrective_steering"  # Correção direção

class CardioWheelGyrGenerator:
    """Gerador de dados de giroscópio para tópico CardioWheel_GYR"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações GYR do settings
        self.gyrConfig = settings.signals.sensorsConfig["gyroscope"]
        self.zmqConfig = settings.zeromq.topicProcessingConfig["CardioWheel_GYR"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração GYR
        self.samplingRate = self.zmqConfig["samplingRate"]              # 100Hz
        self.chunkSize = self.mockConfig.topicChunkSizes["CardioWheel_GYR"]  # 10 samples
        self.chunkDuration = self.chunkSize / self.samplingRate         # 0.1s (100ms)
        
        # Configurações ADC e baselines (giroscópio tem baseline zero)
        self.generatorConfig = self.mockConfig.generatorBaseConfig["gyroscope"]
        self.baselineX = self.generatorConfig["baselineX"]             # 0 ADC
        self.baselineY = self.generatorConfig["baselineY"]             # 0 ADC
        self.baselineZ = self.generatorConfig["baselineZ"]             # 0 ADC
        self.noiseStd = self.generatorConfig["noiseStd"]               # 2 ADC units
        
        # Thresholds de anomalias
        self.rapidRotationThreshold = self.gyrConfig["rapidRotationThreshold"]          # 500 °/s
        self.instabilityThreshold = self.gyrConfig["instabilityThreshold"]              # 100 °/s
        self.spinThreshold = self.gyrConfig["spinThreshold"]                            # 1000 °/s
        self.angularMagnitudeThreshold = self.gyrConfig["angularMagnitudeThreshold"]    # 800 °/s
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["CardioWheel_GYR"]  # 4%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = GyrAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado de rotação simulada
        self.currentTurningPattern = TurningPattern.STRAIGHT
        self.patternStartTime = 0.0
        self.patternDuration = 0.0
        self.angularVelocity = np.array([0.0, 0.0, 0.0])  # Velocidade angular atual °/s
        
        # Parâmetros de rotação
        self.maxYawRate = 120.0           # °/s yaw máximo normal
        self.maxPitchRate = 30.0          # °/s pitch máximo normal
        self.maxRollRate = 40.0           # °/s roll máximo normal
        self.steeringSmoothing = 0.85     # Factor de suavização
        
        # Deriva do sensor (comum em giroscópios)
        self.driftRate = np.array([0.0, 0.0, 0.0])        # °/s deriva acumulada
        self.driftAccumulation = np.array([0.0, 0.0, 0.0]) # Deriva total
        
        # Oscilações de alta frequência (vibração)
        self.vibrationFrequency = 25.0    # Hz frequência de vibração
        self.vibrationPhase = 0.0
        
        self.logger.info(f"CardioWheelGyrGenerator initialized - {self.samplingRate}Hz, chunks of {self.chunkSize}")
    
    def generateChunk(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um chunk de dados de giroscópio (10 samples).
        
        Args:
            baseTimestamp: Timestamp base para o chunk (usa interno se None)
            
        Returns:
            Dict com arrays de dados GYR x,y,z para formatação
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Atualizar padrão de rotação
            self._updateTurningPattern()
            
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Gerar samples GYR para o chunk
            xSamples = []
            ySamples = []
            zSamples = []
            
            for i in range(self.chunkSize):
                # Gerar sample GYR baseado no padrão e anomalia atual
                gyrX, gyrY, gyrZ = self._generateGyrSample(i)
                
                xSamples.append(gyrX)  # ADC values são inteiros
                ySamples.append(gyrY)
                zSamples.append(gyrZ)
                
                # Avançar contadores
                self.sampleCounter += 1
                self.vibrationPhase += (2 * np.pi * self.vibrationFrequency) / self.samplingRate
                
                if self.vibrationPhase >= 2 * np.pi:
                    self.vibrationPhase -= 2 * np.pi
                
                # Atualizar deriva gradual
                self._updateDrift()
            
            # Avançar timestamp para próximo chunk
            self.currentTimestamp += self.chunkDuration
            
            # Calcular magnitude angular para debug
            angularMagnitudes = []
            for i in range(self.chunkSize):
                # Converter para °/s para cálculo de magnitude
                physX = (xSamples[i] - self.baselineX) * self.gyrConfig["conversionFactor"]
                physY = (ySamples[i] - self.baselineY) * self.gyrConfig["conversionFactor"]
                physZ = (zSamples[i] - self.baselineZ) * self.gyrConfig["conversionFactor"]
                magnitude = (physX**2 + physY**2 + physZ**2)**0.5
                angularMagnitudes.append(round(magnitude, 1))
            
            result = {
                "x": xSamples,
                "y": ySamples,
                "z": zSamples,
                "chunkTimestamp": self.currentTimestamp - self.chunkDuration,
                "anomalyType": self.currentAnomalyType.value,
                "turningPattern": self.currentTurningPattern.value,
                "angularMagnitudes": angularMagnitudes,  # Para debug
                "samplingRate": self.samplingRate,
                "chunkSize": self.chunkSize
            }
            
            self.logger.debug(f"Generated GYR chunk: {len(xSamples)} samples, pattern: {self.currentTurningPattern.value}, anomaly: {self.currentAnomalyType.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating GYR chunk: {e}")
            raise
    
    def _generateGyrSample(self, sampleIndex: int) -> Tuple[float, float, float]:
        """
        Gera um sample de giroscópio individual.
        
        Args:
            sampleIndex: Índice do sample no chunk (0-9)
            
        Returns:
            Tuplo (gyrX, gyrY, gyrZ) em ADC units
        """
        
        # Obter velocidade angular baseada no padrão de rotação
        physGyrX, physGyrY, physGyrZ = self._getTurningAngularVelocity()
        
        # Aplicar anomalias se ativas
        if self.currentAnomalyType != GyrAnomalyType.NORMAL:
            physGyrX, physGyrY, physGyrZ = self._applyAnomalies(physGyrX, physGyrY, physGyrZ, sampleIndex)
        
        # Adicionar vibração de alta frequência (sempre presente)
        vibration = self._getAngularVibration()
        physGyrX += vibration[0]
        physGyrY += vibration[1]
        physGyrZ += vibration[2]
        
        # Adicionar deriva do sensor
        physGyrX += self.driftAccumulation[0]
        physGyrY += self.driftAccumulation[1]
        physGyrZ += self.driftAccumulation[2]
        
        # Adicionar ruído gaussiano
        noise = np.random.normal(0, self.noiseStd, 3)
        
        # Converter para ADC units e adicionar baselines
        adcX = self.baselineX + (physGyrX / self.gyrConfig["conversionFactor"]) + noise[0]
        adcY = self.baselineY + (physGyrY / self.gyrConfig["conversionFactor"]) + noise[1]
        adcZ = self.baselineZ + (physGyrZ / self.gyrConfig["conversionFactor"]) + noise[2]
        
        # Clipar para range ADC 16-bit
        adcX = np.clip(adcX, -32768, 32767)
        adcY = np.clip(adcY, -32768, 32767)
        adcZ = np.clip(adcZ, -32768, 32767)
        
        return adcX, adcY, adcZ
    
    def _getTurningAngularVelocity(self) -> Tuple[float, float, float]:
        """
        Calcula velocidade angular baseada no padrão de rotação atual.
        
        Returns:
            Tuplo (gyrX, gyrY, gyrZ) em °/s
            X = Roll, Y = Pitch, Z = Yaw
        """
        
        if self.currentTurningPattern == TurningPattern.STRAIGHT:
            # Em linha reta - apenas pequenas correções
            gyrX = np.random.normal(0, 2.0)   # Roll mínimo
            gyrY = np.random.normal(0, 1.0)   # Pitch mínimo
            gyrZ = np.random.normal(0, 3.0)   # Yaw correções pequenas
            
        elif self.currentTurningPattern == TurningPattern.GENTLE_LEFT:
            # Curva suave à esquerda
            gyrX = np.random.normal(-5, 3)    # Roll ligeiro para dentro
            gyrY = np.random.normal(0, 2)     # Pitch neutro
            gyrZ = np.random.uniform(-40, -15) # Yaw negativo (esquerda)
            
        elif self.currentTurningPattern == TurningPattern.GENTLE_RIGHT:
            # Curva suave à direita
            gyrX = np.random.normal(5, 3)     # Roll ligeiro para dentro
            gyrY = np.random.normal(0, 2)     # Pitch neutro
            gyrZ = np.random.uniform(15, 40)  # Yaw positivo (direita)
            
        elif self.currentTurningPattern == TurningPattern.SHARP_LEFT:
            # Curva apertada à esquerda
            gyrX = np.random.normal(-15, 5)   # Roll mais acentuado
            gyrY = np.random.normal(-3, 3)    # Pitch ligeiro
            gyrZ = np.random.uniform(-80, -40) # Yaw forte (esquerda)
            
        elif self.currentTurningPattern == TurningPattern.SHARP_RIGHT:
            # Curva apertada à direita
            gyrX = np.random.normal(15, 5)    # Roll mais acentuado
            gyrY = np.random.normal(-3, 3)    # Pitch ligeiro
            gyrZ = np.random.uniform(40, 80)  # Yaw forte (direita)
            
        elif self.currentTurningPattern == TurningPattern.LANE_CHANGE_LEFT:
            # Mudança de faixa à esquerda - movimento rápido mas suave
            progress = (self.currentTimestamp - self.patternStartTime) / self.patternDuration
            if progress < 0.3:
                gyrZ = np.random.uniform(-60, -30)  # Início rotação
            elif progress < 0.7:
                gyrZ = np.random.uniform(30, 60)    # Correção oposta
            else:
                gyrZ = np.random.normal(0, 5)       # Estabilização
            gyrX = np.random.normal(-8, 4)
            gyrY = np.random.normal(0, 2)
            
        elif self.currentTurningPattern == TurningPattern.LANE_CHANGE_RIGHT:
            # Mudança de faixa à direita
            progress = (self.currentTimestamp - self.patternStartTime) / self.patternDuration
            if progress < 0.3:
                gyrZ = np.random.uniform(30, 60)    # Início rotação
            elif progress < 0.7:
                gyrZ = np.random.uniform(-60, -30)  # Correção oposta
            else:
                gyrZ = np.random.normal(0, 5)       # Estabilização
            gyrX = np.random.normal(8, 4)
            gyrY = np.random.normal(0, 2)
            
        elif self.currentTurningPattern == TurningPattern.ROUNDABOUT:
            # Rotunda - rotação contínua
            gyrX = np.random.normal(10, 5)     # Roll constante
            gyrY = np.random.normal(-2, 3)     # Pitch ligeiro
            gyrZ = np.random.uniform(30, 70)   # Yaw contínuo (direita)
            
        elif self.currentTurningPattern == TurningPattern.PARKING_MANEUVER:
            # Manobra de estacionamento - rotações lentas mas pronunciadas
            gyrX = np.random.normal(0, 8)      # Roll variável
            gyrY = np.random.normal(0, 5)      # Pitch variável
            gyrZ = np.random.uniform(-90, 90)  # Yaw amplo (ambas direções)
            
        elif self.currentTurningPattern == TurningPattern.CORRECTIVE_STEERING:
            # Correções rápidas do volante
            direction = np.random.choice([-1, 1])
            gyrX = np.random.normal(direction * 12, 6)
            gyrY = np.random.normal(0, 4)
            gyrZ = np.random.uniform(direction * 50, direction * 20)
            
        else:
            gyrX = gyrY = gyrZ = 0.0
        
        # Suavizar mudanças bruscas
        alpha = self.steeringSmoothing
        self.angularVelocity = alpha * self.angularVelocity + (1 - alpha) * np.array([gyrX, gyrY, gyrZ])
        
        return self.angularVelocity[0], self.angularVelocity[1], self.angularVelocity[2]
    
    def _getAngularVibration(self) -> np.ndarray:
        """
        Simula vibração angular de alta frequência.
        
        Returns:
            Array com vibração em °/s para [X, Y, Z]
        """
        
        # Vibração de alta frequência (motor, estrada)
        amplitude = 3.0  # °/s
        
        vibX = amplitude * 0.8 * np.sin(self.vibrationPhase * 1.1)
        vibY = amplitude * 0.6 * np.sin(self.vibrationPhase * 0.9)
        vibZ = amplitude * 1.0 * np.sin(self.vibrationPhase * 1.2)
        
        return np.array([vibX, vibY, vibZ])
    
    def _updateDrift(self):
        """
        Atualiza deriva gradual do sensor (comum em giroscópios).
        """
        
        # Deriva muito lenta (typical para giroscópios MEMS)
        driftRate = 0.001  # °/s por sample
        
        self.driftAccumulation += np.random.normal(0, driftRate, 3)
        
        # Limitar deriva total
        self.driftAccumulation = np.clip(self.driftAccumulation, -10.0, 10.0)
    
    def _applyAnomalies(self, gyrX: float, gyrY: float, gyrZ: float, sampleIndex: int) -> Tuple[float, float, float]:
        """
        Aplica anomalias específicas à velocidade angular.
        
        Args:
            gyrX, gyrY, gyrZ: Velocidade angular base em °/s
            sampleIndex: Índice do sample no chunk
            
        Returns:
            Velocidade angular modificada pela anomalia
        """
        
        if self.currentAnomalyType == GyrAnomalyType.RAPID_ROTATION:
            # Rotação rápida - valores elevados
            magnitude = np.random.uniform(300, 600)
            direction = np.random.choice([-1, 1], 3)
            gyrX += magnitude * 0.3 * direction[0]  # Roll moderado
            gyrY += magnitude * 0.2 * direction[1]  # Pitch ligeiro
            gyrZ += magnitude * 1.0 * direction[2]  # Yaw principal
            
        elif self.currentAnomalyType == GyrAnomalyType.SPIN_SLIP:
            # Spin/derrapagem - rotação extrema principalmente em Z
            if sampleIndex < 5:  # Primeira metade do chunk
                gyrX += np.random.uniform(-200, 200)
                gyrY += np.random.uniform(-100, 100)
                gyrZ += np.random.uniform(-1200, 1200)  # Spin extremo
                
        elif self.currentAnomalyType == GyrAnomalyType.INSTABILITY:
            # Instabilidade - variação alta e irregular
            instabilityMagnitude = np.random.uniform(80, 150)
            gyrX += np.random.uniform(-instabilityMagnitude, instabilityMagnitude)
            gyrY += np.random.uniform(-instabilityMagnitude*0.6, instabilityMagnitude*0.6)
            gyrZ += np.random.uniform(-instabilityMagnitude*1.2, instabilityMagnitude*1.2)
            
        elif self.currentAnomalyType == GyrAnomalyType.EXCESSIVE_YAW:
            # Yaw excessivo - problema específico no eixo Z
            excessiveYaw = np.random.uniform(-400, 400)
            gyrZ += excessiveYaw
            
        elif self.currentAnomalyType == GyrAnomalyType.SENSOR_DRIFT:
            # Deriva acelerada do sensor
            driftAcceleration = np.random.normal(0, 5, 3)
            self.driftAccumulation += driftAcceleration
            
        elif self.currentAnomalyType == GyrAnomalyType.HIGH_FREQUENCY_NOISE:
            # Ruído de alta frequência
            noiseAmplitude = np.random.uniform(20, 50)
            highFreqNoise = np.random.normal(0, noiseAmplitude, 3)
            gyrX += highFreqNoise[0]
            gyrY += highFreqNoise[1]
            gyrZ += highFreqNoise[2]
            
        elif self.currentAnomalyType == GyrAnomalyType.SENSOR_STUCK:
            # Sensor travado - valores constantes
            gyrX = gyrY = gyrZ = 0  # Override para zero
            
        elif self.currentAnomalyType == GyrAnomalyType.CALIBRATION_ERROR:
            # Erro de calibração - offset constante
            calibrationOffset = np.array([50, 30, 80])  # °/s offset
            gyrX += calibrationOffset[0]
            gyrY += calibrationOffset[1]
            gyrZ += calibrationOffset[2]
        
        return gyrX, gyrY, gyrZ
    
    def _updateTurningPattern(self):
        """
        Atualiza padrão de rotação baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Verificar se deve mudar padrão
        if currentTime - self.patternStartTime >= self.patternDuration:
            # Escolher novo padrão baseado em probabilidades
            patterns = [
                TurningPattern.STRAIGHT,             # 40% - Mais comum
                TurningPattern.GENTLE_LEFT,          # 12%
                TurningPattern.GENTLE_RIGHT,         # 12%
                TurningPattern.SHARP_LEFT,           # 8%
                TurningPattern.SHARP_RIGHT,          # 8%
                TurningPattern.LANE_CHANGE_LEFT,     # 5%
                TurningPattern.LANE_CHANGE_RIGHT,    # 5%
                TurningPattern.CORRECTIVE_STEERING,  # 5%
                TurningPattern.ROUNDABOUT,           # 3%
                TurningPattern.PARKING_MANEUVER      # 2%
            ]
            
            weights = [0.40, 0.12, 0.12, 0.08, 0.08, 0.05, 0.05, 0.05, 0.03, 0.02]
            self.currentTurningPattern = np.random.choice(patterns, p=weights)
            
            self.patternStartTime = currentTime
            
            # Duração do padrão baseada no tipo
            if self.currentTurningPattern == TurningPattern.STRAIGHT:
                self.patternDuration = np.random.uniform(15.0, 45.0)
            elif self.currentTurningPattern in [TurningPattern.LANE_CHANGE_LEFT, TurningPattern.LANE_CHANGE_RIGHT]:
                self.patternDuration = np.random.uniform(3.0, 8.0)
            elif self.currentTurningPattern == TurningPattern.CORRECTIVE_STEERING:
                self.patternDuration = np.random.uniform(1.0, 4.0)
            elif self.currentTurningPattern == TurningPattern.ROUNDABOUT:
                self.patternDuration = np.random.uniform(8.0, 20.0)
            elif self.currentTurningPattern == TurningPattern.PARKING_MANEUVER:
                self.patternDuration = np.random.uniform(10.0, 30.0)
            else:
                self.patternDuration = np.random.uniform(4.0, 15.0)
            
            self.logger.debug(f"Turning pattern changed to: {self.currentTurningPattern.value} for {self.patternDuration:.1f}s")
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != GyrAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = GyrAnomalyType.NORMAL
                self.logger.debug(f"GYR anomaly ended at {currentTime:.3f}s")
            return
        
        # Verificar se deve injetar nova anomalia
        if not self.anomalyConfig["enabled"]:
            return
        
        # Intervalo mínimo entre anomalias
        if currentTime - self.lastAnomalyTime < self.anomalyConfig["minInterval"]:
            return
        
        # Probabilidade de anomalia
        if np.random.random() < self.anomalyChance:
            # Escolher tipo de anomalia com pesos específicos
            anomalyTypes = [
                GyrAnomalyType.RAPID_ROTATION,        # 20% - Comum
                GyrAnomalyType.INSTABILITY,           # 18% - Comum
                GyrAnomalyType.EXCESSIVE_YAW,         # 15% - Moderado
                GyrAnomalyType.HIGH_FREQUENCY_NOISE,  # 12% - Moderado
                GyrAnomalyType.SENSOR_DRIFT,          # 10% - Moderado
                GyrAnomalyType.SPIN_SLIP,             # 10% - Raro mas crítico
                GyrAnomalyType.CALIBRATION_ERROR,     # 8% - Raro
                GyrAnomalyType.SENSOR_STUCK           # 7% - Raro
            ]
            
            weights = [0.20, 0.18, 0.15, 0.12, 0.10, 0.10, 0.08, 0.07]
            self.currentAnomalyType = np.random.choice(anomalyTypes, p=weights)
            
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia baseada no tipo
            if self.currentAnomalyType == GyrAnomalyType.SPIN_SLIP:
                self.anomalyDuration = np.random.uniform(0.5, 3.0)    # Muito curta
            elif self.currentAnomalyType == GyrAnomalyType.SENSOR_STUCK:
                self.anomalyDuration = np.random.uniform(2.0, 8.0)    # Curta
            elif self.currentAnomalyType == GyrAnomalyType.SENSOR_DRIFT:
                self.anomalyDuration = np.random.uniform(20.0, 60.0)  # Longa
            else:
                self.anomalyDuration = np.random.uniform(3.0, 15.0)   # Normal
            
            self.logger.warning(f"GYR anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 5.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("rapid_rotation", "spin_slip", etc.)
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = GyrAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced GYR anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown GYR anomaly type: {anomalyType}")
    
    def forceTurningPattern(self, pattern: str, duration: float = 10.0):
        """
        Força padrão de rotação específico.
        
        Args:
            pattern: Padrão de rotação ("straight", "sharp_left", etc.)
            duration: Duração do padrão em segundos
        """
        
        try:
            self.currentTurningPattern = TurningPattern(pattern)
            self.patternStartTime = self.currentTimestamp
            self.patternDuration = duration
            
            self.logger.info(f"Forced turning pattern: {pattern} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown turning pattern: {pattern}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "CardioWheelGYR",
            "samplingRate": self.samplingRate,
            "chunkSize": self.chunkSize,
            "currentTimestamp": self.currentTimestamp,
            "sampleCounter": self.sampleCounter,
            "currentTurningPattern": self.currentTurningPattern.value,
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != GyrAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "patternTimeRemaining": max(0, (self.patternStartTime + self.patternDuration) - self.currentTimestamp),
            "angularVelocity": {
                "x": round(self.angularVelocity[0], 1),
                "y": round(self.angularVelocity[1], 1), 
                "z": round(self.angularVelocity[2], 1)
            },
            "driftAccumulation": {
                "x": round(self.driftAccumulation[0], 3),
                "y": round(self.driftAccumulation[1], 3),
                "z": round(self.driftAccumulation[2], 3)
            },
            "config": {
                "baselines": {"x": self.baselineX, "y": self.baselineY, "z": self.baselineZ},
                "noiseStd": self.noiseStd,
                "anomalyChance": self.anomalyChance,
                "conversionFactor": self.gyrConfig["conversionFactor"],
                "maxYawRate": self.maxYawRate,
                "maxPitchRate": self.maxPitchRate,
                "maxRollRate": self.maxRollRate
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = GyrAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentTurningPattern = TurningPattern.STRAIGHT
        self.patternStartTime = 0.0
        self.patternDuration = 0.0
        self.angularVelocity = np.array([0.0, 0.0, 0.0])
        self.driftAccumulation = np.array([0.0, 0.0, 0.0])
        self.vibrationPhase = 0.0
        
        self.logger.info("CardioWheelGyrGenerator reset")

# Instância global
cardioWheelGyrGenerator = CardioWheelGyrGenerator()