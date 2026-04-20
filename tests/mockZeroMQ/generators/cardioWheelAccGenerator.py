"""
CardioWheelAccGenerator - Gerador de dados de acelerómetro realistas

Resumo:
Gera dados de acelerómetro simulando o CardioWheel real com:
- Valores ADC 16-bit realistas com baselines observados (X~7500, Y~0, Z~3100)
- Chunks de 10 samples (100ms @ 100Hz real)
- Simulação de movimentos de condução: aceleração, travagem, curvas, vibração
- Anomalias específicas: movimento brusco, impacto, vibração excessiva
- Gravidade simulada no eixo Z
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from app.core import settings

class AccAnomalyType(Enum):
    """Tipos de anomalias de acelerómetro"""
    NORMAL = "normal"
    SUDDEN_MOVEMENT = "sudden_movement"       # Movimento brusco
    IMPACT = "impact"                         # Impacto severo
    EXCESSIVE_VIBRATION = "excessive_vibration"  # Vibração excessiva
    AGGRESSIVE_DRIVING = "aggressive_driving"    # Condução agressiva
    EMERGENCY_BRAKING = "emergency_braking"      # Travagem de emergência
    RAPID_ACCELERATION = "rapid_acceleration"    # Aceleração rápida
    SENSOR_STUCK = "sensor_stuck"            # Sensor travado
    HIGH_NOISE = "high_noise"                # Ruído elevado

class DrivingPattern(Enum):
    """Padrões de condução simulados"""
    STEADY = "steady"                        # Condução estável
    ACCELERATING = "accelerating"            # A acelerar
    BRAKING = "braking"                      # A travar
    CORNERING_LEFT = "cornering_left"        # Curva à esquerda
    CORNERING_RIGHT = "cornering_right"      # Curva à direita
    CITY_DRIVING = "city_driving"            # Condução urbana
    HIGHWAY = "highway"                      # Autoestrada
    PARKING = "parking"                      # Estacionamento

class CardioWheelAccGenerator:
    """Gerador de dados de acelerómetro para tópico CardioWheel_ACC"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações ACC do settings
        self.accConfig = settings.signals.sensorsConfig["accelerometer"]
        self.zmqConfig = settings.zeromq.topicProcessingConfig["CardioWheel_ACC"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração ACC
        self.samplingRate = self.zmqConfig["samplingRate"]              # 100Hz
        self.chunkSize = self.mockConfig.topicChunkSizes["CardioWheel_ACC"]  # 10 samples
        self.chunkDuration = self.chunkSize / self.samplingRate         # 0.1s (100ms)
        
        # Configurações ADC e baselines observados nos dados reais
        self.generatorConfig = self.mockConfig.generatorBaseConfig["accelerometer"]
        self.baselineX = self.generatorConfig["baselineX"]             # 7500 ADC
        self.baselineY = self.generatorConfig["baselineY"]             # 0 ADC
        self.baselineZ = self.generatorConfig["baselineZ"]             # 3100 ADC (com gravidade)
        self.noiseStd = self.generatorConfig["noiseStd"]               # 5 ADC units
        
        # Thresholds de anomalias
        self.suddenMovementThreshold = self.accConfig["suddenMovementThreshold"]      # 50 m/s²
        self.impactThreshold = self.accConfig["impactThreshold"]                      # 120 m/s²
        self.highVibrationsThreshold = self.accConfig["highVibrationsThreshold"]      # 20 m/s²
        self.magnitudeThreshold = self.accConfig["magnitudeThreshold"]                # 100 m/s²
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["CardioWheel_ACC"]  # 4%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = AccAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado de condução simulada
        self.currentDrivingPattern = DrivingPattern.STEADY
        self.patternStartTime = 0.0
        self.patternDuration = 0.0
        self.drivingVelocity = np.array([0.0, 0.0, 0.0])  # Velocidade simulada m/s
        self.drivingAcceleration = np.array([0.0, 0.0, 0.0])  # Aceleração simulada m/s²
        
        # Parâmetros de movimento
        self.maxSpeed = 30.0          # m/s (~100 km/h)
        self.typicalAcceleration = 2.0  # m/s² aceleração normal
        self.typicalBraking = -4.0    # m/s² travagem normal
        self.corneringAcceleration = 3.0  # m/s² aceleração lateral em curvas
        
        # Vibração de estrada
        self.vibrationFrequency = 15.0  # Hz frequência de vibração da estrada
        self.vibrationPhase = 0.0
        
        self.logger.info(f"CardioWheelAccGenerator initialized - {self.samplingRate}Hz, chunks of {self.chunkSize}")
    
    def generateChunk(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um chunk de dados de acelerómetro (10 samples).
        
        Args:
            baseTimestamp: Timestamp base para o chunk (usa interno se None)
            
        Returns:
            Dict com arrays de dados ACC x,y,z para formatação
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Atualizar padrão de condução
            self._updateDrivingPattern()
            
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Gerar samples ACC para o chunk
            xSamples = []
            ySamples = []
            zSamples = []
            
            for i in range(self.chunkSize):
                # Gerar sample ACC baseado no padrão e anomalia atual
                accX, accY, accZ = self._generateAccSample(i)
                
                xSamples.append(int(accX))  # ADC values são inteiros
                ySamples.append(int(accY))
                zSamples.append(int(accZ))
                
                # Avançar contadores
                self.sampleCounter += 1
                self.vibrationPhase += (2 * np.pi * self.vibrationFrequency) / self.samplingRate
                
                if self.vibrationPhase >= 2 * np.pi:
                    self.vibrationPhase -= 2 * np.pi
            
            # Avançar timestamp para próximo chunk
            self.currentTimestamp += self.chunkDuration
            
            # Calcular magnitude para debug
            magnitudes = []
            for i in range(self.chunkSize):
                # Converter para m/s² para cálculo de magnitude
                physX = (xSamples[i] - self.baselineX) * self.accConfig["conversionFactor"]
                physY = (ySamples[i] - self.baselineY) * self.accConfig["conversionFactor"]
                physZ = (zSamples[i] - self.baselineZ) * self.accConfig["conversionFactor"]
                magnitude = (physX**2 + physY**2 + physZ**2)**0.5
                magnitudes.append(round(magnitude, 2))
            
            result = {
                "x": xSamples,
                "y": ySamples,
                "z": zSamples,
                "chunkTimestamp": self.currentTimestamp - self.chunkDuration,
                "anomalyType": self.currentAnomalyType.value,
                "drivingPattern": self.currentDrivingPattern.value,
                "magnitudes": magnitudes,  # Para debug
                "samplingRate": self.samplingRate,
                "chunkSize": self.chunkSize
            }
            
            self.logger.debug(f"Generated ACC chunk: {len(xSamples)} samples, pattern: {self.currentDrivingPattern.value}, anomaly: {self.currentAnomalyType.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating ACC chunk: {e}")
            raise
    
    def _generateAccSample(self, sampleIndex: int) -> Tuple[float, float, float]:
        """
        Gera um sample de acelerómetro individual.
        
        Args:
            sampleIndex: Índice do sample no chunk (0-9)
            
        Returns:
            Tuplo (accX, accY, accZ) em ADC units
        """
        
        # Obter aceleração física baseada no padrão de condução
        physAccX, physAccY, physAccZ = self._getDrivingAcceleration()
        
        # Aplicar anomalias se ativas
        if self.currentAnomalyType != AccAnomalyType.NORMAL:
            physAccX, physAccY, physAccZ = self._applyAnomalies(physAccX, physAccY, physAccZ, sampleIndex)
        
        # Adicionar vibração da estrada (sempre presente)
        roadVibration = self._getRoadVibration()
        physAccX += roadVibration[0]
        physAccY += roadVibration[1]
        physAccZ += roadVibration[2]
        
        # Adicionar ruído gaussiano
        noise = np.random.normal(0, self.noiseStd, 3)
        
        # Converter para ADC units e adicionar baselines
        adcX = self.baselineX + (physAccX / self.accConfig["conversionFactor"]) + noise[0]
        adcY = self.baselineY + (physAccY / self.accConfig["conversionFactor"]) + noise[1]
        adcZ = self.baselineZ + (physAccZ / self.accConfig["conversionFactor"]) + noise[2]
        
        # Clipar para range ADC 16-bit
        adcX = np.clip(adcX, -32768, 32767)
        adcY = np.clip(adcY, -32768, 32767)
        adcZ = np.clip(adcZ, -32768, 32767)
        
        return adcX, adcY, adcZ
    
    def _getDrivingAcceleration(self) -> Tuple[float, float, float]:
        """
        Calcula aceleração baseada no padrão de condução atual.
        
        Returns:
            Tuplo (accX, accY, accZ) em m/s²
        """
        
        if self.currentDrivingPattern == DrivingPattern.STEADY:
            # Condução estável - apenas pequenas variações
            accX = np.random.normal(0, 0.5)  # Variação lateral mínima
            accY = np.random.normal(0, 0.3)  # Variação longitudinal mínima
            accZ = np.random.normal(0, 0.2)  # Variação vertical mínima
            
        elif self.currentDrivingPattern == DrivingPattern.ACCELERATING:
            # Aceleração para frente
            accX = np.random.normal(0, 1.0)
            accY = np.random.uniform(1.0, 3.0)  # Aceleração positiva
            accZ = np.random.normal(0, 0.5)
            
        elif self.currentDrivingPattern == DrivingPattern.BRAKING:
            # Travagem (desaceleração)
            accX = np.random.normal(0, 1.5)
            accY = np.random.uniform(-5.0, -1.0)  # Desaceleração
            accZ = np.random.normal(0, 0.8)
            
        elif self.currentDrivingPattern == DrivingPattern.CORNERING_LEFT:
            # Curva à esquerda - aceleração lateral
            accX = np.random.uniform(-4.0, -1.0)  # Força centrífuga para direita
            accY = np.random.normal(0, 1.0)
            accZ = np.random.normal(0, 0.5)
            
        elif self.currentDrivingPattern == DrivingPattern.CORNERING_RIGHT:
            # Curva à direita - aceleração lateral
            accX = np.random.uniform(1.0, 4.0)    # Força centrífuga para esquerda
            accY = np.random.normal(0, 1.0)
            accZ = np.random.normal(0, 0.5)
            
        elif self.currentDrivingPattern == DrivingPattern.CITY_DRIVING:
            # Condução urbana - variações frequentes
            accX = np.random.normal(0, 2.0)
            accY = np.random.normal(0, 2.5)
            accZ = np.random.normal(0, 1.0)
            
        elif self.currentDrivingPattern == DrivingPattern.HIGHWAY:
            # Autoestrada - mais estável
            accX = np.random.normal(0, 0.8)
            accY = np.random.normal(0, 0.5)
            accZ = np.random.normal(0, 0.3)
            
        elif self.currentDrivingPattern == DrivingPattern.PARKING:
            # Estacionamento - movimentos lentos e precisos
            accX = np.random.normal(0, 1.5)
            accY = np.random.uniform(-1.0, 1.0)
            accZ = np.random.normal(0, 0.5)
            
        else:
            accX = accY = accZ = 0.0
        
        return accX, accY, accZ
    
    def _getRoadVibration(self) -> np.ndarray:
        """
        Simula vibração contínua da estrada.
        
        Returns:
            Array com vibração em m/s² para [X, Y, Z]
        """
        
        # Vibração base da estrada (sempre presente)
        amplitude = 0.5  # m/s²
        
        vibX = amplitude * 0.7 * np.sin(self.vibrationPhase)
        vibY = amplitude * 0.5 * np.sin(self.vibrationPhase * 1.3)
        vibZ = amplitude * 1.0 * np.sin(self.vibrationPhase * 0.8)
        
        return np.array([vibX, vibY, vibZ])
    
    def _applyAnomalies(self, accX: float, accY: float, accZ: float, sampleIndex: int) -> Tuple[float, float, float]:
        """
        Aplica anomalias específicas à aceleração.
        
        Args:
            accX, accY, accZ: Aceleração base em m/s²
            sampleIndex: Índice do sample no chunk
            
        Returns:
            Aceleração modificada pela anomalia
        """
        
        if self.currentAnomalyType == AccAnomalyType.SUDDEN_MOVEMENT:
            # Movimento brusco - pico repentino
            if sampleIndex < 3:  # Primeiros samples do chunk
                magnitude = np.random.uniform(30, 60)
                direction = np.random.choice([-1, 1], 3)
                accX += magnitude * direction[0]
                accY += magnitude * direction[1]
                accZ += magnitude * direction[2]
                
        elif self.currentAnomalyType == AccAnomalyType.IMPACT:
            # Impacto severo - muito breve mas intenso
            if sampleIndex == 0:  # Só no primeiro sample
                magnitude = np.random.uniform(80, 150)
                direction = np.random.choice([-1, 1], 3)
                accX += magnitude * direction[0]
                accY += magnitude * direction[1]
                accZ += magnitude * direction[2]
                
        elif self.currentAnomalyType == AccAnomalyType.EXCESSIVE_VIBRATION:
            # Vibração excessiva - amplitude muito alta
            vibMagnitude = np.random.uniform(10, 25)
            highFreqPhase = self.sampleCounter * 0.5  # Frequência mais alta
            accX += vibMagnitude * np.sin(highFreqPhase)
            accY += vibMagnitude * np.sin(highFreqPhase * 1.2)
            accZ += vibMagnitude * np.sin(highFreqPhase * 0.9)
            
        elif self.currentAnomalyType == AccAnomalyType.AGGRESSIVE_DRIVING:
            # Condução agressiva - acelerações altas mantidas
            magnitude = np.random.uniform(15, 35)
            accX += np.random.uniform(-magnitude, magnitude)
            accY += np.random.uniform(-magnitude, magnitude)
            accZ += np.random.uniform(-magnitude*0.5, magnitude*0.5)
            
        elif self.currentAnomalyType == AccAnomalyType.EMERGENCY_BRAKING:
            # Travagem de emergência - forte desaceleração Y
            accX += np.random.normal(0, 5)
            accY += np.random.uniform(-80, -40)  # Forte desaceleração
            accZ += np.random.normal(0, 3)
            
        elif self.currentAnomalyType == AccAnomalyType.RAPID_ACCELERATION:
            # Aceleração rápida - forte aceleração Y
            accX += np.random.normal(0, 3)
            accY += np.random.uniform(25, 50)   # Forte aceleração
            accZ += np.random.normal(0, 2)
            
        elif self.currentAnomalyType == AccAnomalyType.SENSOR_STUCK:
            # Sensor travado - valores constantes
            accX = accY = accZ = 0  # Override para zero
            
        elif self.currentAnomalyType == AccAnomalyType.HIGH_NOISE:
            # Ruído elevado - variação aleatória alta
            noise = np.random.normal(0, 15, 3)
            accX += noise[0]
            accY += noise[1]
            accZ += noise[2]
        
        return accX, accY, accZ
    
    def _updateDrivingPattern(self):
        """
        Atualiza padrão de condução baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Verificar se deve mudar padrão
        if currentTime - self.patternStartTime >= self.patternDuration:
            # Escolher novo padrão baseado em probabilidades
            patterns = [
                DrivingPattern.STEADY,           # 30% - Mais comum
                DrivingPattern.ACCELERATING,     # 15%
                DrivingPattern.BRAKING,          # 15%
                DrivingPattern.CORNERING_LEFT,   # 10%
                DrivingPattern.CORNERING_RIGHT,  # 10%
                DrivingPattern.CITY_DRIVING,     # 10%
                DrivingPattern.HIGHWAY,          # 8%
                DrivingPattern.PARKING           # 2%
            ]
            
            weights = [0.30, 0.15, 0.15, 0.10, 0.10, 0.10, 0.08, 0.02]
            self.currentDrivingPattern = np.random.choice(patterns, p=weights)
            
            self.patternStartTime = currentTime
            
            # Duração do padrão baseada no tipo
            if self.currentDrivingPattern == DrivingPattern.STEADY:
                self.patternDuration = np.random.uniform(10.0, 30.0)
            elif self.currentDrivingPattern in [DrivingPattern.CORNERING_LEFT, DrivingPattern.CORNERING_RIGHT]:
                self.patternDuration = np.random.uniform(2.0, 8.0)
            elif self.currentDrivingPattern == DrivingPattern.PARKING:
                self.patternDuration = np.random.uniform(5.0, 15.0)
            else:
                self.patternDuration = np.random.uniform(3.0, 12.0)
            
            self.logger.debug(f"Driving pattern changed to: {self.currentDrivingPattern.value} for {self.patternDuration:.1f}s")
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != AccAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = AccAnomalyType.NORMAL
                self.logger.debug(f"ACC anomaly ended at {currentTime:.3f}s")
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
                AccAnomalyType.SUDDEN_MOVEMENT,      # 25% - Comum
                AccAnomalyType.EXCESSIVE_VIBRATION,  # 20% - Comum
                AccAnomalyType.AGGRESSIVE_DRIVING,   # 15% - Moderado
                AccAnomalyType.EMERGENCY_BRAKING,    # 12% - Moderado
                AccAnomalyType.RAPID_ACCELERATION,   # 10% - Moderado
                AccAnomalyType.HIGH_NOISE,           # 8% - Raro
                AccAnomalyType.IMPACT,               # 5% - Raro
                AccAnomalyType.SENSOR_STUCK          # 5% - Raro
            ]
            
            weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.05, 0.05]
            self.currentAnomalyType = np.random.choice(anomalyTypes, p=weights)
            
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia baseada no tipo
            if self.currentAnomalyType in [AccAnomalyType.IMPACT, AccAnomalyType.SUDDEN_MOVEMENT]:
                self.anomalyDuration = np.random.uniform(0.5, 3.0)    # Muito curta
            elif self.currentAnomalyType == AccAnomalyType.SENSOR_STUCK:
                self.anomalyDuration = np.random.uniform(2.0, 8.0)    # Curta
            else:
                self.anomalyDuration = np.random.uniform(3.0, 15.0)   # Normal
            
            self.logger.warning(f"ACC anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 5.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("sudden_movement", "impact", etc.)
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = AccAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced ACC anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown ACC anomaly type: {anomalyType}")
    
    def forceDrivingPattern(self, pattern: str, duration: float = 10.0):
        """
        Força padrão de condução específico.
        
        Args:
            pattern: Padrão de condução ("steady", "accelerating", etc.)
            duration: Duração do padrão em segundos
        """
        
        try:
            self.currentDrivingPattern = DrivingPattern(pattern)
            self.patternStartTime = self.currentTimestamp
            self.patternDuration = duration
            
            self.logger.info(f"Forced driving pattern: {pattern} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown driving pattern: {pattern}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "CardioWheelACC",
            "samplingRate": self.samplingRate,
            "chunkSize": self.chunkSize,
            "currentTimestamp": self.currentTimestamp,
            "sampleCounter": self.sampleCounter,
            "currentDrivingPattern": self.currentDrivingPattern.value,
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != AccAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "patternTimeRemaining": max(0, (self.patternStartTime + self.patternDuration) - self.currentTimestamp),
            "config": {
                "baselines": {"x": self.baselineX, "y": self.baselineY, "z": self.baselineZ},
                "noiseStd": self.noiseStd,
                "anomalyChance": self.anomalyChance,
                "conversionFactor": self.accConfig["conversionFactor"]
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = AccAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentDrivingPattern = DrivingPattern.STEADY
        self.patternStartTime = 0.0
        self.patternDuration = 0.0
        self.vibrationPhase = 0.0
        
        self.logger.info("CardioWheelAccGenerator reset")

# Instância global
cardioWheelAccGenerator = CardioWheelAccGenerator()