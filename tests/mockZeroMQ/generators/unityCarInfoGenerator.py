"""
UnityCarInfoGenerator - Gerador de dados de informação do carro

Resumo:
Gera dados realistas de velocidade e centralidade na estrada para simulação de condução.
Simula diferentes padrões de condução: cidade, estrada, autoestrada.
Frequência de 1Hz com correlação ao nível de álcool (mais álcool = pior condução).
Detecta anomalias como excesso de velocidade e saída de faixa.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional, Tuple
from enum import Enum

from app.core import settings
from .unityAlcoholGenerator import unityAlcoholGenerator, AlcoholState

class DrivingMode(Enum):
    """Modos de condução simulados"""
    CITY = "city"              # Cidade: 0-60 km/h
    ROAD = "road"              # Estrada: 60-90 km/h  
    HIGHWAY = "highway"        # Autoestrada: 90-120 km/h
    PARKING = "parking"        # Estacionamento: 0-20 km/h

class CarAnomalyType(Enum):
    """Tipos de anomalias de condução"""
    NORMAL = "normal"
    SPEEDING = "speeding"                    # Excesso velocidade (>100 km/h)
    DANGEROUS_SPEED = "dangerous_speed"      # Velocidade perigosa (>150 km/h)
    LANE_DEPARTURE = "lane_departure"        # Saída de faixa (<0.4)
    DANGEROUS_LANE = "dangerous_lane"        # Fora da faixa (<0.2)
    ERRATIC_DRIVING = "erratic_driving"      # Condução errática
    SUDDEN_SPEED_CHANGE = "sudden_speed_change"  # Mudança súbita velocidade

class UnityCarInfoGenerator:
    """Gerador de dados de informação do carro para tópico Unity_CarInfo"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações CarInfo do settings
        self.carConfig = settings.signals.unityConfig["car_information"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração
        self.samplingRate = self.carConfig["samplingRate"]  # 1Hz
        
        # Configurações de velocidade
        self.speedConfig = self.carConfig["speed"]
        self.speedingThreshold = self.speedConfig["speedingThreshold"]      # 100 km/h
        self.dangerSpeedThreshold = self.speedConfig["dangerSpeedThreshold"] # 150 km/h
        self.suddenChangeThreshold = self.speedConfig["suddenChangeThreshold"] # 20 km/h
        
        # Configurações de centralidade
        self.laneCentralityConfig = self.carConfig["lane_centrality"]
        self.warningThreshold = self.laneCentralityConfig["warningThreshold"]    # 0.4
        self.dangerThreshold = self.laneCentralityConfig["dangerThreshold"]      # 0.2
        self.stabilityThreshold = self.laneCentralityConfig["stabilityThreshold"] # 0.1
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["Unity_CarInfo"]  # 3%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = CarAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado de condução simulada
        self.currentDrivingMode = DrivingMode.CITY
        self.modeStartTime = 0.0
        self.modeDuration = 0.0
        self.currentSpeed = 30.0
        self.currentLaneCentrality = 0.8
        self.lastSpeed = 30.0
        
        # Parâmetros de movimento
        self.targetSpeed = 30.0
        self.speedTrend = 0.0  # -1 (diminuindo) a +1 (aumentando)
        self.accelerationRate = 2.0  # km/h por segundo
        
        # Probabilidades de modo de condução
        self.modeWeights = {
            DrivingMode.CITY: 0.50,     # 50% - cidade mais comum
            DrivingMode.ROAD: 0.30,     # 30% - estrada
            DrivingMode.HIGHWAY: 0.15,  # 15% - autoestrada
            DrivingMode.PARKING: 0.05   # 5% - estacionamento
        }
        
        # Referência ao gerador de álcool para correlação
        self.alcoholGenerator = unityAlcoholGenerator
        
        self.logger.info(f"UnityCarInfoGenerator initialized - 1Hz, speed threshold: {self.speedingThreshold} km/h")
    
    def generateEvent(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um evento de informação do carro (velocidade + centralidade).
        
        Args:
            baseTimestamp: Timestamp base (usa interno se None)
            
        Returns:
            Dict com velocidade e centralidade para formatação ZeroMQ
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Atualizar modo de condução
            self._updateDrivingMode()
            
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Gerar velocidade e centralidade correlacionadas
            speed, laneCentrality = self._generateCarData()
            
            # Avançar contadores
            self.sampleCounter += 1
            self.currentTimestamp += 1.0  # +1 segundo
            
            result = {
                "speed": round(speed, 1),                    # 1 casa decimal
                "lane_centrality": round(laneCentrality, 3), # 3 casas decimais
                "eventTimestamp": self.currentTimestamp - 1.0,
                "anomalyType": self.currentAnomalyType.value,
                "drivingMode": self.currentDrivingMode.value,
                "samplingRate": self.samplingRate
            }
            
            self.logger.debug(f"Generated car data: {speed:.1f} km/h, centrality: {laneCentrality:.3f}, mode: {self.currentDrivingMode.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating car information: {e}")
            raise
    
    def _generateCarData(self) -> Tuple[float, float]:
        """
        Gera velocidade e centralidade baseadas no modo atual e correlação com álcool.
        
        Returns:
            Tuplo (speed, lane_centrality)
        """
        
        # Gerar velocidade baseada no modo
        speed = self._generateSpeed()
        
        # Gerar centralidade correlacionada com álcool
        laneCentrality = self._generateLaneCentrality(speed)
        
        # Aplicar anomalias se ativas
        if self.currentAnomalyType != CarAnomalyType.NORMAL:
            speed, laneCentrality = self._applyAnomalies(speed, laneCentrality)
        
        # Atualizar estado para próxima iteração
        self.lastSpeed = self.currentSpeed
        self.currentSpeed = speed
        self.currentLaneCentrality = laneCentrality
        
        return speed, laneCentrality
    
    def _generateSpeed(self) -> float:
        """
        Gera velocidade baseada no modo de condução atual.
        
        Returns:
            Velocidade em km/h
        """
        
        # Obter range de velocidade para o modo atual
        speedRange = self._getSpeedRangeForMode()
        
        # Calcular velocidade alvo se mudou de modo
        if self.targetSpeed < speedRange[0] or self.targetSpeed > speedRange[1]:
            self.targetSpeed = np.random.uniform(speedRange[0], speedRange[1])
        
        # Movimento gradual em direção à velocidade alvo
        speedDiff = self.targetSpeed - self.currentSpeed
        
        if abs(speedDiff) > self.accelerationRate:
            # Acelerar/desacelerar gradualmente
            if speedDiff > 0:
                newSpeed = self.currentSpeed + self.accelerationRate
            else:
                newSpeed = self.currentSpeed - self.accelerationRate
        else:
            # Próximo da velocidade alvo
            newSpeed = self.targetSpeed
            
            # Escolher nova velocidade alvo ocasionalmente
            if np.random.random() < 0.1:  # 10% chance
                self.targetSpeed = np.random.uniform(speedRange[0], speedRange[1])
        
        # Adicionar variação aleatória pequena
        noise = np.random.normal(0, 1.0)  # ±1 km/h de ruído
        newSpeed += noise
        
        # Clipar para range válido
        newSpeed = max(0, min(newSpeed, 200))  # 0-200 km/h máximo
        
        return newSpeed
    
    def _generateLaneCentrality(self, speed: float) -> float:
        """
        Gera centralidade correlacionada com álcool e velocidade.
        
        Args:
            speed: Velocidade atual em km/h
            
        Returns:
            Centralidade da faixa (0-1)
        """
        
        # Obter nível de álcool atual para correlação
        alcoholLevel = getattr(self.alcoholGenerator, 'currentAlcoholLevel', 0.0)
        
        # Centralidade base baseada no modo de condução
        baseCentrality = self._getBaseCentralityForMode()
        
        # Impacto do álcool na centralidade (mais álcool = pior centralidade)
        alcoholImpact = self._calculateAlcoholImpact(alcoholLevel)
        
        # Impacto da velocidade (velocidade muito alta = ligeiramente pior centralidade)
        speedImpact = self._calculateSpeedImpact(speed)
        
        # Calcular centralidade final
        finalCentrality = baseCentrality - alcoholImpact - speedImpact
        
        # Adicionar variação natural
        naturalVariation = np.random.normal(0, self.stabilityThreshold / 2)
        finalCentrality += naturalVariation
        
        # Clipar para range válido
        finalCentrality = max(0.0, min(finalCentrality, 1.0))
        
        return finalCentrality
    
    def _getSpeedRangeForMode(self) -> Tuple[float, float]:
        """Retorna range de velocidade para o modo atual"""
        
        if self.currentDrivingMode == DrivingMode.CITY:
            return (10, 60)   # Cidade
        elif self.currentDrivingMode == DrivingMode.ROAD:
            return (50, 90)   # Estrada
        elif self.currentDrivingMode == DrivingMode.HIGHWAY:
            return (80, 120)  # Autoestrada
        elif self.currentDrivingMode == DrivingMode.PARKING:
            return (0, 20)    # Estacionamento
        else:
            return (0, 50)
    
    def _getBaseCentralityForMode(self) -> float:
        """Retorna centralidade base para o modo atual"""
        
        if self.currentDrivingMode == DrivingMode.CITY:
            return np.random.uniform(0.6, 0.9)   # Cidade - trânsito
        elif self.currentDrivingMode == DrivingMode.ROAD:
            return np.random.uniform(0.7, 1.0)   # Estrada - melhor
        elif self.currentDrivingMode == DrivingMode.HIGHWAY:
            return np.random.uniform(0.8, 1.0)   # Autoestrada - óptimo
        elif self.currentDrivingMode == DrivingMode.PARKING:
            return np.random.uniform(0.4, 0.8)   # Estacionamento - manobras
        else:
            return 0.7
    
    def _calculateAlcoholImpact(self, alcoholLevel: float) -> float:
        """
        Calcula impacto do álcool na centralidade da faixa.
        
        Args:
            alcoholLevel: Nível de álcool em g/L
            
        Returns:
            Redução na centralidade (0-0.6)
        """
        
        if alcoholLevel <= 0.1:
            return 0.0  # Sóbrio - sem impacto
        elif alcoholLevel <= 0.5:
            # Ligeiro impacto linear
            return (alcoholLevel - 0.1) * 0.25  # Máximo -0.1
        elif alcoholLevel <= 0.8:
            # Impacto moderado
            return 0.1 + (alcoholLevel - 0.5) * 0.5  # -0.1 a -0.25
        else:
            # Impacto severo
            return 0.25 + (alcoholLevel - 0.8) * 0.7  # -0.25 a -0.6+
    
    def _calculateSpeedImpact(self, speed: float) -> float:
        """
        Calcula impacto da velocidade na centralidade.
        
        Args:
            speed: Velocidade em km/h
            
        Returns:
            Redução na centralidade (0-0.2)
        """
        
        if speed <= 100:
            return 0.0  # Velocidade normal - sem impacto
        elif speed <= 150:
            # Impacto ligeiro por velocidade alta
            return (speed - 100) * 0.002  # Máximo -0.1
        else:
            # Impacto maior por velocidade muito alta
            return 0.1 + (speed - 150) * 0.004  # -0.1 a -0.3+
    
    def _applyAnomalies(self, speed: float, laneCentrality: float) -> Tuple[float, float]:
        """
        Aplica anomalias específicas aos dados do carro.
        
        Args:
            speed: Velocidade base em km/h
            laneCentrality: Centralidade base
            
        Returns:
            Dados modificados pela anomalia
        """
        
        if self.currentAnomalyType == CarAnomalyType.SPEEDING:
            # Excesso de velocidade
            speed = max(speed, np.random.uniform(100, 130))
            
        elif self.currentAnomalyType == CarAnomalyType.DANGEROUS_SPEED:
            # Velocidade perigosa
            speed = max(speed, np.random.uniform(150, 180))
            
        elif self.currentAnomalyType == CarAnomalyType.LANE_DEPARTURE:
            # Saída de faixa
            laneCentrality = min(laneCentrality, np.random.uniform(0.2, 0.4))
            
        elif self.currentAnomalyType == CarAnomalyType.DANGEROUS_LANE:
            # Fora da faixa
            laneCentrality = min(laneCentrality, np.random.uniform(0.0, 0.2))
            
        elif self.currentAnomalyType == CarAnomalyType.ERRATIC_DRIVING:
            # Condução errática - ambos afetados
            speed += np.random.uniform(-20, 30)
            laneCentrality *= np.random.uniform(0.5, 0.9)
            
        elif self.currentAnomalyType == CarAnomalyType.SUDDEN_SPEED_CHANGE:
            # Mudança súbita de velocidade
            speedChange = np.random.choice([-40, -30, 30, 40])
            speed = max(0, speed + speedChange)
        
        return speed, laneCentrality
    
    def _updateDrivingMode(self):
        """
        Atualiza modo de condução baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Verificar se deve mudar modo
        if currentTime - self.modeStartTime >= self.modeDuration:
            # Escolher novo modo baseado em pesos
            modes = list(self.modeWeights.keys())
            weights = list(self.modeWeights.values())
            self.currentDrivingMode = np.random.choice(modes, p=weights)
            
            self.modeStartTime = currentTime
            
            # Duração do modo baseada no tipo
            if self.currentDrivingMode == DrivingMode.CITY:
                self.modeDuration = np.random.uniform(300.0, 1200.0)  # 5-20 min
            elif self.currentDrivingMode == DrivingMode.ROAD:
                self.modeDuration = np.random.uniform(600.0, 1800.0)  # 10-30 min
            elif self.currentDrivingMode == DrivingMode.HIGHWAY:
                self.modeDuration = np.random.uniform(900.0, 2400.0)  # 15-40 min
            elif self.currentDrivingMode == DrivingMode.PARKING:
                self.modeDuration = np.random.uniform(60.0, 300.0)    # 1-5 min
            
            self.logger.debug(f"Driving mode changed to: {self.currentDrivingMode.value} for {self.modeDuration:.1f}s")
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != CarAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = CarAnomalyType.NORMAL
                self.logger.debug(f"Car anomaly ended at {currentTime:.1f}s")
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
                CarAnomalyType.SPEEDING,             # 30% - excesso velocidade
                CarAnomalyType.LANE_DEPARTURE,       # 25% - saída faixa
                CarAnomalyType.ERRATIC_DRIVING,      # 20% - condução errática
                CarAnomalyType.SUDDEN_SPEED_CHANGE,  # 15% - mudança súbita
                CarAnomalyType.DANGEROUS_SPEED,      # 7% - velocidade perigosa
                CarAnomalyType.DANGEROUS_LANE        # 3% - fora da faixa
            ]
            
            weights = [0.30, 0.25, 0.20, 0.15, 0.07, 0.03]
            self.currentAnomalyType = np.random.choice(anomalyTypes, p=weights)
            
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia baseada no tipo
            if self.currentAnomalyType in [CarAnomalyType.SUDDEN_SPEED_CHANGE]:
                self.anomalyDuration = np.random.uniform(5.0, 30.0)     # Muito curta
            elif self.currentAnomalyType in [CarAnomalyType.DANGEROUS_SPEED, CarAnomalyType.DANGEROUS_LANE]:
                self.anomalyDuration = np.random.uniform(10.0, 60.0)    # Curta
            else:
                self.anomalyDuration = np.random.uniform(30.0, 180.0)   # Normal
            
            self.logger.warning(f"Car anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 60.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = CarAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced car anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown car anomaly type: {anomalyType}")
    
    def forceDrivingMode(self, mode: str, duration: float = 600.0):
        """
        Força modo de condução específico.
        
        Args:
            mode: Modo de condução ("city", "road", "highway", "parking")
            duration: Duração do modo em segundos
        """
        
        try:
            self.currentDrivingMode = DrivingMode(mode)
            self.modeStartTime = self.currentTimestamp
            self.modeDuration = duration
            
            self.logger.info(f"Forced driving mode: {mode} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown driving mode: {mode}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "UnityCarInfo",
            "samplingRate": self.samplingRate,
            "currentTimestamp": self.currentTimestamp,
            "sampleCounter": self.sampleCounter,
            "currentDrivingMode": self.currentDrivingMode.value,
            "currentSpeed": round(self.currentSpeed, 1),
            "currentLaneCentrality": round(self.currentLaneCentrality, 3),
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != CarAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "modeTimeRemaining": max(0, (self.modeStartTime + self.modeDuration) - self.currentTimestamp),
            "alcoholCorrelation": {
                "currentAlcoholLevel": getattr(self.alcoholGenerator, 'currentAlcoholLevel', 0.0),
                "alcoholImpact": self._calculateAlcoholImpact(getattr(self.alcoholGenerator, 'currentAlcoholLevel', 0.0))
            },
            "config": {
                "speedingThreshold": self.speedingThreshold,
                "dangerSpeedThreshold": self.dangerSpeedThreshold,
                "warningThreshold": self.warningThreshold,
                "dangerThreshold": self.dangerThreshold,
                "anomalyChance": self.anomalyChance,
                "modeWeights": {mode.value: weight for mode, weight in self.modeWeights.items()}
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = CarAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentDrivingMode = DrivingMode.CITY
        self.modeStartTime = 0.0
        self.modeDuration = 0.0
        self.currentSpeed = 30.0
        self.currentLaneCentrality = 0.8
        self.lastSpeed = 30.0
        self.targetSpeed = 30.0
        self.speedTrend = 0.0
        
        self.logger.info("UnityCarInfoGenerator reset")

# Instância global
unityCarInfoGenerator = UnityCarInfoGenerator()