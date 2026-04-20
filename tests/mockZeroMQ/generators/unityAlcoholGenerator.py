"""
UnityAlcoholGenerator - Gerador de dados de nível de álcool

Resumo:
Gera dados realistas de nível de álcool no sangue para simulação de condução.
Simula diferentes cenários: sóbrio, ligeiro, acima do limite legal, e perigoso.
Frequência de 1Hz com anomalias ocasionais acima do limite de 0.8 g/L.
Mantém estados persistentes para simular variação gradual ao longo do tempo.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, Any, Optional
from enum import Enum

from app.core import settings

class AlcoholState(Enum):
    """Estados de nível de álcool"""
    SOBER = "sober"              # 0.0-0.1 g/L
    LIGHT = "light"              # 0.1-0.5 g/L  
    ABOVE_LEGAL = "above_legal"  # 0.5-0.8 g/L
    DANGEROUS = "dangerous"      # >0.8 g/L (anomalia)

class AlcoholAnomalyType(Enum):
    """Tipos de anomalias de álcool"""
    NORMAL = "normal"
    HIGH_ALCOHOL = "high_alcohol"        # Acima limite legal (>0.5)
    DANGEROUS_ALCOHOL = "dangerous_alcohol"  # Perigoso (>0.8)
    EXTREME_ALCOHOL = "extreme_alcohol"  # Extremo (>1.2)

class UnityAlcoholGenerator:
    """Gerador de dados de nível de álcool para tópico Unity_Alcohol"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações Alcohol do settings
        self.alcoholConfig = settings.signals.unityConfig["alcohol_level"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração
        self.samplingRate = self.alcoholConfig["samplingRate"]  # 1Hz
        self.normalRange = self.alcoholConfig["normalRange"]
        self.legalLimit = self.alcoholConfig["legalLimit"]      # 0.5 g/L
        self.dangerLimit = self.alcoholConfig["dangerLimit"]    # 0.8 g/L
        self.criticalLimit = self.alcoholConfig.get("criticalLimit", 1.2)  # 1.2 g/L
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["Unity_Alcohol"]  # 2%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = AlcoholAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado de álcool simulado
        self.currentAlcoholState = AlcoholState.SOBER
        self.stateStartTime = 0.0
        self.stateDuration = 0.0
        self.currentAlcoholLevel = 0.0
        
        # Parâmetros de variação gradual
        self.alcoholTrend = 0.0  # -1 (diminuindo) a +1 (aumentando)
        self.variationRate = 0.001  # g/L por segundo (variação gradual)
        
        # Probabilidades de estado (realistas)
        self.stateWeights = {
            AlcoholState.SOBER: 0.80,      # 80% - maioria do tempo sóbrio
            AlcoholState.LIGHT: 0.15,      # 15% - ligeiramente alcoolizado  
            AlcoholState.ABOVE_LEGAL: 0.04, # 4% - acima limite legal
            AlcoholState.DANGEROUS: 0.01   # 1% - perigoso (anomalia)
        }
        
        self.logger.info(f"UnityAlcoholGenerator initialized - 1Hz, legal limit: {self.legalLimit} g/L")
    
    def generateEvent(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um evento de nível de álcool (1 valor por segundo).
        
        Args:
            baseTimestamp: Timestamp base (usa interno se None)
            
        Returns:
            Dict com nível de álcool para formatação ZeroMQ
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Atualizar estado de álcool
            self._updateAlcoholState()
            
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Gerar nível de álcool baseado no estado atual
            alcoholLevel = self._generateAlcoholLevel()
            
            # Avançar contadores
            self.sampleCounter += 1
            self.currentTimestamp += 1.0  # +1 segundo
            
            result = {
                "alcohol_level": round(alcoholLevel, 3),  # 3 casas decimais
                "eventTimestamp": self.currentTimestamp - 1.0,
                "anomalyType": self.currentAnomalyType.value,
                "alcoholState": self.currentAlcoholState.value,
                "samplingRate": self.samplingRate
            }
            
            self.logger.debug(f"Generated alcohol level: {alcoholLevel:.3f} g/L, state: {self.currentAlcoholState.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating alcohol level: {e}")
            raise
    
    def _generateAlcoholLevel(self) -> float:
        """
        Gera nível de álcool baseado no estado atual e anomalias.
        
        Returns:
            Nível de álcool em g/L
        """
        
        # Obter nível base do estado atual
        baseLevel = self._getBaseLevelForState()
        
        # Aplicar variação gradual
        baseLevel += self._getGradualVariation()
        
        # Aplicar anomalias se ativas
        if self.currentAnomalyType != AlcoholAnomalyType.NORMAL:
            baseLevel = self._applyAnomalies(baseLevel)
        
        # Adicionar pequena variação aleatória (ruído)
        noise = np.random.normal(0, 0.005)  # ±5mg/L de ruído
        finalLevel = baseLevel + noise
        
        # Clipar para range válido
        finalLevel = max(0.0, min(finalLevel, 3.0))  # 0-3.0 g/L máximo
        
        # Atualizar nível atual para próxima iteração
        self.currentAlcoholLevel = finalLevel
        
        return finalLevel
    
    def _getBaseLevelForState(self) -> float:
        """
        Retorna nível base para o estado atual de álcool.
        
        Returns:
            Nível base em g/L
        """
        
        if self.currentAlcoholState == AlcoholState.SOBER:
            return np.random.uniform(0.0, 0.1)
        elif self.currentAlcoholState == AlcoholState.LIGHT:
            return np.random.uniform(0.1, 0.5)
        elif self.currentAlcoholState == AlcoholState.ABOVE_LEGAL:
            return np.random.uniform(0.5, 0.8)
        elif self.currentAlcoholState == AlcoholState.DANGEROUS:
            return np.random.uniform(0.8, 1.2)
        else:
            return 0.0
    
    def _getGradualVariation(self) -> float:
        """
        Simula variação gradual do álcool (absorção/metabolismo).
        
        Returns:
            Variação a aplicar em g/L
        """
        
        # Simular metabolismo (diminui gradualmente)
        metabolismRate = -0.0002  # -0.2 mg/L por segundo (realista)
        
        # Simular absorção ocasional (se em estado de consumo)
        absorptionRate = 0.0
        if self.currentAlcoholState in [AlcoholState.LIGHT, AlcoholState.ABOVE_LEGAL]:
            # 10% chance de absorção por segundo
            if np.random.random() < 0.1:
                absorptionRate = np.random.uniform(0.001, 0.005)
        
        return metabolismRate + absorptionRate
    
    def _applyAnomalies(self, baseLevel: float) -> float:
        """
        Aplica anomalias específicas ao nível de álcool.
        
        Args:
            baseLevel: Nível base em g/L
            
        Returns:
            Nível modificado pela anomalia
        """
        
        if self.currentAnomalyType == AlcoholAnomalyType.HIGH_ALCOHOL:
            # Acima limite legal (0.5-0.8)
            return max(baseLevel, np.random.uniform(0.5, 0.8))
            
        elif self.currentAnomalyType == AlcoholAnomalyType.DANGEROUS_ALCOHOL:
            # Perigoso (0.8-1.2)
            return max(baseLevel, np.random.uniform(0.8, 1.2))
            
        elif self.currentAnomalyType == AlcoholAnomalyType.EXTREME_ALCOHOL:
            # Extremo (>1.2)
            return max(baseLevel, np.random.uniform(1.2, 2.0))
        
        return baseLevel
    
    def _updateAlcoholState(self):
        """
        Atualiza estado de álcool baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Verificar se deve mudar estado
        if currentTime - self.stateStartTime >= self.stateDuration:
            # Escolher novo estado baseado em pesos
            states = list(self.stateWeights.keys())
            weights = list(self.stateWeights.values())
            self.currentAlcoholState = np.random.choice(states, p=weights)
            
            self.stateStartTime = currentTime
            
            # Duração do estado baseada no tipo
            if self.currentAlcoholState == AlcoholState.SOBER:
                self.stateDuration = np.random.uniform(300.0, 1800.0)  # 5-30 min
            elif self.currentAlcoholState == AlcoholState.LIGHT:
                self.stateDuration = np.random.uniform(600.0, 3600.0)  # 10-60 min
            elif self.currentAlcoholState == AlcoholState.ABOVE_LEGAL:
                self.stateDuration = np.random.uniform(300.0, 1800.0)  # 5-30 min
            elif self.currentAlcoholState == AlcoholState.DANGEROUS:
                self.stateDuration = np.random.uniform(120.0, 600.0)   # 2-10 min
            
            self.logger.debug(f"Alcohol state changed to: {self.currentAlcoholState.value} for {self.stateDuration:.1f}s")
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != AlcoholAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = AlcoholAnomalyType.NORMAL
                self.logger.debug(f"Alcohol anomaly ended at {currentTime:.1f}s")
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
                AlcoholAnomalyType.HIGH_ALCOHOL,      # 60% - acima limite legal
                AlcoholAnomalyType.DANGEROUS_ALCOHOL, # 35% - perigoso
                AlcoholAnomalyType.EXTREME_ALCOHOL    # 5% - extremo
            ]
            
            weights = [0.60, 0.35, 0.05]
            self.currentAnomalyType = np.random.choice(anomalyTypes, p=weights)
            
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia baseada no tipo
            if self.currentAnomalyType == AlcoholAnomalyType.HIGH_ALCOHOL:
                self.anomalyDuration = np.random.uniform(600.0, 1800.0)   # 10-30 min
            elif self.currentAnomalyType == AlcoholAnomalyType.DANGEROUS_ALCOHOL:
                self.anomalyDuration = np.random.uniform(300.0, 900.0)    # 5-15 min
            elif self.currentAnomalyType == AlcoholAnomalyType.EXTREME_ALCOHOL:
                self.anomalyDuration = np.random.uniform(120.0, 300.0)    # 2-5 min
            
            self.logger.warning(f"Alcohol anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 300.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("high_alcohol", "dangerous_alcohol", "extreme_alcohol")
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = AlcoholAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced alcohol anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown alcohol anomaly type: {anomalyType}")
    
    def forceAlcoholState(self, state: str, duration: float = 600.0):
        """
        Força estado de álcool específico.
        
        Args:
            state: Estado de álcool ("sober", "light", "above_legal", "dangerous")
            duration: Duração do estado em segundos
        """
        
        try:
            self.currentAlcoholState = AlcoholState(state)
            self.stateStartTime = self.currentTimestamp
            self.stateDuration = duration
            
            self.logger.info(f"Forced alcohol state: {state} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown alcohol state: {state}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "UnityAlcohol",
            "samplingRate": self.samplingRate,
            "currentTimestamp": self.currentTimestamp,
            "sampleCounter": self.sampleCounter,
            "currentAlcoholState": self.currentAlcoholState.value,
            "currentAlcoholLevel": round(self.currentAlcoholLevel, 3),
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != AlcoholAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "stateTimeRemaining": max(0, (self.stateStartTime + self.stateDuration) - self.currentTimestamp),
            "config": {
                "legalLimit": self.legalLimit,
                "dangerLimit": self.dangerLimit,
                "criticalLimit": self.criticalLimit,
                "anomalyChance": self.anomalyChance,
                "stateWeights": {state.value: weight for state, weight in self.stateWeights.items()}
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = AlcoholAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentAlcoholState = AlcoholState.SOBER
        self.stateStartTime = 0.0
        self.stateDuration = 0.0
        self.currentAlcoholLevel = 0.0
        self.alcoholTrend = 0.0
        
        self.logger.info("UnityAlcoholGenerator reset")

# Instância global
unityAlcoholGenerator = UnityAlcoholGenerator()