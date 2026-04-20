"""
PolarPpiGenerator - Gerador de eventos PPI do Polar ARM Band

Resumo:
Gera eventos PPI (Peak-to-Peak Interval) simulando o Polar ARM Band real com:
- Intervalos PPI realistas em milissegundos (300-2000ms)
- Variação natural de HR baseada em fisiologia
- Eventos discretos (não chunks contínuos como ECG)
- Anomalias específicas: bradicardia, taquicardia, arritmias
- Error_ms e flags apropriados por evento
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from app.core import settings

class PpiAnomalyType(Enum):
    """Tipos de anomalias PPI/HR"""
    NORMAL = "normal"
    BRADYCARDIA = "bradycardia"           # HR muito baixo (<60 BPM)
    TACHYCARDIA = "tachycardia"           # HR muito alto (>100 BPM)
    SEVERE_BRADYCARDIA = "severe_bradycardia"  # HR crítico (<40 BPM)
    SEVERE_TACHYCARDIA = "severe_tachycardia"  # HR crítico (>150 BPM)
    ARRHYTHMIA = "arrhythmia"             # Variação irregular
    HIGH_ERROR = "high_error"             # Erro de medição elevado
    SENSOR_ERROR = "sensor_error"         # Falha do sensor

class PolarPpiGenerator:
    """Gerador de eventos PPI para tópico Polar_PPI"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações PPI do settings
        self.ppiConfig = settings.signals.cardiacConfig["ppi"]
        self.hrConfig = settings.signals.cardiacConfig["hr"]
        self.zmqConfig = settings.zeromq.topicProcessingConfig["Polar_PPI"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração HR/PPI
        self.frequency = self.mockConfig.topicFrequencies["Polar_PPI"]  # 1Hz eventos
        self.eventInterval = 1.0 / self.frequency                      # 1.0s entre checks
        
        # Configurações de HR base
        self.generatorConfig = self.mockConfig.generatorBaseConfig["cardiac"]
        self.baseHr = self.generatorConfig["baseHr"]                   # 75 BPM
        self.hrVariationStd = self.generatorConfig["hrVariationStd"]   # ±5 BPM
        
        # Ranges de validação
        self.validPpiRange = self.zmqConfig["validPpiRange"]           # (300, 2000) ms
        self.normalHrRange = self.hrConfig["normalRange"]              # (60, 100) BPM
        self.criticalHrRange = self.hrConfig["criticalRange"]          # (30, 200) BPM
        
        # Thresholds de anomalias
        self.bradycardiaThreshold = self.hrConfig["bradycardiaThreshold"]         # 60 BPM
        self.tachycardiaThreshold = self.hrConfig["tachycardiaThreshold"]         # 100 BPM
        self.severeBradycardiaThreshold = self.hrConfig["severeBradycardiaThreshold"]  # 40 BPM
        self.severeTachycardiaThreshold = self.hrConfig["severeTachycardiaThreshold"]  # 150 BPM
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["Polar_PPI"]  # 5%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.eventCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = PpiAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado fisiológico
        self.currentHr = self.baseHr
        self.lastPpi = self._hrToPpi(self.baseHr)
        self.hrTrend = 0.0  # Tendência de subida/descida gradual
        
        # Configurações de erro do sensor
        self.baseErrorMs = 10                                          # Erro base em ms
        self.maxErrorMs = 100                                          # Erro máximo
        
        self.logger.info(f"PolarPpiGenerator initialized - {self.frequency}Hz events, base HR: {self.baseHr} BPM")
    
    def generateEvent(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um evento PPI individual.
        
        Args:
            baseTimestamp: Timestamp base para o evento (usa interno se None)
            
        Returns:
            Dict com dados PPI, error_ms e flags para formatação
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Gerar HR baseado no estado atual
            hr = self._generateHeartRate()
            
            # Converter HR para PPI
            ppi = self._hrToPpi(hr)
            
            # Adicionar variação natural pequena
            ppiVariation = np.random.normal(0, 20)  # ±20ms variação natural
            ppi += ppiVariation
            
            # Garantir que PPI está no range válido
            ppi = np.clip(ppi, self.validPpiRange[0], self.validPpiRange[1])
            
            # Gerar erro de medição e flags
            errorMs, flags = self._generateErrorAndFlags()
            
            # Avançar contadores
            self.eventCounter += 1
            self.currentTimestamp += self.eventInterval
            self.lastPpi = ppi
            
            result = {
                "ppi": int(round(ppi)),     # PPI em ms (inteiro)
                "error_ms": int(errorMs),   # Erro em ms
                "flags": int(flags),        # Flags de estado
                "eventTimestamp": self.currentTimestamp - self.eventInterval,
                "calculatedHr": round(60000.0 / ppi, 1),  # HR calculado para debug
                "anomalyType": self.currentAnomalyType.value
            }
            
            self.logger.debug(f"Generated PPI event: {ppi}ms ({result['calculatedHr']} BPM), anomaly: {self.currentAnomalyType.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating PPI event: {e}")
            raise
    
    def _generateHeartRate(self) -> float:
        """
        Gera HR baseado no estado de anomalia atual.
        
        Returns:
            HR em BPM
        """
        
        if self.currentAnomalyType == PpiAnomalyType.NORMAL:
            # HR normal com variação gradual e natural
            hr = self._generateNormalHr()
            
        elif self.currentAnomalyType == PpiAnomalyType.BRADYCARDIA:
            # Bradicardia moderada (50-59 BPM)
            hr = np.random.uniform(50, 59)
            
        elif self.currentAnomalyType == PpiAnomalyType.TACHYCARDIA:
            # Taquicardia moderada (101-130 BPM)
            hr = np.random.uniform(101, 130)
            
        elif self.currentAnomalyType == PpiAnomalyType.SEVERE_BRADYCARDIA:
            # Bradicardia severa (35-45 BPM)
            hr = np.random.uniform(35, 45)
            
        elif self.currentAnomalyType == PpiAnomalyType.SEVERE_TACHYCARDIA:
            # Taquicardia severa (150-180 BPM)
            hr = np.random.uniform(150, 180)
            
        elif self.currentAnomalyType == PpiAnomalyType.ARRHYTHMIA:
            # Arritmia com variação irregular
            baseHr = self._generateNormalHr()
            irregularVariation = np.random.uniform(-30, 30)  # Variação extrema
            hr = baseHr + irregularVariation
            
        elif self.currentAnomalyType == PpiAnomalyType.SENSOR_ERROR:
            # Valores inconsistentes por falha do sensor
            hr = np.random.choice([
                np.random.uniform(200, 250),  # Valores impossíveis
                np.random.uniform(20, 30),    # Valores impossíveis baixos
                self.currentHr                # Ou valor travado
            ])
            
        else:
            hr = self._generateNormalHr()
        
        # Clipar para range crítico para segurança
        hr = np.clip(hr, self.criticalHrRange[0], self.criticalHrRange[1])
        
        self.currentHr = hr
        return hr
    
    def _generateNormalHr(self) -> float:
        """
        Gera HR normal com tendências graduais realistas.
        
        Returns:
            HR normal em BPM
        """
        
        # Atualizar tendência gradual (-1 a +1 BPM por evento)
        trendChange = np.random.normal(0, 0.1)
        self.hrTrend += trendChange
        self.hrTrend = np.clip(self.hrTrend, -2.0, 2.0)  # Limitar tendência
        
        # HR base com tendência e variação natural
        targetHr = self.baseHr + self.hrTrend
        hrVariation = np.random.normal(0, self.hrVariationStd * 0.3)  # Variação suave
        
        # Suavizar mudanças (não saltar muito entre eventos)
        alpha = 0.8  # Factor de suavização
        newHr = alpha * self.currentHr + (1 - alpha) * (targetHr + hrVariation)
        
        # Manter no range normal na maioria das vezes
        return np.clip(newHr, self.normalHrRange[0], self.normalHrRange[1])
    
    def _hrToPpi(self, hr: float) -> float:
        """
        Converte HR (BPM) para PPI (ms).
        
        Args:
            hr: Heart Rate em BPM
            
        Returns:
            PPI em milissegundos
        """
        
        if hr <= 0:
            return self.validPpiRange[1]  # PPI máximo para HR inválido
        
        return 60000.0 / hr  # 60 segundos * 1000 ms / BPM
    
    def _generateErrorAndFlags(self) -> tuple[float, int]:
        """
        Gera erro de medição e flags baseados no estado atual.
        
        Returns:
            Tupla (error_ms, flags)
        """
        
        if self.currentAnomalyType == PpiAnomalyType.HIGH_ERROR:
            # Erro de medição elevado
            errorMs = np.random.uniform(50, self.maxErrorMs)
            flags = 1  # Flag de erro de medição
            
        elif self.currentAnomalyType == PpiAnomalyType.SENSOR_ERROR:
            # Erro crítico do sensor
            errorMs = self.maxErrorMs
            flags = 3  # Flags combinados (erro + sensor)
            
        elif self.currentAnomalyType in [PpiAnomalyType.SEVERE_BRADYCARDIA, PpiAnomalyType.SEVERE_TACHYCARDIA]:
            # Erro ligeiramente elevado em condições extremas
            errorMs = np.random.uniform(15, 30)
            flags = 0
            
        else:
            # Erro normal
            errorMs = np.random.uniform(5, self.baseErrorMs)
            flags = 0
        
        return errorMs, flags
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != PpiAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = PpiAnomalyType.NORMAL
                self.logger.debug(f"PPI anomaly ended at {currentTime:.3f}s")
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
                PpiAnomalyType.BRADYCARDIA,        # 25% - Comum
                PpiAnomalyType.TACHYCARDIA,        # 25% - Comum
                PpiAnomalyType.ARRHYTHMIA,         # 20% - Moderadamente comum
                PpiAnomalyType.SEVERE_BRADYCARDIA, # 10% - Menos comum
                PpiAnomalyType.SEVERE_TACHYCARDIA, # 10% - Menos comum
                PpiAnomalyType.HIGH_ERROR,         # 8% - Raro
                PpiAnomalyType.SENSOR_ERROR        # 2% - Muito raro
            ]
            
            weights = [0.25, 0.25, 0.20, 0.10, 0.10, 0.08, 0.02]
            self.currentAnomalyType = np.random.choice(anomalyTypes, p=weights)
            
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia baseada no tipo
            if self.currentAnomalyType in [PpiAnomalyType.SEVERE_BRADYCARDIA, PpiAnomalyType.SEVERE_TACHYCARDIA]:
                self.anomalyDuration = np.random.uniform(5.0, 15.0)   # Mais curta para severas
            elif self.currentAnomalyType == PpiAnomalyType.SENSOR_ERROR:
                self.anomalyDuration = np.random.uniform(2.0, 8.0)    # Curta para erro sensor
            else:
                self.anomalyDuration = np.random.uniform(10.0, 30.0)  # Normal para outras
            
            self.logger.warning(f"PPI anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 10.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("bradycardia", "tachycardia", etc.)
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = PpiAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced PPI anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown PPI anomaly type: {anomalyType}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "PolarPPI",
            "frequency": self.frequency,
            "currentTimestamp": self.currentTimestamp,
            "eventCounter": self.eventCounter,
            "currentHr": self.currentHr,
            "lastPpi": self.lastPpi,
            "hrTrend": self.hrTrend,
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != PpiAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "config": {
                "baseHr": self.baseHr,
                "hrVariationStd": self.hrVariationStd,
                "anomalyChance": self.anomalyChance,
                "validPpiRange": self.validPpiRange,
                "normalHrRange": self.normalHrRange
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.eventCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = PpiAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentHr = self.baseHr
        self.lastPpi = self._hrToPpi(self.baseHr)
        self.hrTrend = 0.0
        
        self.logger.info("PolarPpiGenerator reset")

# Instância global
polarPpiGenerator = PolarPpiGenerator()