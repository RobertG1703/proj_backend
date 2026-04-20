"""
UnitySignal - Implementação para dados Unity (álcool + informação do carro)

Resumo:
Processa dados de condução vindos do Unity incluindo nível de álcool no sangue,
velocidade do veículo e centralidade na faixa de rodagem. Recebe dois tipos de
dados: alcohol_level (g/L) e car_information (speed em km/h + lane_centrality 0-1).
Detecta anomalias relacionadas com condução sob influência do álcool e padrões
de condução perigosos.

Para detecção de anomalias, identifica níveis de álcool acima do limite legal (>0.5 g/L),
velocidades excessivas (>100 km/h), saída de faixa (<0.4 centralidade), combinação
perigosa de álcool elevado com condução instável, e mudanças súbitas nos padrões.

Os métodos principais incluem getLatestAlcoholLevel() para nível atual de álcool,
getLatestCarInfo() para velocidade e centralidade mais recentes, getAlcoholStats()
para estatísticas de álcool, getSpeedStats() para análise de velocidade, 
getDrivingQuality() que avalia qualidade geral da condução baseada na combinação
de todos os fatores, e getUnityStatus() que fornece resumo completo do estado.
"""

import numpy as np
from typing import List, Optional, Any, Dict, Union
from datetime import datetime

from ..base import BaseSignal
from ..dataPoint import SignalPoint
from ...core import settings
from ...core.exceptions import SignalValidationError

class UnitySignal(BaseSignal):
    """Sinal Unity para dados de condução - álcool + informação do carro"""
    
    def __init__(self):
        # Configuração Unity: 30 pontos = 30s * 1Hz
        unityConfig = settings.signals.unityConfig
        
        super().__init__(
            signalName="unity",
            bufferSize=30,  # 30s * 1Hz
            samplingRate=1  # 1Hz
        )
        
        # Configurações específicas
        self.alcoholConfig = unityConfig["alcohol_level"]
        self.carConfig = unityConfig["car_information"]
        
        # Thresholds de álcool
        self.legalLimit = self.alcoholConfig["legalLimit"]        # 0.5 g/L
        self.dangerLimit = self.alcoholConfig["dangerLimit"]      # 0.8 g/L
        self.detectionThreshold = self.alcoholConfig["detectionThreshold"]  # 0.1 g/L
        
        # Thresholds de velocidade
        self.speedConfig = self.carConfig["speed"]
        self.speedingThreshold = self.speedConfig["speedingThreshold"]      # 100 km/h
        self.dangerSpeedThreshold = self.speedConfig["dangerSpeedThreshold"] # 150 km/h
        self.suddenChangeThreshold = self.speedConfig["suddenChangeThreshold"] # 20 km/h
        
        # Thresholds de centralidade
        self.centralityConfig = self.carConfig["lane_centrality"] 
        self.warningThreshold = self.centralityConfig["warningThreshold"]    # 0.4
        self.dangerThreshold = self.centralityConfig["dangerThreshold"]      # 0.2
        self.stabilityThreshold = self.centralityConfig["stabilityThreshold"] # 0.1
        
        self.logger.info(f"UnitySignal initialized - Legal limit: {self.legalLimit} g/L, Speed limit: {self.speedingThreshold} km/h")
    
    def validateValue(self, value: Any) -> bool:
        """Valida valores de álcool ou informação do carro"""
        
        # Alcohol level data
        if isinstance(value, dict) and "alcohol_level" in value:
            try:
                alcoholLevel = value["alcohol_level"]
                
                if not isinstance(alcoholLevel, (int, float)):
                    raise SignalValidationError(
                        signalType="unity",
                        value=str(alcoholLevel),
                        reason="Alcohol level must be numeric"
                    )
                
                # Verificar range usando configurações
                alcoholRange = self.alcoholConfig["normalRange"]
                if not (alcoholRange[0] <= alcoholLevel <= alcoholRange[1]):  # Range absoluto
                    raise SignalValidationError(
                        signalType="unity",
                        value=f"{alcoholLevel} g/L",
                        reason=f"Alcohol level fora do range absoluto [{alcoholRange[0]}, {alcoholRange[1]}] g/L"
                    )
                
                return True
                
            except SignalValidationError:
                raise
            except Exception as e:
                raise SignalValidationError(
                    signalType="unity",
                    value=str(value)[:100],
                    reason=f"Erro ao validar alcohol level: {e}"
                )
        
        # Car information data
        elif isinstance(value, dict) and "car_information" in value:
            try:
                carInfo = value["car_information"]
                
                # Verificar se tem speed e lane_centrality
                if "speed" not in carInfo or "lane_centrality" not in carInfo:
                    raise SignalValidationError(
                        signalType="unity",
                        value="Missing fields",
                        reason="Car information deve ter 'speed' e 'lane_centrality'"
                    )
                
                speed = carInfo["speed"]
                laneCentrality = carInfo["lane_centrality"]
                
                # Validar tipos
                if not isinstance(speed, (int, float)) or not isinstance(laneCentrality, (int, float)):
                    raise SignalValidationError(
                        signalType="unity",
                        value=f"speed={speed}, lane_centrality={laneCentrality}",
                        reason="Speed e lane_centrality devem ser numéricos"
                    )
                
                # Verificar ranges usando configurações
                speedRange = self.speedConfig["normalRange"]
                if not (speedRange[0] <= speed <= speedRange[1]):  # Range absoluto
                    raise SignalValidationError(
                        signalType="unity",
                        value=f"{speed} km/h",
                        reason=f"Speed fora do range absoluto [{speedRange[0]}, {speedRange[1]}] km/h"
                    )
                
                centralityRange = self.centralityConfig["normalRange"]
                if not (centralityRange[0] <= laneCentrality <= centralityRange[1]):  # Range absoluto
                    raise SignalValidationError(
                        signalType="unity",
                        value=f"{laneCentrality}",
                        reason=f"Lane centrality fora do range [{centralityRange[0]}, {centralityRange[1]}]"
                    )
                
                return True
                
            except SignalValidationError:
                raise
            except Exception as e:
                raise SignalValidationError(
                    signalType="unity",
                    value=str(value)[:100],
                    reason=f"Erro ao validar car information: {e}"
                )
        
        # Tipo não reconhecido
        raise SignalValidationError(
            signalType="unity",
            value=type(value).__name__,
            reason="Valor deve ser dict com 'alcohol_level' ou 'car_information'"
        )
    
    def getNormalRange(self) -> tuple:
        """Range normal para Unity (retorna range do álcool)"""
        return self.alcoholConfig["normalRange"]
    
    def detectAnomalies(self, recentPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias em dados Unity"""
        anomalies = []
        
        if len(recentPoints) < 1:
            return anomalies
        
        # Separar dados por tipo
        alcoholPoints = [point for point in recentPoints if isinstance(point.value, dict) and "alcohol_level" in point.value]
        carPoints = [point for point in recentPoints if isinstance(point.value, dict) and "car_information" in point.value]
        
        # Anomalias de álcool
        if alcoholPoints:
            anomalies.extend(self._detectAlcoholAnomalies(alcoholPoints))
        
        # Anomalias de condução
        if carPoints:
            anomalies.extend(self._detectCarAnomalies(carPoints))
        
        # Anomalias combinadas (álcool + condução)
        if alcoholPoints and carPoints:
            anomalies.extend(self._detectCombinedAnomalies(alcoholPoints, carPoints))
        
        return anomalies
    
    def _detectAlcoholAnomalies(self, alcoholPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias no nível de álcool"""
        anomalies = []
        
        if not alcoholPoints:
            return anomalies
        
        latestAlcohol = alcoholPoints[-1].value["alcohol_level"]
        
        # Álcool acima do limite legal
        if latestAlcohol > self.legalLimit:
            if latestAlcohol > self.dangerLimit:
                anomalies.append(f"Nível de álcool perigoso: {latestAlcohol:.3f} g/L (limite: {self.dangerLimit})")
            else:
                anomalies.append(f"Álcool acima do limite legal: {latestAlcohol:.3f} g/L (limite: {self.legalLimit})")
        
        # Verificar aumento súbito de álcool
        if len(alcoholPoints) >= 5:
            recentLevels = [point.value["alcohol_level"] for point in alcoholPoints[-5:]]
            if len(recentLevels) >= 2:
                increase = max(recentLevels) - min(recentLevels)
                if increase > 0.3:  # Aumento >0.3 g/L em 5 segundos
                    anomalies.append(f"Aumento súbito de álcool: +{increase:.2f} g/L em poucos segundos")
        
        return anomalies
    
    def _detectCarAnomalies(self, carPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias na condução"""
        anomalies = []
        
        if not carPoints:
            return anomalies
        
        latestCar = carPoints[-1].value["car_information"]
        speed = latestCar["speed"]
        laneCentrality = latestCar["lane_centrality"]
        
        # Excesso de velocidade
        if speed > self.speedingThreshold:
            if speed > self.dangerSpeedThreshold:
                anomalies.append(f"Velocidade muito perigosa: {speed:.1f} km/h (limite: {self.dangerSpeedThreshold})")
            else:
                anomalies.append(f"Excesso de velocidade: {speed:.1f} km/h (limite: {self.speedingThreshold})")
        
        # Saída de faixa
        if laneCentrality < self.warningThreshold:
            if laneCentrality < self.dangerThreshold:
                anomalies.append(f"Fora da faixa de rodagem: centralidade {laneCentrality:.2f} (mínimo: {self.dangerThreshold})")
            else:
                anomalies.append(f"Próximo da saída de faixa: centralidade {laneCentrality:.2f} (aviso: {self.warningThreshold})")
        
        # Verificar mudanças súbitas de velocidade
        if len(carPoints) >= 3:
            recentSpeeds = [point.value["car_information"]["speed"] for point in carPoints[-3:]]
            speedChanges = [abs(recentSpeeds[i] - recentSpeeds[i-1]) for i in range(1, len(recentSpeeds))]
            maxChange = max(speedChanges) if speedChanges else 0
            
            if maxChange > self.suddenChangeThreshold:
                anomalies.append(f"Mudança súbita de velocidade: {maxChange:.1f} km/h/s")
        
        # Verificar instabilidade de centralidade
        if len(carPoints) >= 10:
            recentCentralities = [point.value["car_information"]["lane_centrality"] for point in carPoints[-10:]]
            centralityStd = np.std(recentCentralities)
            
            if centralityStd > self.stabilityThreshold * 2:  # 2x o threshold
                anomalies.append(f"Condução instável: variação de centralidade {centralityStd:.3f}")
        
        return anomalies
    
    def _detectCombinedAnomalies(self, alcoholPoints: List[SignalPoint], carPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias combinadas álcool + condução"""
        anomalies = []
        
        if not alcoholPoints or not carPoints:
            return anomalies
        
        # Pegar dados mais recentes
        latestAlcohol = alcoholPoints[-1].value["alcohol_level"]
        latestCar = carPoints[-1].value["car_information"]
        speed = latestCar["speed"]
        laneCentrality = latestCar["lane_centrality"]
        
        # Condução sob influência do álcool
        if latestAlcohol > self.detectionThreshold:  # Qualquer álcool detectável
            if speed > self.speedingThreshold:
                anomalies.append(f"Condução perigosa: álcool {latestAlcohol:.3f} g/L + velocidade {speed:.1f} km/h")
            
            if laneCentrality < self.warningThreshold:
                anomalies.append(f"Condução instável: álcool {latestAlcohol:.3f} g/L + centralidade baixa {laneCentrality:.2f}")
        
        # Correlação perigosa: álcool alto + condução muito instável
        if latestAlcohol > self.legalLimit and laneCentrality < self.dangerThreshold:
            anomalies.append(f"PERIGO CRÍTICO: álcool {latestAlcohol:.3f} g/L + fora da faixa (centralidade {laneCentrality:.2f})")
        
        return anomalies
    
    # Métodos específicos para UnitySignal
    
    def getLatestAlcoholLevel(self) -> Optional[Dict[str, Any]]:
        """Retorna nível de álcool mais recente"""
        allPoints = self.getAllData()
        
        for point in reversed(allPoints):
            if isinstance(point.value, dict) and "alcohol_level" in point.value:
                return {
                    "timestamp": point.timestamp,
                    "alcohol_level": point.value["alcohol_level"],
                    "units": "g/L",
                    "legalLimit": self.legalLimit,
                    "dangerLimit": self.dangerLimit,
                    "isLegal": point.value["alcohol_level"] <= self.legalLimit
                }
        
        return None
    
    def getLatestCarInfo(self) -> Optional[Dict[str, Any]]:
        """Retorna informação do carro mais recente"""
        allPoints = self.getAllData()
        
        for point in reversed(allPoints):
            if isinstance(point.value, dict) and "car_information" in point.value:
                carInfo = point.value["car_information"]
                return {
                    "timestamp": point.timestamp,
                    "speed": carInfo["speed"],
                    "lane_centrality": carInfo["lane_centrality"],
                    "speedUnits": "km/h",
                    "speedingThreshold": self.speedingThreshold,
                    "centralityWarning": self.warningThreshold,
                    "isSpeeding": carInfo["speed"] > self.speedingThreshold,
                    "isStable": carInfo["lane_centrality"] >= self.warningThreshold
                }
        
        return None
    
    def getAlcoholStats(self, durationSeconds: float = 30.0) -> Optional[Dict[str, Any]]:
        """Calcula estatísticas de álcool dos últimos X segundos"""
        allPoints = self.getAllData()
        
        # Filtrar pontos de álcool recentes
        cutoffTime = datetime.now().timestamp() - durationSeconds
        alcoholPoints = []
        
        for point in allPoints:
            if (point.timestamp >= cutoffTime and 
                isinstance(point.value, dict) and 
                "alcohol_level" in point.value):
                alcoholPoints.append(point.value["alcohol_level"])
        
        if len(alcoholPoints) < 2:
            return None
        
        alcoholArray = np.array(alcoholPoints)
        
        return {
            "duration": durationSeconds,
            "sampleCount": len(alcoholPoints),
            "mean": float(np.mean(alcoholArray)),
            "std": float(np.std(alcoholArray)),
            "min": float(np.min(alcoholArray)),
            "max": float(np.max(alcoholArray)),
            "current": alcoholPoints[-1],
            "trend": "increasing" if len(alcoholPoints) >= 2 and alcoholPoints[-1] > alcoholPoints[-2] else "stable_or_decreasing",
            "timesAboveLegal": sum(1 for level in alcoholPoints if level > self.legalLimit),
            "percentageAboveLegal": (sum(1 for level in alcoholPoints if level > self.legalLimit) / len(alcoholPoints)) * 100,
            "units": "g/L"
        }
    
    def getSpeedStats(self, durationSeconds: float = 30.0) -> Optional[Dict[str, Any]]:
        """Calcula estatísticas de velocidade dos últimos X segundos"""
        allPoints = self.getAllData()
        
        # Filtrar pontos de velocidade recentes
        cutoffTime = datetime.now().timestamp() - durationSeconds
        speedPoints = []
        
        for point in allPoints:
            if (point.timestamp >= cutoffTime and 
                isinstance(point.value, dict) and 
                "car_information" in point.value):
                speedPoints.append(point.value["car_information"]["speed"])
        
        if len(speedPoints) < 2:
            return None
        
        speedArray = np.array(speedPoints)
        
        return {
            "duration": durationSeconds,
            "sampleCount": len(speedPoints),
            "mean": float(np.mean(speedArray)),
            "std": float(np.std(speedArray)),
            "min": float(np.min(speedArray)),
            "max": float(np.max(speedArray)),
            "current": speedPoints[-1],
            "timesSpeeding": sum(1 for speed in speedPoints if speed > self.speedingThreshold),
            "percentageSpeeding": (sum(1 for speed in speedPoints if speed > self.speedingThreshold) / len(speedPoints)) * 100,
            "avgSpeedChange": float(np.mean([abs(speedPoints[i] - speedPoints[i-1]) for i in range(1, len(speedPoints))])),
            "units": "km/h"
        }
    
    def getDrivingQuality(self) -> Dict[str, Any]:
        """Avalia qualidade geral da condução baseada em álcool + condução"""
        alcoholData = self.getLatestAlcoholLevel()
        carData = self.getLatestCarInfo()
        
        if not alcoholData or not carData:
            return {"quality": "unknown", "reason": "Insufficient data"}
        
        # Calcular score de qualidade (0-100)
        qualityScore = 100
        issues = []
        
        # Penalizar por álcool
        alcoholLevel = alcoholData["alcohol_level"]
        if alcoholLevel > self.dangerLimit:
            qualityScore -= 60
            issues.append(f"Álcool perigoso ({alcoholLevel:.3f} g/L)")
        elif alcoholLevel > self.legalLimit:
            qualityScore -= 30
            issues.append(f"Álcool acima limite legal ({alcoholLevel:.3f} g/L)")
        elif alcoholLevel > self.detectionThreshold:
            qualityScore -= 10
            issues.append(f"Álcool detetado ({alcoholLevel:.3f} g/L)")
        
        # Penalizar por velocidade
        speed = carData["speed"]
        if speed > self.dangerSpeedThreshold:
            qualityScore -= 40
            issues.append(f"Velocidade muito alta ({speed:.1f} km/h)")
        elif speed > self.speedingThreshold:
            qualityScore -= 20
            issues.append(f"Excesso velocidade ({speed:.1f} km/h)")
        
        # Penalizar por centralidade
        centrality = carData["lane_centrality"]
        if centrality < self.dangerThreshold:
            qualityScore -= 50
            issues.append(f"Fora da faixa (centralidade {centrality:.2f})")
        elif centrality < self.warningThreshold:
            qualityScore -= 15
            issues.append(f"Próximo saída faixa (centralidade {centrality:.2f})")
        
        # Classificar qualidade
        qualityScore = max(0, qualityScore)
        
        if qualityScore >= 80:
            quality = "excellent"
        elif qualityScore >= 60:
            quality = "good"
        elif qualityScore >= 40:
            quality = "moderate"
        elif qualityScore >= 20:
            quality = "poor"
        else:
            quality = "critical"
        
        return {
            "quality": quality,
            "score": qualityScore,
            "issues": issues,
            "alcoholLevel": alcoholLevel,
            "speed": speed,
            "laneCentrality": centrality,
            "isLegalDriving": alcoholLevel <= self.legalLimit and speed <= self.speedingThreshold and centrality >= self.warningThreshold,
            "timestamp": datetime.now().isoformat()
        }
    
    def getUnityStatus(self) -> Dict[str, Any]:
        """Status geral dos dados Unity"""
        baseStatus = self.getStatus()
        
        # Informações específicas Unity
        latestAlcohol = self.getLatestAlcoholLevel()
        latestCar = self.getLatestCarInfo()
        drivingQuality = self.getDrivingQuality()
        
        # Estatísticas por tipo
        alcoholStats = self.getAlcoholStats(30.0)
        speedStats = self.getSpeedStats(30.0)
        
        unityStatus = {
            **baseStatus,
            "latestAlcoholLevel": latestAlcohol,
            "latestCarInfo": latestCar,
            "drivingQuality": drivingQuality,
            "alcoholStats": alcoholStats,
            "speedStats": speedStats,
            "dataAvailable": {
                "alcohol": latestAlcohol is not None,
                "carInfo": latestCar is not None
            },
            "configuration": {
                "alcoholLegalLimit": self.legalLimit,
                "alcoholDangerLimit": self.dangerLimit,
                "speedingThreshold": self.speedingThreshold,
                "dangerSpeedThreshold": self.dangerSpeedThreshold,
                "centralityWarningThreshold": self.warningThreshold,
                "centralityDangerThreshold": self.dangerThreshold,
                "bufferDuration": f"{self.bufferSize}s"
            }
        }
        
        return unityStatus