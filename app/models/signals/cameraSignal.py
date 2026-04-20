"""
CameraSignal - Validação e processamento de dados de câmera

Resumo:
Processa dados de face landmarks (478 pontos), gaze tracking, Eye Aspect Ratio
e blink detection provenientes de sistemas como MediaPipe. Valida estrutura dos dados,
detecta anomalias relacionadas com sonolência e distração, e mantém estatísticas
de qualidade de deteção seguindo o padrão da arquitetura de sinais.
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple, Union

from ..base import BaseSignal
from ..dataPoint import SignalPoint
from app.core import settings, SignalValidationError

class CameraSignal(BaseSignal):
    """Signal para dados de face landmarks, gaze tracking e blink detection"""
    
    def __init__(self):
        # Configurações baseadas nos settings
        cameraConfig = settings.signals.cameraConfig
        
        super().__init__(
            signalName="camera",
            bufferSize=cameraConfig["faceLandmarks"]["bufferSize"],    # 15 frames (30s * 0.5Hz)
            samplingRate=cameraConfig["faceLandmarks"]["fps"]          # 0.5Hz
        )
        
        # Configurações específicas de câmera
        self.landmarksConfig = cameraConfig["faceLandmarks"]
        self.gazeConfig = cameraConfig["gaze"] 
        self.blinkConfig = cameraConfig["blink_rate"]
        self.earConfig = cameraConfig["ear"]
        
        # Parâmetros de validação
        self.expectedLandmarksCount = self.landmarksConfig["landmarksCount"]      # 478
        self.landmarksDimensions = self.landmarksConfig["landmarksDimensions"]    # 3 (x,y,z)
        
        # Ranges e thresholds de validação
        self.gazeNormalRange = self.gazeConfig["normalRange"]                    # (-1.0, 1.0)
        self.blinkRateNormalRange = self.blinkConfig["normalRange"]              # (10, 30)
        self.earNormalRange = self.earConfig["normalRange"]                      # (0.15, 0.4)
        
        # Thresholds de anomalias
        self.drowsinessBlinkThreshold = self.blinkConfig["drowsinessThreshold"]  # 8 blinks/min
        self.hyperBlinkThreshold = self.blinkConfig["hyperBlinkThreshold"]       # 40 blinks/min
        self.earBlinkThreshold = self.earConfig["blinkThreshold"]                # 0.12
        self.earDrowsyThreshold = self.earConfig["drowsyThreshold"]              # 0.18
        self.gazeStabilityThreshold = self.gazeConfig["stabilityThreshold"]      # 0.1
        
        # Estado interno para análise de tendências
        self.lastEar: Optional[float] = None
        self.lastBlinkRate: Optional[float] = None
        self.lastGazeVector: Optional[Dict[str, float]] = None
        self.recentEarValues: List[float] = []
        self.consecutiveLowEarCount = 0
        
        self.logger.info(f"CameraSignal initialized - {self.expectedLandmarksCount} landmarks")
    
    def getNormalRange(self):
        return super().getNormalRange()

    def validateValue(self, value: Any) -> bool:
        """
        Valida estrutura completa dos dados de câmera.
        
        Args:
            value: Dict com dados de câmera processados
            
        Returns:
            True se dados são válidos
            
        Raises:
            SignalValidationError: Se validação falhar
        """
        
        if not isinstance(value, dict):
            raise SignalValidationError(
                signalType="camera",
                value=type(value).__name__,
                reason="Camera data must be a dictionary"
            )
        
        # Verificar campos obrigatórios
        requiredFields = ["landmarks", "gaze_vector", "ear", "blink_rate", "blink_counter"]
        for field in requiredFields:
            if field not in value:
                raise SignalValidationError(
                    signalType="camera",
                    value=f"missing_{field}",
                    reason=f"Required field '{field}' missing in camera data"
                )
        
        # Validar landmarks
        landmarks = value["landmarks"]
        if not isinstance(landmarks, (list, np.ndarray)):
            raise SignalValidationError(
                signalType="camera",
                value=type(landmarks).__name__,
                reason="Landmarks must be a list or numpy array"
            )
        
        landmarksArray = np.array(landmarks)
        if landmarksArray.shape != (self.expectedLandmarksCount, self.landmarksDimensions):
            raise SignalValidationError(
                signalType="camera",
                value=f"shape_{landmarksArray.shape}",
                reason=f"Landmarks must have shape ({self.expectedLandmarksCount}, {self.landmarksDimensions})"
            )
        
        # Validar coordenadas normalizadas (0-1)
        if not np.all((landmarksArray >= 0.0) & (landmarksArray <= 1.0)):
            raise SignalValidationError(
                signalType="camera",
                value="coordinates_out_of_range",
                reason="Landmark coordinates must be normalized between 0.0 and 1.0"
            )
        
        # Validar gaze vector
        gazeVector = value["gaze_vector"]
        if not isinstance(gazeVector, dict) or "dx" not in gazeVector or "dy" not in gazeVector:
            raise SignalValidationError(
                signalType="camera",
                value="invalid_gaze_structure",
                reason="Gaze vector must be dict with 'dx' and 'dy' keys"
            )
        
        dx, dy = gazeVector["dx"], gazeVector["dy"]
        if not (self.gazeNormalRange[0] <= dx <= self.gazeNormalRange[1] and
                self.gazeNormalRange[0] <= dy <= self.gazeNormalRange[1]):
            raise SignalValidationError(
                signalType="camera",
                value=f"gaze({dx:.2f},{dy:.2f})",
                reason=f"Gaze vector outside normal range {self.gazeNormalRange}"
            )
        
        # Validar EAR
        ear = value["ear"]
        if not isinstance(ear, (int, float)) or not (0.0 <= ear <= 1.0):
            raise SignalValidationError(
                signalType="camera",
                value=ear,
                reason="EAR must be a number between 0.0 and 1.0"
            )
        
        # Validar blink rate
        blinkRate = value["blink_rate"]
        if not isinstance(blinkRate, (int, float)) or not (0 <= blinkRate <= 120):
            raise SignalValidationError(
                signalType="camera",
                value=blinkRate,
                reason="Blink rate must be between 0 and 120 bpm"
            )
        
        blinkCounter = value["blink_counter"]
        if not isinstance(blinkCounter, int) or blinkCounter < 0:
            raise SignalValidationError(
                signalType="camera",
                value=blinkCounter,
                reason="Blink counter must be a non-negative integer"
            )
                
        return True
    
    def detectAnomalies(self, recentPoints: List[SignalPoint]) -> List[str]:
        """
        Detecta anomalias nos dados de câmera recentes.
        
        Args:
            recentPoints: Lista de pontos recentes do signal
            
        Returns:
            Lista de mensagens de anomalias detectadas
        """
        anomalies = []
        
        if len(recentPoints) < 1:
            return anomalies
        
        # Analisar ponto mais recente
        latestPoint = recentPoints[-1]
        if not isinstance(latestPoint.value, dict):
            return anomalies
        
        data = latestPoint.value
        
        # Extrair valores principais
        ear = data.get("ear")
        blinkRate = data.get("blink_rate")
        gazeVector = data.get("gaze_vector", {})
        
        # Anomalia: Taxa de piscadelas baixa (sonolência)
        if blinkRate is not None and blinkRate < self.drowsinessBlinkThreshold:
            severity = "crítica" if blinkRate < 5 else "moderada"
            anomalies.append(f"Taxa de piscadelas baixa detectada: {blinkRate:.1f} bpm (sonolência {severity})")
        
        # Anomalia: Taxa de piscadelas muito alta (stress/irritação)
        if blinkRate is not None and blinkRate > self.hyperBlinkThreshold:
            severity = "alta" if blinkRate > self.hyperBlinkThreshold * 1.5 else "moderada"
            anomalies.append(f"Taxa de piscadelas excessiva: {blinkRate:.1f} bpm (stress {severity})")
        
        # Anomalia: EAR baixo prolongado (olhos fechados/sonolência)
        if ear is not None and ear < self.earDrowsyThreshold:
            # Atualizar contador de EAR baixo consecutivo
            self.consecutiveLowEarCount += 1
            if self.consecutiveLowEarCount >= 3:  # 3 leituras consecutivas (6 segundos)
                severity = "crítica" if ear < 0.1 else "moderada"
                anomalies.append(f"EAR baixo prolongado: {ear:.3f} (sonolência {severity})")
        else:
            self.consecutiveLowEarCount = 0
        
        # Anomalia: Olhar muito desviado (distração)
        if gazeVector:
            dx = gazeVector.get("dx", 0)
            dy = gazeVector.get("dy", 0)
            gazeMagnitude = np.sqrt(dx**2 + dy**2)
            
            if gazeMagnitude > 0.7:  # Olhar muito afastado do centro
                anomalies.append(f"Olhar desviado detectado: magnitude {gazeMagnitude:.2f} (distração)")
        
        # Anomalia: Variação súbita de EAR (instabilidade)
        if len(recentPoints) >= 3 and ear is not None:
            recentEars = [p.value.get("ear") for p in recentPoints[-3:] 
                          if isinstance(p.value, dict) and p.value.get("ear") is not None]
            
            if len(recentEars) >= 3:
                earVariation = max(recentEars) - min(recentEars)
                if earVariation > 0.2:  # Variação > 20% do range normal
                    anomalies.append(f"Variação súbita no EAR: {earVariation:.3f} (instabilidade)")
        
        # Anomalia: Drift de gaze (movimento errático)
        if len(recentPoints) >= 2 and gazeVector:
            prev_point = recentPoints[-2]
            if isinstance(prev_point.value, dict):
                prevGaze = prev_point.value.get("gaze_vector", {})
                if prevGaze:
                    dxChange = abs(gazeVector.get("dx", 0) - prevGaze.get("dx", 0))
                    dyChange = abs(gazeVector.get("dy", 0) - prevGaze.get("dy", 0))
                    gazeChange = np.sqrt(dxChange**2 + dyChange**2)
                    
                    if gazeChange > self.gazeStabilityThreshold:
                        anomalies.append(f"Movimento errático do olhar: mudança {gazeChange:.2f}")
        
        # Atualizar histórico interno
        self._updateInternalHistory(ear, blinkRate, gazeVector)
        
        return anomalies
    
    def _updateInternalHistory(self, ear: Optional[float], blinkRate: Optional[float], 
                              gazeVector: Optional[Dict[str, float]]) -> None:
        """
        Atualiza histórico interno para análise de tendências.
        
        Args:
            ear: Eye Aspect Ratio atual
            blinkRate: Taxa de piscadelas atual
            gazeVector: Vetor de direção do olhar atual
        """
        
        # Atualizar valores anteriores
        self.lastEar = ear
        self.lastBlinkRate = blinkRate
        self.lastGazeVector = gazeVector.copy() if gazeVector else None
        
        # Manter histórico de EAR (últimos 10 valores)
        if ear is not None:
            self.recentEarValues.append(ear)
            if len(self.recentEarValues) > 10:
                self.recentEarValues.pop(0)
    
    def _extractNumericValues(self, points: List[SignalPoint]) -> List[float]:
        """
        Extrai valores numéricos dos pontos para cálculo de métricas.
        Usa blinkcounter como valor numérico principal.
        
        Args:
            points: Lista de pontos do signal
            
        Returns:
            Lista de valores de blinkCounter
        """
        
        numericValues = []
        for point in points:
            if isinstance(point.value, dict):
                blinkCounter = point.value.get("blink_counter")
                if blinkCounter is not None and isinstance(blinkCounter, (int, float)):
                    numericValues.append(float(blinkCounter))
        
        return numericValues
    
    def getCameraStatus(self) -> Dict[str, Any]:
        """
        Status específico do signal de câmera incluindo métricas detalhadas.
        
        Returns:
            Status detalhado com informações específicas de câmera
        """
        
        baseStatus = self.getStatus()
        
        # Calcular métricas específicas de câmera
        recentPoints = self.getLatest(5)
        cameraMetrics = self._calculateCameraMetrics(recentPoints)
        
        camera_status = {
            **baseStatus,
            "cameraMetrics": cameraMetrics,
            "lastValues": {
                "ear": self.lastEar,
                "blink_rate": self.lastBlinkRate,
                "gaze_vector": self.lastGazeVector,
                "consecutiveLowEar": self.consecutiveLowEarCount
            },
            "thresholds": {
                "drowsinessBlinkRate": self.drowsinessBlinkThreshold,
                "hyperBlinkRate": self.hyperBlinkThreshold,
                "earBlink": self.earBlinkThreshold,
                "earDrowsy": self.earDrowsyThreshold,
                "gazeStability": self.gazeStabilityThreshold
            },
            "ranges": {
                "gaze": self.gazeNormalRange,
                "blink_rate": self.blinkRateNormalRange,
                "ear": self.earNormalRange
            },
            "config": {
                "expectedLandmarks": self.expectedLandmarksCount,
                "landmarksDimensions": self.landmarksDimensions,
                "samplingRate": self.samplingRate
            }
        }
        
        return camera_status
    
    def _calculateCameraMetrics(self, recentPoints: List[SignalPoint]) -> Dict[str, Any]:
        """
        Calcula métricas específicas dos dados de câmera.
        
        Args:
            recentPoints: Pontos recentes para análise
            
        Returns:
            Métricas detalhadas de qualidade e performance
        """
        
        if not recentPoints:
            return {"available": False, "reason": "No recent data"}
        
        # Extrair dados dos pontos recentes
        ears = []
        blinkRate = []
        gazeMagnitude = []
        
        for point in recentPoints:
            if isinstance(point.value, dict):
                data = point.value
                
                if "ear" in data:
                    ears.append(data["ear"])
                
                if "blink_rate" in data:
                    blinkRate.append(data["blink_rate"])
                
                if "gaze_vector" in data:
                    gv = data["gaze_vector"]
                    if isinstance(gv, dict) and "dx" in gv and "dy" in gv:
                        magnitude = np.sqrt(gv["dx"]**2 + gv["dy"]**2)
                        gazeMagnitude.append(magnitude)
        
        metrics = {
            "available": True,
            "sampleCount": len(recentPoints),
            "qualityMetrics": {},
            "behaviorMetrics": {},
            "stabilityMetrics": {}
        }
        
        
        # Métricas comportamentais
        if ears and blinkRate:
            metrics["behaviorMetrics"] = {
                "averageEar": round(np.mean(ears), 3),
                "averageBlinkRate": round(np.mean(blinkRate), 1),
                "minEar": round(np.min(ears), 3),
                "maxBlinkRate": round(np.max(blinkRate), 1),
                "drowsinessIndicators": sum(1 for e in ears if e < self.earDrowsyThreshold),
                "alertnessLevel": self._calculateAlertnessLevel(ears, blinkRate)
            }
        
        # Métricas de estabilidade
        if gazeMagnitude:
            metrics["stabilityMetrics"] = {
                "averageGazeMagnitude": round(np.mean(gazeMagnitude), 3),
                "gazeStability": round(1.0 - np.std(gazeMagnitude), 3),
                "centeredGazeFrames": sum(1 for g in gazeMagnitude if g <= 0.2),
                "distractedFrames": sum(1 for g in gazeMagnitude if g > 0.7)
            }
        
        return metrics
    
    def _calculateAlertnessLevel(self, ears: List[float], blinkRate: List[float]) -> str:
        """
        Calcula nível de alerta baseado em EAR e blink rate.
        
        Args:
            ears: Lista de valores EAR
            blinkRate: Lista de taxas de piscadelas
            
        Returns:
            Nível de alerta ("alert", "normal", "drowsy", "critical")
        """
        
        if not ears or not blinkRate:
            return "unknown"
        
        avgEar = np.mean(ears)
        avgBlinkRate = np.mean(blinkRate)
        
        # Lógica de classificação
        if avgEar < 0.15 or avgBlinkRate < 5:
            return "critical"
        elif avgEar < self.earDrowsyThreshold or avgBlinkRate < self.drowsinessBlinkThreshold:
            return "drowsy"
        elif avgEar > 0.35 and self.blinkRateNormalRange[0] <= avgBlinkRate <= self.blinkRateNormalRange[1]:
            return "alert"
        else:
            return "normal"
    
    def reset(self) -> None:
        """Reset completo do signal de câmera"""
        
        super().reset()
        
        # Reset estado interno específico
        self.lastEar = None
        self.lastBlinkRate = None
        self.lastGazeVector = None
        self.recentEarValues.clear()
        self.consecutiveLowEarCount = 0
        
        self.logger.info("CameraSignal reset completed")