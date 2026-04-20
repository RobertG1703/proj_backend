"""
SensorsSignal - Implementação para sensores de movimento (ACC + GYR do CardioWheel)

Resumo:
Processa todos os dados de movimento vindos do CardioWheel (acelerómetro e giroscópio).
Recebe dois tipos de dados principais: aceleração linear em 3 eixos (X, Y, Z) em m/s² 
e velocidade angular em 3 eixos (X, Y, Z) em °/s. Verifica se os valores estão nos ranges
físicos esperados e detecta padrões de condução.

Para detecção de anomalias, consegue identificar movimentos bruscos (aceleração >50 m/s²),
vibração excessiva (std >20 m/s²), impactos (>120 m/s²), rotação rápida (>500 °/s),
instabilidade (std >100 °/s), possível spin/derrapagem (>1000 °/s) e padrões combinados
como condução agressiva (alta aceleração + alta rotação simultâneas), travagem de emergência
(forte desaceleração sem rotação) e curvas apertadas (aceleração lateral + rotação yaw).

De forma resumida o getLatestAcceleration() serve para obter os dados mais recentes dos 3 eixos,
getLatestRotation() para as velocidades angulares, calculateMagnitude() para magnitude total
dos vetores, detectMovementPatterns() que classifica padrões de condução como normal, agressivo,
emergência ou instável baseado na combinação de dados, analyzeVehicleStability() que avalia
estabilidade geral do veículo, e getSensorsStatus() que dá um resumo completo do estado dos sensores.

A classe mantém um buffer circular de 3.000 pontos (30 segundos a 100Hz) e consegue interpretar
automaticamente padrões de condução do condutor. Foi feita para suportar qualquer fonte de 
dados de movimento, desde que enviem os 3 eixos de aceleração e rotação no formato esperado.
"""

import numpy as np
from typing import List, Optional, Any, Dict, Union
from datetime import datetime

from ..base import BaseSignal
from ..dataPoint import SignalPoint
from ...core import settings, SignalValidationError

class SensorsSignal(BaseSignal):
    """Sinal de sensores de movimento - ACC + GYR (CardioWheel)"""
    
    def __init__(self):
        # Configuração sensores: 3000 pontos = 30s * 100Hz
        sensorsConfig = settings.signals.sensorsConfig
        
        super().__init__(
            signalName="sensors",
            bufferSize=sensorsConfig["accelerometer"]["bufferSize"],  # 3000
            samplingRate=sensorsConfig["accelerometer"]["samplingRate"]  # 100Hz
        )
        
        # Configurações do acelerómetro
        self.accConfig = sensorsConfig["accelerometer"]
        self.accPhysicalRange = self.accConfig["physicalRange"]
        self.accSamplingRate = self.accConfig["samplingRate"]
        
        # Configurações do giroscópio
        self.gyrConfig = sensorsConfig["gyroscope"]
        self.gyrPhysicalRange = self.gyrConfig["physicalRange"]
        self.gyrSamplingRate = self.gyrConfig["samplingRate"]
        
        # Configurações combinadas
        self.combinedConfig = sensorsConfig["combined"]
        
        # Thresholds de anomalias ACC
        self.suddenMovementThreshold = self.accConfig["suddenMovementThreshold"]
        self.highVibrationsThreshold = self.accConfig["highVibrationsThreshold"]
        self.magnitudeThreshold = self.accConfig["magnitudeThreshold"]
        self.lowActivityThreshold = self.accConfig["lowActivityThreshold"]
        self.impactThreshold = self.accConfig["impactThreshold"]
        self.sustainedAccelThreshold = self.accConfig["sustainedAccelThreshold"]
        
        # Thresholds de anomalias GYR
        self.rapidRotationThreshold = self.gyrConfig["rapidRotationThreshold"]
        self.instabilityThreshold = self.gyrConfig["instabilityThreshold"]
        self.angularMagnitudeThreshold = self.gyrConfig["angularMagnitudeThreshold"]
        self.lowGyrActivityThreshold = self.gyrConfig["lowActivityThreshold"]
        self.spinThreshold = self.gyrConfig["spinThreshold"]
        self.sustainedRotationThreshold = self.gyrConfig["sustainedRotationThreshold"]
        
        self.logger.info(f"SensorsSignal initialized - ACC range: {self.accPhysicalRange} m/s², GYR range: {self.gyrPhysicalRange} °/s")
    
    def validateValue(self, value: Any) -> bool:
        """Valida valores de acelerómetro ou giroscópio"""
        
        # Accelerometer data (dict com eixos + magnitude)
        if isinstance(value, dict) and "accelerometer" in value:
            try:
                accData = value["accelerometer"]
                
                # Verificar se tem os 3 eixos
                requiredAxes = ["x", "y", "z"]
                for axis in requiredAxes:
                    if axis not in accData:
                        raise SignalValidationError(
                            signalType="sensors",
                            value=f"Missing axis: {axis}",
                            reason=f"Accelerometer deve ter eixos {requiredAxes}"
                        )
                
                # Verificar cada eixo
                for axis in requiredAxes:
                    axisData = accData[axis]
                    if isinstance(axisData, list):
                        # Array de valores
                        axisArray = np.array(axisData)
                        minVal, maxVal = np.min(axisArray), np.max(axisArray)
                    else:
                        # Valor único
                        minVal = maxVal = float(axisData)
                    
                    # Verificar range físico
                    if not (self.accPhysicalRange[0] <= minVal and maxVal <= self.accPhysicalRange[1]):
                        raise SignalValidationError(
                            signalType="sensors",
                            value=f"{axis}: [{minVal:.2f}, {maxVal:.2f}] m/s²",
                            reason=f"Accelerometer {axis} fora do range {self.accPhysicalRange} m/s²"
                        )
                
                return True
                
            except SignalValidationError:
                # Reenvia a exceção tal como foi lançada
                raise
            except Exception as e:
                raise SignalValidationError(
                    signalType="sensors",
                    value=str(value)[:100],
                    reason=f"Erro ao validar accelerometer: {e}"
                )
        
        # Gyroscope data (dict com eixos + magnitude angular)
        elif isinstance(value, dict) and "gyroscope" in value:
            try:
                gyrData = value["gyroscope"]
                
                # Verificar se tem os 3 eixos
                requiredAxes = ["x", "y", "z"]
                for axis in requiredAxes:
                    if axis not in gyrData:
                        raise SignalValidationError(
                            signalType="sensors",
                            value=f"Missing axis: {axis}",
                            reason=f"Gyroscope deve ter eixos {requiredAxes}"
                        )
                
                # Verificar cada eixo
                for axis in requiredAxes:
                    axisData = gyrData[axis]
                    if isinstance(axisData, list):
                        # Array de valores
                        axisArray = np.array(axisData)
                        minVal, maxVal = np.min(axisArray), np.max(axisArray)
                    else:
                        # Valor único
                        minVal = maxVal = float(axisData)
                    
                    # Verificar range físico
                    if not (self.gyrPhysicalRange[0] <= minVal and maxVal <= self.gyrPhysicalRange[1]):
                        raise SignalValidationError(
                            signalType="sensors",
                            value=f"{axis}: [{minVal:.1f}, {maxVal:.1f}] °/s",
                            reason=f"Gyroscope {axis} fora do range {self.gyrPhysicalRange} °/s"
                        )
                
                return True
                
            except SignalValidationError:
                # Reenvia a exceção tal como foi lançada
                raise
            except Exception as e:
                raise SignalValidationError(
                    signalType="sensors",
                    value=str(value)[:100],
                    reason=f"Erro ao validar gyroscope: {e}"
                )
        
        # Tipo não reconhecido
        raise SignalValidationError(
            signalType="sensors",
            value=type(value).__name__,
            reason="Valor deve ser dict com 'accelerometer' ou 'gyroscope'"
        )
    
    def getNormalRange(self) -> tuple:
        """Range normal para sensores (retorna range do acelerómetro)"""
        return self.accPhysicalRange
    
    def detectAnomalies(self, recentPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias em sensores de movimento"""
        anomalies = []
        
        if len(recentPoints) < 1:
            return anomalies
        
        # Separar dados ACC e GYR
        accPoints = [point for point in recentPoints if isinstance(point.value, dict) and "accelerometer" in point.value]
        gyrPoints = [point for point in recentPoints if isinstance(point.value, dict) and "gyroscope" in point.value]
        
        # Anomalias do acelerómetro
        if accPoints:
            anomalies.extend(self._detectAccelerometerAnomalies(accPoints))
        
        # Anomalias do giroscópio
        if gyrPoints:
            anomalies.extend(self._detectGyroscopeAnomalies(gyrPoints))
        
        # Anomalias combinadas (ACC + GYR)
        if accPoints and gyrPoints:
            anomalies.extend(self._detectCombinedAnomalies(accPoints, gyrPoints))
        
        return anomalies
    
    def _detectAccelerometerAnomalies(self, accPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias no acelerómetro"""
        anomalies = []
        
        if not accPoints:
            return anomalies
        
        latestAcc = accPoints[-1].value["accelerometer"]
        
        # Extrair magnitude se disponível
        magnitude = None
        if "magnitude" in accPoints[-1].value:
            magnitudeArray = accPoints[-1].value["magnitude"]
            magnitude = magnitudeArray[-1] if isinstance(magnitudeArray, list) else magnitudeArray
        else:
            # Calcular magnitude a partir dos eixos
            try:
                x = latestAcc["x"][-1] if isinstance(latestAcc["x"], list) else latestAcc["x"]
                y = latestAcc["y"][-1] if isinstance(latestAcc["y"], list) else latestAcc["y"]
                z = latestAcc["z"][-1] if isinstance(latestAcc["z"], list) else latestAcc["z"]
                magnitude = (x**2 + y**2 + z**2)**0.5
            except (KeyError, IndexError, TypeError):
                magnitude = 0
        
        # Movimento brusco (magnitude excessiva)
        if magnitude and magnitude > self.suddenMovementThreshold:
            anomalies.append(f"Movimento brusco detectado: {magnitude:.1f} m/s² (limite: {self.suddenMovementThreshold})")
        
        # Impacto severo
        if magnitude and magnitude > self.impactThreshold:
            anomalies.append(f"Possível impacto detectado: {magnitude:.1f} m/s²")
        
        # Verificar vibração excessiva (múltiplos pontos)
        if len(accPoints) >= 10:
            # Coletar magnitudes dos últimos 10 pontos
            recentMagnitudes = []
            for point in accPoints[-10:]:
                pointMag = None
                if "magnitude" in point.value:
                    magArray = point.value["magnitude"]
                    pointMag = magArray[-1] if isinstance(magArray, list) else magArray
                
                if pointMag:
                    recentMagnitudes.append(pointMag)
            
            if len(recentMagnitudes) >= 5:
                magStd = np.std(recentMagnitudes)
                if magStd > self.highVibrationsThreshold:
                    anomalies.append(f"Vibração excessiva: std={magStd:.1f} m/s²")
        
        # Verificar atividade muito baixa (sinal plano)
        if magnitude and magnitude < self.lowActivityThreshold:
            anomalies.append(f"Atividade muito baixa: {magnitude:.2f} m/s² (possível sensor com problemas)")
        
        # Verificar aceleração sustentada (condução agressiva)
        if len(accPoints) >= 5:
            sustainedCount = 0
            for point in accPoints[-5:]:
                pointMag = None
                if "magnitude" in point.value:
                    magArray = point.value["magnitude"]
                    pointMag = magArray[-1] if isinstance(magArray, list) else magArray
                
                if pointMag and pointMag > self.sustainedAccelThreshold:
                    sustainedCount += 1
            
            if sustainedCount >= 4:  # 4 de 5 pontos
                anomalies.append(f"Aceleração sustentada detectada (condução agressiva)")
        
        return anomalies
    
    def _detectGyroscopeAnomalies(self, gyrPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias no giroscópio"""
        anomalies = []
        
        if not gyrPoints:
            return anomalies
        
        latestGyr = gyrPoints[-1].value["gyroscope"]
        
        # Extrair magnitude angular se disponível
        angularMagnitude = None
        if "angularMagnitude" in gyrPoints[-1].value:
            magArray = gyrPoints[-1].value["angularMagnitude"]
            angularMagnitude = magArray[-1] if isinstance(magArray, list) else magArray
        else:
            # Calcular magnitude a partir dos eixos
            try:
                x = latestGyr["x"][-1] if isinstance(latestGyr["x"], list) else latestGyr["x"]
                y = latestGyr["y"][-1] if isinstance(latestGyr["y"], list) else latestGyr["y"]
                z = latestGyr["z"][-1] if isinstance(latestGyr["z"], list) else latestGyr["z"]
                angularMagnitude = (x**2 + y**2 + z**2)**0.5
            except (KeyError, IndexError, TypeError):
                angularMagnitude = 0
        
        # Rotação rápida
        if angularMagnitude and angularMagnitude > self.rapidRotationThreshold:
            anomalies.append(f"Rotação rápida detectada: {angularMagnitude:.1f} °/s (limite: {self.rapidRotationThreshold})")
        
        # Possível spin/derrapagem
        if angularMagnitude and angularMagnitude > self.spinThreshold:
            anomalies.append(f"Possível spin/derrapagem: {angularMagnitude:.1f} °/s")
        
        # Verificar instabilidade (múltiplos pontos)
        if len(gyrPoints) >= 10:
            # Coletar magnitudes angulares dos últimos 10 pontos
            recentAngularMags = []
            for point in gyrPoints[-10:]:
                pointMag = None
                if "angularMagnitude" in point.value:
                    magArray = point.value["angularMagnitude"]
                    pointMag = magArray[-1] if isinstance(magArray, list) else magArray
                
                if pointMag:
                    recentAngularMags.append(pointMag)
            
            if len(recentAngularMags) >= 5:
                magStd = np.std(recentAngularMags)
                if magStd > self.instabilityThreshold:
                    anomalies.append(f"Instabilidade detectada: std={magStd:.1f} °/s")
        
        # Verificar atividade muito baixa
        if angularMagnitude and angularMagnitude < self.lowGyrActivityThreshold:
            anomalies.append(f"Atividade angular muito baixa: {angularMagnitude:.1f} °/s (possível sensor com problemas)")
        
        # Verificar rotação sustentada
        if len(gyrPoints) >= 5:
            sustainedCount = 0
            for point in gyrPoints[-5:]:
                pointMag = None
                if "angularMagnitude" in point.value:
                    magArray = point.value["angularMagnitude"]
                    pointMag = magArray[-1] if isinstance(magArray, list) else magArray
                
                if pointMag and pointMag > self.sustainedRotationThreshold:
                    sustainedCount += 1
            
            if sustainedCount >= 4:  # 4 de 5 pontos
                anomalies.append(f"Rotação sustentada detectada")
        
        return anomalies
    
    def _detectCombinedAnomalies(self, accPoints: List[SignalPoint], gyrPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias combinadas ACC + GYR"""
        anomalies = []
        
        if not accPoints or not gyrPoints:
            return anomalies
        
        # Pegar pontos mais recentes de ambos os sensores
        latestAcc = accPoints[-1].value
        latestGyr = gyrPoints[-1].value
        
        # Calcular magnitudes
        accMagnitude = None
        if "magnitude" in latestAcc:
            magArray = latestAcc["magnitude"]
            accMagnitude = magArray[-1] if isinstance(magArray, list) else magArray
        
        gyrMagnitude = None
        if "angularMagnitude" in latestGyr:
            magArray = latestGyr["angularMagnitude"]
            gyrMagnitude = magArray[-1] if isinstance(magArray, list) else magArray
        
        if not accMagnitude or not gyrMagnitude:
            return anomalies
        
        # Condução agressiva (alta aceleração + alta rotação)
        aggressiveConfig = self.combinedConfig["aggressiveDrivingCombo"]
        if (accMagnitude > aggressiveConfig["accThreshold"] and 
            gyrMagnitude > aggressiveConfig["gyrThreshold"]):
            anomalies.append(f"Condução agressiva detectada: ACC={accMagnitude:.1f} m/s², GYR={gyrMagnitude:.1f} °/s")
        
        # Travagem de emergência (forte desaceleração sem rotação)
        emergencyConfig = self.combinedConfig["emergencyBrakingCombo"]
        
        # Verificar se há aceleração negativa significativa (assumindo eixo Y = frente/trás)
        try:
            accY = latestAcc["accelerometer"]["y"]
            accYValue = accY[-1] if isinstance(accY, list) else accY
            
            if (accYValue < emergencyConfig["accThreshold"] and  # Desaceleração forte
                gyrMagnitude < emergencyConfig["gyrStabilityMax"]):  # Pouca rotação
                anomalies.append(f"Possível travagem de emergência: desaceleração={accYValue:.1f} m/s²")
        except (KeyError, IndexError, TypeError):
            pass
        
        # Curva apertada (aceleração lateral + rotação yaw)
        corneringConfig = self.combinedConfig["cornering"]
        
        try:
            # Aceleração lateral (assumindo eixo X)
            accX = latestAcc["accelerometer"]["x"]
            accXValue = abs(accX[-1] if isinstance(accX, list) else accX)
            
            # Rotação yaw (assumindo eixo Z)
            gyrZ = latestGyr["gyroscope"]["z"]
            gyrZValue = abs(gyrZ[-1] if isinstance(gyrZ, list) else gyrZ)
            
            if (accXValue > corneringConfig["accLateralMin"] and 
                gyrZValue > corneringConfig["gyrYawMin"]):
                anomalies.append(f"Curva apertada detectada: ACC lateral={accXValue:.1f} m/s², rotação yaw={gyrZValue:.1f} °/s")
        except (KeyError, IndexError, TypeError):
            pass
        
        return anomalies
    
    # Métodos específicos para SensorsSignal
    
    def getLatestAcceleration(self) -> Optional[Dict[str, Any]]:
        """Retorna dados de aceleração mais recentes"""
        allPoints = self.getAllData()
        
        for point in reversed(allPoints):
            if isinstance(point.value, dict) and "accelerometer" in point.value:
                return {
                    "timestamp": point.timestamp,
                    "accelerometer": point.value["accelerometer"],
                    "magnitude": point.value.get("magnitude"),
                    "units": "m/s²"
                }
        
        return None
    
    def getLatestRotation(self) -> Optional[Dict[str, Any]]:
        """Retorna dados de rotação mais recentes"""
        allPoints = self.getAllData()
        
        for point in reversed(allPoints):
            if isinstance(point.value, dict) and "gyroscope" in point.value:
                return {
                    "timestamp": point.timestamp,
                    "gyroscope": point.value["gyroscope"],
                    "angularMagnitude": point.value.get("angularMagnitude"),
                    "units": "°/s"
                }
        
        return None
    
    def calculateMagnitude(self, sensorType: str, durationSeconds: float = 5.0) -> Optional[Dict[str, Any]]:
        """Calcula magnitude de movimento dos últimos X segundos"""
        if sensorType not in ["accelerometer", "gyroscope"]:
            raise SignalValidationError(
                signalType="sensors",
                value=sensorType,
                reason="sensorType deve ser 'accelerometer' ou 'gyroscope'"
            )
        
        allPoints = self.getAllData()
        
        # Filtrar pontos do tipo de sensor e período desejado
        cutoffTime = datetime.now().timestamp() - durationSeconds
        relevantPoints = []
        
        for point in allPoints:
            if (point.timestamp >= cutoffTime and 
                isinstance(point.value, dict) and 
                sensorType in point.value):
                relevantPoints.append(point)
        
        if len(relevantPoints) < 10:  # Mínimo de pontos
            return None
        
        # Extrair magnitudes
        magnitudes = []
        for point in relevantPoints:
            if sensorType == "accelerometer" and "magnitude" in point.value:
                magArray = point.value["magnitude"]
                magnitude = magArray[-1] if isinstance(magArray, list) else magArray
                magnitudes.append(magnitude)
            elif sensorType == "gyroscope" and "angularMagnitude" in point.value:
                magArray = point.value["angularMagnitude"]
                magnitude = magArray[-1] if isinstance(magArray, list) else magArray
                magnitudes.append(magnitude)
        
        if not magnitudes:
            return None
        
        magnitudeArray = np.array(magnitudes)
        
        return {
            "sensorType": sensorType,
            "duration": durationSeconds,
            "sampleCount": len(magnitudes),
            "mean": float(np.mean(magnitudeArray)),
            "std": float(np.std(magnitudeArray)),
            "min": float(np.min(magnitudeArray)),
            "max": float(np.max(magnitudeArray)),
            "rms": float(np.sqrt(np.mean(magnitudeArray**2))),
            "units": "m/s²" if sensorType == "accelerometer" else "°/s"
        }
    
    def detectMovementPatterns(self) -> Dict[str, Any]:
        """Deteta padrões de movimento e classifica condução"""
        latestAcc = self.getLatestAcceleration()
        latestGyr = self.getLatestRotation()
        
        if not latestAcc or not latestGyr:
            return {"pattern": "unknown", "reason": "Insufficient data"}
        
        # Calcular métricas dos últimos 10 segundos
        accStats = self.calculateMagnitude("accelerometer", 10.0)
        gyrStats = self.calculateMagnitude("gyroscope", 10.0)
        
        if not accStats or not gyrStats:
            return {"pattern": "unknown", "reason": "Insufficient data"}
        
        # Classificar baseado nas estatísticas
        accMean = accStats["mean"]
        gyrMean = gyrStats["mean"]
        accStd = accStats["std"]
        gyrStd = gyrStats["std"]
        
        # Critérios para classificação
        if accMean > 30 and gyrMean > 200:
            pattern = "aggressive"
            confidence = min((accMean + gyrMean) / 100, 1.0)
        elif accStd > 20 or gyrStd > 100:
            pattern = "unstable"
            confidence = min((accStd + gyrStd) / 50, 1.0)
        elif accMean < 5 and gyrMean < 10:
            pattern = "steady"
            confidence = 0.8
        elif accMean > 50 and gyrMean < 50:
            pattern = "emergency"
            confidence = min(accMean / 50, 1.0)
        else:
            pattern = "normal"
            confidence = 0.6
        
        return {
            "pattern": pattern,
            "confidence": confidence,
            "metrics": {
                "accelerationMean": accMean,
                "rotationMean": gyrMean,
                "accelerationStd": accStd,
                "rotationStd": gyrStd
            },
            "interpretation": self._interpretMovementPattern(pattern, accStats, gyrStats)
        }
    
    def _interpretMovementPattern(self, pattern: str, accStats: Dict, gyrStats: Dict) -> str:
        """Interpretação textual do padrão de movimento"""
        interpretations = {
            "aggressive": f"Condução agressiva com alta aceleração ({accStats['mean']:.1f} m/s²) e rotação ({gyrStats['mean']:.1f} °/s)",
            "unstable": f"Condução instável com alta variabilidade (ACC std: {accStats['std']:.1f}, GYR std: {gyrStats['std']:.1f})",
            "steady": f"Condução estável com baixa atividade (ACC: {accStats['mean']:.1f} m/s², GYR: {gyrStats['mean']:.1f} °/s)",
            "emergency": f"Possível situação de emergência com alta aceleração ({accStats['mean']:.1f} m/s²) e pouca rotação",
            "normal": "Condução normal sem padrões específicos detectados"
        }
        
        return interpretations.get(pattern, "Padrão não classificado")
    
    def analyzeVehicleStability(self) -> Dict[str, Any]:
        """Análise da estabilidade geral do veículo"""
        # Obter dados dos últimos 30 segundos
        accStats = self.calculateMagnitude("accelerometer", 30.0)
        gyrStats = self.calculateMagnitude("gyroscope", 30.0)
        
        if not accStats or not gyrStats:
            return {"stability": "unknown", "reason": "Insufficient data"}
        
        # Calcular índice de estabilidade (0-1, 1 = mais estável)
        accStabilityFactor = max(0, 1 - (accStats["std"] / 50))  # Normalizar std para 0-1
        gyrStabilityFactor = max(0, 1 - (gyrStats["std"] / 200))  # Normalizar std para 0-1
        
        # Índice combinado
        stabilityIndex = (accStabilityFactor + gyrStabilityFactor) / 2
        
        # Classificar estabilidade
        if stabilityIndex > 0.8:
            stability = "excellent"
        elif stabilityIndex > 0.6:
            stability = "good"
        elif stabilityIndex > 0.4:
            stability = "moderate"
        elif stabilityIndex > 0.2:
            stability = "poor"
        else:
            stability = "critical"
        
        return {
            "stability": stability,
            "stabilityIndex": stabilityIndex,
            "factors": {
                "accelerationStability": accStabilityFactor,
                "rotationStability": gyrStabilityFactor
            },
            "stats": {
                "acceleration": accStats,
                "rotation": gyrStats
            },
            "recommendations": self._getStabilityRecommendations(stability, accStats, gyrStats)
        }
    
    def _getStabilityRecommendations(self, stability: str, accStats: Dict, gyrStats: Dict) -> List[str]:
        """Recomendações baseadas na estabilidade"""
        recommendations = []
        
        if stability in ["poor", "critical"]:
            if accStats["std"] > 30:
                recommendations.append("Reduzir variabilidade nas acelerações")
            if gyrStats["std"] > 150:
                recommendations.append("Evitar mudanças bruscas de direção")
            if accStats["mean"] > 40:
                recommendations.append("Reduzir velocidade e intensidade de manobras")
            recommendations.append("Verificar condições da estrada e do veículo")
        elif stability == "moderate":
            recommendations.append("Manter condução suave e constante")
            if accStats["max"] > 60:
                recommendations.append("Evitar acelerações/travagens bruscas")
        else:
            recommendations.append("Condução estável - manter padrão atual")
        
        return recommendations
    
    def getSensorsStatus(self) -> Dict[str, Any]:
        """Status geral dos sensores de movimento"""
        baseStatus = self.getStatus()
        
        # Informações específicas dos sensores
        latestAcc = self.getLatestAcceleration()
        latestGyr = self.getLatestRotation()
        movementPattern = self.detectMovementPatterns()
        vehicleStability = self.analyzeVehicleStability()
        
        # Estatísticas por sensor
        accStats = self.calculateMagnitude("accelerometer", 30.0)
        gyrStats = self.calculateMagnitude("gyroscope", 30.0)
        
        # Qualidade dos sensores
        sensorQuality = self._assessSensorQuality()
        
        sensorsStatus = {
            **baseStatus,
            "latestAcceleration": latestAcc,
            "latestRotation": latestGyr,
            "movementPattern": movementPattern,
            "vehicleStability": vehicleStability,
            "accelerometerStats": accStats,
            "gyroscopeStats": gyrStats,
            "sensorQuality": sensorQuality,
            "dataAvailable": {
                "accelerometer": latestAcc is not None,
                "gyroscope": latestGyr is not None
            },
            "configuration": {
                "accelerometerRange": self.accPhysicalRange,
                "gyroscopeRange": self.gyrPhysicalRange,
                "samplingRate": self.accSamplingRate,
                "bufferDuration": f"{self.bufferSize / self.accSamplingRate:.1f}s",
                "thresholds": {
                    "suddenMovement": self.suddenMovementThreshold,
                    "impact": self.impactThreshold,
                    "rapidRotation": self.rapidRotationThreshold,
                    "spin": self.spinThreshold
                }
            }
        }
        
        return sensorsStatus
    
    def _assessSensorQuality(self) -> Dict[str, str]:
        """Avalia qualidade dos sensores baseado em atividade e consistência"""
        quality = {}
        
        # Avaliar acelerómetro
        accStats = self.calculateMagnitude("accelerometer", 10.0)
        if not accStats:
            quality["accelerometer"] = "no_data"
        elif accStats["mean"] < self.lowActivityThreshold:
            quality["accelerometer"] = "poor"  # Muito pouca atividade
        elif accStats["std"] > self.highVibrationsThreshold * 2:
            quality["accelerometer"] = "noisy"  # Muito ruído
        elif self.lowActivityThreshold <= accStats["mean"] <= 50 and accStats["std"] <= 30:
            quality["accelerometer"] = "good"
        else:
            quality["accelerometer"] = "ok"
        
        # Avaliar giroscópio
        gyrStats = self.calculateMagnitude("gyroscope", 10.0)
        if not gyrStats:
            quality["gyroscope"] = "no_data"
        elif gyrStats["mean"] < self.lowGyrActivityThreshold:
            quality["gyroscope"] = "poor"  # Muito pouca atividade
        elif gyrStats["std"] > self.instabilityThreshold * 2:
            quality["gyroscope"] = "noisy"  # Muito ruído
        elif self.lowGyrActivityThreshold <= gyrStats["mean"] <= 300 and gyrStats["std"] <= 150:
            quality["gyroscope"] = "good"
        else:
            quality["gyroscope"] = "ok"
        
        return quality
    
    def detectSensorFaults(self) -> List[Dict[str, Any]]:
        """Detecta falhas específicas nos sensores"""
        faults = []
        
        # Verificar dados recentes
        recentPoints = self.getLatest(50)  # Últimos 50 pontos
        
        if len(recentPoints) < 10:
            faults.append({
                "type": "insufficient_data",
                "severity": "warning",
                "message": "Dados insuficientes para análise de falhas",
                "sensor": "both"
            })
            return faults
        
        # Separar dados por sensor
        accPoints = [p for p in recentPoints if "accelerometer" in p.value]
        gyrPoints = [p for p in recentPoints if "gyroscope" in p.value]
        
        # Verificar falhas do acelerómetro
        if accPoints:
            accFaults = self._checkAccelerometerFaults(accPoints)
            faults.extend(accFaults)
        
        # Verificar falhas do giroscópio
        if gyrPoints:
            gyrFaults = self._checkGyroscopeFaults(gyrPoints)
            faults.extend(gyrFaults)
        
        return faults
    
    def _checkAccelerometerFaults(self, accPoints: List[SignalPoint]) -> List[Dict[str, Any]]:
        """Verifica falhas específicas do acelerómetro"""
        faults = []
        
        # Extrair magnitudes
        magnitudes = []
        for point in accPoints:
            if "magnitude" in point.value:
                magArray = point.value["magnitude"]
                magnitude = magArray[-1] if isinstance(magArray, list) else magArray
                magnitudes.append(magnitude)
        
        if len(magnitudes) < 5:
            return faults
        
        magnitudeArray = np.array(magnitudes)
        
        # Verificar sinal plano (falha de sensor)
        if np.std(magnitudeArray) < 0.1:
            faults.append({
                "type": "flat_signal",
                "severity": "critical",
                "message": f"Sinal do acelerómetro muito plano (std: {np.std(magnitudeArray):.3f})",
                "sensor": "accelerometer"
            })
        
        # Verificar valores constantes (sensor travado)
        uniqueValues = len(np.unique(np.round(magnitudeArray, 1)))
        if uniqueValues < 3:
            faults.append({
                "type": "stuck_sensor",
                "severity": "critical",
                "message": f"Acelerómetro com valores repetidos ({uniqueValues} valores únicos)",
                "sensor": "accelerometer"
            })
        
        # Verificar saturação
        if np.any(magnitudeArray > self.accPhysicalRange[1] * 0.9):
            faults.append({
                "type": "saturation",
                "severity": "warning",
                "message": "Possível saturação do acelerómetro",
                "sensor": "accelerometer"
            })
        
        return faults
    
    def _checkGyroscopeFaults(self, gyrPoints: List[SignalPoint]) -> List[Dict[str, Any]]:
        """Verifica falhas específicas do giroscópio"""
        faults = []
        
        # Extrair magnitudes angulares
        angularMagnitudes = []
        for point in gyrPoints:
            if "angularMagnitude" in point.value:
                magArray = point.value["angularMagnitude"]
                magnitude = magArray[-1] if isinstance(magArray, list) else magArray
                angularMagnitudes.append(magnitude)
        
        if len(angularMagnitudes) < 5:
            return faults
        
        magnitudeArray = np.array(angularMagnitudes)
        
        # Verificar sinal plano (falha de sensor)
        if np.std(magnitudeArray) < 0.5:
            faults.append({
                "type": "flat_signal",
                "severity": "critical",
                "message": f"Sinal do giroscópio muito plano (std: {np.std(magnitudeArray):.3f})",
                "sensor": "gyroscope"
            })
        
        # Verificar valores constantes (sensor travado)
        uniqueValues = len(np.unique(np.round(magnitudeArray, 1)))
        if uniqueValues < 3:
            faults.append({
                "type": "stuck_sensor",
                "severity": "critical",
                "message": f"Giroscópio com valores repetidos ({uniqueValues} valores únicos)",
                "sensor": "gyroscope"
            })
        
        # Verificar saturação
        if np.any(magnitudeArray > self.gyrPhysicalRange[1] * 0.9):
            faults.append({
                "type": "saturation",
                "severity": "warning",
                "message": "Possível saturação do giroscópio",
                "sensor": "gyroscope"
            })
        
        return faults
    
    def getDrivingInsights(self) -> Dict[str, Any]:
        """Análise avançada dos padrões de condução"""
        # Combinar análises existentes
        movementPattern = self.detectMovementPatterns()
        vehicleStability = self.analyzeVehicleStability()
        sensorFaults = self.detectSensorFaults()
        
        # Calcular score geral de condução (0-100)
        stabilityScore = vehicleStability.get("stabilityIndex", 0) * 100
        
        # Penalizar por padrões agressivos
        patternPenalty = 0
        if movementPattern.get("pattern") == "aggressive":
            patternPenalty = 30
        elif movementPattern.get("pattern") == "unstable":
            patternPenalty = 20
        elif movementPattern.get("pattern") == "emergency":
            patternPenalty = 10
        
        # Penalizar por falhas críticas
        faultPenalty = sum(20 for fault in sensorFaults if fault["severity"] == "critical")
        faultPenalty += sum(5 for fault in sensorFaults if fault["severity"] == "warning")
        
        drivingScore = max(0, stabilityScore - patternPenalty - faultPenalty)
        
        # Classificar nível de condução
        if drivingScore >= 80:
            drivingLevel = "excellent"
        elif drivingScore >= 60:
            drivingLevel = "good"
        elif drivingScore >= 40:
            drivingLevel = "moderate"
        elif drivingScore >= 20:
            drivingLevel = "poor"
        else:
            drivingLevel = "critical"
        
        return {
            "drivingScore": round(drivingScore, 1),
            "drivingLevel": drivingLevel,
            "breakdown": {
                "stabilityScore": round(stabilityScore, 1),
                "patternPenalty": patternPenalty,
                "faultPenalty": faultPenalty
            },
            "movementPattern": movementPattern,
            "vehicleStability": vehicleStability,
            "sensorFaults": sensorFaults,
            "recommendations": self._getDrivingRecommendations(drivingLevel, movementPattern, sensorFaults),
            "timestamp": datetime.now().isoformat()
        }
    
    def _getDrivingRecommendations(self, drivingLevel: str, movementPattern: Dict, faults: List) -> List[str]:
        """Recomendações específicas baseadas na análise"""
        recommendations = []
        
        # Recomendações baseadas no nível
        if drivingLevel in ["poor", "critical"]:
            recommendations.append("Reduzir significativamente velocidade e intensidade de manobras")
            recommendations.append("Fazer pausa para avaliar condições de condução")
        elif drivingLevel == "moderate":
            recommendations.append("Melhorar suavidade nas acelerações e curvas")
        
        # Recomendações baseadas no padrão
        pattern = movementPattern.get("pattern", "unknown")
        if pattern == "aggressive":
            recommendations.append("Adoptar estilo de condução mais defensivo")
            recommendations.append("Reduzir acelerações e travagens bruscas")
        elif pattern == "unstable":
            recommendations.append("Manter trajectória mais consistente")
            recommendations.append("Verificar condições da estrada")
        
        # Recomendações baseadas em falhas
        criticalFaults = [f for f in faults if f["severity"] == "critical"]
        if criticalFaults:
            recommendations.append("Verificar funcionamento dos sensores do veículo")
            recommendations.append("Contactar assistência técnica se problemas persistirem")
        
        return recommendations