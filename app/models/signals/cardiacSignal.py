"""
CardiacSignal - Implementação para ECG e HR 

Resumo:
Processa todos os dados cardíacos vindos do CardioWheel e do Polar Armband. 

Recebe dois tipos de dados principais: o ECG e eventos de HR (batimentos cardíacos que chegam irregularmente quando o coração bate).
Verifica se os valores de HR estão entre 30-250 bpm (para não aceitar valores impossíveis como 500 bpm) e se o ECG está no range normal de -5 a 5 milivolts. 
Para detecção de anomalias, consegue identificar bradicardia (coração lento, <50 bpm), taquicardia (coração rápido, >120 bpm), 
variabilidade excessiva (quando o HR 'salta' muito entre leituras), mudanças súbitas (diferenças >30 bpm entre batimentos consecutivos o que não é realista), 
e problemas no ECG como amplitude muito baixa (eletrodo solto), amplitude muito alta (saturação), sinal plano (sem variação) e deriva da linha de base.

De forma resumida o getLatestHr() server para saber o último batimento, getLatestEcgSegment() para obter os últimos segundos de ECG, 
calculateHrStatistics() para estatísticas dos últimos minutos (média, desvio padrão, mínimo, máximo), 
detectArrhythmia() que procura arritmias baseado na variabilidade, e getCardiacStatus() que dá um resumo completo do estado cardíaco. 

A classe mantém um buffer circular de 30.000 pontos (30 segundos de ECG a 1000Hz) e consegue classificar automaticamente se o batimento cardiaco (HR) está normal, 
em bradicardia ou taquicardia.
Feito com o objetivo de suportar com qualquer fonte de dados cardíacos, seja CardioWheel, Polar ou outros sensores, desde que enviem dados no formato esperado.

"""

import numpy as np
from typing import List, Optional, Any
from datetime import datetime

from ..base import BaseSignal
from ..dataPoint import SignalPoint
from ...core import settings, SignalValidationError

class CardiacSignal(BaseSignal):
    """Sinal cardíaco - ECG e HR (CardioWheel + Polar)"""
    
    def __init__(self):
        # Configuração baseada nos settings reais
        cardiacConfig = settings.signals.cardiacConfig
        
        super().__init__(
            signalName="cardiac",
            bufferSize=cardiacConfig["ecg"]["bufferSize"],          # 30000 (30s * 1000Hz)
            samplingRate=cardiacConfig["ecg"]["samplingRate"]       # 1000Hz
        )

        self.hrNormalRange = cardiacConfig["hr"]["normalRange"]
        self.criticalBpmRange = cardiacConfig["hr"]["criticalRange"]
        self.normalEcgRange = cardiacConfig["ecg"]["normalEcgRange"]
        
        # Thresholds HR
        self.bradycardiaTreshold = cardiacConfig["hr"]["bradycardiaThreshold"]
        self.tachycardiaThreshold = cardiacConfig["hr"]["tachycardiaThreshold"]
        self.severeBradycardiaThreshold = cardiacConfig["hr"]["severeBradycardiaThreshold"]
        self.severeTachycardiaThreshold = cardiacConfig["hr"]["severeTachycardiaThreshold"]
        self.highVariabilityThreshold = cardiacConfig["hr"]["highVariabilityThreshold"]
        self.suddenChangeThreshold = cardiacConfig["hr"]["suddenChangeThreshold"]
        
        # Thresholds ECG
        self.ecgLowAmplitudeThreshold = cardiacConfig["ecg"]["lowAmplitudeThreshold"]
        self.ecgHighAmplitudeThreshold = cardiacConfig["ecg"]["highAmplitudeThreshold"]
        self.ecgFlatThreshold = cardiacConfig["ecg"]["flatThreshold"]
        self.ecgDriftThreshold = cardiacConfig["ecg"]["driftThreshold"]
        
        self.logger.info(f"CardiacSignal initialized - Normal HR Range: {self.hrNormalRange}")
    
    def validateValue(self, value: Any) -> bool:
        """Valida valores de ECG ou HR"""
        
        # HR (single value)
        if isinstance(value, (int, float)):
            if not (self.criticalBpmRange[0] <= value <= self.criticalBpmRange[1]):
                raise SignalValidationError(
                    signalType="cardiac",
                    value=value,
                    reason=f"HR {value} fora do range crítico {self.criticalBpmRange}"
                )
            return True
        
        # ECG (array de samples)
        elif isinstance(value, (list, np.ndarray)):
            if len(value) == 0:
                raise SignalValidationError(
                    signalType="cardiac",
                    value=value,
                    reason="ECG array está vazio"
                )
            
            # Verificar se todos os samples estão no range
            arr = np.array(value)
            if not np.all((arr >= self.normalEcgRange[0]) & (arr <= self.normalEcgRange[1])):
                minVal, maxVal = np.min(arr), np.max(arr)
                raise SignalValidationError(
                    signalType="cardiac",
                    value=f"ECG range [{minVal:.2f}, {maxVal:.2f}]",
                    reason=f"ECG fora do range normal {self.normalEcgRange}"
                )
            return True
        
        # Tipo não suportado
        raise SignalValidationError(
            signalType="cardiac",
            value=type(value).__name__,
            reason="Tipo de valor não suportado (deve ser float para HR ou array para ECG)"
        )
    
    def getNormalRange(self) -> tuple:
        """Range normal para HR (ECG não tem range único)"""
        return self.normalBpmRange
    
    def detectAnomalies(self, recentPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias cardíacas"""
        anomalies = []
        
        if len(recentPoints) < 1:
            return anomalies
        
        # Separar HR e ECG points
        hrPoints = [point for point in recentPoints if isinstance(point.value, (int, float))]
        ecgPoints = [point for point in recentPoints if isinstance(point.value, (list, np.ndarray))]
        
        # Anomalias de HR
        if hrPoints:
            anomalies.extend(self._detectHrAnomalies(hrPoints))
        
        # Anomalias de ECG
        if ecgPoints:
            anomalies.extend(self._detectEcgAnomalies(ecgPoints))
        
        return anomalies
    
    def _detectHrAnomalies(self, hrPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias na frequência cardíaca"""
        anomalies = []
        
        latestHr = hrPoints[-1].value
        
        # Bradicardia 
        if latestHr < self.bradycardiaTreshold:
            severity = "severe" if latestHr < self.severeBradycardiaThreshold else "moderate"
            anomalies.append(f"Bradicardia detectada: {latestHr:.1f} bpm ({severity})")
        
        # Taquicardia 
        elif latestHr > self.tachycardiaThreshold:
            severity = "severe" if latestHr > self.severeTachycardiaThreshold else "moderate"
            anomalies.append(f"Taquicardia detectada: {latestHr:.1f} bpm ({severity})")
        
        # Variabilidade extrema (se temos houver já pelo menos 5 leituras)
        if len(hrPoints) >= 5:
            recentValues = [point.value for point in hrPoints[-5:]]
            variability = max(recentValues) - min(recentValues)
            
            if variability > self.highVariabilityThreshold:
                anomalies.append(f"Alta variabilidade cardíaca: {variability:.1f} bpm")
        
        # Mudança súbita
        if len(hrPoints) >= 2:
            previousHr = hrPoints[-2].value
            change = abs(latestHr - previousHr)
            
            if change > self.suddenChangeThreshold:
                anomalies.append(f"Mudança súbita na frequência cardiaca: {change:.1f} bpm")
        
        return anomalies
    
    def _detectEcgAnomalies(self, ecgPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias no sinal ECG"""
        anomalies = []
        
        if not ecgPoints:
            return anomalies
        
        latestEcg = np.array(ecgPoints[-1].value)
        
        # Amplitude muito baixa (eletrodo solto?)
        amplitude = np.max(latestEcg) - np.min(latestEcg)
        #if amplitude < self.ecgLowAmplitudeThreshold:
            #anomalies.append(f"Amplitude ECG muito baixa: {amplitude:.3f} mV (possível eletrodo solto/ mau contacto)")  #TODO Averiguar como tornar realiable, quando está num ponto entre waves dá sempre trigger nisto.
        
        # Amplitude muito alta (saturação?)
        if amplitude > self.ecgHighAmplitudeThreshold:
            anomalies.append(f"Amplitude ECG muito alta: {amplitude:.3f} mV (intereferência elétrica/ saturação)")
        
        # Sinal muito plano (sem variação) provavelmente algum problema na leitura
        std = np.std(latestEcg)
        #if std < self.ecgFlatThreshold:
            #anomalies.append(f"Sinal ECG muito plano: std={std:.4f} mV") #TODO Averiguar como tornar realiable, quando está num ponto entre waves dá sempre trigger nisto.
        
        # Baseline drift (comparar com pontos anteriores) , às vezes derivado de pior contacto ao longo do tempo devido a suor ou assim
        if len(ecgPoints) >= 3:
            previousEcg = np.array(ecgPoints[-3].value)
            currentBaseline = np.mean(latestEcg)
            previousBaseline = np.mean(previousEcg)
            
            drift = abs(currentBaseline - previousBaseline)
            if drift > self.ecgDriftThreshold:
                anomalies.append(f"Deriva da linha de base: {drift:.2f} mV")
        
        return anomalies
    
    # Métodos específicos para CardiacSignal
    
    def getLatestHr(self) -> Optional[float]:
        """Retorna a última frequência cardíaca"""
        allPoints = self.getAllData()
        
        # Procurar o último ponto de HR (valor único)
        for point in reversed(allPoints):
            if isinstance(point.value, (int, float)):
                return point.value
        
        return None
    
    def getLatestEcgSegment(self, durationSeconds: float = 5.0) -> Optional[np.ndarray]:
        """Retorna segmento ECG dos últimos X segundos"""
        allPoints = self.getAllData()
        
        if not allPoints:
            return None
        
        # Calcular quantos pontos representam a duração desejada
        samplesNeeded = int(durationSeconds * self.samplingRate)
        
        # Coletar pontos ECG recentes
        ecgSamples = []
        for point in reversed(allPoints):
            if isinstance(point.value, (list, np.ndarray)):
                ecgSamples.extend(point.value)
                if len(ecgSamples) >= samplesNeeded:
                    break
        
        if not ecgSamples:
            return None
        
        # Retornar os últimos N samples
        return np.array(ecgSamples[-samplesNeeded:])
    
    def calculateHrStatistics(self, lastMinutes: int = 5) -> Optional[dict]:
        """Calcula estatísticas de HR dos últimos X minutos"""
        allPoints = self.getAllData()
        
        # Filtrar pontos de HR dos últimos X minutos
        cutoffTime = datetime.now().timestamp() - (lastMinutes * 60)
        hrValues = []
        
        for point in allPoints:
            if (point.timestamp >= cutoffTime and 
                isinstance(point.value, (int, float))):
                hrValues.append(point.value)
        
        if len(hrValues) < 3:
            return None
        
        hrArray = np.array(hrValues)
        
        return {
            "mean": float(np.mean(hrArray)),
            "std": float(np.std(hrArray)),
            "min": float(np.min(hrArray)),
            "max": float(np.max(hrArray)),
            "median": float(np.median(hrArray)),
            "sampleCount": len(hrValues),
            "timeRange": f"{lastMinutes} minutes",
            "normalRange": self.hrNormalRange,
            "withinNormalRange": self._countInRange(hrValues, self.hrNormalRange)
        }
    
    def _countInRange(self, values: List[float], range_tuple: tuple) -> dict:
        """Conta quantos valores estão dentro do range normal"""
        inRange = sum(1 for value in values if range_tuple[0] <= value <= range_tuple[1])
        total = len(values)
        
        return {
            "count": inRange,
            "total": total,
            "percentage": (inRange / total * 100) if total > 0 else 0
        }
    
    def detectArrhythmia(self) -> dict:
        """Detecção simples de arritmias baseada em variabilidade"""
        hrStats = self.calculateHrStatistics(lastMinutes=2)
        
        if not hrStats:
            return {"detected": False, "reason": "Insufficient data"}
        
        # Critérios simples para arritmia
        highVariability = hrStats["std"] > 20  # Desvio padrão alto
        extremeValues = (hrStats["min"] < self.bradycardiaTreshold or 
                        hrStats["max"] > self.tachycardiaThreshold)
        
        if highVariability and extremeValues:
            return {
                "detected": True,
                "type": "possible_arrhythmia",
                "confidence": "low",  # Detecção simples
                "details": {
                    "variability": hrStats["std"],
                    "range": (hrStats["min"], hrStats["max"]),
                    "normalRange": self.hrNormalRange
                }
            }
        
        return {"detected": False, "reason": "Normal rhythm"}
    
    def getCardiacStatus(self) -> dict:
        """Status geral do sinal cardíaco"""
        baseStatus = self.getStatus()
        
        # Adicionar informações específicas
        latestHr = self.getLatestHr()
        hrStats = self.calculateHrStatistics(lastMinutes=1)
        arrhythmia = self.detectArrhythmia()
        
        cardiacStatus = {
            **baseStatus,
            "latestHr": latestHr,
            "hrClassification": self._classifyHr(latestHr) if latestHr else None,
            "hrStatistics": hrStats,
            "arrhythmiaDetection": arrhythmia,
            "ecgSegmentAvailable": self.getLatestEcgSegment() is not None,
            "thresholds": {
                "bradycardia": self.bradycardiaTreshold,
                "tachycardia": self.tachycardiaThreshold,
                "hrNormalRange": self.hrNormalRange
            }
        }
        
        return cardiacStatus
    
    def _classifyHr(self, hr: float) -> str:
        """Classifica frequência cardíaca"""
        if hr < self.bradycardiaTreshold:
            return "bradycardia"
        elif hr > self.tachycardiaThreshold:
            return "tachycardia"
        elif self.hrNormalRange[0] <= hr <= self.hrNormalRange[1]:
            return "normal"
        else:
            return "unknown"