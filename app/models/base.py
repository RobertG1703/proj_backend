"""
Classe base abstrata para todos os sinais
"""

from abc import ABC, abstractmethod
from typing import List, Optional, Dict, Any, Union
from datetime import datetime, timedelta
import logging
import numpy as np

from .dataPoint import SignalPoint, DataBuffer, SignalMetrics

class BaseSignal(ABC):
    """Classe base para todos os tipos de sinais"""
    
    def __init__(self, signalName: str, bufferSize: int, samplingRate: Union[int, str] = None):
        self.signalName = signalName
        self.bufferSize = bufferSize
        self.samplingRate = samplingRate  # Hz ou "event" para event-based
        self.buffer = DataBuffer(bufferSize)
        self.isActive = False
        self.lastUpdate: Optional[datetime] = None
        self.anomalies: List[str] = []
        self.logger = logging.getLogger(f"{__name__}.{signalName}")
        
        self.logger.info(f"Signal {signalName} initialized - Buffer: {bufferSize}, Rate: {samplingRate}")
    
    @abstractmethod
    def validateValue(self, value: Any) -> bool:
        """Valida se o valor está dentro dos limites esperados"""
        pass
    
    @abstractmethod
    def getNormalRange(self) -> Optional[tuple]:
        """Retorna o range normal para este sinal (se aplicável)"""
        pass
    
    @abstractmethod
    def detectAnomalies(self, recentPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias baseado nos pontos recentes"""
        pass
    
    def addPoint(self, point: SignalPoint) -> bool:
        """Adiciona ponto ao sinal após validação"""
        try:
            # Validar valor
            if not self.validateValue(point.value):
                self.logger.warning(f"Invalid value for {self.signalName}: {point.value}")
                return False
            
            # Adicionar ao buffer
            self.buffer.add(point)
            
            # Atualizar estado
            self.lastUpdate = datetime.now()
            self.isActive = True
            
            # Detectar anomalias
            self._checkForAnomalies()
            
            self.logger.debug(f"Point added to {self.signalName}: {point.value}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error adding point to {self.signalName}: {e}")
            return False
    
    def getLatest(self, count: int = 1) -> List[SignalPoint]:
        """Retorna os últimos N pontos"""
        return self.buffer.getLatest(count)
    
    def getLatestValue(self) -> Optional[SignalPoint]:
        """Retorna o último ponto"""
        latest = self.buffer.getLatest(1)
        return latest[0] if latest else None
    
    def getAllData(self) -> List[SignalPoint]:
        """Retorna todos os dados do buffer"""
        return self.buffer.getAll()
    
    def getMetrics(self, lastN: Optional[int] = None) -> Optional[SignalMetrics]:
        """Calcula métricas estatísticas"""
        data = self.buffer.getLatest(lastN) if lastN else self.buffer.getAll()
        
        if not data:
            return None
        
        # Extrair valores numéricos (se possível)
        numericValues = self._extractNumericValues(data)
        
        timeRange = (data[0].timestamp, data[-1].timestamp)
        
        metrics = SignalMetrics(
            signalName=self.signalName,
            sampleCount=len(data),
            timeRange=timeRange,
            lastUpdate=self.lastUpdate or datetime.now(),
            quality=np.mean([point.quality for point in data]),
            anomalyCount=len(self.anomalies)
        )
        
        # Calcular stats se há valores numéricos
        if numericValues:
            metrics.mean = float(np.mean(numericValues))
            metrics.std = float(np.std(numericValues))
            metrics.minValue = float(np.min(numericValues))
            metrics.maxValue = float(np.max(numericValues))
        
        return metrics
    
    def _extractNumericValues(self, points: List[SignalPoint]) -> List[float]:
        """Extrai valores numéricos dos pontos (implementação base)"""
        numericValues = []
        for point in points:
            if isinstance(point.value, (int, float)):
                numericValues.append(float(point.value))
            elif isinstance(point.value, (list, np.ndarray)):
                # Para arrays, usar a média
                try:
                    numericValues.append(float(np.mean(point.value)))
                except:
                    pass
        return numericValues
    
    def _checkForAnomalies(self) -> None:
        """Verifica anomalias nos dados recentes"""
        recentPoints = self.buffer.getLatest(20)
        
        if len(recentPoints) < 1:
            return
        
        currentAnomalies = self.detectAnomalies(recentPoints)
        
        # Limpar anomalias se não há problemas atuais
        if not currentAnomalies:
            if self.anomalies:  # Só log se havia anomalias antes
                self.logger.info(f"Anomalias resolvidas para {self.signalName}")
                self.anomalies.clear()
            return
        
        # Adicionar novas anomalias
        for anomaly in currentAnomalies:
            if anomaly not in self.anomalies:
                self.anomalies.append(anomaly)
                self.logger.warning(f"NOVA anomalia detectada em {self.signalName}: {anomaly}")
        
        # Manter limite de 10 anomalias
        if len(self.anomalies) > 10:
            self.anomalies = self.anomalies[-10:]
    
    def getRecentAnomalies(self, maxAge: timedelta = None) -> List[str]:
        """Retorna anomalias recentes""" #TODO 
        #if maxAge is None:
            #maxAge = timedelta(minutes=5)
        
        # Para simplicidade retornar todas as anomalias por enqautno
        return self.anomalies.copy()
    
    def clearAnomalies(self) -> None:
        """Limpa histórico de anomalias"""
        self.anomalies.clear()
        self.logger.info(f"Anomalies cleared for {self.signalName}")
    
    def getStatus(self) -> Dict[str, Any]:
        """Status geral do sinal"""
        timeSinceUpdate = None
        if self.lastUpdate:
            timeSinceUpdate = (datetime.now() - self.lastUpdate).total_seconds()
        
        return {
            "signalName": self.signalName,
            "isActive": self.isActive,
            "bufferSize": self.buffer.size(),
            "bufferCapacity": self.bufferSize,
            "lastUpdate": self.lastUpdate.isoformat() if self.lastUpdate else None,
            "timeSinceUpdate": timeSinceUpdate,
            "anomalyCount": len(self.anomalies),
            "samplingRate": self.samplingRate
        }
    
    def reset(self) -> None:
        """Reset completo do sinal"""
        self.buffer.clear()
        self.anomalies.clear()
        self.isActive = False
        self.lastUpdate = None
        self.logger.info(f"Signal {self.signalName} reset")