"""
Estruturas de dados básicas para sinais #TODO comentar melhor, averiguar qualidade
"""

from pydantic import BaseModel, Field
from typing import Dict, List, Any, Optional
from datetime import datetime
import numpy as np

class SignalPoint(BaseModel):
    """Ponto individual de um sinal"""
    timestamp: float = Field(..., description="Timestamp em segundos")                              # Campo obrigatorio
    value: Any = Field(..., description="Valor do sinal (float, array, dict)")                      # Campo obrigatorio
    quality: float = Field(default=1.0, ge=0.0, le=1.0, description="Qualidade do sinal (0-1)")     # Seria interessante para podermos qualificar a qualidade( ou melhor ainda se for fornecida pelo sensor)
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Metadados adicionais")      
    
    class Config:
        # Permitir numpy arrays
        arbitrary_types_allowed = True

class DataBuffer:
    """Buffer circular para armazenar pontos de sinal"""
    
    def __init__(self, maxSize: int):
        self.maxSize = maxSize
        self.data: List[SignalPoint] = []
        self._currentIndex = 0
        self._isFull = False
    
    def add(self, point: SignalPoint) -> None:
        """Adiciona ponto ao buffer (circular)"""
        if len(self.data) < self.maxSize:
            self.data.append(point)
        else:
            # Buffer cheio - substituir o mais antigo
            self.data[self._currentIndex] = point
            self._currentIndex = (self._currentIndex + 1) % self.maxSize
            self._isFull = True
    
    def getLatest(self, count: int = 1) -> List[SignalPoint]:
        """Retorna os últimos N pontos"""
        if not self.data:
            return []
        
        if not self._isFull:
            # Buffer ainda não está cheio
            return self.data[-count:] if count <= len(self.data) else self.data
        else:
            # Buffer circular cheio
            if count >= self.maxSize:
                return self.getAll()
            
            # Obter últimos N pontos em ordem cronológica
            endIndex = self._currentIndex
            startIndex = (endIndex - count) % self.maxSize
            
            if startIndex < endIndex:
                return self.data[startIndex:endIndex]
            else:
                return self.data[startIndex:] + self.data[:endIndex]
    
    def getAll(self) -> List[SignalPoint]:
        """Retorna todos os pontos em ordem cronológica"""
        if not self._isFull:
            return self.data.copy()
        else:
            # Reordenar buffer circular
            return self.data[self._currentIndex:] + self.data[:self._currentIndex]
    
    def clear(self) -> None:
        """Limpa o buffer"""
        self.data.clear()
        self._currentIndex = 0
        self._isFull = False
    
    def size(self) -> int:
        """Número atual de pontos no buffer"""
        return len(self.data)
    
    def isFull(self) -> bool:
        """Verifica se buffer está cheio"""
        return self._isFull

class SignalMetrics(BaseModel):
    """Métricas estatísticas de um sinal"""
    signalName: str
    sampleCount: int
    timeRange: tuple[float, float]  # (start_timestamp, end_timestamp)
    mean: Optional[float] = None
    std: Optional[float] = None
    minValue: Optional[float] = None
    maxValue: Optional[float] = None
    lastUpdate: datetime
    quality: float = Field(ge=0.0, le=1.0)
    anomalyCount: int = Field(default=0)