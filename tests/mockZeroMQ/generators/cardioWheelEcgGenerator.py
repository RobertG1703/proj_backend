"""
CardioWheelEcgGenerator - Gerador de dados ECG realistas

Resumo:
Gera dados ECG simulando o CardioWheel real com:
- Valores ADC 16-bit realistas (baseline ~1650)
- Ruído gaussiano apropriado
- Chunks de 20 samples (20ms @ 1000Hz real)
- Anomalias específicas: amplitude baixa, saturação, deriva, sinal plano
- Timestamps incrementais corretos para chunks sequenciais
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional
from enum import Enum

from app.core import settings

class EcgAnomalyType(Enum):
    """Tipos de anomalias ECG"""
    NORMAL = "normal"
    LOW_AMPLITUDE = "low_amplitude"           # Eletrodo solto
    HIGH_AMPLITUDE = "high_amplitude"         # Saturação/interferência  
    FLAT_SIGNAL = "flat_signal"               # Sinal plano
    BASELINE_DRIFT = "baseline_drift"         # Deriva DC
    NOISE_BURST = "noise_burst"               # Rajada de ruído

class CardioWheelEcgGenerator:
    """Gerador de dados ECG para tópico CardioWheel_ECG"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações ECG do settings
        self.ecgConfig = settings.signals.cardiacConfig["ecg"]
        self.zmqConfig = settings.zeromq.topicProcessingConfig["CardioWheel_ECG"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração ECG
        self.samplingRate = self.zmqConfig["samplingRate"]              # 1000Hz
        self.chunkSize = self.mockConfig.topicChunkSizes["CardioWheel_ECG"]  # 20 samples
        self.chunkDuration = self.chunkSize / self.samplingRate         # 0.02s (20ms)
        
        # Configurações ADC e conversão
        self.generatorConfig = self.mockConfig.generatorBaseConfig["cardiac"]
        self.baselineValue = self.generatorConfig["ecgAmplitudeBase"]   # 1650 ADC
        self.noiseStd = self.generatorConfig["ecgNoiseStd"]             # 10 ADC units
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["CardioWheel_ECG"]  # 3%
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = EcgAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Parâmetros para simulação de ECG
        self.baseHr = self.generatorConfig["baseHr"]                    # 75 BPM
        self.hrVariation = self.generatorConfig["hrVariationStd"]       # ±5 BPM
        self.currentHr = self.baseHr
        
        # Estados para gerar formas de onda ECG realistas
        self.ecgPhase = 0.0  # Fase atual na forma de onda
        
        self.logger.info(f"CardioWheelEcgGenerator initialized - {self.samplingRate}Hz, chunks of {self.chunkSize}")
    
    def generateChunk(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um chunk de dados ECG (20 samples).
        
        Args:
            baseTimestamp: Timestamp base para o chunk (usa interno se None)
            
        Returns:
            Dict com arrays de dados ECG e LOD para formatação
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Gerar samples ECG para o chunk
            ecgSamples = []
            lodSamples = []
            
            for i in range(self.chunkSize):
                # Gerar sample ECG baseado no tipo atual
                ecgValue = self._generateEcgSample()
                lodValue = self._generateLodSample()
                
                ecgSamples.append(int(ecgValue))  # ADC values são inteiros
                lodSamples.append(lodValue)
                
                # Avançar contadores
                self.sampleCounter += 1
                self.ecgPhase += (2 * np.pi * self.currentHr / 60) / self.samplingRate
                
                # Manter fase no range [0, 2π]
                if self.ecgPhase >= 2 * np.pi:
                    self.ecgPhase -= 2 * np.pi
                    # Variar HR ligeiramente a cada batimento
                    self._updateHeartRate()


            # Avançar timestamp para próximo chunk
            self.currentTimestamp += self.chunkDuration
            
            result = {
                "ecg": ecgSamples,
                "lod": lodSamples,
                "chunkTimestamp": self.currentTimestamp - self.chunkDuration,
                "anomalyType": self.currentAnomalyType.value,
                "samplingRate": self.samplingRate,
                "chunkSize": self.chunkSize
            }
            
            self.logger.debug(f"Generated ECG chunk: {len(ecgSamples)} samples, anomaly: {self.currentAnomalyType.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating ECG chunk: {e}")
            raise
    
    def _generateEcgSample(self) -> float:
        """
        Gera um sample ECG individual baseado no estado atual.
        
        Returns:
            Valor ECG em ADC units
        """
        
        # Forma de onda ECG básica (simplificada)
        if self.currentAnomalyType == EcgAnomalyType.NORMAL:
            # ECG normal com QRS, P, T waves simuladas
            ecgWave = self._generateNormalEcgWave()
            
        elif self.currentAnomalyType == EcgAnomalyType.LOW_AMPLITUDE:
            # Amplitude muito baixa (eletrodo solto)
            ecgWave = self._generateNormalEcgWave() * 0.05  # 5% da amplitude normal
            
        elif self.currentAnomalyType == EcgAnomalyType.HIGH_AMPLITUDE:
            # Saturação ou interferência
            ecgWave = self._generateNormalEcgWave() * 5.0   # 5x amplitude normal
            # Clipar no máximo ADC
            ecgWave = np.clip(ecgWave, -500, 500)
            
        elif self.currentAnomalyType == EcgAnomalyType.FLAT_SIGNAL:
            # Sinal completamente plano
            ecgWave = 0.0
            
        elif self.currentAnomalyType == EcgAnomalyType.BASELINE_DRIFT:
            # Deriva da linha de base
            driftAmount = 100 * np.sin(self.sampleCounter * 0.001)  # Deriva lenta
            ecgWave = self._generateNormalEcgWave() + driftAmount
            
        elif self.currentAnomalyType == EcgAnomalyType.NOISE_BURST:
            # Rajada de ruído
            ecgWave = self._generateNormalEcgWave() + np.random.normal(0, self.noiseStd * 3)
            
        else:
            ecgWave = self._generateNormalEcgWave()
        
        # Adicionar ruído gaussiano base
        noise = np.random.normal(0, self.noiseStd)
        
        # Valor final ADC
        adcValue = self.baselineValue + ecgWave + noise
        # Clipar para range ADC 16-bit 
        return np.clip(adcValue, -32768, 32767)
    
    def _generateNormalEcgWave(self) -> float:
        """
        Gera forma de onda ECG normal simplificada.
        
        Returns:
            Amplitude ECG relativa ao baseline
        """
        # Fatores de escala (1 mV = 6400 ADC)
        P_SCALE = 6400    # 0.2 mV * 6400 = 1280 ADC
        QRS_SCALE = 6400  # 1.5 mV * 6400 = 9600 ADC
        T_SCALE = 6400    # 0.3 mV * 6400 = 1920 ADC

        
        # ECG simplificado com 3 componentes principais
        # P wave (pequena, antes do QRS)
        pWave = 0.2 * P_SCALE * np.exp(-((self.ecgPhase - 0.2) / 0.1)**2) if 0.1 < self.ecgPhase < 0.3 else 0
        
        # QRS complex (grande, sharp)
        if 0.8 < self.ecgPhase < 1.2:
            qrsPhase = (self.ecgPhase - 1.0) / 0.2
            qrsWave = 1.5 * QRS_SCALE * np.exp(-(qrsPhase**2) * 10)
        else:
            qrsWave = 0
            
        
        # T wave (média, depois do QRS)
        tWave = 0.3 * T_SCALE * np.exp(-((self.ecgPhase - 1.8) / 0.3)**2) if 1.4 < self.ecgPhase < 2.2 else 0
        
        return pWave + qrsWave + tWave
    
    def _generateLodSample(self) -> int:
        """
        Gera sample LOD (Lead-Off Detection).
        
        Returns:
            0 para eletrodo conectado, 1 para solto
        """
        
        # LOD ativo se anomalia de baixa amplitude (eletrodo solto)
        if self.currentAnomalyType == EcgAnomalyType.LOW_AMPLITUDE:
            return 1
        else:
            return 0
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != EcgAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = EcgAnomalyType.NORMAL
                self.logger.debug(f"ECG anomaly ended at {currentTime:.3f}s")
            return
        
        # Verificar se deve injetar nova anomalia
        if not self.anomalyConfig["enabled"]:
            return
        
        # Intervalo mínimo entre anomalias
        if currentTime - self.lastAnomalyTime < self.anomalyConfig["minInterval"]:
            return
        
        # Probabilidade de anomalia
        if np.random.random() < self.anomalyChance:
            # Escolher tipo de anomalia aleatoriamente
            anomalyTypes = [
                EcgAnomalyType.LOW_AMPLITUDE,
                EcgAnomalyType.HIGH_AMPLITUDE, 
                EcgAnomalyType.FLAT_SIGNAL,
                EcgAnomalyType.BASELINE_DRIFT,
                EcgAnomalyType.NOISE_BURST
            ]
            
            self.currentAnomalyType = np.random.choice(anomalyTypes)
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia (2-10 segundos)
            self.anomalyDuration = np.random.uniform(2.0, 10.0)
            
            self.logger.warning(f"ECG anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def _updateHeartRate(self):
        """
        Atualiza HR com variação natural.
        """
        
        # Variação gradual do HR
        hrChange = np.random.normal(0, self.hrVariation * 0.1)  # Mudança pequena
        self.currentHr += hrChange
        
        # Manter HR em range razoável
        self.currentHr = np.clip(self.currentHr, 50, 120)
    
    def forceAnomaly(self, anomalyType: str, duration: float = 5.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("low_amplitude", "high_amplitude", etc.)
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = EcgAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced ECG anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown ECG anomaly type: {anomalyType}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "CardioWheelECG",
            "samplingRate": self.samplingRate,
            "chunkSize": self.chunkSize,
            "currentTimestamp": self.currentTimestamp,
            "sampleCounter": self.sampleCounter,
            "currentHr": self.currentHr,
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != EcgAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "config": {
                "baselineValue": self.baselineValue,
                "noiseStd": self.noiseStd,
                "anomalyChance": self.anomalyChance,
                "baseHr": self.baseHr
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = EcgAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.ecgPhase = 0.0
        self.currentHr = self.baseHr
        
        self.logger.info("CardioWheelEcgGenerator reset")

# Instância global
cardioWheelEcgGenerator = CardioWheelEcgGenerator()