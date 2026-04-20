"""
BrainAccessEegGenerator - Gerador de dados EEG realistas

Resumo:
Gera dados EEG simulando o BrainAccess Halo real com:
- EEG raw data para 4 canais (ch0-ch3) @ 250Hz em chunks de 10 samples
- Power bands (delta, theta, alpha, beta, gamma) que somam ~1.0
- Estados cerebrais realísticos: relaxed, alert, drowsy, sleepy, neutral
- Anomalias específicas: eletrodo solto, saturação, artefactos, mudanças de estado
- Alternância entre raw data e power bands baseada em probabilidade
"""

import logging
import numpy as np
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from enum import Enum

from app.core import settings

class EegAnomalyType(Enum):
    """Tipos de anomalias EEG"""
    NORMAL = "normal"
    ELECTRODE_LOOSE = "electrode_loose"       # Eletrodo solto
    SATURATION = "saturation"                 # Saturação do sinal
    MOVEMENT_ARTIFACT = "movement_artifact"   # Artefacto de movimento
    EYE_BLINK_ARTIFACT = "eye_blink_artifact" # Artefacto de piscar olhos
    MUSCLE_ARTIFACT = "muscle_artifact"       # Artefacto muscular
    DC_DRIFT = "dc_drift"                     # Deriva DC
    HIGH_NOISE = "high_noise"                 # Ruído elevado
    CHANNEL_FAILURE = "channel_failure"       # Falha de canal

class BrainState(Enum):
    """Estados cerebrais simulados"""
    RELAXED = "relaxed"       # Relaxado (alta alfa)
    ALERT = "alert"           # Alerta (alta beta)
    DROWSY = "drowsy"         # Muito sonolento (alta delta)
    SLEEPY = "sleepy"         # Sonolento/cansado (alta theta)
    NEUTRAL = "neutral"       # Estado neutro/normal

class DataType(Enum):
    """Tipos de dados EEG a gerar"""
    RAW_EEG = "raw_eeg"       # Dados raw dos 4 canais
    POWER_BANDS = "power_bands"  # Power bands calculadas

class BrainAccessEegGenerator:
    """Gerador de dados EEG para tópico BrainAcess_EEG"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações EEG do settings
        self.eegConfig = settings.signals.eegConfig
        self.zmqConfig = settings.zeromq.topicProcessingConfig["BrainAcess_EEG"]
        self.mockConfig = settings.mockZeromq
        
        # Parâmetros de geração EEG
        self.samplingRate = self.zmqConfig["samplingRate"]              # 250Hz
        self.chunkSize = self.mockConfig.topicChunkSizes["BrainAcess_EEG"]  # 10 samples
        self.chunkDuration = self.chunkSize / self.samplingRate         # 0.04s (40ms)
        
        # Configurações dos canais
        self.channelCount = self.eegConfig["raw"]["channels"]           # 4
        self.channelNames = self.eegConfig["raw"]["channelNames"]       # ["ch0", "ch1", "ch2", "ch3"]
        self.normalRange = self.eegConfig["raw"]["normalRange"]         # (-200, 200) μV
        
        # Configurações power bands
        self.bandNames = self.eegConfig["bands"]["bandNames"]           # ["delta", "theta", "alpha", "beta", "gamma"]
        self.brainStateTemplates = self.eegConfig["brainStates"]["stateBandTemplates"]
        
        # Configurações de anomalias
        self.anomalyConfig = self.mockConfig.anomalyInjection
        self.anomalyChance = self.anomalyConfig["topicChances"]["CardioWheel_ECG"]  # 3% (usar ECG como base)
        
        # Estado interno do gerador
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = EegAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        
        # Estado cerebral
        self.currentBrainState = BrainState.NEUTRAL
        self.stateStartTime = 0.0
        self.stateDuration = 0.0
        self.stateTransitionProgress = 0.0
        
        # Configurações de geração
        self.baseAmplitude = 20.0                  # μV amplitude base
        self.noiseStd = 8.0                        # μV ruído gaussiano
        self.powerBandsGenerationChance = 0        # #TODO (Supostamente isto nunca vai ser gerado no sim...) 0% chance de gerar power bands vs raw
        
        # Fases para oscilações por canal (para variação entre canais)
        self.channelPhases = [0.0, 0.1, 0.2, 0.3]  # Diferenças de fase entre canais
        self.channelOffsets = [0.0, -3.0, 2.0, -1.0]  # Offsets DC por canal
        
        self.logger.info(f"BrainAccessEegGenerator initialized - {self.samplingRate}Hz, {self.channelCount} channels, chunks of {self.chunkSize}")
    
    def generateChunk(self, baseTimestamp: Optional[float] = None) -> Dict[str, Any]:
        """
        Gera um chunk de dados EEG (raw ou power bands).
        
        Args:
            baseTimestamp: Timestamp base para o chunk (usa interno se None)
            
        Returns:
            Dict com dados EEG formatados
        """
        
        if baseTimestamp is not None:
            self.currentTimestamp = baseTimestamp
        
        try:
            # Atualizar estado cerebral
            self._updateBrainState()
            
            # Verificar se deve injetar anomalia
            self._updateAnomalyState()
            
            # Decidir tipo de dados a gerar
            if np.random.random() < self.powerBandsGenerationChance:
                dataType = DataType.POWER_BANDS
                result = self._generatePowerBandsChunk()
            else:
                dataType = DataType.RAW_EEG
                result = self._generateRawEegChunk()
            
            # Adicionar metadados comuns (só para power bands, raw EEG vai direto para formatter)
            if dataType == DataType.POWER_BANDS:
                result.update({
                    "chunkTimestamp": self.currentTimestamp,
                    "dataType": dataType.value,
                    "brainState": self.currentBrainState.value,
                    "anomalyType": self.currentAnomalyType.value,
                    "samplingRate": self.samplingRate,
                    "chunkSize": self.chunkSize
                })
            
            # Avançar timestamp para próximo chunk
            self.currentTimestamp += self.chunkDuration
            self.sampleCounter += self.chunkSize
            
            self.logger.debug(f"Generated EEG chunk: {dataType.value}, state: {self.currentBrainState.value}, anomaly: {self.currentAnomalyType.value}")
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error generating EEG chunk: {e}")
            raise
    
    def _generateRawEegChunk(self) -> Dict[str, Any]:
        """
        Gera chunk de dados EEG raw para os 4 canais.
        
        Returns:
            Dict com dados raw dos canais no formato esperado pelo formatter
        """
        
        # Obter template do estado cerebral atual
        stateTemplate = self._getCurrentStateAmplitudes()
        
        # Gerar dados por canal - formato direto para o formatter
        channelData = {}
        
        for i, channel in enumerate(self.channelNames):
            channelSamples = []
            
            for sampleIdx in range(self.chunkSize):
                # Gerar sample EEG baseado no estado cerebral
                sample = self._generateEegSample(i, sampleIdx, stateTemplate)
                
                # Aplicar anomalias se ativas
                if self.currentAnomalyType != EegAnomalyType.NORMAL:
                    sample = self._applyRawAnomalies(sample, i, sampleIdx)
                
                # Clipar para range normal
                sample = np.clip(sample, self.normalRange[0], self.normalRange[1])
                channelSamples.append(round(sample, 2))
            
            # Adicionar diretamente ao resultado (sem wrapper "eegRaw")
            channelData[channel] = channelSamples
        
        return channelData
    
    def _generatePowerBandsChunk(self) -> Dict[str, Any]:
        """
        Gera chunk de power bands baseado no estado cerebral.
        
        Returns:
            Dict com power bands que somam ~1.0
        """
        
        # Obter template base do estado cerebral
        baseTemplate = self.brainStateTemplates[self.currentBrainState.value]
        
        # Adicionar variação natural
        powerBands = {}
        totalPower = 0.0
        
        for band in self.bandNames:
            basePower = baseTemplate[band]
            # Variação de ±5% do valor base
            variation = np.random.normal(0, 0.05)
            power = max(0.01, basePower + variation)  # Mínimo 1%
            powerBands[band] = power
            totalPower += power
        
        # Normalizar para somar ~1.0
        if totalPower > 0:
            for band in self.bandNames:
                powerBands[band] = powerBands[band] / totalPower
        
        # Aplicar anomalias nas power bands
        if self.currentAnomalyType != EegAnomalyType.NORMAL:
            powerBands = self._applyPowerBandAnomalies(powerBands)
        
        # Arredondar para 3 casas decimais
        for band in self.bandNames:
            powerBands[band] = round(powerBands[band], 3)
        
        return {
            "eegBands": powerBands,
            "totalPower": round(sum(powerBands.values()), 3)
        }
    
    def _generateEegSample(self, channelIndex: int, sampleIndex: int, stateTemplate: Dict) -> float:
        """
        Gera um sample EEG individual para um canal.
        
        Args:
            channelIndex: Índice do canal (0-3)
            sampleIndex: Índice do sample no chunk
            stateTemplate: Template de amplitudes por frequência
            
        Returns:
            Valor EEG em μV
        """
        
        # Timestamp atual do sample
        sampleTime = self.currentTimestamp + (sampleIndex / self.samplingRate)
        
        # Componentes de frequência baseadas no estado cerebral
        eegValue = 0.0
        
        # Adicionar componentes principais baseadas no estado
        for freqName, amplitude in stateTemplate.items():
            if freqName in ["primary", "secondary"]:
                freq = stateTemplate.get("frequencies", [10, 8])[0 if freqName == "primary" else 1]
                phase = self.channelPhases[channelIndex]
                eegValue += amplitude * np.sin(2 * np.pi * freq * sampleTime + phase)
        
        # Adicionar componentes adicionais se definidas
        if "amplitudes" in stateTemplate and "frequencies" in stateTemplate:
            for amp, freq in zip(stateTemplate["amplitudes"], stateTemplate["frequencies"]):
                phase = self.channelPhases[channelIndex]
                eegValue += amp * np.sin(2 * np.pi * freq * sampleTime + phase)
        
        # Adicionar offset DC por canal
        eegValue += self.channelOffsets[channelIndex]
        
        # Adicionar ruído gaussiano
        noise = np.random.normal(0, self.noiseStd)
        eegValue += noise
        
        return eegValue
    
    def _getCurrentStateAmplitudes(self) -> Dict:
        """
        Retorna amplitudes para o estado cerebral atual.
        
        Returns:
            Dict com amplitudes e frequências
        """
        
        # Templates de amplitude por estado cerebral
        stateAmplitudes = {
            BrainState.RELAXED: {
                "primary": 30, "secondary": 15, "frequencies": [10, 8]  # Alfa dominante
            },
            BrainState.ALERT: {
                "primary": 25, "secondary": 20, "frequencies": [18, 22]  # Beta dominante
            },
            BrainState.DROWSY: {
                "primary": 40, "secondary": 25, "frequencies": [2, 3]  # Delta dominante
            },
            BrainState.SLEEPY: {
                "primary": 35, "secondary": 20, "frequencies": [6, 7]  # Theta dominante
            },
            BrainState.NEUTRAL: {
                "amplitudes": [20, 15, 10], "frequencies": [10, 18, 3]  # Misto
            }
        }
        
        return stateAmplitudes[self.currentBrainState]
    
    def _applyRawAnomalies(self, sample: float, channelIndex: int, sampleIndex: int) -> float:
        """
        Aplica anomalias ao sample de EEG raw.
        
        Args:
            sample: Valor EEG base
            channelIndex: Índice do canal
            sampleIndex: Índice do sample
            
        Returns:
            Valor modificado pela anomalia
        """
        
        if self.currentAnomalyType == EegAnomalyType.ELECTRODE_LOOSE:
            # Eletrodo solto - amplitude muito baixa
            return sample * 0.05
            
        elif self.currentAnomalyType == EegAnomalyType.SATURATION:
            # Saturação - clipar no máximo
            return np.sign(sample) * (self.normalRange[1] - 1)
            
        elif self.currentAnomalyType == EegAnomalyType.MOVEMENT_ARTIFACT:
            # Artefacto de movimento - picos altos
            if sampleIndex < 3:  # Primeiros samples
                artifactAmplitude = np.random.uniform(100, 180)
                direction = np.random.choice([-1, 1])
                return sample + (artifactAmplitude * direction)
                
        elif self.currentAnomalyType == EegAnomalyType.EYE_BLINK_ARTIFACT:
            # Artefacto de piscar - principalmente em ch0 e ch1 (frontais)
            if channelIndex < 2 and sampleIndex in [2, 3, 4]:
                blinkAmplitude = np.random.uniform(50, 120)
                return sample + blinkAmplitude
                
        elif self.currentAnomalyType == EegAnomalyType.MUSCLE_ARTIFACT:
            # Artefacto muscular - alta frequência
            muscleNoise = np.random.normal(0, 30)
            return sample + muscleNoise
            
        elif self.currentAnomalyType == EegAnomalyType.DC_DRIFT:
            # Deriva DC - offset crescente
            drift = (self.sampleCounter + sampleIndex) * 0.1
            return sample + drift
            
        elif self.currentAnomalyType == EegAnomalyType.HIGH_NOISE:
            # Ruído elevado
            highNoise = np.random.normal(0, 25)
            return sample + highNoise
            
        elif self.currentAnomalyType == EegAnomalyType.CHANNEL_FAILURE:
            # Falha de canal - só afetar um canal específico
            if channelIndex == 1:  # Afetar ch1
                return 0.0
        
        return sample
    
    def _applyPowerBandAnomalies(self, powerBands: Dict[str, float]) -> Dict[str, float]:
        """
        Aplica anomalias às power bands.
        
        Args:
            powerBands: Power bands base
            
        Returns:
            Power bands modificadas
        """
        
        if self.currentAnomalyType == EegAnomalyType.ELECTRODE_LOOSE:
            # Eletrodo solto - dominância de ruído (alta gamma)
            powerBands["gamma"] = 0.7
            remaining = 0.3
            for band in ["delta", "theta", "alpha", "beta"]:
                powerBands[band] = remaining / 4
                
        elif self.currentAnomalyType == EegAnomalyType.MOVEMENT_ARTIFACT:
            # Movimento - dominância de baixas frequências
            powerBands["delta"] = 0.6
            powerBands["theta"] = 0.25
            powerBands["alpha"] = 0.1
            powerBands["beta"] = 0.04
            powerBands["gamma"] = 0.01
            
        elif self.currentAnomalyType == EegAnomalyType.MUSCLE_ARTIFACT:
            # Músculo - dominância de altas frequências
            powerBands["beta"] = 0.4
            powerBands["gamma"] = 0.35
            powerBands["alpha"] = 0.15
            powerBands["theta"] = 0.07
            powerBands["delta"] = 0.03
        
        # Renormalizar
        total = sum(powerBands.values())
        if total > 0:
            for band in powerBands:
                powerBands[band] = powerBands[band] / total
        
        return powerBands
    
    def _updateBrainState(self):
        """
        Atualiza estado cerebral baseado em timing e probabilidades.
        """
        
        currentTime = self.currentTimestamp
        
        # Verificar se deve mudar estado
        if currentTime - self.stateStartTime >= self.stateDuration:
            # Escolher novo estado baseado em probabilidades
            states = [
                BrainState.NEUTRAL,    # 40% - Mais comum
                BrainState.RELAXED,    # 25%
                BrainState.ALERT,      # 20%
                BrainState.SLEEPY,     # 10%
                BrainState.DROWSY      # 5%
            ]
            
            weights = [0.40, 0.25, 0.20, 0.10, 0.05]
            self.currentBrainState = np.random.choice(states, p=weights)
            
            self.stateStartTime = currentTime
            
            # Duração do estado baseada no tipo
            if self.currentBrainState == BrainState.DROWSY:
                self.stateDuration = np.random.uniform(5.0, 15.0)   # Mais curta
            elif self.currentBrainState == BrainState.ALERT:
                self.stateDuration = np.random.uniform(20.0, 60.0)  # Mais longa
            else:
                self.stateDuration = np.random.uniform(15.0, 45.0)  # Normal
            
            self.logger.debug(f"Brain state changed to: {self.currentBrainState.value} for {self.stateDuration:.1f}s")
    
    def _updateAnomalyState(self):
        """
        Atualiza estado de anomalias baseado em probabilidades e timing.
        """
        
        currentTime = self.currentTimestamp
        
        # Se já há uma anomalia ativa, verificar se deve terminar
        if self.currentAnomalyType != EegAnomalyType.NORMAL:
            if currentTime - self.anomalyStartTime >= self.anomalyDuration:
                self.currentAnomalyType = EegAnomalyType.NORMAL
                self.logger.debug(f"EEG anomaly ended at {currentTime:.3f}s")
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
                EegAnomalyType.MOVEMENT_ARTIFACT,    # 25% - Mais comum
                EegAnomalyType.EYE_BLINK_ARTIFACT,   # 20%
                EegAnomalyType.MUSCLE_ARTIFACT,      # 15%
                EegAnomalyType.HIGH_NOISE,           # 12%
                EegAnomalyType.ELECTRODE_LOOSE,      # 10%
                EegAnomalyType.DC_DRIFT,             # 8%
                EegAnomalyType.SATURATION,           # 6%
                EegAnomalyType.CHANNEL_FAILURE       # 4%
            ]
            
            weights = [0.25, 0.20, 0.15, 0.12, 0.10, 0.08, 0.06, 0.04]
            self.currentAnomalyType = np.random.choice(anomalyTypes, p=weights)
            
            self.anomalyStartTime = currentTime
            self.lastAnomalyTime = currentTime
            
            # Duração da anomalia baseada no tipo
            if self.currentAnomalyType in [EegAnomalyType.EYE_BLINK_ARTIFACT, EegAnomalyType.MOVEMENT_ARTIFACT]:
                self.anomalyDuration = np.random.uniform(0.5, 3.0)    # Muito curta
            elif self.currentAnomalyType == EegAnomalyType.CHANNEL_FAILURE:
                self.anomalyDuration = np.random.uniform(5.0, 20.0)   # Longa
            else:
                self.anomalyDuration = np.random.uniform(2.0, 10.0)   # Normal
            
            self.logger.warning(f"EEG anomaly started: {self.currentAnomalyType.value} for {self.anomalyDuration:.1f}s")
    
    def forceAnomaly(self, anomalyType: str, duration: float = 5.0):
        """
        Força injeção de anomalia específica.
        
        Args:
            anomalyType: Tipo de anomalia ("electrode_loose", "saturation", etc.)
            duration: Duração da anomalia em segundos
        """
        
        try:
            self.currentAnomalyType = EegAnomalyType(anomalyType)
            self.anomalyStartTime = self.currentTimestamp
            self.lastAnomalyTime = self.currentTimestamp
            self.anomalyDuration = duration
            
            self.logger.warning(f"Forced EEG anomaly: {anomalyType} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown EEG anomaly type: {anomalyType}")
    
    def forceBrainState(self, state: str, duration: float = 30.0):
        """
        Força estado cerebral específico.
        
        Args:
            state: Estado cerebral ("relaxed", "alert", etc.)
            duration: Duração do estado em segundos
        """
        
        try:
            self.currentBrainState = BrainState(state)
            self.stateStartTime = self.currentTimestamp
            self.stateDuration = duration
            
            self.logger.info(f"Forced brain state: {state} for {duration}s")
            
        except ValueError:
            self.logger.error(f"Unknown brain state: {state}")
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status atual do gerador.
        
        Returns:
            Status detalhado do gerador
        """
        
        return {
            "generatorType": "BrainAccessEEG",
            "samplingRate": self.samplingRate,
            "chunkSize": self.chunkSize,
            "channelCount": self.channelCount,
            "currentTimestamp": self.currentTimestamp,
            "sampleCounter": self.sampleCounter,
            "currentBrainState": self.currentBrainState.value,
            "currentAnomalyType": self.currentAnomalyType.value,
            "anomalyActive": self.currentAnomalyType != EegAnomalyType.NORMAL,
            "anomalyTimeRemaining": max(0, (self.anomalyStartTime + self.anomalyDuration) - self.currentTimestamp),
            "stateTimeRemaining": max(0, (self.stateStartTime + self.stateDuration) - self.currentTimestamp),
            "powerBandsGenerationChance": self.powerBandsGenerationChance,
            "config": {
                "channelNames": self.channelNames,
                "normalRange": self.normalRange,
                "bandNames": self.bandNames,
                "baseAmplitude": self.baseAmplitude,
                "noiseStd": self.noiseStd,
                "anomalyChance": self.anomalyChance
            }
        }
    
    def reset(self):
        """
        Reset do estado interno do gerador.
        """
        
        self.currentTimestamp = 0.0
        self.sampleCounter = 0
        self.lastAnomalyTime = 0.0
        self.currentAnomalyType = EegAnomalyType.NORMAL
        self.anomalyDuration = 0.0
        self.anomalyStartTime = 0.0
        self.currentBrainState = BrainState.NEUTRAL
        self.stateStartTime = 0.0
        self.stateDuration = 0.0
        self.channelPhases = [0.0, 0.1, 0.2, 0.3]
        
        self.logger.info("BrainAccessEegGenerator reset")

# Instância global
brainAccessEegGenerator = BrainAccessEegGenerator()