"""
EEGSignal - Implementação para sinais EEG (Halo BrainAccess) (CORRIGIDO)

Resumo:
Processa todos os dados cerebrais vindos do Halo BrainAccess. 
Recebe dois tipos de dados principais: o EEG raw (sinais eléctricos de 4 canais cerebrais a 250Hz cada) 
e power bands (análise de frequências delta, theta, alpha, beta, gamma que chegam a cada 1-5Hz). 
Verifica se os valores de EEG estão entre -200 a 200 microvolts (range típico da actividade cerebral) 
e se as power bands somam aproximadamente 1.0 (representam percentagens da actividade total).

Para detecção de anomalias, consegue identificar eletrodos soltos (sinal muito plano com pouca variação), saturação (amplitude constante no máximo), 
movimentação (picos muito altos >180μV), deriva DC (mudança da linha de base), dominância excessiva de delta (possível sonolência com >70% de ondas lentas), 
ausência de alfa (possível stress com <5% de relaxamento), e beta excessivo (possível ansiedade com >60% de ondas rápidas). 
Também detecta mudanças súbitas nas power bands (variações >30% entre leituras consecutivas).

De forma resumida o getLatestRawEeg() serve para obter os dados mais recentes dos 4 canais, getLatestPowerBands() para as percentagens de cada banda de frequência, 
calculateChannelStatistics() para estatísticas de cada canal (amplitude, variabilidade, qualidade), analyzeBrainState() que classifica o estado mental 
como relaxed (relaxado), alert (alerta), drowsy (muito sonolento), sleepy (sonolento/cansado) ou neutral (normal) baseado na dominância das bandas, 
detectElectrodeQuality() que avalia se os contactos estão bons, e getEegStatus() que dá um resumo completo do estado cerebral.

A classe mantém um buffer circular de 30.000 pontos (30 segundos de dados de 4 canais a 250Hz) e consegue interpretar de forma limitada automaticamente o estado cerebral do condutor. 
Foi feita com o objectivo de suportar qualquer fonte de dados EEG, desde que enviem os 4 canais raw e as 5 power bands no formato esperado.
"""

import numpy as np
from typing import List, Optional, Any, Dict, Union
from datetime import datetime

from ..base import BaseSignal
from ..dataPoint import SignalPoint
from ...core import settings, SignalValidationError

class EEGSignal(BaseSignal):
    """Sinal EEG - 4 canais + power bands (Halo BrainAccess)"""
    
    def __init__(self):
        # Configuração EEG: 4 canais * 250Hz * 30s = 30000 pontos por canal
        eegConfig = settings.signals.eegConfig
        
        super().__init__(
            signalName="eeg",
            bufferSize=eegConfig["raw"]["bufferSize"],      # 30000 (30s * 250Hz * 4 canais)
            samplingRate=eegConfig["raw"]["samplingRate"]   # 250Hz
        )
        
        # Channels
        self.channelCount = eegConfig["raw"]["channels"]
        self.channelNames = eegConfig["raw"]["channelNames"]
        self.normalEegRange = eegConfig["raw"]["normalRange"]
        self.saturationThreshold = eegConfig["raw"]["saturationThreshold"]
        
        # Power bands
        self.bandsSamplingRate = eegConfig["bands"]["samplingRate"]
        self.bandsBufferSize = eegConfig["bands"]["bufferSize"]
        self.bandNames = eegConfig["bands"]["bandNames"]
        self.powerBandsTolerance = eegConfig["bands"]["powerBandsTolerance"]
        self.expectedBandRanges = eegConfig["bands"]["expectedBandRanges"]
        
        # Anomalias 
        self.minChannelStd = eegConfig["raw"]["minChannelStd"]
        self.maxChannelAmplitude = eegConfig["raw"]["maxChannelAmplitude"]
        self.maxBaselineDrift = eegConfig["raw"]["maxBaselineDrift"]
        self.deltaExcessThreshold = eegConfig["bands"]["deltaExcessThreshold"]
        self.alphaDeficitThreshold = eegConfig["bands"]["alphaDeficitThreshold"]
        self.betaExcessThreshold = eegConfig["bands"]["betaExcessThreshold"]
        self.bandChangeThreshold = eegConfig["bands"]["bandChangeThreshold"]
        
        # Calcular tolerâncias automaticamente
        self.minPowerBandsSum = 1.0 - self.powerBandsTolerance
        self.maxPowerBandsSum = 1.0 + self.powerBandsTolerance
        
        self.logger.info(f"EEGSignal initialized - {self.channelCount} channels @ {self.samplingRate}Hz, bands @ {self.bandsSamplingRate}Hz")
        
    def validateValue(self, value: Any) -> bool:
        """Valida valores de EEG raw ou power bands"""
        
        # EEG Raw (dict com canais)
        if isinstance(value, dict) and any(f"ch{i+1}" in value for i in range(self.channelCount)):
            try:
                # Verificar se os canais necessários existem
                missingChannels = []
                for channel in self.channelNames:
                    if channel not in value:
                        missingChannels.append(channel)
                
                if missingChannels:
                    raise SignalValidationError(
                        signalType="eeg",
                        value=f"Missing channels: {missingChannels}",
                        reason=f"EEG raw deve ter {self.channelCount} canais: {self.channelNames}"
                    )
                
                # Verificar cada canal
                for channel in self.channelNames:
                    channelData = np.array(value[channel])
                    
                    if len(channelData) == 0:
                        raise SignalValidationError(
                            signalType="eeg",
                            value=f"{channel}: empty array",
                            reason=f"Canal {channel} não pode estar vazio"
                        )
                    
                    # Verificar range de amplitude
                    minVal, maxVal = np.min(channelData), np.max(channelData)
                    if not (self.normalEegRange[0] <= minVal and maxVal <= self.normalEegRange[1]):
                        raise SignalValidationError(
                            signalType="eeg",
                            value=f"{channel}: [{minVal:.1f}, {maxVal:.1f}] μV",
                            reason=f"Canal {channel} fora do range normal {self.normalEegRange} μV"
                        )
                
                return True
                
            except SignalValidationError:
                # Reenvia a exceção SignalValidationError tal como foi lançada, de modo a preservar a mensagem original e de onde surgiu o erro
                raise

            except Exception as e:
                raise SignalValidationError(
                    signalType="eeg",
                    value=str(value)[:100],
                    reason=f"Erro ao validar EEG raw: {e}"
                )
        
        # Power Bands (dict com bandas)
        elif isinstance(value, dict) and any(band in value for band in self.bandNames):
            try:
                # Verificar se todas as bandas existem
                missingBands = []
                for band in self.bandNames:
                    if band not in value:
                        missingBands.append(band)
                
                if missingBands:
                    raise SignalValidationError(
                        signalType="eeg",
                        value=f"Missing bands: {missingBands}",
                        reason=f"Power bands deve ter todas as bandas: {self.bandNames}"
                    )
                
                # Verificar cada banda
                invalidBands = []
                for band in self.bandNames:
                    try:
                        bandValue = float(value[band])
                        if not (0.0 <= bandValue <= 1.0):
                            invalidBands.append(f"{band}={bandValue}")
                    except (ValueError, TypeError):
                        invalidBands.append(f"{band}=non-numeric")
                
                if invalidBands:
                    raise SignalValidationError(
                        signalType="eeg",
                        value=f"Invalid bands: {invalidBands}",
                        reason="Power bands devem ser valores numéricos entre 0.0 e 1.0"
                    )
                
                # Verificar se soma é aproximadamente 1.0 (em teoria devia ser 1 exatamente mas obviamente os equipamentos estão sujeitos a imperfeições, damos 10% de margem)
                # TODO verificar se margem está boa ou não quando testarmos no sim
                totalPower = sum(float(value[band]) for band in self.bandNames)
                if not (self.minPowerBandsSum <= totalPower <= self.maxPowerBandsSum):
                    raise SignalValidationError(
                        signalType="eeg",
                        value=f"Total power: {totalPower:.3f}",
                        reason=f"Soma das power bands deve estar entre {self.minPowerBandsSum:.1f} e {self.maxPowerBandsSum:.1f}"
                    )
                
                return True
                
            except SignalValidationError:
                # Reenvia a exceção SignalValidationError tal como foi lançada, preservando a mensagem original e de onde surgiu o erro
                raise
            except Exception as e:
                raise SignalValidationError(
                    signalType="eeg",
                    value=str(value)[:100],
                    reason=f"Erro ao validar power bands: {e}"
                )
        
        #TODO SE CHEGAR AQUI TAMOS COMPLETAMENTE COOKED
        # Tipo não reconhecido
        raise SignalValidationError(
            signalType="eeg",
            value=type(value).__name__,
            reason=f"Valor deve ser dict com canais EEG {self.channelNames} ou power bands {self.bandNames}"
        )
    
    def getNormalRange(self) -> tuple:
        """Range normal para amplitude EEG"""
        return self.normalEegRange
    
    def detectAnomalies(self, recentPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias em sinais EEG"""
        anomalies = []
        
        if len(recentPoints) < 1:
            return anomalies
        
        # Separar EEG raw e power bands
        rawPoints = [point for point in recentPoints if isinstance(point.value, dict) and "ch1" in point.value]
        bandPoints = [point for point in recentPoints if isinstance(point.value, dict) and "delta" in point.value]
        
        # Anomalias em EEG raw
        if rawPoints:
            anomalies.extend(self._detectRawEegAnomalies(rawPoints))
        
        # Anomalias em power bands
        if bandPoints:
            anomalies.extend(self._detectPowerBandAnomalies(bandPoints))
        
        return anomalies
    
    def _detectRawEegAnomalies(self, rawPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias no EEG raw - usa thresholds das configurações"""
        anomalies = []
        
        if not rawPoints:
            return anomalies
        
        latestEeg = rawPoints[-1].value
        
        # Verificar cada canal
        for channel in self.channelNames:
            if channel not in latestEeg:
                continue
                
            channelData = np.array(latestEeg[channel])
            
            # Saturação (amplitude constante no máximo)
            if np.all(np.abs(channelData) > self.saturationThreshold):
                anomalies.append(f"Saturação detectada no {channel}")
            
            # Sinal muito plano (eletrodo solto ou mal contato geral)
            std = np.std(channelData)
            if std < self.minChannelStd:  # μV - muito baixo para EEG ativo
                anomalies.append(f"Eletrodo possivelmente solto no {channel}: std={std:.3f}μV")
            
            # Provalvelmnete causado pelo movimento (amplitude muito alta)
            maxAmplitude = np.max(np.abs(channelData))
            if maxAmplitude > self.maxChannelAmplitude:  # μV
                anomalies.append(f"Possível movimento brusco do sujeito {channel}: {maxAmplitude:.1f}μV")
            
            # Deriva DC (baseline drift)
            baseline = np.mean(channelData)
            if abs(baseline) > self.maxBaselineDrift:  # μV
                anomalies.append(f"Deriva DC detectada no {channel}: {baseline:.1f}μV")
        
        # Anomalias entre canais
        if len(self.channelNames) == self.channelCount:
            # Verificar se algum canal está muito diferente dos outros
            channelStds = []
            for channel in self.channelNames:
                if channel in latestEeg:
                    channelStds.append(np.std(latestEeg[channel]))
            
            if len(channelStds) >= 3:
                meanStd = np.mean(channelStds)
                for i, std in enumerate(channelStds):
                    if std > meanStd * 3:  # Canal 3x mais ativo que média
                        anomalies.append(f"Atividade anómala elevada no ch{i+1}: {std:.1f}μV vs média {meanStd:.1f}μV")
        
        return anomalies
    
    def _detectPowerBandAnomalies(self, bandPoints: List[SignalPoint]) -> List[str]:
        """Detecta anomalias nas power bands"""
        anomalies = []
        
        if not bandPoints:
            return anomalies
        
        latestBands = bandPoints[-1].value
        
        # Verificar cada banda individualmente
        for band, (minVal, maxVal) in self.expectedBandRanges.items():
            if band not in latestBands:
                continue
            
            value = latestBands[band]
            
            if value < minVal:
                anomalies.append(f"Poder {band} muito baixo: {value:.3f} (esperado ≥{minVal:.3f})")
            elif value > maxVal:
                anomalies.append(f"Poder {band} muito alto: {value:.3f} (esperado ≤{maxVal:.3f})")
        
        
        # Dominância excessiva de delta (possível sonolência)
        if latestBands.get("delta", 0) > self.deltaExcessThreshold:
            anomalies.append(f"Dominância excessiva de ondas delta: {latestBands['delta']:.1%}")
        
        # Ausência de alfa (possível stress)
        if latestBands.get("alpha", 0) < self.alphaDeficitThreshold:
            anomalies.append(f"Actividade alfa muito baixa: {latestBands['alpha']:.1%}")
        
        # Beta excessivo (possível ansiedade/tensão)
        if latestBands.get("beta", 0) > self.betaExcessThreshold:
            anomalies.append(f"Atividade beta excessiva: {latestBands['beta']:.1%}")
        
        # Comparação temporal (entre leitura anterior e atual)
        if len(bandPoints) >= 2:
            # Verificar mudanças súbitas nas bandas
            previousBands = bandPoints[-2].value
            
            for band in ["alpha", "beta", "delta"]:
                if band in latestBands and band in previousBands:
                    change = abs(latestBands[band] - previousBands[band])
                    if change > self.bandChangeThreshold:  # Mudança configurável
                        anomalies.append(f"Mudança súbita em {band}: {change:.1%}")
        
        return anomalies
    
    # Métodos específicos para EEGSignal
    
    def getLatestRawEeg(self, channel: Optional[str] = None) -> Optional[Union[np.ndarray, Dict[str, np.ndarray]]]:
        """Retorna dados EEG raw mais recentes"""
        allPoints = self.getAllData()
        
        # Procurar último ponto de EEG raw
        for point in reversed(allPoints):
            if isinstance(point.value, dict) and "ch1" in point.value:
                if channel:
                    if channel in point.value:
                        return np.array(point.value[channel])
                    else:
                        return None
                else:
                    # Retornar todos os canais
                    return {ch: np.array(point.value.get(ch, [])) 
                           for ch in self.channelNames if ch in point.value}
        
        return None
    
    def getLatestPowerBands(self) -> Optional[Dict[str, float]]:
        """Retorna power bands mais recentes"""
        allPoints = self.getAllData()
        
        for point in reversed(allPoints):
            if isinstance(point.value, dict) and "delta" in point.value:
                return point.value.copy()
        
        return None
    
    def calculateChannelStatistics(self, channel: str, durationSeconds: float = 30.0) -> Optional[dict]:
        """Calcula estatísticas de um canal específico"""
        if channel not in self.channelNames:
            raise SignalValidationError(
                signalType="eeg",
                value=channel,
                reason=f"Canal deve ser um de: {self.channelNames}"
            )
        
        allPoints = self.getAllData()
        
        # Coletar dados do canal dos últimos X segundos
        cutoffTime = datetime.now().timestamp() - durationSeconds
        channelSamples = []
        
        for point in allPoints:
            if (point.timestamp >= cutoffTime and 
                isinstance(point.value, dict) and 
                channel in point.value):
                channelSamples.extend(point.value[channel])
        
        if len(channelSamples) < 100:  # Mínimo de amostras
            return None
        
        data = np.array(channelSamples)
        
        return {
            "channel": channel,
            "mean": float(np.mean(data)),
            "std": float(np.std(data)),
            "min": float(np.min(data)),
            "max": float(np.max(data)),
            "rms": float(np.sqrt(np.mean(data**2))),
            "sampleCount": len(channelSamples),
            "duration": durationSeconds,
            "amplitude": float(np.max(data) - np.min(data)),
            "quality": self._assessChannelQuality(data),
            "normalRange": self.normalEegRange
        }
    
    def _assessChannelQuality(self, data: np.ndarray) -> str:
        """Avalia qualidade de um canal baseado em amplitude e variabilidade"""
        std = np.std(data)
        amplitude = np.max(data) - np.min(data)
        baseline = abs(np.mean(data))
        
        if std < self.minChannelStd:
            return "poor"  # Muito plano
        elif amplitude > self.maxChannelAmplitude:
            return "noisy"  # Muito ruído
        elif baseline > self.maxBaselineDrift:
            return "drift"  # Deriva DC
        elif self.minChannelStd <= std <= 50.0 and 10.0 <= amplitude <= self.maxChannelAmplitude:
            return "good"
        else:
            return "ok"
    
    def analyzeBrainState(self) -> dict:
        """Análise simples do estado cerebral baseada em power bands"""
        latestBands = self.getLatestPowerBands()
        
        if not latestBands:
            return {"state": "unknown", "reason": "No power band data"}
        
        # Classificações simples baseadas em dominância de bandas
        alpha = latestBands.get("alpha", 0)
        beta = latestBands.get("beta", 0)
        delta = latestBands.get("delta", 0)
        theta = latestBands.get("theta", 0)
        gamma = latestBands.get("gamma", 0)
        
        # Estados aproximados (usa thresholds das configurações)
        if alpha > 0.4:
            state = "relaxed"
            confidence = min(alpha * 2, 1.0)
        elif beta > 0.5:
            state = "alert"
            confidence = min(beta * 1.5, 1.0)
        elif delta > self.deltaExcessThreshold:
            state = "drowsy"
            confidence = min(delta * 1.3, 1.0)
        elif theta > 0.4:
            state = "sleepy"
            confidence = min(theta * 2, 1.0)
        else:
            state = "neutral"
            confidence = 0.5
        
        return {
            "state": state,
            "confidence": confidence,
            "powerBands": latestBands,
            "dominantBand": max(latestBands.keys(), key=lambda k: latestBands[k]),
            "analysis": self._interpretBrainState(state, latestBands),
            "thresholds": {
                "deltaExcess": self.deltaExcessThreshold,
                "alphaDeficit": self.alphaDeficitThreshold,
                "betaExcess": self.betaExcessThreshold
            }
        }
    
    def _interpretBrainState(self, state: str, bands: Dict[str, float]) -> str:
        """Interpretação textual do estado cerebral"""
        interpretations = {
            "relaxed": f"Estado relaxado com alta atividade alfa ({bands.get('alpha', 0):.1%})",
            "alert": f"Estado alerta com alta atividade beta ({bands.get('beta', 0):.1%})",
            "drowsy": f"Estado muito sonolento com alta atividade delta ({bands.get('delta', 0):.1%})",
            "sleepy": f"Estado relativamente sonolento com alta atividade theta ({bands.get('theta', 0):.1%})",
            "neutral": "Estado neutro sem dominância clara de nenhuma banda"
        }
        
        return interpretations.get(state, "Estado não classificado")
    
    def getEegStatus(self) -> dict:
        """Status geral do sinal EEG"""
        baseStatus = self.getStatus()
        
        # Informações específicas EEG
        latestBands = self.getLatestPowerBands()
        brainState = self.analyzeBrainState()
        
        # Estatísticas por canal
        channelStats = {}
        for channel in self.channelNames:
            try:
                stats = self.calculateChannelStatistics(channel, durationSeconds=10.0)
                if stats:
                    channelStats[channel] = stats
            except Exception as e:
                self.logger.warning(f"Error calculating stats for {channel}: {e}")
        
        # Qualidade dos eletrodos
        electrodeQuality = self.detectElectrodeQuality()
        
        eegStatus = {
            **baseStatus,
            "latestPowerBands": latestBands,
            "brainState": brainState,
            "channelStatistics": channelStats,
            "electrodeQuality": electrodeQuality,
            "activeChannels": len(channelStats),
            "rawDataAvailable": self.getLatestRawEeg() is not None,
            "configuration": {
                "channels": self.channelCount,
                "samplingRate": self.samplingRate,
                "bandsSamplingRate": self.bandsSamplingRate,
                "expectedBands": self.bandNames,
                "normalRange": self.normalEegRange
            }
        }
        
        return eegStatus
    
    def detectElectrodeQuality(self) -> Dict[str, str]:
        """Avalia qualidade de contacto dos eletrodos"""
        quality = {}
        
        for channel in self.channelNames:
            try:
                stats = self.calculateChannelStatistics(channel, durationSeconds=5.0)
                
                if not stats:
                    quality[channel] = "no_data"
                    continue
                
                # Usar assessment já implementado
                quality[channel] = stats["quality"]
                
            except Exception as e:
                self.logger.warning(f"Error assessing quality for {channel}: {e}")
                quality[channel] = "error"
        
        return quality