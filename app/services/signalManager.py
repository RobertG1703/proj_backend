"""
SignalManager 

Resumo:
Coordena todos os tipos de sinais (CardiacSignal, EEGSignal, SensorsSignal) e processa dados 
vindos de diferentes fontes (ZeroMQ, DataStreamer). Funciona como um "manager"
que recebe dados brutos, os distribui pelos sinais corretos, e emite eventos 
quando processados. Inclui validação, gestão de erros, e avaliação da saúde geral do sistema.
Inclui controlo granular de sinais através do sistema Signal Control para filtering
por signal types individuais.
"""

import logging
from typing import Dict, List, Optional, Any, Set
from datetime import datetime

from ..models.signals.cardiacSignal import CardiacSignal
from ..models.signals.eegSignal import EEGSignal
from ..models.signals.sensorSignal import SensorsSignal
from ..models.signals.cameraSignal import CameraSignal
from ..models.signals.unitySignal import UnitySignal
from ..models.base import SignalPoint
from ..core import eventManager, settings
from ..core.signalControl import SignalControlInterface, SignalState, ComponentState, signalControlManager

class SignalManager(SignalControlInterface):
    """Manager central para coordenar sinais com controlo de sinais"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Sinais implementados: cardiac + EEG + sensors
        self.signals: Dict[str, Any] = {
            "cardiac": CardiacSignal(),
            "eeg": EEGSignal(),
            "sensors": SensorsSignal(),
            "camera": CameraSignal(),
            "unity": UnitySignal()
        }
        
        # Mapeamento de data types por sinal 
        self.dataTypeMappings = {
            "cardiac": ["ecg", "hr"],
            "eeg": ["eegRaw", "eegBands"],
            "sensors": ["accelerometer", "gyroscope"],
            "camera": ["faceLandmarks"],
            "unity": ["alcohol_level", "car_information"]
        }
        
        # Signal Control properties
        self.availableSignals = settings.signalControl.signalTypes.copy()
        defaultActiveStates = settings.signalControl.defaultActiveStates["manager"]
        self.activeSignals: Set[str] = {signal for signal, active in defaultActiveStates.items() if active}
        
        # Mapeamento de métodos específicos de status por sinal
        self.statusMethods = {
            "cardiac": "getCardiacStatus",
            "eeg": "getEegStatus",
            "sensors": "getSensorsStatus",
            "camera": "getCameraStatus"
        }
        
        # Estatísticas do manager
        self.stats = {
            "totalDataProcessed": 0,
            "totalFiltered": 0,
            "dataProcessedBySignal": {signal: 0 for signal in self.signals.keys()},
            "dataFilteredBySignal": {signal: 0 for signal in self.signals.keys()},
            "totalErrors": 0,
            "lastProcessedTime": None,
            "startTime": datetime.now().isoformat()
        }
        
        # Registar no manager central de Signal Control
        signalControlManager.registerComponent("manager", self)
        
        self.logger.info(f"SignalManager initialized with Signal Control - signals: {list(self.signals.keys())}")
    
    # Signal Control Interface Implementation
    
    def getAvailableSignals(self) -> List[str]:
        """Retorna lista de signal types disponíveis para processamento"""
        return self.availableSignals.copy()
    
    def getActiveSignals(self) -> List[str]:
        """Retorna lista de signal types atualmente ativos"""
        return list(self.activeSignals)
    
    async def enableSignal(self, signal: str) -> bool:
        """Ativa processamento de um signal type específico"""
        if signal not in self.availableSignals:
            self.logger.warning(f"Signal Control: Cannot enable unknown signal {signal}")
            return False
        
        self.activeSignals.add(signal)
        self.logger.info(f"Signal Control: Enabled signal type {signal}")
        return True
    
    async def disableSignal(self, signal: str) -> bool:
        """Desativa processamento de um signal type específico"""
        self.activeSignals.discard(signal)
        self.logger.info(f"Signal Control: Disabled signal type {signal}")
        return True
    
    def getSignalState(self, signal: str) -> SignalState:
        """Retorna estado atual de um signal type"""
        if signal not in self.availableSignals:
            return SignalState.UNKNOWN
        
        if signal in self.activeSignals:
            return SignalState.ACTIVE
        else:
            return SignalState.INACTIVE
    
    def getComponentState(self) -> ComponentState:
        """Retorna estado atual do componente"""
        return ComponentState.RUNNING  # SignalManager é sempre considerado running se inicializado
    
    # Core Manager Methods
    
    async def addSignalData(self, signalType: str, dataType: str, value: Any, 
                           timestamp: Optional[float] = None) -> bool:
        """
        Adiciona dados a um sinal específico com filtering por Signal Control
        
        Args:
            signalType: "cardiac", "eeg", "sensors", "camera"
            dataType: "ecg", "hr", "eegRaw", "eegBands", "accelerometer", "gyroscope", "faceLandmarks"
            value: Valor do sinal
            timestamp: Timestamp opcional
        """
        
        # Verificar se sinal existe
        if signalType not in self.signals:
            self.logger.warning(f"Unknown signal type: {signalType}")
            self.stats["totalErrors"] += 1
            return False
        
        # Verificar se dataType é válido para o sinal
        if dataType not in self.dataTypeMappings.get(signalType, []):
            self.logger.warning(f"Invalid data type {dataType} for signal {signalType}. Valid types: {self.dataTypeMappings.get(signalType, [])}")
            self.stats["totalErrors"] += 1
            return False
        
        # Filtering via Signal Control por signal type individual
        if dataType not in self.activeSignals:
            self.stats["totalFiltered"] += 1
            self.stats["dataFilteredBySignal"][signalType] += 1
            self.logger.debug(f"Signal Control: Signal type {dataType} filtered")
            return True  # Retorna True mas não processa (filtering silencioso)
        
        try:
            # Criar SignalPoint
            point = SignalPoint(
                timestamp=timestamp or datetime.now().timestamp(),
                value=value,
                quality=1.0,  # Por agora qualidade fixa
                metadata={"dataType": dataType, "source": "signal_manager"}
            )
            
            # Obter anomalias antes de adicionar
            signal = self.signals[signalType]
            previousAnomalies = set(signal.getRecentAnomalies())

            # Adicionar ao sinal
            success = signal.addPoint(point)

            if success:
                # Obter anomalias depois de adicionar
                currentAnomalies = signal.getRecentAnomalies()
                newAnomalies = [a for a in currentAnomalies if a not in previousAnomalies]
                
                # Atualizar estatísticas
                self.stats["totalDataProcessed"] += 1
                self.stats["dataProcessedBySignal"][signalType] += 1
                self.stats["lastProcessedTime"] = datetime.now().isoformat()
                
                self.logger.debug(f"Added {dataType} data to {signalType}")
                
                # Emitir evento normal (sempre)
                await eventManager.emit("signal.processed", {
                    "signalType": signalType,
                    "dataType": dataType,
                    "value": value,
                    "timestamp": point.timestamp,
                })
                
                # Emitir evento para novas anomalias
                for anomaly in newAnomalies:
                    await self._emitAnomalyDetected(signalType, anomaly, value)
                
                return True
            else:
                self.logger.warning(f"Failed to add {dataType} to {signalType}")
                self.stats["totalErrors"] += 1
                return False
                
        except Exception as e:
            self.logger.error(f"Error adding signal data: {e}")
            self.stats["totalErrors"] += 1
            return False
    
    async def processZeroMQData(self, rawData: Dict[str, Any]) -> bool:
        """
        Processa dados vindos do ZeroMQ com filtering por Signal Control
        
        Formato esperado:
        {
            "timestamp": 1653123456.789,
            "source": "cardiowheel|halo|camera|unity",
            "data": {
                # Cardiac
                "ecg": [sample1, sample2, ...],
                "hr": 75.5,
                
                # EEG
                "eegRaw": {
                    "ch1": [samples...], "ch2": [samples...], 
                    "ch3": [samples...], "ch4": [samples...]
                },
                "eegBands": {
                    "delta": 0.25, "theta": 0.33, "alpha": 0.42, 
                    "beta": 0.18, "gamma": 0.05
                },
                
                # Sensors
                "accelerometer": {
                    "x": [val1, val2, ...], "y": [...], "z": [...]
                },
                "gyroscope": {
                    "x": [val1, val2, ...], "y": [...], "z": [...]
                },
                
                # Camera
                "faceLandmarks": {
                    "landmarks": [[x1,y1,z1], [x2,y2,z2], ...],
                    "gaze_vector": {"dx": 0.2, "dy": -0.1},
                    "ear": 0.25,
                    "blink_rate": 18,
                    "blink_counter": 34,
                    "frame_b64": ,
                }
            }
        }
        """
        
        try:
            # Validação básica da estrutura
            if not isinstance(rawData, dict):
                self.logger.error("ZeroMQ data must be a dictionary")
                self.stats["totalErrors"] += 1
                return False
            
            timestamp = rawData.get("timestamp")
            source = rawData.get("source", "unknown")
            data = rawData.get("data", {})
            
            if not isinstance(data, dict):
                self.logger.error("ZeroMQ data.data must be a dictionary")
                self.stats["totalErrors"] += 1
                return False
            
            self.logger.debug(f"Processing ZeroMQ data from {source} with keys: {list(data.keys())}")
            
            overallSuccess = True
            processedCount = 0
            errors = []
            
            # Processar dados cardíacos
            if "ecg" in data or "hr" in data:
                try:
                    cardiacSuccess = await self._processCardiacData(data, timestamp)
                    overallSuccess = overallSuccess and cardiacSuccess
                    if cardiacSuccess:
                        processedCount += 1
                except Exception as e:
                    errors.append(f"Cardiac processing failed: {e}")
                    overallSuccess = False
            
            # Processar dados EEG
            if "eegRaw" in data or "eegBands" in data:
                try:
                    eegSuccess = await self._processEegData(data, timestamp)
                    overallSuccess = overallSuccess and eegSuccess
                    if eegSuccess:
                        processedCount += 1
                except Exception as e:
                    errors.append(f"EEG processing failed: {e}")
                    overallSuccess = False
            
            # Processar dados de sensores
            if "accelerometer" in data or "gyroscope" in data:
                try:
                    sensorsSuccess = await self._processSensorsData(data, timestamp)
                    overallSuccess = overallSuccess and sensorsSuccess
                    if sensorsSuccess:
                        processedCount += 1
                except Exception as e:
                    errors.append(f"Sensors processing failed: {e}")
                    overallSuccess = False
            
            # Processar dados de câmera
            if "faceLandmarks" in data:
                try:
                    cameraSuccess = await self._processCameraData(data, timestamp)
                    overallSuccess = overallSuccess and cameraSuccess
                    if cameraSuccess:
                        processedCount += 1
                except Exception as e:
                    errors.append(f"Camera processing failed: {e}")
                    overallSuccess = False

            # Processar dados Unity
            if "alcohol_level" in data or "car_information" in data:
                try:
                    unitySuccess = await self._processUnityData(data, timestamp)
                    overallSuccess = overallSuccess and unitySuccess
                    if unitySuccess:
                        processedCount += 1
                except Exception as e:
                    errors.append(f"Unity processing failed: {e}")
                    overallSuccess = False
                        
            # Verificar se processamos alguma coisa
            if processedCount > 0:
                self.logger.debug(f"Successfully processed {processedCount} signal types from {source}")
            else:
                self.logger.warning(f"No recognizable data types in message from {source}. Available keys: {list(data.keys())}")
                overallSuccess = False
            
            # Log de erros se houver
            if errors:
                for error in errors:
                    self.logger.error(error)
                self.stats["totalErrors"] += len(errors)
            
            return overallSuccess
            
        except Exception as e:
            self.logger.error(f"Error processing ZeroMQ data: {e}")
            self.stats["totalErrors"] += 1
            return False

    async def _processUnityData(self, data: Dict[str, Any], timestamp: Optional[float]) -> bool:
        """Processa dados Unity específicos com validação e filtering"""
        success = True
        
        # Processar Alcohol Level se presente
        if "alcohol_level" in data:
            try:
                alcoholSuccess = await self.addSignalData(
                    signalType="unity",
                    dataType="alcohol_level", 
                    value={"alcohol_level": data["alcohol_level"]},
                    timestamp=timestamp
                )
                success = success and alcoholSuccess
                if not alcoholSuccess:
                    self.logger.warning(f"Failed to process alcohol level data: {data['alcohol_level']}")
            except Exception as e:
                self.logger.error(f"Error processing alcohol level data: {e}")
                success = False
        
        # Processar Car Information se presente
        if "car_information" in data:
            try:
                carSuccess = await self.addSignalData(
                    signalType="unity",
                    dataType="car_information",
                    value={"car_information": data["car_information"]},
                    timestamp=timestamp
                )
                success = success and carSuccess
                if not carSuccess:
                    self.logger.warning(f"Failed to process car information data")
            except Exception as e:
                self.logger.error(f"Error processing car information data: {e}")
                success = False
        
        return success

    async def _processCameraData(self, data: Dict[str, Any], timestamp: Optional[float]) -> bool:
        """Processa dados de câmera específicos com validação e filtering"""
        success = True
        
        # Processar Face Landmarks se presente
        if "faceLandmarks" in data:
            try:
                cameraSuccess = await self.addSignalData(
                    signalType="camera",
                    dataType="faceLandmarks",
                    value=data["faceLandmarks"],
                    timestamp=timestamp
                )
                success = success and cameraSuccess
                if not cameraSuccess:
                    self.logger.warning(f"Failed to process face landmarks data")
            except Exception as e:
                self.logger.error(f"Error processing face landmarks data: {e}")
                success = False
        
        return success
  
    async def _processCardiacData(self, data: Dict[str, Any], timestamp: Optional[float]) -> bool:
        """Processa dados cardíacos específicos com validação e filtering"""
        success = True
        
        # Processar ECG se presente
        if "ecg" in data:
            try:
                ecgSuccess = await self.addSignalData(
                    signalType="cardiac",
                    dataType="ecg", 
                    value=data["ecg"],
                    timestamp=timestamp
                )
                success = success and ecgSuccess
                if not ecgSuccess:
                    self.logger.warning(f"Failed to process ECG data")
            except Exception as e:
                self.logger.error(f"Error processing ECG data: {e}")
                success = False
        
        # Processar HR se presente
        if "hr" in data:
            try:
                hrSuccess = await self.addSignalData(
                    signalType="cardiac",
                    dataType="hr",
                    value=data["hr"],
                    timestamp=timestamp
                )
                success = success and hrSuccess
                if not hrSuccess:
                    self.logger.warning(f"Failed to process HR data: {data['hr']}")
            except Exception as e:
                self.logger.error(f"Error processing HR data: {e}")
                success = False
        
        return success
    
    async def _processEegData(self, data: Dict[str, Any], timestamp: Optional[float]) -> bool:
        """Processa dados EEG específicos com validação e filtering"""
        success = True
        
        # Processar EEG Raw se presente
        if "eegRaw" in data:
            try:
                rawSuccess = await self.addSignalData(
                    signalType="eeg",
                    dataType="eegRaw",
                    value=data["eegRaw"],
                    timestamp=timestamp
                )
                success = success and rawSuccess
                if not rawSuccess:
                    self.logger.warning(f"Failed to process EEG raw data")
            except Exception as e:
                self.logger.error(f"Error processing EEG raw data: {e}")
                success = False
        
        # Processar EEG Bands se presente
        if "eegBands" in data:
            try:
                bandsSuccess = await self.addSignalData(
                    signalType="eeg",
                    dataType="eegBands",
                    value=data["eegBands"],
                    timestamp=timestamp
                )
                success = success and bandsSuccess
                if not bandsSuccess:
                    self.logger.warning(f"Failed to process EEG bands data")
            except Exception as e:
                self.logger.error(f"Error processing EEG bands data: {e}")
                success = False
        
        return success
    
    async def _processSensorsData(self, data: Dict[str, Any], timestamp: Optional[float]) -> bool:
        """Processa dados de sensores específicos com validação e filtering"""
        success = True
        
        # Processar Accelerometer se presente
        if "accelerometer" in data:
            try:
                accSuccess = await self.addSignalData(
                    signalType="sensors",
                    dataType="accelerometer",
                    value={"accelerometer": data["accelerometer"]},  # Wrap no formato esperado
                    timestamp=timestamp
                )
                success = success and accSuccess
                if not accSuccess:
                    self.logger.warning(f"Failed to process accelerometer data")
            except Exception as e:
                self.logger.error(f"Error processing accelerometer data: {e}")
                success = False
        
        # Processar Gyroscope se presente
        if "gyroscope" in data:
            try:
                gyrSuccess = await self.addSignalData(
                    signalType="sensors",
                    dataType="gyroscope",
                    value={"gyroscope": data["gyroscope"]},  # Wrap no formato esperado
                    timestamp=timestamp
                )
                success = success and gyrSuccess
                if not gyrSuccess:
                    self.logger.warning(f"Failed to process gyroscope data")
            except Exception as e:
                self.logger.error(f"Error processing gyroscope data: {e}")
                success = False
        
        return success
    
    def getLatestData(self) -> Dict[str, Any]:
        """Retorna dados mais recentes de todos os sinais"""
        result = {}
        
        for signalName, signal in self.signals.items():
            try:
                latest = signal.getLatestValue()
                if latest:
                    result[signalName] = {
                        "timestamp": latest.timestamp,
                        "value": latest.value,
                        "quality": latest.quality,
                        "dataType": latest.metadata.get("dataType", "unknown")
                    }
                else:
                    result[signalName] = None
            except Exception as e:
                self.logger.error(f"Error getting latest data for {signalName}: {e}")
                result[signalName] = {"error": str(e)}
        
        return result
    
    def getSignalStatus(self, signalType: str) -> Optional[Dict[str, Any]]:
        """Status de um sinal específico - com verificação de métodos"""
        if signalType not in self.signals:
            return None
        
        signal = self.signals[signalType]
        
        try:
            # Verificar se o sinal tem método específico de status
            statusMethodName = self.statusMethods.get(signalType)
            if statusMethodName and hasattr(signal, statusMethodName):
                statusMethod = getattr(signal, statusMethodName)
                return statusMethod()
            else:
                # Fallback para método base
                return signal.getStatus()
                
        except Exception as e:
            self.logger.error(f"Error getting status for {signalType}: {e}")
            return {
                "signalName": signalType,
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def getAllSignalsStatus(self) -> Dict[str, Any]:
        """Status de todos os sinais"""
        status = {}
        
        for signalName in self.signals:
            try:
                signalStatus = self.getSignalStatus(signalName)
                status[signalName] = signalStatus
            except Exception as e:
                self.logger.error(f"Error getting status for {signalName}: {e}")
                status[signalName] = {"error": str(e)}
        
        return status
    
    def getSystemHealth(self) -> Dict[str, Any]:
        """Avalia saúde geral do sistema"""
        try:
            allSignalsStatus = self.getAllSignalsStatus()
            
            health = "healthy"
            issues = []
            warnings = []
            
            # Verificar cada sinal
            activeSignals = 0
            totalAnomalies = 0
            
            for signalName, status in allSignalsStatus.items():
                if not status:
                    health = "critical"
                    issues.append(f"{signalName}: no status available")
                    continue
                
                if "error" in status:
                    health = "warning" if health == "healthy" else health
                    issues.append(f"{signalName}: {status['error']}")
                    continue
                
                # Verificar se está ativo
                if not status.get("isActive", False):
                    health = "warning" if health == "healthy" else health
                    warnings.append(f"{signalName}: not active")
                else:
                    activeSignals += 1
                
                # Contar anomalias
                anomalyCount = status.get("anomalyCount", 0)
                totalAnomalies += anomalyCount
                
                # Verificar tempo desde última atualização
                timeSinceUpdate = status.get("timeSinceUpdate")
                if timeSinceUpdate and timeSinceUpdate > 30:  # 30 segundos
                    health = "warning" if health == "healthy" else health
                    warnings.append(f"{signalName}: no updates for {timeSinceUpdate:.1f}s")
            
            # Verificar estatísticas gerais
            errorRate = self.stats["totalErrors"] / max(1, self.stats["totalDataProcessed"])
            if errorRate > 0.1:  # >10% erro
                health = "warning" if health == "healthy" else health
                warnings.append(f"High error rate: {errorRate:.1%}")
            
            # Verificar se há sinais ativos
            if activeSignals == 0:
                health = "critical"
                issues.append("No active signals")
            elif activeSignals < len(self.signals):
                health = "warning" if health == "healthy" else health
                warnings.append(f"Only {activeSignals}/{len(self.signals)} signals active")
            
            # Verificar anomalias excessivas
            if totalAnomalies > 5:
                health = "warning" if health == "healthy" else health
                warnings.append(f"High anomaly count: {totalAnomalies}")
            
            # Verificar filtering excessivo
            if self.stats["totalFiltered"] > 0:
                filterRate = self.stats["totalFiltered"] / max(1, self.stats["totalDataProcessed"] + self.stats["totalFiltered"])
                if filterRate > 0.5:  # >50% filtrado
                    health = "warning" if health == "healthy" else health
                    warnings.append(f"High filter rate: {filterRate:.1%}")
            
            # Calcular uptime
            startTime = datetime.fromisoformat(self.stats["startTime"])
            uptime = (datetime.now() - startTime).total_seconds()
            
            return {
                "health": health,
                "issues": issues,
                "warnings": warnings,
                "summary": {
                    "activeSignals": activeSignals,
                    "totalSignals": len(self.signals),
                    "totalAnomalies": totalAnomalies,
                    "errorRate": errorRate,
                    "filterRate": self.stats["totalFiltered"] / max(1, self.stats["totalDataProcessed"] + self.stats["totalFiltered"]),
                    "uptime": uptime
                },
                "stats": self.stats.copy(),
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            self.logger.error(f"Error assessing system health: {e}")
            return {
                "health": "critical",
                "issues": [f"Health assessment failed: {e}"],
                "warnings": [],
                "timestamp": datetime.now().isoformat()
            }
    
    def getSignalMetrics(self, signalType: str, lastN: Optional[int] = None) -> Optional[Dict]:
        """Métricas de um sinal"""
        if signalType not in self.signals:
            return None
        
        try:
            signal = self.signals[signalType]
            return signal.getMetrics(lastN)
        except Exception as e:
            self.logger.error(f"Error getting metrics for {signalType}: {e}")
            return None
    
    def checkAnomalies(self) -> List[Dict[str, Any]]:
        """Verifica anomalias em todos os sinais"""
        allAnomalies = []
        
        for signalName, signal in self.signals.items():
            try:
                anomalies = signal.getRecentAnomalies()
                
                for anomaly in anomalies:
                    allAnomalies.append({
                        "signalType": signalName,
                        "message": anomaly,
                        "timestamp": datetime.now().isoformat(),
                        "severity": self._classifyAnomalySeverity(anomaly)
                    })
                    
            except Exception as e:
                self.logger.error(f"Error checking anomalies for {signalName}: {e}")
                allAnomalies.append({
                    "signalType": signalName,
                    "message": f"Error checking anomalies: {e}",
                    "timestamp": datetime.now().isoformat(),
                    "severity": "critical"
                })
        
        return allAnomalies
    
    async def _emitAnomalyDetected(self, signalType: str, anomalyMessage: str, value: Any):
        """Emite evento específico de anomalia detectada"""
        
        # Extrair informações da mensagem de anomalia
        anomalyInfo = self._parseAnomalyMessage(anomalyMessage)
        
        await eventManager.emit("anomaly.detected", {
            "signalType": signalType,
            "anomalyType": anomalyInfo["type"],
            "severity": anomalyInfo["severity"],
            "message": anomalyMessage,
            "timestamp": datetime.now().isoformat(),
            "value": value,
            "threshold": anomalyInfo.get("threshold")
        })

    def _parseAnomalyMessage(self, message: str) -> Dict[str, Any]:
        """Extrai informações da mensagem de anomalia"""
        message_lower = message.lower()
        
        # Detectar tipo de anomalia cardiac
        if "bradicardia" in message_lower:
            anomaly_type = "bradycardia"
            threshold = settings.signals.cardiacConfig["hr"]["bradycardiaThreshold"]
        elif "taquicardia" in message_lower:
            anomaly_type = "tachycardia" 
            threshold = settings.signals.cardiacConfig["hr"]["tachycardiaThreshold"]
        elif "eletrodo" in message_lower and "solto" in message_lower:
            anomaly_type = "electrode_loose"
            threshold = None
        elif "amplitude" in message_lower and "baixa" in message_lower:
            anomaly_type = "low_amplitude"
            threshold = settings.signals.cardiacConfig["ecg"]["lowAmplitudeThreshold"]
        
        # Detectar tipo de anomalia EEG
        elif "saturação" in message_lower:
            anomaly_type = "saturation"
            threshold = settings.signals.eegConfig["raw"]["saturationThreshold"]
        elif "dominância" in message_lower and "delta" in message_lower:
            anomaly_type = "delta_dominance"
            threshold = settings.signals.eegConfig["bands"]["deltaExcessThreshold"]
        
        # Detectar tipo de anomalia sensors
        elif "movimento" in message_lower and "brusco" in message_lower:
            anomaly_type = "sudden_movement"
            threshold = settings.signals.sensorsConfig["accelerometer"]["suddenMovementThreshold"]
        elif "impacto" in message_lower:
            anomaly_type = "impact"
            threshold = settings.signals.sensorsConfig["accelerometer"]["impactThreshold"]
        elif "vibração" in message_lower and "excessiva" in message_lower:
            anomaly_type = "excessive_vibration"
            threshold = settings.signals.sensorsConfig["accelerometer"]["highVibrationsThreshold"]
        elif "rotação" in message_lower and "rápida" in message_lower:
            anomaly_type = "rapid_rotation"
            threshold = settings.signals.sensorsConfig["gyroscope"]["rapidRotationThreshold"]
        elif "spin" in message_lower or "derrapagem" in message_lower:
            anomaly_type = "spin_slip"
            threshold = settings.signals.sensorsConfig["gyroscope"]["spinThreshold"]
        elif "condução" in message_lower and "agressiva" in message_lower:
            anomaly_type = "aggressive_driving"
            threshold = None
        elif "travagem" in message_lower and "emergência" in message_lower:
            anomaly_type = "emergency_braking"
            threshold = None
        elif "instabilidade" in message_lower:
            anomaly_type = "instability"
            threshold = settings.signals.sensorsConfig["gyroscope"]["instabilityThreshold"]
        
        # Detectar tipo de anomalia camera
        elif "piscadelas" in message_lower and "baixa" in message_lower:
            anomaly_type = "low_blink_rate"
            threshold = settings.signals.cameraConfig["blinkRate"]["drowsinessThreshold"]
        elif "piscadelas" in message_lower and "excessiva" in message_lower:
            anomaly_type = "high_blink_rate"
            threshold = settings.signals.cameraConfig["blinkRate"]["hyperBlinkThreshold"]
        elif "ear" in message_lower and "baixo" in message_lower:
            anomaly_type = "low_ear"
            threshold = settings.signals.cameraConfig["ear"]["drowsyThreshold"]
        elif "confiança" in message_lower and "baixa" in message_lower:
            anomaly_type = "low_detection_confidence"
            threshold = settings.signals.cameraConfig["faceLandmarks"]["detectionThreshold"]
        elif "olhar" in message_lower and "desviado" in message_lower:
            anomaly_type = "gaze_drift"
            threshold = 0.7  # Valor hardcoded na detecção
        elif "movimento" in message_lower and "errático" in message_lower:
            anomaly_type = "erratic_gaze"
            threshold = settings.signals.cameraConfig["gaze"]["stabilityThreshold"]
        elif "variação" in message_lower and "ear" in message_lower:
            anomaly_type = "ear_instability"
            threshold = 0.2  # Valor hardcoded na detecção

        # Detectar tipo de anomalia Unity
        elif "álcool" in message_lower and "limite" in message_lower:
            if "perigoso" in message_lower:
                anomaly_type = "dangerous_alcohol"
                threshold = settings.signals.unityConfig["alcohol_level"]["dangerLimit"]
            else:
                anomaly_type = "high_alcohol"
                threshold = settings.signals.unityConfig["alcohol_level"]["legalLimit"]
        elif "velocidade" in message_lower and ("excessiva" in message_lower or "alta" in message_lower):
            if "perigosa" in message_lower:
                anomaly_type = "dangerous_speed"
                threshold = settings.signals.unityConfig["car_information"]["speed"]["dangerSpeedThreshold"]
            else:
                anomaly_type = "speeding"
                threshold = settings.signals.unityConfig["car_information"]["speed"]["speedingThreshold"]
        elif "faixa" in message_lower and ("saída" in message_lower or "fora" in message_lower):
            if "fora" in message_lower:
                anomaly_type = "lane_departure_critical"
                threshold = settings.signals.unityConfig["car_information"]["lane_centrality"]["dangerThreshold"]
            else:
                anomaly_type = "lane_departure_warning"
                threshold = settings.signals.unityConfig["car_information"]["lane_centrality"]["warningThreshold"]
        elif "condução" in message_lower and ("perigosa" in message_lower or "instável" in message_lower):
            anomaly_type = "dangerous_driving"
            threshold = None
        elif "perigo crítico" in message_lower:
            anomaly_type = "critical_driving_danger"
            threshold = None
                
        else:
            anomaly_type = "unknown"
            threshold = None
        
        # Detectar severidade
        severity = self._classifyAnomalySeverity(message)
        
        return {
            "type": anomaly_type,
            "severity": severity,
            "threshold": threshold
        }
        
    def _classifyAnomalySeverity(self, anomalyMessage: str) -> str:
        """Classifica severidade de anomalia"""
        message = anomalyMessage.lower()
        
        # Crítico
        if any(word in message for word in [
            "severe", "crítico", "crítica", "saturação", "solto", "muito baixa", "muito alta",
            "error", "failed", "connection", "timeout", "impacto", "spin", "derrapagem",
            "emergência", "travagem", "sonolência crítica", "confiança baixa", "qualidade alta",
            "perigo crítico", "álcool perigoso", "velocidade muito perigosa", "fora da faixa",
            "nível de álcool perigoso"
        ]):
            return "critical"
        
        # Aviso
        elif any(word in message for word in [
            "moderate", "moderada", "alta", "súbita", "dominância", "excessiva", "warning",
            "drift", "artefacto", "movimento", "variabilidade", "brusco", "rápida",
            "agressiva", "vibração", "rotação", "instabilidade", "sonolência moderada",
            "piscadelas baixa", "piscadelas excessiva", "olhar desviado", "errático",
            "qualidade moderada", "álcool acima", "excesso de velocidade", "próximo da saída", "condução perigosa",
            "condução instável", "mudança súbita", "aumento súbito"
        ]):
            return "warning"
        
        # Info
        else:
            return "info"
    
    def getControlSummary(self) -> Dict[str, Any]:
        """
        Retorna resumo do estado de controlo do componente.
        
        Returns:
            Resumo com estatísticas e estado atual
        """
        available = self.getAvailableSignals()
        active = self.getActiveSignals()
        
        return {
            "componentState": self.getComponentState().value,
            "totalSignals": len(available),
            "activeSignals": len(active),
            "inactiveSignals": len(available) - len(active),
            "availableSignals": available,
            "activeSignalsList": active,
            "inactiveSignalsList": [s for s in available if s not in active],
            "filtering": {
                "totalFiltered": self.stats["totalFiltered"],
                "filteringRate": self.stats["totalFiltered"] / max(1, 
                    self.stats["totalDataProcessed"] + self.stats["totalFiltered"]
                )
            },
            "lastUpdate": datetime.now().isoformat()
        }
    
    def getManagerStats(self) -> Dict[str, Any]:
        """Estatísticas do SignalManager incluindo Signal Control"""
        uptime = (datetime.now() - datetime.fromisoformat(self.stats["startTime"])).total_seconds()
        
        return {
            **self.stats,
            "uptime": uptime,
            "availableSignals": list(self.signals.keys()),
            "dataTypeMappings": self.dataTypeMappings,
            "averageProcessingRate": self.stats["totalDataProcessed"] / max(1, uptime),
            "successRate": 1 - (self.stats["totalErrors"] / max(1, self.stats["totalDataProcessed"])),
            "filterRate": self.stats["totalFiltered"] / max(1, self.stats["totalDataProcessed"] + self.stats["totalFiltered"]),
            "signalControl": {
                "availableSignals": self.getAvailableSignals(),
                "activeSignals": self.getActiveSignals(),
                "componentState": self.getComponentState().value,
                "filteredSignals": [signal for signal in self.availableSignals if signal not in self.activeSignals]
            }
        }
    
    def reset(self) -> None:
        """Reset de todos os sinais e estatísticas"""
        try:
            for signal in self.signals.values():
                signal.reset()
            
            # Reset das estatísticas
            self.stats = {
                "totalDataProcessed": 0,
                "totalFiltered": 0,
                "dataProcessedBySignal": {signal: 0 for signal in self.signals.keys()},
                "dataFilteredBySignal": {signal: 0 for signal in self.signals.keys()},
                "totalErrors": 0,
                "lastProcessedTime": None,
                "startTime": datetime.now().isoformat()
            }
            
            self.logger.info("All signals and statistics reset")
            
        except Exception as e:
            self.logger.error(f"Error during reset: {e}")

# Instância global
signalManager = SignalManager()