"""
ZeroMQProcessor - Processamento de dados recebidos via ZeroMQ

Resumo:
Processa dados brutos recebidos do ZeroMQListener e converte-os para o formato esperado
pelo SignalManager. Cada tópico ZeroMQ tem o seu próprio método de processamento específico
que lida com as particularidades dos dados (conversões, validações, formatação).
Inclui controlo granular de sinais através do sistema Signal Control para filtering
por tópicos ZeroMQ individuais.

Funcionalidades principais:
- Processamento específico por tópico (Polar_PPI, CardioWheel_ECG, BrainAccess_EEG, etc.)
- Validação de dados baseada nas configurações centralizadas
- Conversão de formatos (ex: PPI para HR, chunks de dados para arrays temporais)
- Mapeamento automático de tópicos para tipos de sinais do SignalManager
- Gestão de timestamps e reconstrução de ordem temporal em chunks
- Logging detalhado para debugging no simulador
- Tratamento de exceções específicas por tipo de dados
- Controlo granular por tópico através do Signal Control

O processor atua como adaptador entre o formato de dados ZeroMQ e a arquitetura
interna do backend, garantindo que todos os dados chegam formatados corretamente
ao SignalManager independentemente da fonte original.

FORMATO DOS DADOS ZEROMQ:
{
    "ts": "timestamp_string",
    "labels": ["label1", "label2", ...],
    "data": [[val1, val2, ...], [val1, val2, ...], ...]
}
"""

import json
import logging
import msgpack
from datetime import datetime
from typing import Dict, Any, Optional, List, Union, Set
import numpy as np

from ..core import settings, eventManager
from ..core.exceptions import ZeroMQProcessingError, TopicValidationError, UnknownTopicError
from ..core.signalControl import SignalControlInterface, SignalState, ComponentState, signalControlManager  

class ZeroMQProcessor(SignalControlInterface):
    """Processador de dados ZeroMQ para conversão e formatação com controlo de sinais"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Carregar configurações centralizadas
        zmqConfig = settings.zeromq
        self.topicSignalMapping = zmqConfig.topicToSignalMapping
        self.processingConfig = zmqConfig.topicProcessingConfig  
        self.validationConfig = zmqConfig.topicValidationConfig
        
        # Signal Control properties
        self.availableSignals = settings.signalControl.zeroMQTopics.copy()
        defaultActiveStates = settings.signalControl.defaultActiveStates["processor"]
        self.activeSignals: Set[str] = {signal for signal, active in defaultActiveStates.items() if active}
        
        # Estatísticas de processamento por tópico
        self.processingStats = {
            "totalProcessed": 0,
            "totalErrors": 0,
            "totalFiltered": 0,
            "byTopic": {topic: {
                "processed": 0,
                "errors": 0,
                "filtered": 0,
                "lastProcessed": None,
                "lastError": None
            } for topic in self.topicSignalMapping.keys()}
        }
        
        # Cache para timestamps e ordem de chunks
        self.chunkCache = {}
        
        # Registar no manager central de Signal Control
        signalControlManager.registerComponent("processor", self)
        
        self.logger.info(f"ZeroMQProcessor initialized with Signal Control for topics: {list(self.topicSignalMapping.keys())}")
        self.logger.debug(f"Processing config loaded: {len(self.processingConfig)} topic configs")
    
    # Signal Control Interface Implementation
    
    def getAvailableSignals(self) -> List[str]:
        """Retorna lista de tópicos disponíveis para processamento"""
        return self.availableSignals.copy()
    
    def getActiveSignals(self) -> List[str]:
        """Retorna lista de tópicos atualmente ativos"""
        return list(self.activeSignals)
    
    async def enableSignal(self, signal: str) -> bool:
        """Ativa processamento de um tópico específico"""
        if signal not in self.availableSignals:
            self.logger.warning(f"Signal Control: Cannot enable unknown signal {signal}")
            return False
        
        self.activeSignals.add(signal)
        self.logger.info(f"Signal Control: Enabled topic {signal}")
        return True
    
    async def disableSignal(self, signal: str) -> bool:
        """Desativa processamento de um tópico específico"""
        self.activeSignals.discard(signal)
        self.logger.info(f"Signal Control: Disabled topic {signal}")
        return True
    
    def getSignalState(self, signal: str) -> SignalState:
        """Retorna estado atual de um sinal"""
        if signal not in self.availableSignals:
            return SignalState.UNKNOWN
        
        if signal in self.activeSignals:
            return SignalState.ACTIVE
        else:
            return SignalState.INACTIVE
    
    def getComponentState(self) -> ComponentState:
        """Retorna estado atual do componente"""
        return ComponentState.RUNNING  # Processor é sempre considerado running se inicializado
    
    # Core Processing Methods
    
    async def processTopicData(self, topic: str, rawData: bytes) -> Optional[Dict[str, Any]]:
        """
        Processa dados de um tópico específico recebidos via ZeroMQ.
        
        Args:
            topic: Nome do tópico ZeroMQ (ex: "Polar_PPI", "CardioWheel_ECG")
            rawData: Dados brutos em bytes (msgpack ou JSON)
            
        Returns:
            Dados formatados para o SignalManager ou None se erro/inválido/filtrado
            
        Raises:
            UnknownTopicError: Se tópico não é reconhecido
            ZeroMQProcessingError: Se falha no processamento dos dados
        """
        
        startTime = datetime.now()
        
        try:
            # Verificar se tópico é reconhecido
            if topic not in self.topicSignalMapping:
                availableTopics = list(self.topicSignalMapping.keys())
                raise UnknownTopicError(topic, availableTopics)
            
            # Filtering via Signal Control
            if topic not in self.activeSignals:
                self.processingStats["totalFiltered"] += 1
                self.processingStats["byTopic"][topic]["filtered"] += 1
                self.logger.debug(f"Signal Control: Topic {topic} filtered")
                return None
            
            self.logger.debug(f"Processing data from topic: {topic}")
            
            # Descodificar dados msgpack
            try:
                decodedData = msgpack.unpackb(rawData, raw=False)
                self.logger.debug(f"Successfully decoded msgpack data for {topic}")
            except Exception as e:
                raise ZeroMQProcessingError(
                    topic=topic,
                    operation="msgpack_decode", 
                    reason=f"Failed to decode msgpack: {e}",
                    rawData=rawData
                )

            # Validar estrutura básica dos dados
            validatedData = await self._validateTopicData(topic, decodedData)

            
            # Processar dados específicos do tópico
            processedData = await self._processSpecificTopic(topic, validatedData)

            
            if processedData:
                # Atualizar estatísticas de sucesso
                self.processingStats["totalProcessed"] += 1
                self.processingStats["byTopic"][topic]["processed"] += 1
                self.processingStats["byTopic"][topic]["lastProcessed"] = datetime.now().isoformat()
                
                # Calcular tempo de processamento
                processingTime = (datetime.now() - startTime).total_seconds()
                
                self.logger.debug(f"Successfully processed {topic} data in {processingTime:.3f}s")
                
                # Emitir evento de processamento bem-sucedido
                await eventManager.emit("zmq.data_processed", {
                    "topic": topic,
                    "processingTime": processingTime,
                    "dataSize": len(rawData),
                    "outputSignalType": processedData.get("signalType"),
                    "outputDataType": processedData.get("dataType"),
                    "timestamp": datetime.now().isoformat()
                })
                
                return processedData
            else:
                raise ZeroMQProcessingError(
                    topic=topic,
                    operation="topic_processing",
                    reason="Processor returned None"
                )
        
        except (UnknownTopicError, ZeroMQProcessingError, TopicValidationError):
            # Reenviar exceções específicas tal como estão
            raise
            
        except Exception as e:
            # Capturar erros inesperados
            self._recordError(topic, str(e))
            raise ZeroMQProcessingError(
                topic=topic,
                operation="general_processing",
                reason=f"Unexpected error: {e}",
                rawData=rawData
            )
    
    async def _validateTopicData(self, topic: str, data: Any) -> None:
        """
        Valida dados recebidos baseado na configuração do tópico.
        FORMATO: {"ts": "...", "labels": [...], "data": [[...], [...]]}
        
        Args:
            topic: Nome do tópico
            data: Dados descodificados para validar
            
        Raises:
            TopicValidationError: Se dados não passam validação
        """
        
        if topic not in self.validationConfig:
            self.logger.warning(f"No validation config for topic {topic}, skipping validation")
            return
        
        config = self.validationConfig[topic]
        
        # Converter string JSON se necessário
        if isinstance(data, str):
            try:
                data = json.loads(data)
            except json.JSONDecodeError as e:
                raise TopicValidationError(
                    topic=topic,
                    field="data_format",
                    value="invalid_json",
                    expectedRange=("valid_json",)
                )
        
        self.logger.debug(f"Validating data structure for {topic}: {data}")
        
        # Verificar se dados são um dicionário
        if not isinstance(data, dict):
            raise TopicValidationError(
                topic=topic,
                field="data_type",
                value=type(data).__name__,
                expectedRange=("dict",)
            )
        
        # Verificar campos obrigatórios (ts, labels, data)
        requiredFields = config.get("requiredFields", [])
        for field in requiredFields:
            if field not in data:
                raise TopicValidationError(
                    topic=topic,
                    field=field,
                    value="missing",
                    expectedRange=("required",)
                )
        
        # Validar estrutura específica do novo formato
        if "labels" in requiredFields and "data" in requiredFields:
            # Verificar se labels é uma lista
            if not isinstance(data.get("labels"), list):
                raise TopicValidationError(
                    topic=topic,
                    field="labels",
                    value=type(data.get("labels")).__name__,
                    expectedRange=("list",)
                )
            
            # Verificar se data é uma lista
            if not isinstance(data.get("data"), list):
                raise TopicValidationError(
                    topic=topic,
                    field="data",
                    value=type(data.get("data")).__name__,
                    expectedRange=("list",)
                )
            
            # Verificar se labels esperadas estão presentes
            expectedLabels = config.get("expectedLabels", [])
            actualLabels = data.get("labels", [])
            
            for expectedLabel in expectedLabels:
                if expectedLabel not in actualLabels:
                    self.logger.warning(f"Expected label '{expectedLabel}' not found in {actualLabels} for topic {topic}")
            
            # Validar dimensões dos dados
            dataArray = data.get("data", [])
            labelsArray = data.get("labels", [])
            
            if dataArray and labelsArray:
                if len(dataArray) > 0 and isinstance(dataArray[0], list):
                    # Verificar se cada linha de dados tem o mesmo número de elementos que labels
                    expectedColumns = len(labelsArray)
                    for i, row in enumerate(dataArray):
                        if len(row) != expectedColumns:
                            raise TopicValidationError(
                                topic=topic,
                                field=f"data_row_{i}",
                                value=f"length_{len(row)}",
                                expectedRange=(f"length_{expectedColumns}",)
                            )
        
        self.logger.debug(f"Data validation passed for topic {topic}")

        return data
    
    async def _processSpecificTopic(self, topic: str, data: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Redireciona para processamento específico baseado no tópico.
        
        Args:
            topic: Nome do tópico
            data: Dados validados do tópico
            
        Returns:
            Dados formatados para SignalManager
        """

        # Mapeamento de tópicos para métodos de processamento
        topicProcessors = {
            "Polar_PPI": self._processPolarPPI,
            "CardioWheel_ECG": self._processCardioWheelECG,
            "CardioWheel_ACC": self._processCardioWheelACC,
            "CardioWheel_GYR": self._processCardioWheelGYR,
            "BrainAcess_EEG": self._processBrainAccessEEG,
            "Camera_FaceLandmarks": self._processCameraFaceLandmarks,
            "Unity_Alcohol": self._processUnityAlcohol,
            "Unity_CarInfo": self._processUnityCarInfo,
            "Control": self._processSystemControl,
            "Timestamp": self._processSystemTimestamp,
            "Cfg": self._processSystemConfig
        }
        
        self.logger.debug(f"Processing {topic} with data structure: {list(data.keys())}")
        
        processor = topicProcessors.get(topic)
        if not processor:
            raise ZeroMQProcessingError(
                topic=topic,
                operation="processor_lookup",
                reason=f"No processor found for topic {topic}"
            )
        
        return await processor(data)
    
    async def _processCameraFaceLandmarks(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados de face landmarks da câmera.
        Converte formato ZeroMQ para estrutura esperada pelo SignalManager.
        
        FORMATO INPUT ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["landmarks", "gaze_dx", "gaze_dy", "ear", "blink_rate", "blink_counter", "frame_b64"],,
            "data": [[flattened_landmarks, gaze_dx, gaze_dy, ear, blink_rate, blink_counter, frame_b64]]
        }
        
        FORMATO OUTPUT SignalManager:
        {
            "signalType": "camera",
            "dataType": "faceLandmarks", 
            "timestamp": timestamp,
            "value": {
                "landmarks": [[x1,y1,z1], [x2,y2,z2], ...],  # 478x3
                "gaze_vector": {"dx": 0.2, "dy": -0.1},
                "ear": 0.25,
                "blink_rate": 18,
                "blink_counter": 34,
                "frame_b64": "base64_string"
            }
        }
        
        Args:
            data: Dados formatados do tópico Camera_FaceLandmarks
            
        Returns:
            Dados processados para SignalManager ou None se inválidos
        """
        
        try:
            # Extrair timestamp
            timestamp = float(data.get("ts", 0))
            
            # Extrair dados das labels/data
            labels = data.get("labels", [])
            dataArray = data.get("data", [])
            
            if not dataArray or len(dataArray) == 0:
                self.logger.warning(f"Empty data array in Camera_FaceLandmarks")
                return None
            
            # Processar primeira linha de dados (formato ZeroMQ sempre tem uma linha)
            firstRow = dataArray[0]
            
            if len(firstRow) != len(labels):
                self.logger.error(f"Data row length ({len(firstRow)}) doesn't match labels length ({len(labels)})")
                return None
            
            # Inicializar estrutura de saída
            processedData = {
                "landmarks": None,
                "gaze_vector": {},
                "ear": None,
                "blink_rate": None,
                "blink_counter": None,
                "frame_b64": None
            }
            
            # Mapear dados baseado nas labels
            for i, label in enumerate(labels):
                if i >= len(firstRow):
                    continue
                    
                if label == "landmarks":
                    # Landmarks são array flattened [x1,y1,z1,x2,y2,z2,...]
                    landmarksFlat = firstRow[i]
                    if isinstance(landmarksFlat, list) and len(landmarksFlat) == 1434:  # 478 * 3
                        # Reshape para [[x1,y1,z1], [x2,y2,z2], ...]
                        landmarks = []
                        for j in range(0, len(landmarksFlat), 3):
                            landmarks.append([
                                landmarksFlat[j],      # x
                                landmarksFlat[j+1],    # y
                                landmarksFlat[j+2]     # z
                            ])
                        processedData["landmarks"] = landmarks
                    else:
                        self.logger.warning(f"Invalid landmarks length: expected 1434, got {len(landmarksFlat) if isinstance(landmarksFlat, list) else 'not list'}")
                        return None
                        
                elif label == "gaze_dx":
                    processedData["gaze_vector"]["dx"] = float(firstRow[i])
                elif label == "gaze_dy":  
                    processedData["gaze_vector"]["dy"] = float(firstRow[i])
                elif label == "ear":
                    processedData["ear"] = float(firstRow[i])
                elif label == "blink_rate":
                    processedData["blink_rate"] = float(firstRow[i])
                elif label == "blink_counter":
                    processedData["blink_counter"] = int(firstRow[i])
                elif label == "frame_b64":
                    processedData["frame_b64"] = str(firstRow[i])
            
            # Validações básicas de integridade
            if processedData["landmarks"] is None:
                self.logger.warning(f"Missing landmarks in Camera_FaceLandmarks")
                return None
                
            if not processedData["gaze_vector"] or "dx" not in processedData["gaze_vector"] or "dy" not in processedData["gaze_vector"]:
                self.logger.warning(f"Missing gaze vector components in Camera_FaceLandmarks")
                return None
                
            if any(val is None for val in [processedData["ear"], processedData["blink_rate"], processedData["blink_counter"]]):
                self.logger.warning(f"Missing required fields in Camera_FaceLandmarks")
                return None
            
            # Validações de range básicas
            if not (0.0 <= processedData["ear"] <= 1.0):
                self.logger.warning(f"EAR out of range: {processedData['ear']}")
                return None
                
            if not (0 <= processedData["blink_rate"] <= 120):
                self.logger.warning(f"Blink rate out of range: {processedData['blink_rate']}")
                return None
                
            # Validar gaze vector range
            dx = processedData["gaze_vector"]["dx"]
            dy = processedData["gaze_vector"]["dy"]
            if not (-1.0 <= dx <= 1.0 and -1.0 <= dy <= 1.0):
                self.logger.warning(f"Gaze vector out of range: ({dx:.2f}, {dy:.2f})")
                return None
            
            # Estrutura final para SignalManager
            signalMapping = self.topicSignalMapping["Camera_FaceLandmarks"]
            
            return {
                "timestamp": timestamp,
                "source": "camera",
                "signalType": signalMapping["signalType"],  # "camera"
                "dataType": signalMapping["dataType"],      # "faceLandmarks"
                "data": {
                    "faceLandmarks": processedData  # Wrap para SignalManager
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Camera_FaceLandmarks",
                operation="camera_processing",
                reason=str(e),
                rawData=data
            )
    
    async def _processPolarPPI(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados do Polar ARM Band (PPI).
        
        FORMATO:
        {
            "ts": "timestamp",
            "labels": ["error_ms", "flags", "value"],
            "data": [[10, 0, 800], [15, 0, 820], ...]  # error_ms, flags, ppi_ms
        }
        
        Args:
            data: Dados PPI do Polar
            
        Returns:
            Dados formatados para SignalManager
        """
        
        config = self.processingConfig["Polar_PPI"]
        
        try:
            # Extrair dados principais
            timestamp = float(data["ts"])
            labels = data["labels"]
            dataArray = data["data"]
            
            self.logger.debug(f"Processing Polar PPI: labels {labels}, {len(dataArray)} data points at {timestamp}")
            
            # Mapear labels para índices
            labelMap = {label: i for i, label in enumerate(labels)}
            
            # Processar cada linha de dados (normalmente será só uma para PPI)
            processedData = []
            
            for row in dataArray:
                pointData = {}
                
                # Extrair PPI (campo "value")
                if "value" in labelMap and len(row) > labelMap["value"]:
                    ppi_ms = row[labelMap["value"]]
                    pointData["ppi"] = ppi_ms
                    
                    # Calcular HR a partir do PPI
                    if config.get("ppiToHrConversion", False) and ppi_ms > 0:
                        factor = config["ppiToHrFactor"]
                        pointData["hr"] = round(factor / ppi_ms, 1)
                        self.logger.debug(f"Converted PPI {ppi_ms}ms to HR {pointData['hr']} BPM")
                
                # Extrair campos opcionais
                if "error_ms" in labelMap and len(row) > labelMap["error_ms"]:
                    pointData["error_ms"] = row[labelMap["error_ms"]]
                
                if "flags" in labelMap and len(row) > labelMap["flags"]:
                    pointData["flags"] = row[labelMap["flags"]]
                
                processedData.append(pointData)
            
            # Mapear para formato SignalManager (usar primeiro ponto se múltiplos)
            signalMapping = self.topicSignalMapping["Polar_PPI"]
            firstPoint = processedData[0] if processedData else {}
            
            outputData = {
                "timestamp": timestamp,
                "source": "polar",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": firstPoint
            }
            
            return outputData
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Polar_PPI",
                operation="ppi_processing",
                reason=str(e),
                rawData=data
            )
    
    async def _processCardioWheelECG(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados de ECG do CardioWheel com conversão de 16-bit ADC para milivolts.
        
        FORMATO:
        {
            "ts": "1255173.683",
            "labels": ["ECG", "LOD"], 
            "data": [[1646, 0], [1650, 0], [1651, 0], ...]
        }
        
        Args:
            data: Dados ECG do CardioWheel
            
        Returns:
            Dados formatados para SignalManager
        """
        
        config = self.processingConfig["CardioWheel_ECG"]
        
        try:
            # Extrair dados principais
            timestamp = float(data["ts"])
            labels = data["labels"]
            dataArray = data["data"]
            
            self.logger.debug(f"Processing CardioWheel ECG: {len(dataArray)} samples with labels {labels} at {timestamp}")
            
            # Encontrar índices das colunas ECG e LOD
            try:
                ecgIndex = labels.index("ECG")
                lodIndex = labels.index("LOD") if "LOD" in labels else None
            except ValueError as e:
                raise ZeroMQProcessingError(
                    topic="CardioWheel_ECG",
                    operation="label_mapping",
                    reason=f"Required label not found: {e}",
                    rawData=data
                )
            
            # Extrair valores ECG e LOD
            ecgRawValues = []
            lodValues = []
            
            for row in dataArray:
                if len(row) > ecgIndex:
                    ecgRawValues.append(row[ecgIndex])
                
                if lodIndex is not None and len(row) > lodIndex:
                    lodValues.append(row[lodIndex])
            
            self.logger.debug(f"Extracted {len(ecgRawValues)} ECG raw values, range: {min(ecgRawValues) if ecgRawValues else 'N/A'} to {max(ecgRawValues) if ecgRawValues else 'N/A'}")
            
            # Conversão de 16-bit ADC para milivolts
            conversionFactor = 5.0 / 32768.0  # mV por ADC unit
            baselineOffset = 1650  # Valor baseline típico observado nos dados
            
            ecgMillivolts = []
            for rawValue in ecgRawValues:
                # Remover offset baseline e converter para mV
                adjustedValue = rawValue - baselineOffset
                millivolts = adjustedValue * conversionFactor
                ecgMillivolts.append(round(millivolts, 3))
            
            self.logger.debug(f"Converted ECG to mV: range {min(ecgMillivolts) if ecgMillivolts else 'N/A'} to {max(ecgMillivolts) if ecgMillivolts else 'N/A'}")
            
            # Gerar timestamps para cada amostra
            timestampIncrement = config["timestampIncrement"]
            timestamps = [timestamp + (i * timestampIncrement) for i in range(len(ecgMillivolts))]
            
            # Preparar dados de saída
            outputData = {
                "timestamp": timestamp,
                "source": "cardiowheel",
                "signalType": "cardiac",
                "dataType": "ecg",
                "data": {
                    "ecg": ecgMillivolts,
                    "timestamps": timestamps,
                    "samplingRate": config["samplingRate"],
                    "conversionApplied": True,
                    "originalRange": [min(ecgRawValues), max(ecgRawValues)] if ecgRawValues else [0, 0],
                    "convertedRange": [min(ecgMillivolts), max(ecgMillivolts)] if ecgMillivolts else [0, 0]
                }
            }
            
            # Adicionar LOD se disponível
            if lodValues:
                outputData["data"]["lod"] = lodValues
            
            return outputData
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="CardioWheel_ECG",
                operation="ecg_processing",
                reason=str(e),
                rawData=data
            )

    
    async def _processBrainAccessEEG(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados EEG do BrainAccess Halo.
        
        FORMATO:
        {
            "ts": "timestamp",
            "labels": ["ch0", "ch1", "ch2", "ch3"],
            "data": [[val0, val1, val2, val3], [val0, val1, val2, val3], ...]
        }
        
        Args:
            data: Dados EEG do BrainAccess
            
        Returns:
            Dados formatados para SignalManager
        """
        
        config = self.processingConfig["BrainAcess_EEG"]
        
        try:
            timestamp = float(data["ts"])
            labels = data["labels"]
            dataArray = data["data"]
            
            self.logger.debug(f"Processing BrainAccess EEG: labels {labels}, {len(dataArray)} samples at {timestamp}")
            
            # Mapear labels para índices
            labelMap = {label: i for i, label in enumerate(labels)}
            
            # Extrair dados por canal
            eegChannelData = {}
            expectedChannels = config["channels"]  # ["ch0", "ch1", "ch2", "ch3"]
            
            for channel in expectedChannels:
                if channel in labelMap:
                    channelIndex = labelMap[channel]
                    channelValues = []
                    
                    # Extrair valores para este canal de todas as amostras
                    for row in dataArray:
                        if len(row) > channelIndex:
                            channelValues.append(row[channelIndex])
                    
                    eegChannelData[channel] = channelValues
                else:
                    self.logger.warning(f"Expected EEG channel '{channel}' not found in labels {labels}")
            
            # Verificar se temos dados de todos os canais esperados
            if len(eegChannelData) != len(expectedChannels):
                missing = set(expectedChannels) - set(eegChannelData.keys())
                self.logger.warning(f"Missing EEG channels: {missing}")
            
            self.logger.debug(f"Extracted EEG data: {len(eegChannelData)} channels, {len(dataArray)} samples each")
            
            signalMapping = self.topicSignalMapping["BrainAcess_EEG"]
            
            return {
                "timestamp": timestamp,
                "source": "halo",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": {
                    "eegRaw": eegChannelData,
                    "samplingRate": config["samplingRate"]
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="BrainAcess_EEG",
                operation="eeg_processing",
                reason=str(e),
                rawData=data
            )
    
    async def _processCardioWheelACC(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados do acelerómetro do CardioWheel com conversão de 16-bit ADC para m/s².
        
        FORMATO:
        {
            "ts": "timestamp",
            "labels": ["X", "Y", "Z"],
            "data": [[7536, 3, 3104], [7520, -6, 3107], ...]
        }
        
        Args:
            data: Dados do acelerómetro
            
        Returns:
            Dados formatados para SignalManager
        """
        
        config = self.processingConfig["CardioWheel_ACC"]
        sensorsConfig = settings.signals.sensorsConfig["accelerometer"]
        
        try:
            timestamp = float(data["ts"])
            labels = data["labels"]
            dataArray = data["data"]
            
            self.logger.debug(f"Processing CardioWheel ACC: labels {labels}, {len(dataArray)} samples at {timestamp}")
            
            # Mapear labels para índices
            labelMap = {label: i for i, label in enumerate(labels)}
            expectedAxes = config["axes"]  # ["X", "Y", "Z"]
            
            # Extrair dados por eixo e converter
            accelerometerRaw = {}
            accelerometerPhysical = {}
            
            # Fatores de conversão e offsets
            conversionFactor = sensorsConfig["conversionFactor"]  # m/s² por ADC unit
            baselineOffset = sensorsConfig["baselineOffset"]
            
            for axis in expectedAxes:
                if axis in labelMap:
                    axisIndex = labelMap[axis]
                    rawValues = []
                    physicalValues = []
                    
                    # Extrair valores para este eixo de todas as amostras
                    for row in dataArray:
                        if len(row) > axisIndex:
                            rawValue = row[axisIndex]
                            rawValues.append(rawValue)
                            
                            # Converter para m/s²
                            if axis.upper() == 'X':
                                adjustedValue = rawValue - 7500  # Baseline observado em X
                            elif axis.upper() == 'Y':
                                adjustedValue = rawValue - 0     # Y já centrado em zero
                            elif axis.upper() == 'Z':
                                adjustedValue = rawValue - 3100  # Z baseline (inclui gravidade)
                            else:
                                adjustedValue = rawValue - baselineOffset
                            
                            physicalValue = adjustedValue * conversionFactor
                            physicalValues.append(round(physicalValue, 2))
                    
                    accelerometerRaw[axis.lower()] = rawValues
                    accelerometerPhysical[axis.lower()] = physicalValues
                    
                    self.logger.debug(f"ACC {axis}: raw range [{min(rawValues)}, {max(rawValues)}] -> physical range [{min(physicalValues):.2f}, {max(physicalValues):.2f}] m/s²")
                else:
                    self.logger.warning(f"Expected accelerometer axis '{axis}' not found in labels {labels}")
            
            # Calcular magnitude total para cada amostra
            magnitudes = []
            sampleCount = len(dataArray)
            
            for i in range(sampleCount):
                try:
                    x = accelerometerPhysical['x'][i] if 'x' in accelerometerPhysical else 0
                    y = accelerometerPhysical['y'][i] if 'y' in accelerometerPhysical else 0
                    z = accelerometerPhysical['z'][i] if 'z' in accelerometerPhysical else 0
                    
                    magnitude = (x**2 + y**2 + z**2)**0.5
                    magnitudes.append(round(magnitude, 2))
                except (IndexError, KeyError):
                    magnitudes.append(0.0)
            
            # Gerar timestamps para cada amostra
            timestampIncrement = 1.0 / config["samplingRate"]  # 1/100Hz = 0.01s
            timestamps = [timestamp + (i * timestampIncrement) for i in range(sampleCount)]
            
            self.logger.debug(f"Extracted accelerometer data: {len(accelerometerPhysical)} axes, {sampleCount} samples each, magnitude range: [{min(magnitudes):.2f}, {max(magnitudes):.2f}] m/s²")
            
            signalMapping = self.topicSignalMapping["CardioWheel_ACC"]
            
            return {
                "timestamp": timestamp,
                "source": "cardiowheel",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": {
                    "accelerometer": accelerometerPhysical,
                    "accelerometerRaw": accelerometerRaw,
                    "magnitude": magnitudes,
                    "timestamps": timestamps,
                    "samplingRate": config["samplingRate"],
                    "units": "m/s²",
                    "conversionApplied": True,
                    "conversionFactor": conversionFactor
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="CardioWheel_ACC",
                operation="accelerometer_processing",
                reason=str(e),
                rawData=data
            )
        
    async def _processCardioWheelGYR(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados do giroscópio do CardioWheel com conversão de 16-bit ADC para °/s.
        
        FORMATO:
        {
            "ts": "timestamp",
            "labels": ["X", "Y", "Z"],
            "data": [[1, 0, 2], [2, -1, 0], ...]
        }
        
        Args:
            data: Dados do giroscópio
            
        Returns:
            Dados formatados para SignalManager
        """
        
        config = self.processingConfig["CardioWheel_GYR"]
        sensorsConfig = settings.signals.sensorsConfig["gyroscope"]
        
        try:
            timestamp = float(data["ts"])
            labels = data["labels"]
            dataArray = data["data"]
            
            self.logger.debug(f"Processing CardioWheel GYR: labels {labels}, {len(dataArray)} samples at {timestamp}")
            
            # Mapear labels para índices
            labelMap = {label: i for i, label in enumerate(labels)}
            expectedAxes = config["axes"]  # ["X", "Y", "Z"]
            
            # Extrair dados por eixo e converter
            gyroscopeRaw = {}
            gyroscopePhysical = {}
            
            # Fatores de conversão e offsets
            conversionFactor = sensorsConfig["conversionFactor"]  # °/s por ADC unit
            baselineOffset = sensorsConfig["baselineOffset"]
            
            for axis in expectedAxes:
                if axis in labelMap:
                    axisIndex = labelMap[axis]
                    rawValues = []
                    physicalValues = []
                    
                    # Extrair valores para este eixo de todas as amostras
                    for row in dataArray:
                        if len(row) > axisIndex:
                            rawValue = row[axisIndex]
                            rawValues.append(rawValue)
                            
                            # Converter para °/s
                            adjustedValue = rawValue - baselineOffset
                            physicalValue = adjustedValue * conversionFactor
                            physicalValues.append(round(physicalValue, 1))
                    
                    gyroscopeRaw[axis.lower()] = rawValues
                    gyroscopePhysical[axis.lower()] = physicalValues
                    
                    self.logger.debug(f"GYR {axis}: raw range [{min(rawValues)}, {max(rawValues)}] -> physical range [{min(physicalValues):.1f}, {max(physicalValues):.1f}] °/s")
                else:
                    self.logger.warning(f"Expected gyroscope axis '{axis}' not found in labels {labels}")
            
            # Calcular magnitude angular total para cada amostra
            angularMagnitudes = []
            sampleCount = len(dataArray)
            
            for i in range(sampleCount):
                try:
                    x = gyroscopePhysical['x'][i] if 'x' in gyroscopePhysical else 0
                    y = gyroscopePhysical['y'][i] if 'y' in gyroscopePhysical else 0
                    z = gyroscopePhysical['z'][i] if 'z' in gyroscopePhysical else 0
                    
                    magnitude = (x**2 + y**2 + z**2)**0.5
                    angularMagnitudes.append(round(magnitude, 1))
                except (IndexError, KeyError):
                    angularMagnitudes.append(0.0)
            
            # Gerar timestamps para cada amostra
            timestampIncrement = 1.0 / config["samplingRate"]  # 1/100Hz = 0.01s
            timestamps = [timestamp + (i * timestampIncrement) for i in range(sampleCount)]
            
            # Calcular estatísticas básicas para detecção de padrões
            if gyroscopePhysical:
                maxRotationRate = max(max(abs(min(values)), abs(max(values))) for values in gyroscopePhysical.values())
                averageMagnitude = sum(angularMagnitudes) / len(angularMagnitudes) if angularMagnitudes else 0
            else:
                maxRotationRate = 0
                averageMagnitude = 0
            
            self.logger.debug(f"Extracted gyroscope data: {len(gyroscopePhysical)} axes, {sampleCount} samples each, angular magnitude range: [{min(angularMagnitudes):.1f}, {max(angularMagnitudes):.1f}] °/s")
            
            signalMapping = self.topicSignalMapping["CardioWheel_GYR"]
            
            return {
                "timestamp": timestamp,
                "source": "cardiowheel",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": {
                    "gyroscope": gyroscopePhysical,
                    "gyroscopeRaw": gyroscopeRaw,
                    "angularMagnitude": angularMagnitudes,
                    "timestamps": timestamps,
                    "samplingRate": config["samplingRate"],
                    "units": "°/s",
                    "conversionApplied": True,
                    "conversionFactor": conversionFactor,
                    "maxRotationRate": maxRotationRate,
                    "averageMagnitude": averageMagnitude
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="CardioWheel_GYR",
                operation="gyroscope_processing",
                reason=str(e),
                rawData=data
            )
        
    async def _processUnityAlcohol(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados de nível de álcool do Unity.
        
        FORMATO INPUT ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["alcohol_level"],
            "data": [[0.3]]
        }
        
        FORMATO OUTPUT SignalManager:
        {
            "signalType": "unity",
            "dataType": "alcohol_level",
            "timestamp": timestamp,
            "value": {
                "alcohol_level": 0.3
            }
        }
        """
        
        try:
            # Extrair timestamp
            timestamp = float(data.get("ts", 0))
            
            # Extrair dados das labels/data
            labels = data.get("labels", [])
            dataArray = data.get("data", [])
            
            if not dataArray or len(dataArray) == 0:
                self.logger.warning(f"Empty data array in Unity_Alcohol")
                return None
            
            # Processar primeira linha de dados
            firstRow = dataArray[0]
            
            if len(firstRow) != len(labels):
                self.logger.error(f"Data row length ({len(firstRow)}) doesn't match labels length ({len(labels)})")
                return None
            
            # Mapear dados baseado nas labels
            alcoholLevel = None
            
            for i, label in enumerate(labels):
                if i >= len(firstRow):
                    continue
                    
                if label == "alcohol_level":
                    alcoholLevel = float(firstRow[i])
            
            # Validação obrigatória
            if alcoholLevel is None:
                self.logger.warning(f"Missing alcohol_level in Unity_Alcohol data")
                return None
            
            # Validação de range usando configurações centralizadas
            alcoholConfig = settings.signals.unityConfig["alcohol_level"]
            validRange = alcoholConfig["normalRange"]
            
            if not (0.0 <= alcoholLevel <= 3.0):  # Range absoluto
                self.logger.warning(f"Alcohol level out of absolute range: {alcoholLevel}")
                return None
            
            # Estrutura final para SignalManager
            signalMapping = self.topicSignalMapping["Unity_Alcohol"]
            
            return {
                "timestamp": timestamp,
                "source": "unity",
                "signalType": signalMapping["signalType"],  # "unity"
                "dataType": signalMapping["dataType"],      # "alcohol_level"
                "data": {
                    "alcohol_level": alcoholLevel
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Unity_Alcohol",
                operation="alcohol_processing",
                reason=str(e),
                rawData=data
            )
    
    async def _processUnityCarInfo(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa dados de informação do carro do Unity.
        
        FORMATO INPUT ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["speed", "lane_centrality"],
            "data": [[65.0, 0.8]]
        }
        
        FORMATO OUTPUT SignalManager:
        {
            "signalType": "unity",
            "dataType": "car_information", 
            "timestamp": timestamp,
            "value": {
                "speed": 65.0,
                "lane_centrality": 0.8
            }
        }
        """
        
        try:
            # Extrair timestamp
            timestamp = float(data.get("ts", 0))
            
            # Extrair dados das labels/data
            labels = data.get("labels", [])
            dataArray = data.get("data", [])
            
            if not dataArray or len(dataArray) == 0:
                self.logger.warning(f"Empty data array in Unity_CarInfo")
                return None
            
            # Processar primeira linha de dados
            firstRow = dataArray[0]
            
            if len(firstRow) != len(labels):
                self.logger.error(f"Data row length ({len(firstRow)}) doesn't match labels length ({len(labels)})")
                return None
            
            # Mapear dados baseado nas labels
            speed = None
            laneCentrality = None
            
            for i, label in enumerate(labels):
                if i >= len(firstRow):
                    continue
                    
                if label == "speed":
                    speed = float(firstRow[i])
                elif label == "lane_centrality":
                    laneCentrality = float(firstRow[i])
            
            # Validações obrigatórias
            if speed is None or laneCentrality is None:
                self.logger.warning(f"Missing required fields in Unity_CarInfo: speed={speed}, lane_centrality={laneCentrality}")
                return None
            
            # Validações de range usando configurações centralizadas
            carConfig = settings.signals.unityConfig["car_information"]
            
            # Validar velocidade
            speedRange = carConfig["speed"]["normalRange"]
            if not (speedRange[0] <= speed <= speedRange[1]):  # Range absoluto
                self.logger.warning(f"Speed out of absolute range: {speed}")
                return None
            
            # Validar centralidade
            centralityRange = carConfig["lane_centrality"]["normalRange"]
            if not (centralityRange[0] <= laneCentrality <= centralityRange[1]):  # Range absoluto
                self.logger.warning(f"Lane centrality out of range: {laneCentrality}")
                return None
            
            # Estrutura final para SignalManager
            signalMapping = self.topicSignalMapping["Unity_CarInfo"]
            
            return {
                "timestamp": timestamp,
                "source": "unity",
                "signalType": signalMapping["signalType"],  # "unity"
                "dataType": signalMapping["dataType"],      # "car_information"
                "data": {
                    "car_information": {
                        "speed": speed,
                        "lane_centrality": laneCentrality
                    }
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Unity_CarInfo",
                operation="car_info_processing",
                reason=str(e),
                rawData=data
            )
            
    async def _processSystemControl(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens de controlo do sistema.
        
        Args:
            data: Dados de controlo
            
        Returns:
            Dados formatados para SignalManager
        """
        
        try:
            signalMapping = self.topicSignalMapping["Control"]
            
            self.logger.debug(f"Processing system control: {data}")
            
            return {
                "timestamp": datetime.now().timestamp(),
                "source": "system",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": {
                    "control": data
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Control",
                operation="control_processing",
                reason=str(e),
                rawData=data
            )
    
    async def _processSystemTimestamp(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens de timestamp do sistema.
        
        Args:
            data: Dados de timestamp
            
        Returns:
            Dados formatados para SignalManager
        """
        
        try:
            signalMapping = self.topicSignalMapping["Timestamp"]
            
            self.logger.debug(f"Processing system timestamp: {data}")
            
            return {
                "timestamp": datetime.now().timestamp(),
                "source": "system",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": {
                    "timestamp": data
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Timestamp",
                operation="timestamp_processing",
                reason=str(e),
                rawData=data
            )
    
    async def _processSystemConfig(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Processa mensagens de configuração do sistema.
        
        Args:
            data: Dados de configuração
            
        Returns:
            Dados formatados para SignalManager
        """
        
        try:
            signalMapping = self.topicSignalMapping["Cfg"]
            
            self.logger.debug(f"Processing system config: {data}")
            
            return {
                "timestamp": datetime.now().timestamp(),
                "source": "system",
                "signalType": signalMapping["signalType"],
                "dataType": signalMapping["dataType"],
                "data": {
                    "config": data
                }
            }
            
        except Exception as e:
            raise ZeroMQProcessingError(
                topic="Cfg",
                operation="config_processing",
                reason=str(e),
                rawData=data
            )
    
    def _recordError(self, topic: str, error: str) -> None:
        """
        Regista erro nas estatísticas para debugging.
        
        Args:
            topic: Tópico onde ocorreu o erro
            error: Descrição do erro
        """
        
        self.processingStats["totalErrors"] += 1
        
        if topic in self.processingStats["byTopic"]:
            self.processingStats["byTopic"][topic]["errors"] += 1
            self.processingStats["byTopic"][topic]["lastError"] = {
                "error": error,
                "timestamp": datetime.now().isoformat()
            }
        
        self.logger.error(f"Recorded processing error for {topic}: {error}")
    
    def getProcessingStats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de processamento para monitoring.
        
        Returns:
            Estatísticas detalhadas por tópico incluindo dados de Signal Control
        """
        
        return {
            "totalProcessed": self.processingStats["totalProcessed"],
            "totalErrors": self.processingStats["totalErrors"],
            "totalFiltered": self.processingStats["totalFiltered"],
            "successRate": (
                self.processingStats["totalProcessed"] / 
                max(1, self.processingStats["totalProcessed"] + self.processingStats["totalErrors"])
            ),
            "filterRate": (
                self.processingStats["totalFiltered"] / 
                max(1, self.processingStats["totalProcessed"] + self.processingStats["totalErrors"] + self.processingStats["totalFiltered"])
            ),
            "byTopic": self.processingStats["byTopic"].copy(),
            "supportedTopics": list(self.topicSignalMapping.keys()),
            "signalControl": {
                "availableSignals": self.getAvailableSignals(),
                "activeSignals": self.getActiveSignals(),
                "componentState": self.getComponentState().value,
                "filteredTopics": [topic for topic in self.availableSignals if topic not in self.activeSignals]
            },
            "lastUpdate": datetime.now().isoformat()
        }
    
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
                "totalFiltered": self.processingStats["totalFiltered"],
                "filteringRate": self.processingStats["totalFiltered"] / max(1, 
                    self.processingStats["totalProcessed"] + self.processingStats["totalErrors"] + self.processingStats["totalFiltered"]
                )
            },
            "lastUpdate": datetime.now().isoformat()
        }
    
    def reset(self) -> None:
        """
        Reset das estatísticas de processamento.
        """
        
        self.processingStats = {
            "totalProcessed": 0,
            "totalErrors": 0,
            "totalFiltered": 0,
            "byTopic": {topic: {
                "processed": 0,
                "errors": 0,
                "filtered": 0,
                "lastProcessed": None,
                "lastError": None
            } for topic in self.topicSignalMapping.keys()}
        }
        
        self.chunkCache.clear()
        
        self.logger.info("ZeroMQProcessor statistics reset")

# Instância global
zeroMQProcessor = ZeroMQProcessor()