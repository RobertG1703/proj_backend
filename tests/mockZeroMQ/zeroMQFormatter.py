"""
ZeroMQFormatter - Formatação de dados para protocolo ZeroMQ

Resumo:
Converte dados dos geradores mock para o formato exato esperado pelo ZeroMQProcessor.
Cada tópico tem o seu método específico de formatação que cria a estrutura:
{"ts": "timestamp", "labels": [...], "data": [[val1, val2, ...], [...]]}
Aplica timestamps corretos, serializa com msgpack, e valida estrutura antes de enviar.
"""

import logging
import msgpack
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple
import numpy as np

from app.core import settings
from app.core.exceptions import ZeroMQProcessingError

class ZeroMQFormatter:
    """Formatador de dados mock para protocolo ZeroMQ"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Carregar configurações centralizadas
        self.mockConfig = settings.mockZeromq
        self.zmqConfig = settings.zeromq
        
        # Configurações de timing
        self.timingConfig = self.mockConfig.timingConfig
        self.useRealtimeTimestamps = self.timingConfig["useRealtimeTimestamps"]
        self.timestampPrecision = self.timingConfig["timestampPrecision"]
        
        # Mapeamento de tópicos para métodos de formatação
        self.topicFormatters = {
            "Polar_PPI": self._formatPolarPPI,
            "CardioWheel_ECG": self._formatCardioWheelECG,
            "CardioWheel_ACC": self._formatCardioWheelACC,
            "CardioWheel_GYR": self._formatCardioWheelGYR,
            "BrainAcess_EEG": self._formatBrainAccessEEG,
            "Camera_FaceLandmarks": self._formatCameraFaceLandmarks,
            "Unity_Alcohol": self._formatUnityAlcohol,
            "Unity_CarInfo": self._formatUnityCarInfo,
            "Control": self._formatSystemControl,
            "Timestamp": self._formatSystemTimestamp,
            "Cfg": self._formatSystemConfig
        }
        
        # Validação de configurações
        self.validationConfig = self.zmqConfig.topicValidationConfig
        
        # Estatísticas de formatação
        self.stats = {
            "totalFormatted": 0,
            "totalErrors": 0,
            "byTopic": {topic: {
                "formatted": 0,
                "errors": 0,
                "lastFormatted": None,
                "lastError": None
            } for topic in self.topicFormatters.keys()}
        }
        
        self.logger.info(f"ZeroMQFormatter initialized for {len(self.topicFormatters)} topics")
    
    def formatTopicData(self, topic: str, rawData: Dict[str, Any], 
                       timestamp: Optional[float] = None) -> bytes:
        """
        Formata dados de um tópico específico para protocolo ZeroMQ.
        
        Args:
            topic: Nome do tópico ZeroMQ
            rawData: Dados brutos do gerador
            timestamp: Timestamp opcional (usa tempo atual se None)
            
        Returns:
            Dados serializados em msgpack prontos para envio
            
        Raises:
            ZeroMQProcessingError: Se formatação falhar
        """
        
        if topic not in self.topicFormatters:
            raise ZeroMQProcessingError(
                topic=topic,
                operation="format_lookup",
                reason=f"No formatter found for topic {topic}"
            )
        
        try:
            # Usar timestamp atual se não fornecido
            if timestamp is None:
                timestamp = datetime.now().timestamp()
            
            # Formatar timestamp com precisão configurada
            timestampStr = f"{timestamp:.{self.timestampPrecision}f}"
            
            self.logger.debug(f"Formatting data for topic '{topic}' at timestamp {timestampStr}")
            
            # Chamar formatador específico do tópico
            formatter = self.topicFormatters[topic]
            formattedData = formatter(rawData, timestampStr)
            
            # Validar estrutura antes de serializar
            self._validateFormattedData(topic, formattedData)
            
            # Serializar com msgpack
            serializedData = msgpack.packb(formattedData, use_bin_type=True)
            
            # Atualizar estatísticas
            self._updateStats(topic, success=True)
            
            self.logger.debug(f"Successfully formatted {topic}: {len(serializedData)} bytes")
            
            return serializedData
            
        except Exception as e:
            self._updateStats(topic, success=False, error=str(e))
            raise ZeroMQProcessingError(
                topic=topic,
                operation="formatting",
                reason=str(e),
                rawData=rawData
            )
    
    def _formatCameraFaceLandmarks(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados de face landmarks da câmera.
        
        Input esperado:
        {
            "landmarks": [[x1,y1,z1], [x2,y2,z2], ...],     # 478 pontos faciais
            "gaze_vector": {"dx": 0.2, "dy": -0.1},         # Direção do olhar
            "ear": 0.25,                                    # Eye Aspect Ratio
            "blink_rate": 18,                               # Blinks por minuto
            "blink_counter": 34,                            # Número de blinks
            "frame_b64": "base64_data"                      # Imagem (opcional)
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["landmarks", "gaze_dx", "gaze_dy", "ear", "blink_rate", "blink_counter", "frame_b64"],
            "data": [[flattened_landmarks, gaze_dx, gaze_dy, ear, blink_rate, blink_counter, frame_b64]]
        }
        """
        
        # Extrair dados
        landmarks = rawData.get("landmarks")
        gazeVector = rawData.get("gaze_vector", {})
        ear = rawData.get("ear")
        blinkRate = rawData.get("blink_rate")
        blinkCounter = rawData.get("blink_counter")
        frameB64 = rawData.get("frame_b64", "")  # Imagem opcional
        
        # Validações básicas
        if landmarks is None:
            raise ValueError("Landmarks are required for camera data")
        
        if not isinstance(landmarks, list) or len(landmarks) != 478:
            raise ValueError(f"Expected 478 landmarks, got {len(landmarks) if isinstance(landmarks, list) else 'not list'}")
        
        if ear is None or blinkRate is None:
            raise ValueError("EAR, blink_rate and confidence are required")
        
        if not isinstance(gazeVector, dict) or "dx" not in gazeVector or "dy" not in gazeVector:
            raise ValueError("Gaze vector must contain 'dx' and 'dy' components")
        
        # Validar ranges
        if not (0.0 <= ear <= 1.0):
            raise ValueError(f"EAR {ear} fora do range válido [0.0, 1.0]")
        
        if not (0 <= blinkRate <= 120):
            raise ValueError(f"Blink rate {blinkRate} fora do range válido [0, 120]")
        
        gazeDx = float(gazeVector["dx"])
        gazeDy = float(gazeVector["dy"])
        
        if not (-1.0 <= gazeDx <= 1.0 and -1.0 <= gazeDy <= 1.0):
            raise ValueError(f"Gaze vector ({gazeDx}, {gazeDy}) fora do range válido [-1.0, 1.0]")
        
        # Flatten landmarks de [[x,y,z], ...] para [x1,y1,z1,x2,y2,z2,...]
        landmarksFlat = []
        for point in landmarks:
            if not isinstance(point, list) or len(point) != 3:
                raise ValueError(f"Each landmark must be [x, y, z], got {point}")
            
            # Verificar se coordenadas são válidas (assumindo normalizadas 0-1)
            for coord in point:
                if not (0.0 <= coord <= 1.0):
                    raise ValueError(f"Landmark coordinate {coord} fora do range normalizado [0.0, 1.0]")
            
            landmarksFlat.extend([float(point[0]), float(point[1]), float(point[2])])
        
        # Verificar tamanho final dos landmarks flattened
        if len(landmarksFlat) != 1434:  # 478 * 3
            raise ValueError(f"Flattened landmarks should have 1434 values, got {len(landmarksFlat)}")
        
        return {
            "ts": timestamp,
            "labels": ["landmarks", "gaze_dx", "gaze_dy", "ear", "blink_rate", "blink_counter", "frame_b64"],
            "data": [[
                landmarksFlat,    # 1434 valores [x1,y1,z1,x2,y2,z2,...]
                gazeDx,           # -1.0 a 1.0
                gazeDy,           # -1.0 a 1.0  
                float(ear),       # 0.0 a 1.0
                float(blinkRate), # 0 a 120 bpm
                blinkCounter,     # número de blinks
                frameB64          # imagem em base64
            ]]
        }

    def _formatPolarPPI(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados do Polar ARM Band (PPI).
        
        Input esperado:
        {
            "ppi": 800,           # ms
            "error_ms": 10,       # ms (opcional)
            "flags": 0            # flags (opcional)
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["error_ms", "flags", "value"],
            "data": [[10, 0, 800]]  # error_ms, flags, ppi_ms
        }
        """
        
        # Extrair dados
        ppi = rawData.get("ppi")
        errorMs = rawData.get("error_ms", 10)  # Default 10ms error
        flags = rawData.get("flags", 0)        # Default sem flags
        
        if ppi is None:
            raise ValueError("PPI value is required")
        
        # Validar range de PPI
        validRange = self.zmqConfig.topicProcessingConfig["Polar_PPI"]["validPpiRange"]
        if not (validRange[0] <= ppi <= validRange[1]):
            raise ValueError(f"PPI {ppi} fora do range válido {validRange}")
        
        return {
            "ts": timestamp,
            "labels": ["error_ms", "flags", "value"],
            "data": [[errorMs, flags, ppi]]
        }
    
    def _formatCardioWheelECG(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados ECG do CardioWheel.
        
        Input esperado:
        {
            "ecg": [1646, 1650, 1651, ...],    # Valores ADC 16-bit
            "lod": [0, 0, 0, ...]              # Lead-off detection (opcional)
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["ECG", "LOD"],
            "data": [[1646, 0], [1650, 0], [1651, 0], ...]
        }
        """
        
        # Extrair dados ECG
        ecgValues = rawData.get("ecg")
        lodValues = rawData.get("lod")
        
        if ecgValues is None:
            raise ValueError("ECG values are required")
        
        if not isinstance(ecgValues, (list, np.ndarray)):
            raise ValueError("ECG values must be a list or array")
        
        ecgList = list(ecgValues) if isinstance(ecgValues, np.ndarray) else ecgValues
        
        # LOD values (default to 0 se não fornecido)
        if lodValues is None:
            lodValues = [0] * len(ecgList)
        elif len(lodValues) != len(ecgList):
            # Estender ou truncar LOD para dar match à length do ECG
            lodValues = (list(lodValues) + [0] * len(ecgList))[:len(ecgList)]
        
        # Criar array de dados com ECG e LOD intercalados
        dataArray = []
        for i in range(len(ecgList)):
            dataArray.append([ecgList[i], lodValues[i]])
        
        return {
            "ts": timestamp,
            "labels": ["ECG", "LOD"],
            "data": dataArray
        }
    
    def _formatCardioWheelACC(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados do acelerómetro CardioWheel.
        
        Input esperado:
        {
            "x": [7536, 7520, ...],    # Valores ADC para eixo X
            "y": [3, -6, ...],         # Valores ADC para eixo Y  
            "z": [3104, 3107, ...]     # Valores ADC para eixo Z
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["X", "Y", "Z"],
            "data": [[7536, 3, 3104], [7520, -6, 3107], ...]
        }
        """
        
        # Extrair dados dos eixos
        xValues = rawData.get("x")
        yValues = rawData.get("y")
        zValues = rawData.get("z")
        
        if any(v is None for v in [xValues, yValues, zValues]):
            raise ValueError("X, Y, Z values are required for accelerometer")
        
        # Converter para listas se necessário
        xList = list(xValues) if isinstance(xValues, np.ndarray) else xValues
        yList = list(yValues) if isinstance(yValues, np.ndarray) else yValues
        zList = list(zValues) if isinstance(zValues, np.ndarray) else zValues
        
        # Verificar se todos têm o mesmo comprimento
        if not (len(xList) == len(yList) == len(zList)):
            raise ValueError(f"All axes must have same length: X={len(xList)}, Y={len(yList)}, Z={len(zList)}")
        
        # Criar array de dados
        dataArray = []
        for i in range(len(xList)):
            dataArray.append([xList[i], yList[i], zList[i]])
        
        return {
            "ts": timestamp,
            "labels": ["X", "Y", "Z"],
            "data": dataArray
        }
    
    def _formatCardioWheelGYR(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados do giroscópio CardioWheel.
        
        Input esperado:
        {
            "x": [1, 2, ...],      # Valores ADC para eixo X
            "y": [0, -1, ...],     # Valores ADC para eixo Y
            "z": [2, 0, ...]       # Valores ADC para eixo Z
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp", 
            "labels": ["X", "Y", "Z"],
            "data": [[1, 0, 2], [2, -1, 0], ...]
        }
        """
        
        # Reutilizar lógica do acelerómetro (formato idêntico)
        return self._formatCardioWheelACC(rawData, timestamp)
    
    def _formatBrainAccessEEG(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados EEG do BrainAccess Halo.
        
        Input esperado:
        {
            "ch0": [val1, val2, ...],    # Canal 0 em μV
            "ch1": [val1, val2, ...],    # Canal 1 em μV
            "ch2": [val1, val2, ...],    # Canal 2 em μV
            "ch3": [val1, val2, ...]     # Canal 3 em μV
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["ch0", "ch1", "ch2", "ch3"],
            "data": [[val0, val1, val2, val3], [val0, val1, val2, val3], ...]
        }
        """
        
        # Configuração EEG
        expectedChannels = self.mockConfig.generatorBaseConfig["eeg"]["channelNames"]
        
        # Extrair dados de cada canal
        channelData = {}
        for channel in expectedChannels:
            if channel not in rawData:
                raise ValueError(f"EEG channel '{channel}' is required")
            
            channelValues = rawData[channel]
            channelData[channel] = list(channelValues) if isinstance(channelValues, np.ndarray) else channelValues
        
        # Verificar se todos os canais têm o mesmo comprimento
        lengths = [len(channelData[ch]) for ch in expectedChannels]
        if not all(length == lengths[0] for length in lengths):
            raise ValueError(f"All EEG channels must have same length: {dict(zip(expectedChannels, lengths))}")
        
        # Criar array de dados intercalando canais
        dataArray = []
        sampleCount = lengths[0]
        
        for i in range(sampleCount):
            sample = []
            for channel in expectedChannels:
                sample.append(channelData[channel][i])
            dataArray.append(sample)
        
        return {
            "ts": timestamp,
            "labels": expectedChannels,
            "data": dataArray
        }
    
    def _formatUnityAlcohol(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados de nível de álcool do Unity.
        
        Input esperado:
        {
            "alcohol_level": 0.3
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp",
            "labels": ["alcohol_level"],
            "data": [[0.3]]
        }
        """
        
        # Extrair dados
        alcoholLevel = rawData.get("alcohol_level")
        
        if alcoholLevel is None:
            raise ValueError("Alcohol level is required for Unity alcohol data")
        
        # Validar range usando configurações centralizadas
        validRange = self.zmqConfig.topicValidationConfig["Unity_Alcohol"]["valueRanges"]["alcohol_level"]
        if not (validRange[0] <= alcoholLevel <= validRange[1]):
            raise ValueError(f"Alcohol level {alcoholLevel} fora do range válido {validRange}")
        
        return {
            "ts": timestamp,
            "labels": ["alcohol_level"],
            "data": [[float(alcoholLevel)]]
        }
    
    def _formatUnityCarInfo(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata dados de informação do carro do Unity.
        
        Input esperado:
        {
            "speed": 65.0,
            "lane_centrality": 0.8
        }
        
        Output ZeroMQ:
        {
            "ts": "timestamp", 
            "labels": ["speed", "lane_centrality"],
            "data": [[65.0, 0.8]]
        }
        """
        
        # Extrair dados
        speed = rawData.get("speed")
        laneCentrality = rawData.get("lane_centrality")
        
        if speed is None or laneCentrality is None:
            raise ValueError("Speed and lane_centrality are required for Unity car info")
        
        # Validar ranges usando configurações centralizadas
        carInfoValidation = self.zmqConfig.topicValidationConfig["Unity_CarInfo"]["valueRanges"]
        
        speedRange = carInfoValidation["speed"]
        if not (speedRange[0] <= speed <= speedRange[1]):
            raise ValueError(f"Speed {speed} fora do range válido {speedRange}")
        
        centralityRange = carInfoValidation["lane_centrality"]
        if not (centralityRange[0] <= laneCentrality <= centralityRange[1]):
            raise ValueError(f"Lane centrality {laneCentrality} fora do range válido {centralityRange}")
        
        return {
            "ts": timestamp,
            "labels": ["speed", "lane_centrality"], 
            "data": [[float(speed), float(laneCentrality)]]
        }

    def _formatSystemControl(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata mensagens de controlo do sistema.
        
        Input: qualquer dict
        Output: formato básico com timestamp
        """
        
        return {
            "ts": timestamp,
            "control": rawData
        }
    
    def _formatSystemTimestamp(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata mensagens de timestamp do sistema.
        
        Input: qualquer dict
        Output: formato básico com timestamp
        """
        
        return {
            "ts": timestamp,
            "system_timestamp": rawData
        }
    
    def _formatSystemConfig(self, rawData: Dict[str, Any], timestamp: str) -> Dict[str, Any]:
        """
        Formata mensagens de configuração do sistema.
        
        Input: qualquer dict
        Output: formato básico
        """
        
        return {
            "config": rawData
        }
    
    def _validateFormattedData(self, topic: str, formattedData: Dict[str, Any]) -> None:
        """
        Valida estrutura dos dados formatados antes de serializar.
        
        Args:
            topic: Nome do tópico
            formattedData: Dados formatados para validar
            
        Raises:
            ValueError: Se estrutura inválida
        """
        
        if topic not in self.validationConfig:
            # Se não há config de validação, só verificar estrutura básica
            if not isinstance(formattedData, dict):
                raise ValueError("Formatted data must be a dictionary")
            return
        
        config = self.validationConfig[topic]
        
        # Verificar campos obrigatórios
        requiredFields = config.get("requiredFields", [])
        for field in requiredFields:
            if field not in formattedData:
                raise ValueError(f"Required field '{field}' missing in formatted data")
        
        # Validar estrutura específica (ts, labels, data)
        if "labels" in requiredFields and "data" in requiredFields:
            labels = formattedData.get("labels", [])
            data = formattedData.get("data", [])
            
            if not isinstance(labels, list):
                raise ValueError("Labels must be a list")
            
            if not isinstance(data, list):
                raise ValueError("Data must be a list")
            
            # Verificar se data tem estrutura correta
            if data and isinstance(data[0], list):
                expectedColumns = len(labels)
                for i, row in enumerate(data):
                    if len(row) != expectedColumns:
                        raise ValueError(f"Data row {i} has {len(row)} columns, expected {expectedColumns}")
        
        self.logger.debug(f"Validation passed for topic '{topic}'")
    
    def _updateStats(self, topic: str, success: bool, error: str = None) -> None:
        """
        Atualiza estatísticas de formatação.
        
        Args:
            topic: Tópico processado
            success: Se formatação foi bem-sucedida
            error: Mensagem de erro se falhou
        """
        
        if success:
            self.stats["totalFormatted"] += 1
            if topic in self.stats["byTopic"]:
                self.stats["byTopic"][topic]["formatted"] += 1
                self.stats["byTopic"][topic]["lastFormatted"] = datetime.now().isoformat()
        else:
            self.stats["totalErrors"] += 1
            if topic in self.stats["byTopic"]:
                self.stats["byTopic"][topic]["errors"] += 1
                self.stats["byTopic"][topic]["lastError"] = {
                    "error": error,
                    "timestamp": datetime.now().isoformat()
                }
    
    def getStats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas de formatação.
        
        Returns:
            Estatísticas detalhadas por tópico
        """
        
        return {
            "totalFormatted": self.stats["totalFormatted"],
            "totalErrors": self.stats["totalErrors"],
            "successRate": (
                self.stats["totalFormatted"] / 
                max(1, self.stats["totalFormatted"] + self.stats["totalErrors"])
            ),
            "byTopic": self.stats["byTopic"].copy(),
            "supportedTopics": list(self.topicFormatters.keys()),
            "lastUpdate": datetime.now().isoformat()
        }
    
    def validateTopicSupport(self, topic: str) -> bool:
        """
        Verifica se tópico é suportado pelo formatter.
        
        Args:
            topic: Nome do tópico
            
        Returns:
            True se tópico é suportado
        """
        
        return topic in self.topicFormatters
    
    def reset(self) -> None:
        """
        Reset das estatísticas de formatação.
        """
        
        self.stats = {
            "totalFormatted": 0,
            "totalErrors": 0,
            "byTopic": {topic: {
                "formatted": 0,
                "errors": 0,
                "lastFormatted": None,
                "lastError": None
            } for topic in self.topicFormatters.keys()}
        }
        
        self.logger.info("ZeroMQFormatter statistics reset")

# Instância global
zeroMQFormatter = ZeroMQFormatter()