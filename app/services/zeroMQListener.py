"""
ZeroMQListener - Recepção de dados reais de sensores via PUB/SUB

Resumo:
Sistema de recepção de dados em tempo real dos sensores através do protocolo ZeroMQ PUB/SUB.
Estabelece uma conexão SUB socket que subscreve a tópicos específicos dos sensores
(CardioWheel, Polar ARM Band, Halo EEG) e passa os dados recebidos para o ZeroMQProcessor
para conversão e formatação antes de enviar ao SignalManager. Inclui controlo granular
de sinais através do sistema Signal Control para filtering interno.

Funcionalidades principais:
- Conexão ZeroMQ SUB socket para receber dados de múltiplos sensores PUB
- Subscrição automática a todos os tópicos configurados
- Recepção de mensagens msgpack dos sensores em tempo real
- Filtering interno de tópicos através do Signal Control
- Delegação do processamento de dados ao ZeroMQProcessor
- Reconexão automática em caso de falha de comunicação
- Monitorização contínua da saúde da conexão com métricas detalhadas
- Deteção de timeouts e problemas de comunicação
- Emissão de eventos de estado para monitorização externa
- Processamento assíncrono de mensagens com controlo da performance

IMPORTANTE: Este listener subscreve a todos os tópicos via ZeroMQ mas filtra internamente
quais processa baseado no Signal Control. Todo o processamento, validação e conversão
é delegado ao ZeroMQProcessor para manter separação de responsabilidades.
"""

import asyncio
import logging
import zmq
import zmq.asyncio
from datetime import datetime
from typing import Dict, Any, List, Optional, Set
from enum import Enum

from ..core import settings, eventManager
from ..core.exceptions import ZeroMQError
from ..core.signalControl import SignalControlInterface, SignalState, ComponentState, signalControlManager
from .zeroMQProcessor import zeroMQProcessor
from .signalManager import signalManager

class ListenerState(Enum):
    """Estados possíveis do listener ZeroMQ"""
    STOPPED = "stopped"
    CONNECTING = "connecting"
    CONNECTED = "connected"
    RECONNECTING = "reconnecting"
    ERROR = "error"

class WhoToListenState(Enum):
    """Estados possíveis que o listener pode tomar para decidir que portas deve 'ouvir'"""
    MOCK = "mock"
    REAL = "real"
    CUSTOM = "custom"

class ZeroMQListener(SignalControlInterface):
    """Listener para recepção de dados reais de sensores via ZeroMQ PUB/SUB com controlo de sinais"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Carregar configurações ZeroMQ centralizadas
        self.zmqConfig = settings.zeromq
        self.mockConfig = settings.mockZeromq
        
        # Estado atual do modo
        self.currentMode = WhoToListenState.REAL if settings.useRealSensors else WhoToListenState.MOCK
        
        # Configurar endereços iniciais
        self._updateConnectionConfig()

        self.timeout = self.zmqConfig.timeout
        
        # Tópicos para subscrever (configurados centralmente)
        self.topics = self.zmqConfig.topics.copy()  # Lista de tópicos configurados
        self.subscribedTopics: Set[str] = set()
        
        # Signal Control properties
        self.availableSignals = settings.signalControl.zeroMQTopics.copy()
        defaultActiveStates = settings.signalControl.defaultActiveStates["listener"]
        self.activeSignals: Set[str] = {signal for signal, active in defaultActiveStates.items() if active}
        
        # Configurações de socket
        self.lingerTime = self.zmqConfig.lingerTime
        self.receiveHighWaterMark = self.zmqConfig.receiveHighWaterMark
        self.socketType = "SUB"  # Sempre SUB para este listener
        
        # Configurações de reconexão e timeouts
        self.maxReconnectAttempts = self.zmqConfig.maxReconnectAttempts
        self.reconnectDelay = self.zmqConfig.reconnectDelay
        self.messageTimeout = self.zmqConfig.messageTimeout
        self.heartbeatInterval = self.zmqConfig.heartbeatInterval
        
        # Configurações de performance
        self.processingTimeoutWarning = self.zmqConfig.processingTimeoutWarning
        self.errorRateWarningThreshold = self.zmqConfig.errorRateWarningThreshold
        self.rejectionRateWarningThreshold = self.zmqConfig.rejectionRateWarningThreshold
        
        # Estado da conexão
        self.state = ListenerState.STOPPED
        self.startTime: Optional[datetime] = None
        self.lastMessageTime: Optional[datetime] = None
        self.lastMessageByTopic: Dict[str, datetime] = {}
        self.reconnectAttempts = 0
        
        # Componentes ZeroMQ
        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        self.listenerTask: Optional[asyncio.Task] = None
        self.heartbeatTask: Optional[asyncio.Task] = None
        
        # Estatísticas de monitorização por tópico
        self.stats = {
            "messagesReceived": 0,
            "messagesProcessed": 0,
            "messagesRejected": 0,
            "messagesFiltered": 0,  
            "lastMessageTimestamp": None,
            "connectionUptime": 0.0,
            "reconnections": 0,
            "errors": 0,
            "averageProcessingTime": 0.0,
            "topicStats": {topic: {
                "received": 0,
                "processed": 0,
                "rejected": 0,
                "filtered": 0,
                "lastMessage": None
            } for topic in self.topics},
            "dataTypeStats": {}
        }
        
        # Registar no manager central de Signal Control
        signalControlManager.registerComponent("listener", self)
        
        self.logger.info(f"ZeroMQListener initialized with Signal Control - Publisher: {self.subscriberUrl}, Topics: {self.topics}")
    
    # Signal Control Interface Implementation
    
    def getAvailableSignals(self) -> List[str]:
        """Retorna lista de tópicos disponíveis para recepção"""
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
        if self.state == ListenerState.CONNECTED:
            return ComponentState.RUNNING
        elif self.state == ListenerState.STOPPED:
            return ComponentState.STOPPED
        elif self.state == ListenerState.ERROR:
            return ComponentState.ERROR
        else:
            return ComponentState.INITIALIZING
    
    # Core Listener Methods
    
    def _updateConnectionConfig(self):
        """Atualiza configuração de conexão baseada no modo atual"""
        if self.currentMode == WhoToListenState.REAL:
            self.publisherAddress = self.zmqConfig.publisherAddress  # 192.168.1.103
            self.subscriberPort = self.zmqConfig.subscriberPort      # 22881
            self.subscriberUrl = f"tcp://{self.publisherAddress}:{self.subscriberPort}"
            self.logger.info(f"ZeroMQListener CONFIG: Real mode - {self.subscriberUrl}")
        elif self.currentMode == WhoToListenState.MOCK: 
            self.publisherAddress = self.mockConfig.mockPublisherAddress  # 127.0.0.1
            self.subscriberPort = self.mockConfig.mockPublisherPort       # 22882
            self.subscriberUrl = self.mockConfig.mockPublisherUrl         # tcp://127.0.0.1:22882
            self.logger.info(f"ZeroMQListener CONFIG: Mock mode - {self.subscriberUrl}")
        else:
            self.logger.info(f"ZeroMQListener CONFIG: mode desconhecido config mal feita. - {self.subscriberUrl}")

    async def start(self):
        """
        Inicia o listener ZeroMQ e estabelece conexão com publisher.
        
        Configura o socket SUB, subscreve aos tópicos configurados,
        inicia as tasks de recepção e monitorização.
        
        Raises:
            ZeroMQError: Se falhar ao estabelecer conexão inicial
        """
        if self.state in [ListenerState.CONNECTING, ListenerState.CONNECTED]:
            self.logger.warning("ZeroMQListener already running")
            return
        
        self.logger.info("Starting ZeroMQ SUB listener...")
        self.startTime = datetime.now()
        self.reconnectAttempts = 0
        
        try:
            # Estabelecer conexão ZeroMQ
            await self._connect()
            
            # Subscrever aos tópicos
            await self._subscribeToTopics()
            
            # Iniciar tasks de processamento
            self.listenerTask = asyncio.create_task(self._messageLoop())
            self.heartbeatTask = asyncio.create_task(self._heartbeatLoop())
            
            # Emitir evento de início
            await eventManager.emit("zmq.listener_started", {
                "timestamp": datetime.now().isoformat(),
                "subscriberUrl": self.subscriberUrl,
                "socketType": self.socketType,
                "topics": list(self.subscribedTopics),
                "timeout": self.timeout
            })
            
            self.logger.info(f"ZeroMQListener started successfully on {self.subscriberUrl}")
            self.logger.info(f"Subscribed to topics: {list(self.subscribedTopics)}")
            
        except Exception as e:
            self.state = ListenerState.ERROR
            self.stats["errors"] += 1
            await self._emitError("startup_failed", str(e))
            raise ZeroMQError("startup", str(e))
    
    async def stop(self):
        """
        Para o listener ZeroMQ e limpa recursos.
        
        Cancela tasks em execução, fecha socket e contexto ZeroMQ,
        e emite evento de paragem com estatísticas finais.
        """
        if self.state == ListenerState.STOPPED:
            return
        
        self.logger.info("Stopping ZeroMQ SUB listener...")
        self.state = ListenerState.STOPPED
        
        try:
            # Cancelar tasks de processamento
            if self.listenerTask and not self.listenerTask.done():
                self.listenerTask.cancel()
                try:
                    await self.listenerTask
                except asyncio.CancelledError:
                    pass
            
            if self.heartbeatTask and not self.heartbeatTask.done():
                self.heartbeatTask.cancel()
                try:
                    await self.heartbeatTask
                except asyncio.CancelledError:
                    pass
            
            # Fechar conexão ZeroMQ
            await self._disconnect()
            
            # Emitir evento de paragem com estatísticas
            await eventManager.emit("zmq.listener_stopped", {
                "timestamp": datetime.now().isoformat(),
                "uptime": self._getUptime(),
                "finalStats": self.stats.copy()
            })
            
            self.logger.info("ZeroMQListener stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping ZeroMQListener: {e}")
            self.stats["errors"] += 1
    
    async def _connect(self):
        """
        Estabelece conexão ZeroMQ SUB socket para receber dados do publisher.
        
        Configura socket com parâmetros de timeout, buffer e conecta ao
        endereço do publisher especificado.
        
        Raises:
            ZeroMQError: Se falhar ao estabelecer conexão
        """
        try:
            self.state = ListenerState.CONNECTING
            
            # Criar contexto e socket ZeroMQ
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.SUB)
            
            # Configurar parâmetros do socket
            self.socket.setsockopt(zmq.RCVTIMEO, self.timeout)
            self.socket.setsockopt(zmq.LINGER, self.lingerTime)
            self.socket.setsockopt(zmq.RCVHWM, self.receiveHighWaterMark)
            
            # Conectar ao publisher
            self.socket.connect(self.subscriberUrl)
            
            # Atualizar estado e timestamps
            self.state = ListenerState.CONNECTED
            self.lastMessageTime = datetime.now()
            self.reconnectAttempts = 0
            
            # Emitir evento de conexão estabelecida
            await eventManager.emit("zmq.connected", {
                "timestamp": datetime.now().isoformat(),
                "subscriberUrl": self.subscriberUrl,
                "socketType": self.socketType,
                "socketUrl": self.subscriberUrl,  # Adicionar para compatibilidade
                "reconnectAttempt": self.reconnectAttempts
            })
            
            self.logger.info(f"ZeroMQ SUB socket connected to {self.subscriberUrl}")
            
        except Exception as e:
            self.state = ListenerState.ERROR
            self.stats["errors"] += 1
            raise ZeroMQError("connection", str(e))
    
    async def _subscribeToTopics(self):
        """
        Subscreve a todos os tópicos configurados.
        
        Para cada tópico na configuração, estabelece subscrição no socket ZeroMQ
        e actualiza conjunto de tópicos subscritos.
        """
        try:
            for topic in self.topics:
                self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
                self.subscribedTopics.add(topic)
                
                # Inicializar stats por tópico se não existir
                if topic not in self.stats["topicStats"]:
                    self.stats["topicStats"][topic] = {
                        "received": 0,
                        "processed": 0,
                        "rejected": 0,
                        "filtered": 0,
                        "lastMessage": None
                    }
                
                self.logger.debug(f"Subscribed to topic: {topic}")
            
            self.logger.info(f"Successfully subscribed to {len(self.subscribedTopics)} topics")
            
        except Exception as e:
            raise ZeroMQError("subscription", str(e))
    
    async def _disconnect(self):
        """
        Fecha socket e termina contexto ZeroMQ de forma segura.
        """
        try:
            if self.socket:
                # Unsubscribe de todos os tópicos
                for topic in self.subscribedTopics:
                    try:
                        self.socket.setsockopt_string(zmq.UNSUBSCRIBE, topic)
                    except:
                        pass
                
                self.socket.close()
                self.socket = None
            
            if self.context:
                self.context.term()
                self.context = None
            
            self.subscribedTopics.clear()
            self.logger.debug(f"ZeroMQ SUB socket closed")
            
        except Exception as e:
            self.logger.error(f"Error closing ZeroMQ socket: {e}")
            self.stats["errors"] += 1
    
    async def _messageLoop(self):
        """
        Loop principal de recepção e processamento de mensagens.
        
        Executa continuamente enquanto o listener estiver activo, recebendo
        mensagens dos sensores via tópicos e delegando processamento ao ZeroMQProcessor.
        """
        while self.state != ListenerState.STOPPED:
            try:
                if self.state == ListenerState.CONNECTED:
                    await self._receiveMessage()
                else:
                    await self._attemptReconnect()
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error(f"Error in message loop: {e}")
                self.stats["errors"] += 1
                await self._handleConnectionError(str(e))
    
    async def _receiveMessage(self):
        """
        Recebe e processa uma mensagem individual do socket ZeroMQ.
        
        Aguarda mensagem multipart (tópico + dados) com timeout configurado,
        descodifica dados msgpack e delega processamento ao ZeroMQProcessor.
        """
        try:
            # Aguardar mensagem multipart com timeout
            multiPartMsg = await asyncio.wait_for(
                self.socket.recv_multipart(zmq.NOBLOCK),
                timeout=self.timeout / 1000.0
            )
            
            # Extrair tópico e dados
            if len(multiPartMsg) < 2:
                self.logger.warning("Received malformed multipart message")
                self.stats["messagesRejected"] += 1
                return
            
            topic = multiPartMsg[0].decode('utf-8')
            rawData = multiPartMsg[1]  # Manter como bytes para o processor
            
            self.logger.debug(f"Received message from topic: {topic}, size: {len(rawData)} bytes")
            
            # Atualizar contadores de recepção
            self.stats["messagesReceived"] += 1
            if topic in self.stats["topicStats"]:
                self.stats["topicStats"][topic]["received"] += 1
            
            self.lastMessageTime = datetime.now()
            self.lastMessageByTopic[topic] = datetime.now()
            
            # Processar mensagem com medição de tempo
            startTime = datetime.now()
            await self._processMessage(topic, rawData)
            processingTime = (datetime.now() - startTime).total_seconds()
            
            # Verificar se processamento demorou mais que o esperado
            if processingTime > self.processingTimeoutWarning:
                self.logger.warning(f"Slow message processing for topic {topic}: {processingTime:.3f}s")
            
            # Atualizar tempo médio de processamento
            if self.stats["messagesProcessed"] > 0:
                currentAvg = self.stats["averageProcessingTime"]
                processedCount = self.stats["messagesProcessed"]
                self.stats["averageProcessingTime"] = (
                    (currentAvg * (processedCount - 1) + processingTime) / processedCount
                )
            
        except asyncio.TimeoutError:
            # Timeout esperado - verificar se não há mensagens há muito tempo
            await self._checkMessageTimeout()
            
        except zmq.Again:
            # Não há mensagens disponíveis no momento
            await asyncio.sleep(0.01)
            
        except Exception as e:
            self.logger.error(f"Error receiving message: {e}")
            self.stats["errors"] += 1
            await self._handleConnectionError(str(e))
    
    async def _processMessage(self, topic: str, rawData: bytes):
        """
        Processa dados recebidos dos sensores delegando ao ZeroMQProcessor.
        
        Args:
            topic: Tópico ZeroMQ da mensagem
            rawData: Dados brutos em bytes (msgpack)
        """
        try:
            # Filtering via Signal Control
            if topic not in self.activeSignals:
                self.stats["messagesFiltered"] += 1
                if topic in self.stats["topicStats"]:
                    self.stats["topicStats"][topic]["filtered"] += 1
                self.logger.debug(f"Topic {topic} filtered by Signal Control")
                return
            
            self.logger.debug(f"Processing message from topic: {topic}")
            
            # Delegar processamento ao ZeroMQProcessor
            processedData = await zeroMQProcessor.processTopicData(topic, rawData)
            
            if not processedData:
                self.stats["messagesRejected"] += 1
                if topic in self.stats["topicStats"]:
                    self.stats["topicStats"][topic]["rejected"] += 1
                self.logger.warning(f"ZeroMQProcessor returned None for topic {topic}")
                return
            
            # Atualizar timestamp da última mensagem válida
            self.stats["lastMessageTimestamp"] = processedData.get("timestamp")
            if topic in self.stats["topicStats"]:
                self.stats["topicStats"][topic]["lastMessage"] = datetime.now().isoformat()
            
            # Contar tipos de dados recebidos para estatísticas
            dataType = processedData.get("dataType", "unknown")
            if dataType not in self.stats["dataTypeStats"]:
                self.stats["dataTypeStats"][dataType] = 0
            self.stats["dataTypeStats"][dataType] += 1
            
            # Enviar dados processados para o SignalManager
            success = await signalManager.processZeroMQData(processedData)
            
            if success:
                self.stats["messagesProcessed"] += 1
                if topic in self.stats["topicStats"]:
                    self.stats["topicStats"][topic]["processed"] += 1
                self.logger.debug(f"Message successfully processed from topic {topic} -> {dataType}")
            else:
                self.stats["messagesRejected"] += 1
                if topic in self.stats["topicStats"]:
                    self.stats["topicStats"][topic]["rejected"] += 1
                self.logger.warning(f"Message rejected by SignalManager from topic {topic}")
            
            # Emitir evento de mensagem recebida
            await eventManager.emit("zmq.message_received", {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "dataType": dataType,
                "signalType": processedData.get("signalType"),
                "processed": success,
                "messageSize": len(rawData)
            })
            
        except Exception as e:
            self.stats["messagesRejected"] += 1
            if topic in self.stats["topicStats"]:
                self.stats["topicStats"][topic]["rejected"] += 1
            self.stats["errors"] += 1
            self.logger.error(f"Error processing message from topic {topic}: {e}")

    async def switchMode(self, newMode: str):
        """
        Muda dinamicamente entre modos real/mock
        
        Args:
            newMode: "real" ou "mock"
        """
        # Validar string e converter para enum
        try:
            if newMode.lower() == "real":
                newModeEnum = WhoToListenState.REAL
            elif newMode.lower() == "mock":
                newModeEnum = WhoToListenState.MOCK
            else:
                raise ValueError(f"Invalid mode: '{newMode}'. Must be 'real' or 'mock'")
        except AttributeError:
            raise ValueError(f"Mode must be a string, got {type(newMode)}")
        
        # Comparar enums
        if newModeEnum == self.currentMode:
            self.logger.info(f"Already in {newMode} mode")
            return
        
        oldMode = self.currentMode.value  
        self.logger.info(f"Switching from {oldMode} to {newMode} mode...")
        
        # Para conexão atual
        if self.state == ListenerState.CONNECTED:
            await self.stop()
        
        # Atualizar
        self.currentMode = newModeEnum
        self._updateConnectionConfig()
        
        # Reiniciar com nova configuração
        await self.start()
        
        self.logger.info(f"Successfully switched to {newMode} mode")
        
        # Emitir evento de mudança
        await eventManager.emit("zmq.mode_switched", {
            "timestamp": datetime.now().isoformat(),
            "oldMode": oldMode,
            "newMode": newMode,
            "newUrl": self.subscriberUrl
        })
    
    def getCurrentMode(self) -> str:
        """Retorna modo atual"""
        return self.currentMode.value
    
    def getAvailableModes(self) -> List[str]:
        """Retorna modos disponíveis"""
        return [mode.value for mode in WhoToListenState]
    
    async def setCustomAddress(self, address: str, port: int):
        """
        Define endereço customizado temporariamente
        
        Args:
            address: IP address (ex: "192.168.1.100")
            port: Port number (ex: 22883)
        """
        self.logger.info(f"Setting custom address: {address}:{port}")
        
        # Para conexão atual se ativa
        if self.state == ListenerState.CONNECTED:
            await self.stop()
        
        # Definir configuração customizada
        self.currentMode = WhoToListenState.CUSTOM
        self.publisherAddress = address
        self.subscriberPort = port
        self.subscriberUrl = f"tcp://{address}:{port}"
        
        # Reiniciar
        await self.start()
        
        self.logger.info(f"Now listening on custom address: {self.subscriberUrl}")
    
    async def _checkMessageTimeout(self):
        """
        Verifica se há timeout na recepção de mensagens por tópico.
        
        Emite aviso se não foram recebidas mensagens por período superior
        ao timeout configurado para qualquer tópico.
        """
        now = datetime.now()
        
        # Verificar timeout geral
        if self.lastMessageTime:
            timeSinceLastMessage = (now - self.lastMessageTime).total_seconds()
            if timeSinceLastMessage > self.messageTimeout:
                await self._emitWarning("message_timeout", 
                    f"No messages received for {timeSinceLastMessage:.1f}s")
        
        # Verificar timeout por tópico
        for topic, lastTime in self.lastMessageByTopic.items():
            timeSinceTopicMessage = (now - lastTime).total_seconds()
            if timeSinceTopicMessage > self.messageTimeout * 2:  # Timeout mais longo por tópico
                await self._emitWarning("topic_timeout",
                    f"No messages from topic {topic} for {timeSinceTopicMessage:.1f}s")
    
    async def _attemptReconnect(self):
        """
        Tenta restabelecer conexão ZeroMQ em caso de falha.
        
        Implementa backoff exponencial e limite de tentativas para
        evitar sobrecarga em caso de problemas persistentes.
        """
        if self.reconnectAttempts >= self.maxReconnectAttempts:
            self.state = ListenerState.ERROR
            await self._emitError("max_reconnect_attempts", 
                f"Failed to reconnect after {self.maxReconnectAttempts} attempts")
            return
        
        self.state = ListenerState.RECONNECTING
        self.reconnectAttempts += 1
        
        self.logger.info(f"Attempting to reconnect ({self.reconnectAttempts}/{self.maxReconnectAttempts})...")
        
        try:
            # Fechar conexão anterior
            await self._disconnect()
            
            # Aguardar antes de tentar reconectar
            await asyncio.sleep(self.reconnectDelay)
            
            # Tentar restabelecer conexão
            await self._connect()
            await self._subscribeToTopics()
            
            self.stats["reconnections"] += 1
            self.logger.info(f"Reconnection successful after {self.reconnectAttempts} attempts")
            
        except Exception as e:
            self.logger.error(f"Reconnection attempt {self.reconnectAttempts} failed: {e}")
            self.stats["errors"] += 1
            await asyncio.sleep(self.reconnectDelay)
    
    async def _handleConnectionError(self, error: str):
        """
        Gere erros de conexão e inicia processo de reconexão.
        
        Args:
            error: Descrição do erro de conexão
        """
        self.logger.error(f"Connection error: {error}")
        
        if self.state == ListenerState.CONNECTED:
            self.state = ListenerState.RECONNECTING
            await self._emitError("connection_lost", error)
    
    async def _heartbeatLoop(self):
        """
        Loop de monitorização que envia heartbeat periódico.
        
        Emite evento de heartbeat com estatísticas e estado da conexão
        para permitir monitorização externa da saúde do sistema.
        """
        while self.state != ListenerState.STOPPED:
            try:
                if self.state == ListenerState.CONNECTED:
                    await self._sendHeartbeat()
                
                await asyncio.sleep(self.heartbeatInterval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                self.stats["errors"] += 1
    
    async def _sendHeartbeat(self):
        """
        Envia heartbeat com estado actual e métricas de performance.
        
        Inclui estatísticas de mensagens por tópico, saúde da conexão
        e tempo desde última mensagem para monitorização externa.
        """
        try:
            await eventManager.emit("zmq.heartbeat", {
                "timestamp": datetime.now().isoformat(),
                "state": self.state.value,
                "uptime": self._getUptime(),
                "stats": self.stats.copy(),
                "lastMessageAge": self._getLastMessageAge(),
                "topicHealth": self._getTopicHealth(),
                "health": self.getConnectionHealth(),
                "processorStats": zeroMQProcessor.getProcessingStats()
            })
            
        except Exception as e:
            self.logger.error(f"Error sending heartbeat: {e}")
            self.stats["errors"] += 1
    
    def _getTopicHealth(self) -> Dict[str, Any]:
        """
        Avalia saúde individual de cada tópico incluindo estatísticas de filtering.
        
        Returns:
            Dicionário com estado de saúde por tópico
        """
        topicHealth = {}
        now = datetime.now()
        
        for topic in self.subscribedTopics:
            health = "healthy"
            issues = []
            
            topicStats = self.stats["topicStats"].get(topic, {})
            lastMessageTime = self.lastMessageByTopic.get(topic)
            
            # Verificar se recebeu mensagens
            if topicStats.get("received", 0) == 0:
                if self._getUptime() > 10.0:  # Após 10s sem mensagens
                    health = "warning"
                    issues.append("No messages received")
            
            # Verificar timeout específico do tópico
            elif lastMessageTime:
                age = (now - lastMessageTime).total_seconds()
                if age > self.messageTimeout * 2:
                    health = "warning"
                    issues.append(f"No messages for {age:.1f}s")
            
            # Verificar taxa de rejeição
            received = topicStats.get("received", 0)
            rejected = topicStats.get("rejected", 0)
            if received > 0:
                rejection_rate = rejected / received
                if rejection_rate > 0.3:  # >30% rejeição
                    health = "warning"
                    issues.append(f"High rejection rate: {rejection_rate:.1%}")
            
            # Verificar se tópico está sendo filtrado pelo Signal Control
            signal_state = "active" if topic in self.activeSignals else "filtered"
            
            topicHealth[topic] = {
                "health": health,
                "issues": issues,
                "stats": topicStats,
                "signalState": signal_state,
                "lastMessageAge": (now - lastMessageTime).total_seconds() if lastMessageTime else None
            }
        
        return topicHealth
    
    async def _emitError(self, errorType: str, message: str):
        """
        Emite evento de erro com detalhes do problema.
        
        Args:
            errorType: Tipo específico do erro
            message: Descrição detalhada do erro
        """
        await eventManager.emit("zmq.error", {
            "timestamp": datetime.now().isoformat(),
            "errorType": errorType,
            "message": message,
            "state": self.state.value,
            "stats": self.stats.copy(),
            "subscribedTopics": list(self.subscribedTopics)
        })
    
    async def _emitWarning(self, warningType: str, message: str):
        """
        Emite evento de aviso para problemas não críticos.
        
        Args:
            warningType: Tipo específico do aviso
            message: Descrição do problema
        """
        await eventManager.emit("zmq.warning", {
            "timestamp": datetime.now().isoformat(),
            "warningType": warningType,
            "message": message,
            "state": self.state.value,
            "subscribedTopics": list(self.subscribedTopics)
        })
    
    def _getUptime(self) -> float:
        """
        Calcula tempo de funcionamento em segundos.
        
        Returns:
            Uptime em segundos desde o início do listener
        """
        if not self.startTime:
            return 0.0
        return (datetime.now() - self.startTime).total_seconds()
    
    def _getLastMessageAge(self) -> Optional[float]:
        """
        Calcula idade da última mensagem recebida.
        
        Returns:
            Segundos desde última mensagem ou None se nunca recebeu
        """
        if not self.lastMessageTime:
            return None
        return (datetime.now() - self.lastMessageTime).total_seconds()
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna estado completo do listener para monitorização.
        
        Inclui estado da conexão, estatísticas por tópico, configurações,
        métricas de saúde e informações de Signal Control para debug e monitorização externa.
        
        Returns:
            Dicionário com estado completo do sistema
        """
        return {
            "state": self.state.value,
            "subscriberUrl": self.subscriberUrl,
            "socketType": self.socketType,
            "uptime": self._getUptime(),
            "lastMessageAge": self._getLastMessageAge(),
            "reconnectAttempts": self.reconnectAttempts,
            "maxReconnectAttempts": self.maxReconnectAttempts,
            "subscribedTopics": list(self.subscribedTopics),
            "configuredTopics": self.topics.copy(),
            "stats": self.stats.copy(),
            "topicHealth": self._getTopicHealth(),
            "health": self.getConnectionHealth(),
            "processorStats": zeroMQProcessor.getProcessingStats(),
            "signalControl": {
                "availableSignals": self.getAvailableSignals(),
                "activeSignals": self.getActiveSignals(),
                "componentState": self.getComponentState().value,
                "filteredTopics": [topic for topic in self.availableSignals if topic not in self.activeSignals]
            },
            "config": {
                "publisherAddress": self.publisherAddress,
                "subscriberPort": self.subscriberPort,
                "timeout": self.timeout,
                "messageTimeout": self.messageTimeout,
                "reconnectDelay": self.reconnectDelay,
                "heartbeatInterval": self.heartbeatInterval,
                "topics": self.topics
            }
        }
    
    def getConnectionHealth(self) -> Dict[str, Any]:
        """
        Avalia saúde da conexão baseada em métricas e thresholds.
        
        Analisa estado da conexão, taxa de erros, tempo sem mensagens por tópico,
        estatísticas de filtering e outros indicadores para determinar se sistema está saudável.
        
        Returns:
            Dicionário com avaliação de saúde e métricas
        """
        health = "healthy"
        issues = []
        warnings = []
        
        # Verificar estado da conexão
        if self.state == ListenerState.ERROR:
            health = "critical"
            issues.append("Connection in error state")
        elif self.state == ListenerState.RECONNECTING:
            health = "warning"
            warnings.append("Currently reconnecting")
        elif self.state != ListenerState.CONNECTED:
            health = "warning"
            warnings.append(f"Not connected (state: {self.state.value})")
        
        # Verificar timeout de mensagens geral
        lastMessageAge = self._getLastMessageAge()
        if lastMessageAge and lastMessageAge > self.messageTimeout:
            health = "warning" if health == "healthy" else health
            warnings.append(f"No messages for {lastMessageAge:.1f}s")
        elif lastMessageAge is None and self._getUptime() > 10.0:
            health = "warning" if health == "healthy" else health
            warnings.append("Never received messages")
        
        # Verificar saúde individual dos tópicos
        topicHealth = self._getTopicHealth()
        unhealthy_topics = [topic for topic, info in topicHealth.items() 
                           if info["health"] != "healthy"]
        
        if unhealthy_topics:
            if len(unhealthy_topics) == len(self.subscribedTopics):
                health = "critical"
                issues.append(f"All topics unhealthy: {unhealthy_topics}")
            else:
                health = "warning" if health == "healthy" else health
                warnings.append(f"Unhealthy topics: {unhealthy_topics}")
        
        # Verificar tentativas de reconexão
        if self.reconnectAttempts > 0:
            if self.reconnectAttempts >= self.maxReconnectAttempts / 2:
                health = "warning" if health == "healthy" else health
                warnings.append(f"Multiple reconnection attempts ({self.reconnectAttempts})")
        
        # Verificar taxa de erro
        if self.stats["messagesReceived"] > 0:
            errorRate = self.stats["errors"] / self.stats["messagesReceived"]
            if errorRate > self.errorRateWarningThreshold:
                health = "warning" if health == "healthy" else health
                warnings.append(f"High error rate: {errorRate:.1%}")
        
        # Verificar taxa de rejeição
        if self.stats["messagesReceived"] > 0:
            rejectionRate = self.stats["messagesRejected"] / self.stats["messagesReceived"]
            if rejectionRate > self.rejectionRateWarningThreshold:
                health = "warning" if health == "healthy" else health
                warnings.append(f"High rejection rate: {rejectionRate:.1%}")
        
        # Verificar se muitos tópicos estão sendo filtrados
        if len(self.activeSignals) == 0:
            health = "warning" if health == "healthy" else health
            warnings.append("All topics filtered by Signal Control")
        elif len(self.activeSignals) < len(self.availableSignals) / 2:
            filteredCount = len(self.availableSignals) - len(self.activeSignals)
            warnings.append(f"{filteredCount} topics filtered by Signal Control")
        
        return {
            "health": health,
            "issues": issues,
            "warnings": warnings,
            "lastCheck": datetime.now().isoformat(),
            "metrics": {
                "errorRate": self.stats["errors"] / max(1, self.stats["messagesReceived"]),
                "rejectionRate": self.stats["messagesRejected"] / max(1, self.stats["messagesReceived"]),
                "processingRate": self.stats["messagesProcessed"] / max(1, self.stats["messagesReceived"]),
                "filteringRate": self.stats["messagesFiltered"] / max(1, self.stats["messagesReceived"]),
                "topicsHealthy": len([t for t in topicHealth.values() if t["health"] == "healthy"]),
                "totalTopics": len(self.subscribedTopics),
                "activeTopics": len(self.activeSignals),
                "filteredTopics": len(self.availableSignals) - len(self.activeSignals)
            },
            "topicDetails": topicHealth
        }
    
    async def addTopic(self, topic: str):
        """
        Adiciona subscrição a novo tópico dinamicamente.
        
        Args:
            topic: Nome do tópico para subscrever
        """
        if self.state != ListenerState.CONNECTED or not self.socket:
            raise ZeroMQError("add_topic", "Not connected")
        
        try:
            self.socket.setsockopt_string(zmq.SUBSCRIBE, topic)
            self.subscribedTopics.add(topic)
            
            # Adicionar aos sinais disponíveis se não existir
            if topic not in self.availableSignals:
                self.availableSignals.append(topic)
                self.activeSignals.add(topic)  # Ativo por default
            
            # Inicializar stats para o novo tópico
            self.stats["topicStats"][topic] = {
                "received": 0,
                "processed": 0,
                "rejected": 0,
                "filtered": 0,
                "lastMessage": None
            }
            
            self.logger.info(f"Successfully subscribed to new topic: {topic}")
            
            await eventManager.emit("zmq.topic_added", {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "totalTopics": len(self.subscribedTopics)
            })
            
        except Exception as e:
            raise ZeroMQError("add_topic", str(e))
    
    async def removeTopic(self, topic: str):
        """
        Remove subscrição de tópico dinamicamente.
        
        Args:
            topic: Nome do tópico para remover subscrição
        """
        if self.state != ListenerState.CONNECTED or not self.socket:
            raise ZeroMQError("remove_topic", "Not connected")
        
        if topic not in self.subscribedTopics:
            raise ZeroMQError("remove_topic", f"Topic {topic} not subscribed")
        
        try:
            self.socket.setsockopt_string(zmq.UNSUBSCRIBE, topic)
            self.subscribedTopics.remove(topic)
            
            # Remover dos sinais ativos se existir
            self.activeSignals.discard(topic)
            
            # Manter stats para histórico
            if topic in self.lastMessageByTopic:
                del self.lastMessageByTopic[topic]
            
            self.logger.info(f"Successfully unsubscribed from topic: {topic}")
            
            await eventManager.emit("zmq.topic_removed", {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "totalTopics": len(self.subscribedTopics)
            })
            
        except Exception as e:
            raise ZeroMQError("remove_topic", str(e))

# Instância global
zeroMQListener = ZeroMQListener()