"""
ZeroMQPublisher - Publisher mock para simular PC Publisher real

Resumo:
Simula o PC Publisher real que envia dados dos sensores via ZeroMQ PUB socket.
Cria um socket PUB local que publica dados formatados nos tópicos corretos,
permitindo que o ZeroMQListener receba dados como se fossem de sensores reais.
Mantém métricas de publicação e permite controlo granular por tópico.
"""

import asyncio
import logging
import zmq
import zmq.asyncio
from datetime import datetime
from typing import Dict, Any, Optional, Set
from enum import Enum

from app.core import settings, eventManager
from app.core import ZeroMQError

class PublisherState(Enum):
    """Estados possíveis do publisher ZeroMQ"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    STOPPING = "stopping"
    ERROR = "error"

class ZeroMQPublisher:
    """Publisher mock para simular dados de sensores via ZeroMQ"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Carregar configurações mock centralizadas
        mockConfig = settings.mockZeromq
        
        # Configurações de conexão PUB
        self.publisherAddress = mockConfig.mockPublisherAddress  # 127.0.0.1
        self.publisherPort = mockConfig.mockPublisherPort        # 22882
        self.publisherUrl = mockConfig.mockPublisherUrl          # tcp://127.0.0.1:22882
        
        # Configurações de socket
        self.lingerTime = mockConfig.mockLingerTime
        self.sendHighWaterMark = mockConfig.mockSendHighWaterMark
        self.socketType = "PUB"
        
        # Configurações de performance
        self.maxMessagesPerSecond = mockConfig.performanceConfig["maxMessagesPerSecond"]
        self.maxMessageSize = mockConfig.performanceConfig["maxMessageSize"]
        
        # Estado do publisher
        self.state = PublisherState.STOPPED
        self.startTime: Optional[datetime] = None
        self.availableTopics = settings.signalControl.zeroMQTopics.copy()
        defaultActiveStates = settings.signalControl.defaultActiveStates["publisher"]
        self.activeTopics: Set[str] = {signal for signal, active in defaultActiveStates.items() if active}
        
        # Componentes ZeroMQ
        self.context: Optional[zmq.asyncio.Context] = None
        self.socket: Optional[zmq.asyncio.Socket] = None
        
        # Estatísticas de publicação por tópico
        self.stats = {
            "messagesSent": 0,
            "bytesSent": 0,
            "messagesPerSecond": 0.0,
            "lastMessageTimestamp": None,
            "errors": 0,
            "topicStats": {topic: {
                "sent": 0,
                "bytes": 0,
                "lastSent": None,
                "errors": 0
            } for topic in self.availableTopics},
            "startTime": None
        }
        
        # Rate limiting
        self.lastMessageTime = 0.0
        self.messageInterval = 1.0 / self.maxMessagesPerSecond
        
        self.logger.info(f"ZeroMQPublisher initialized - URL: {self.publisherUrl}, Topics: {len(self.availableTopics)}")
    
    async def start(self):
        """
        Inicia o publisher ZeroMQ e cria socket PUB.
        
        Raises:
            ZeroMQError: Se falhar ao estabelecer socket
        """
        if self.state in [PublisherState.STARTING, PublisherState.RUNNING]:
            self.logger.warning("ZeroMQPublisher already running")
            return
        
        self.logger.info("Starting ZeroMQ PUB publisher...")
        self.state = PublisherState.STARTING
        self.startTime = datetime.now()
        
        try:
            # Criar contexto e socket ZeroMQ
            self.context = zmq.asyncio.Context()
            self.socket = self.context.socket(zmq.PUB)
            
            # Configurar parâmetros do socket
            self.socket.setsockopt(zmq.LINGER, self.lingerTime)
            self.socket.setsockopt(zmq.SNDHWM, self.sendHighWaterMark)
            
            # Bind ao endereço configurado
            self.socket.bind(self.publisherUrl)
            
            # Aguardar breve para socket estabilizar
            await asyncio.sleep(0.1)
            
            # Atualizar estado
            self.state = PublisherState.RUNNING
            self.stats["startTime"] = datetime.now().isoformat()
            
            # Emitir evento de início
            await eventManager.emit("mock.publisher_started", {
                "timestamp": datetime.now().isoformat(),
                "publisherUrl": self.publisherUrl,
                "socketType": self.socketType,
                "availableTopics": list(self.availableTopics)
            })
            
            self.logger.info(f"ZeroMQPublisher started successfully on {self.publisherUrl}")
            self.logger.info(f"Available topics: {list(self.availableTopics)}")
            
        except Exception as e:
            self.state = PublisherState.ERROR
            self.stats["errors"] += 1
            await self._emitError("startup_failed", str(e))
            raise ZeroMQError("publisher_startup", str(e))
    
    async def stop(self):
        """
        Para o publisher ZeroMQ e limpa recursos.
        """
        if self.state == PublisherState.STOPPED:
            return
        
        self.logger.info("Stopping ZeroMQ PUB publisher...")
        self.state = PublisherState.STOPPING
        
        try:
            # Fechar socket e contexto
            await self._disconnect()
            
            # Emitir evento de paragem com estatísticas
            await eventManager.emit("mock.publisher_stopped", {
                "timestamp": datetime.now().isoformat(),
                "uptime": self._getUptime(),
                "finalStats": self.stats.copy()
            })
            
            self.state = PublisherState.STOPPED
            self.logger.info("ZeroMQPublisher stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping ZeroMQPublisher: {e}")
            self.stats["errors"] += 1
            self.state = PublisherState.ERROR
    
    async def publishMessage(self, topic: str, data: bytes) -> bool:
        """
        Publica mensagem num tópico específico.
        
        Args:
            topic: Nome do tópico ZeroMQ
            data: Dados serializados (msgpack) para publicar
            
        Returns:
            True se enviado com sucesso, False caso contrário
            
        Raises:
            ZeroMQError: Se publisher não estiver ativo ou erro de envio
        """
        if self.state != PublisherState.RUNNING:
            raise ZeroMQError("publish", f"Publisher not running (state: {self.state.value})")
        
        if topic not in self.availableTopics:
            raise ZeroMQError("publish", f"Unknown topic: {topic}. Available: {list(self.availableTopics)}")
        
        if not isinstance(data, bytes):
            raise ZeroMQError("publish", f"Data must be bytes, got {type(data)}")
        
        if len(data) > self.maxMessageSize:
            raise ZeroMQError("publish", f"Message too large: {len(data)} > {self.maxMessageSize}")
        
        try:
            # Rate limiting simples
            currentTime = asyncio.get_event_loop().time()
            timeSinceLastMessage = currentTime - self.lastMessageTime
            
            if timeSinceLastMessage < self.messageInterval:
                sleepTime = self.messageInterval - timeSinceLastMessage
                await asyncio.sleep(sleepTime)
            
            # Publicar mensagem multipart (tópico + dados)
            await self.socket.send_multipart([
                topic.encode('utf-8'),  # Tópico como bytes
                data                    # Dados já em bytes
            ], zmq.NOBLOCK)
            
            # Atualizar estatísticas
            self._updateStats(topic, len(data))
            self.lastMessageTime = asyncio.get_event_loop().time()
            
            self.logger.debug(f"Published message to topic '{topic}': {len(data)} bytes")
            
            # Emitir evento de mensagem enviada
            await eventManager.emit("mock.message_sent", {
                "timestamp": datetime.now().isoformat(),
                "topic": topic,
                "messageSize": len(data),
                "totalSent": self.stats["messagesSent"]
            })
            
            return True
            
        except zmq.Again:
            # Buffer cheio - não é erro crítico
            self.logger.warning(f"Send buffer full for topic '{topic}', message dropped")
            self.stats["topicStats"][topic]["errors"] += 1
            return False
            
        except Exception as e:
            self.logger.error(f"Error publishing to topic '{topic}': {e}")
            self.stats["errors"] += 1
            self.stats["topicStats"][topic]["errors"] += 1
            await self._emitError("publish_failed", f"Topic '{topic}': {e}")
            return False
    
    async def publishToTopic(self, topic: str, formattedData: Dict[str, Any]) -> bool:
        """
        Conveniência: publica dados já formatados (será serializado automaticamente).
        
        Args:
            topic: Nome do tópico
            formattedData: Dados formatados para serializar
            
        Returns:
            True se enviado com sucesso
        """
        try:
            # Serializar dados para msgpack (será feito pelo ZeroMQFormatter)
            import msgpack
            serializedData = msgpack.packb(formattedData, use_bin_type=True)
            
            return await self.publishMessage(topic, serializedData)
            
        except Exception as e:
            self.logger.error(f"Error serializing data for topic '{topic}': {e}")
            self.stats["errors"] += 1
            return False
    
    async def _disconnect(self):
        """
        Fecha socket e termina contexto ZeroMQ de forma segura.
        """
        try:
            if self.socket:
                self.socket.close()
                self.socket = None
            
            if self.context:
                self.context.term()
                self.context = None
            
            self.activeTopics.clear()
            self.logger.debug("ZeroMQ PUB socket closed")
            
        except Exception as e:
            self.logger.error(f"Error closing ZeroMQ socket: {e}")
            self.stats["errors"] += 1
    
    def _updateStats(self, topic: str, messageSize: int):
        """
        Atualiza estatísticas de publicação.
        
        Args:
            topic: Tópico da mensagem
            messageSize: Tamanho da mensagem em bytes
        """
        now = datetime.now()
        
        # Estatísticas globais
        self.stats["messagesSent"] += 1
        self.stats["bytesSent"] += messageSize
        self.stats["lastMessageTimestamp"] = now.isoformat()
        
        # Estatísticas por tópico
        if topic in self.stats["topicStats"]:
            topicStats = self.stats["topicStats"][topic]
            topicStats["sent"] += 1
            topicStats["bytes"] += messageSize
            topicStats["lastSent"] = now.isoformat()
        
        # Adicionar à lista de tópicos ativos
        self.activeTopics.add(topic)
        
        # Calcular mensagens por segundo (rolling average simples)
        if self.startTime:
            uptime = self._getUptime()
            if uptime > 0:
                self.stats["messagesPerSecond"] = self.stats["messagesSent"] / uptime
    
    async def _emitError(self, errorType: str, message: str):
        """
        Emite evento de erro para monitorização.
        
        Args:
            errorType: Tipo específico do erro
            message: Descrição detalhada do erro
        """
        await eventManager.emit("mock.publisher_error", {
            "timestamp": datetime.now().isoformat(),
            "errorType": errorType,
            "message": message,
            "state": self.state.value,
            "stats": self.stats.copy()
        })
    
    def _getUptime(self) -> float:
        """
        Calcula tempo de funcionamento em segundos.
        
        Returns:
            Uptime em segundos desde o início do publisher
        """
        if not self.startTime:
            return 0.0
        return (datetime.now() - self.startTime).total_seconds()
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna estado completo do publisher para monitorização.
        
        Returns:
            Dicionário com estado completo do sistema
        """
        return {
            "state": self.state.value,
            "publisherUrl": self.publisherUrl,
            "socketType": self.socketType,
            "uptime": self._getUptime(),
            "availableTopics": list(self.availableTopics),
            "activeTopics": list(self.activeTopics),
            "stats": self.stats.copy(),
            "config": {
                "publisherAddress": self.publisherAddress,
                "publisherPort": self.publisherPort,
                "maxMessagesPerSecond": self.maxMessagesPerSecond,
                "maxMessageSize": self.maxMessageSize,
                "lingerTime": self.lingerTime,
                "sendHighWaterMark": self.sendHighWaterMark
            }
        }
    
    def getPublisherHealth(self) -> Dict[str, Any]:
        """
        Avalia saúde do publisher baseada em métricas.
        
        Returns:
            Dicionário com avaliação de saúde
        """
        health = "healthy"
        issues = []
        warnings = []
        
        # Verificar estado
        if self.state == PublisherState.ERROR:
            health = "critical"
            issues.append("Publisher in error state")
        elif self.state != PublisherState.RUNNING:
            health = "warning"
            warnings.append(f"Publisher not running (state: {self.state.value})")
        
        # Verificar tópicos ativos
        if len(self.activeTopics) == 0 and self._getUptime() > 5.0:
            health = "warning" if health == "healthy" else health
            warnings.append("No active topics after 5s")
        
        # Verificar taxa de erro
        if self.stats["messagesSent"] > 0:
            errorRate = self.stats["errors"] / self.stats["messagesSent"]
            if errorRate > 0.1:  # >10% erro
                health = "warning" if health == "healthy" else health
                warnings.append(f"High error rate: {errorRate:.1%}")
        
        return {
            "health": health,
            "issues": issues,
            "warnings": warnings,
            "lastCheck": datetime.now().isoformat(),
            "metrics": {
                "errorRate": self.stats["errors"] / max(1, self.stats["messagesSent"]),
                "messagesPerSecond": self.stats["messagesPerSecond"],
                "activeTopicsCount": len(self.activeTopics),
                "totalTopicsCount": len(self.availableTopics)
            }
        }
    
    def reset(self):
        """
        Reset das estatísticas de publicação.
        """
        self.stats = {
            "messagesSent": 0,
            "bytesSent": 0,
            "messagesPerSecond": 0.0,
            "lastMessageTimestamp": None,
            "errors": 0,
            "topicStats": {topic: {
                "sent": 0,
                "bytes": 0,
                "lastSent": None,
                "errors": 0
            } for topic in self.availableTopics},
            "startTime": datetime.now().isoformat() if self.state == PublisherState.RUNNING else None
        }
        
        self.activeTopics.clear()
        self.logger.info("ZeroMQPublisher statistics reset")

# Instância global
zeroMQPublisher = ZeroMQPublisher()