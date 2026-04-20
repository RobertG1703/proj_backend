"""
WebSocketManager - Coordenador de conexões WebSocket

Resumo:
Gere todas as conexões WebSocket com browsers/clientes. Recebe dados processados dos sinais
através de eventos e envia-os imediatamente para todos os clientes conectados. Mantém lista
de conexões ativas, atribui IDs únicos a cada cliente, e remove automaticamente conexões
mortas. Também envia dados de status do sistema, anomalias detectadas, e heartbeat periódico.
Inclui controlo de sinais através do sistema Signal Control para filtering
por signal types individuais enviados ao frontend.
"""

import asyncio
import logging
from typing import Set, Dict, Any, Optional, List
from datetime import datetime
from fastapi import WebSocket

from ..services.signalManager import signalManager
from ..services.zeroMQListener import zeroMQListener
from ..core import eventManager, settings
from ..core.signalControl import SignalControlInterface, SignalState, ComponentState, signalControlManager

class WebSocketManager(SignalControlInterface):
    """Gere conexões WebSocket e broadcasting de dados com controlo de sinais"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Conexões ativas
        self.activeConnections: Set[WebSocket] = set()
        self.connectionData: Dict[WebSocket, Dict[str, Any]] = {}
        self.connectionCounter = 0
        
        # Configurações
        self.maxConnections = settings.websocket.maxConnections
        self.updateInterval = settings.websocket.updateInterval

        self.heartbeatTask: Optional[asyncio.Task] = None
        self.heartbeatInterval = 10.0  
        
        # Signal Control properties
        self.availableSignals = settings.signalControl.signalTypes.copy()
        defaultActiveStates = settings.signalControl.defaultActiveStates["websocket"]
        self.activeSignals: Set[str] = {signal for signal, active in defaultActiveStates.items() if active}
        
        # Estatísticas de WebSocket incluindo filtering
        self.stats = {
            "messagesSent": 0,
            "messagesFiltered": 0,
            "anomaliesSent": 0,
            "anomaliesFiltered": 0,
            "heartbeatsSent": 0,
            "connectionEvents": 0,
            "errors": 0,
            "bySignalType": {signal: {
                "sent": 0,
                "filtered": 0,
                "lastSent": None
            } for signal in self.availableSignals},
            "startTime": datetime.now().isoformat()
        }
        
        # Subscrever aos eventos do sistema
        self._setupEventSubscriptions()
        
        # Registar no manager central de Signal Control
        signalControlManager.registerComponent("websocket", self)
        
        self.logger.info(f"WebSocketManager initialized with Signal Control - Max connections: {self.maxConnections}")
    
    # Signal Control Interface Implementation
    
    def getAvailableSignals(self) -> List[str]:
        """Retorna lista de signal types disponíveis para broadcasting"""
        return self.availableSignals.copy()
    
    def getActiveSignals(self) -> List[str]:
        """Retorna lista de signal types atualmente ativos"""
        return list(self.activeSignals)
    
    async def enableSignal(self, signal: str) -> bool:
        """Ativa broadcasting de um signal type específico"""
        if signal not in self.availableSignals:
            self.logger.warning(f"Signal Control: Cannot enable unknown signal {signal}")
            return False
        
        self.activeSignals.add(signal)
        self.logger.info(f"Signal Control: Enabled signal type {signal}")
        return True
    
    async def disableSignal(self, signal: str) -> bool:
        """Desativa broadcasting de um signal type específico"""
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
        return ComponentState.RUNNING  # WebSocketManager é sempre considerado running se inicializado
    
    # Core WebSocket Methods
    
    def _setupEventSubscriptions(self):
        """Configura subscriptions aos eventos"""
        # Eventos principais
        eventManager.subscribe("signal.processed", self.onSignalProcessed)
        eventManager.subscribe("anomaly.detected", self.onAnomalyDetected)

        # Eventos ZeroMQ
        eventManager.subscribe("zmq.connected", self.onZmqConnected)
        eventManager.subscribe("zmq.error", self.onZmqError)
        eventManager.subscribe("zmq.warning", self.onZmqWarning)
        eventManager.subscribe("zmq.heartbeat", self.onZmqHeartbeat)
        
        self.logger.info("WebSocket event subscriptions configured")
    
    async def connect(self, websocket: WebSocket, clientInfo: Optional[Dict] = None) -> str:
        """Conecta novo cliente WebSocket"""
        try:
            # Verificar limite de conexões
            if len(self.activeConnections) >= self.maxConnections:
                await websocket.close(code=1008, reason="Maximum connections exceeded")
                raise Exception(f"Connection limit exceeded ({self.maxConnections})")
            
            # Aceitar conexão
            await websocket.accept()
            
            # Gerar ID único
            self.connectionCounter += 1
            clientId = f"client_{self.connectionCounter}"
            
            # Adicionar às conexões ativas
            self.activeConnections.add(websocket)
            self.connectionData[websocket] = {
                "clientId": clientId,
                "connectedAt": datetime.now(),
                "userAgent": clientInfo.get("userAgent", "Unknown") if clientInfo else "Unknown",
                "lastActivity": datetime.now()
            }
            
            self.stats["connectionEvents"] += 1
            
            self.logger.info(f"Client connected: {clientId} (Total: {len(self.activeConnections)})")
            
            # Enviar mensagem de boas-vindas
            await self._sendToClient(websocket, {
                "type": "connection.established",
                "clientId": clientId,
                "serverTime": datetime.now().isoformat(),
              
                "availableSignals": self.availableSignals,
                "activeSignals": self.getActiveSignals(),
                "updateInterval": self.updateInterval
            })
            
            # Emitir evento de conexão
            await eventManager.emit("client.connected", {
                "clientId": clientId,
                "timestamp": datetime.now().isoformat(),
                "totalConnections": len(self.activeConnections),
                "userAgent": self.connectionData[websocket]["userAgent"]
            })

            if len(self.activeConnections) == 1:
                await self._startHeartbeat()
            
            return clientId
            
        except Exception as e:
            self.logger.error(f"Error connecting client: {e}")
            self.stats["errors"] += 1
            raise
    
    async def disconnect(self, websocket: WebSocket, reason: str = "normal_close"):
        """Desconecta cliente WebSocket"""
        try:
            if websocket in self.activeConnections:
                # Obter dados do cliente
                clientData = self.connectionData.get(websocket, {})
                clientId = clientData.get("clientId", "unknown")
                
                # Remover da lista
                self.activeConnections.remove(websocket)
                del self.connectionData[websocket]
                
                self.stats["connectionEvents"] += 1
                
                self.logger.info(f"Client disconnected: {clientId} - {reason} (Remaining: {len(self.activeConnections)})")
                
                # Emitir evento de desconexão
                await eventManager.emit("client.disconnected", {
                    "clientId": clientId,
                    "timestamp": datetime.now().isoformat(),
                    "totalConnections": len(self.activeConnections),
                    "reason": reason,
                    "sessionDuration": (datetime.now() - clientData.get("connectedAt", datetime.now())).total_seconds()
                })

                if len(self.activeConnections) == 0:
                    await self._stopHeartbeat()
                
        except Exception as e:
            self.logger.error(f"Error disconnecting client: {e}")
            self.stats["errors"] += 1

    async def _startHeartbeat(self):
        """Inicia heartbeat autónomo"""
        if self.heartbeatTask and not self.heartbeatTask.done():
            return  # Já está a correr
        
        self.heartbeatTask = asyncio.create_task(self._heartbeatLoop())
        self.logger.info("System heartbeat started (10s interval)")
    
    async def _stopHeartbeat(self):
        """Para heartbeat"""
        if self.heartbeatTask and not self.heartbeatTask.done():
            self.heartbeatTask.cancel()
            try:
                await self.heartbeatTask
            except asyncio.CancelledError:
                pass
        
        self.logger.info("System heartbeat stopped")
    
    async def _heartbeatLoop(self):
        """Loop de heartbeat"""
        while len(self.activeConnections) > 0:
            try:
                await self.sendSystemHeartbeat()
                await asyncio.sleep(self.heartbeatInterval)
                
            except asyncio.CancelledError:
                break
                
            except Exception as e:
                self.logger.error(f"Error in heartbeat loop: {e}")
                self.stats["errors"] += 1
                await asyncio.sleep(self.heartbeatInterval)
            
    async def sendSystemHeartbeat(self):
        """Mandar overall system health"""
        try:
            # Status dos sinais 
            systemHealth = signalManager.getSystemHealth()
            allSignalsStatus = signalManager.getAllSignalsStatus()

            # Detectar fonte de dados dinamicamente
            dataSource = "real_sensors" if settings.useRealSensors else "mock_data"
            
            sourceStatus = zeroMQListener.getStatus()
            sourceUptime = zeroMQListener._getUptime()
            sourceCounters = sourceStatus["stats"]
            
            # Status geral do sistema
            message = {
                "type": "system.heartbeat",
                "timestamp": datetime.now().isoformat(),
                
                # Estado geral
                "systemHealth": systemHealth,
                "signalStatuses": allSignalsStatus,
                
                # Fonte de dados
                "dataSource": dataSource,
                "sourceStatus": sourceStatus["state"] if settings.useRealSensors else sourceStatus["state"],
                "sourceUptime": sourceUptime,
                "activeSignals": list(signalManager.signals.keys()),
                "counters": sourceCounters,
                
                # WebSocket info
                "activeConnections": len(self.activeConnections),
                "maxConnections": self.maxConnections,
                
                # Signal Control info
                "websocketSignalControl": {
                    "availableSignals": self.getAvailableSignals(),
                    "activeSignals": self.getActiveSignals(),
                    "filteringStats": self.stats["messagesFiltered"]
                },
                
                # Server info
                "debugMode": settings.debugMode
            }
            
            await self.broadcast(message)
            self.stats["heartbeatsSent"] += 1
            
        except Exception as e:
            self.logger.error(f"Error sending system heartbeat: {e}")
            self.stats["errors"] += 1
    
    async def onSignalProcessed(self, event):
        """Reage a dados de sinais processados com filtering por Signal Control"""
        data = event.data
        dataType = data.get("dataType")
        
        # Filtering via Signal Control por signal type
        if dataType not in self.activeSignals:
            self.stats["messagesFiltered"] += 1
            if dataType in self.stats["bySignalType"]:
                self.stats["bySignalType"][dataType]["filtered"] += 1
            self.logger.debug(f"Signal Control: Signal type {dataType} filtered from WebSocket")
            return
        
        # Criar mensagem para frontend
        message = {
            "type": "signal.update",
            "signalType": data["signalType"],
            "dataType": dataType,
            "timestamp": data["timestamp"],
            "value": data["value"]
        }
        
        # Enviar para todos os clientes
        await self.broadcast(message)
        
        # Atualizar estatísticas
        self.stats["messagesSent"] += 1
        if dataType in self.stats["bySignalType"]:
            self.stats["bySignalType"][dataType]["sent"] += 1
            self.stats["bySignalType"][dataType]["lastSent"] = datetime.now().isoformat()
        
        self.logger.debug(f"Broadcasted signal update: {data['signalType']}.{dataType}")
    
    async def onAnomalyDetected(self, event):
        """Reage a anomalias detectadas com filtering por Signal Control"""
        data = event.data
        signalType = data.get("signalType")
        
        # Determinar dataType baseado no signalType para filtering
        dataType = None
        if signalType == "cardiac":
            # Pode ser hr ou ecg - assumir baseado na anomalia
            anomalyType = data.get("anomalyType", "")
            if "bradycardia" in anomalyType or "tachycardia" in anomalyType:
                dataType = "hr"
            else:
                dataType = "ecg"
        elif signalType == "eeg":
            dataType = "eegRaw"
        elif signalType == "sensors":
            anomalyType = data.get("anomalyType", "")
            if "rotation" in anomalyType or "spin" in anomalyType:
                dataType = "gyroscope"
            else:
                dataType = "accelerometer"
        
        # Filtering via Signal Control
        if dataType and dataType not in self.activeSignals:
            self.stats["anomaliesFiltered"] += 1
            self.logger.debug(f"Signal Control: Anomaly for {dataType} filtered from WebSocket")
            return
        
        message = {
            "type": "anomaly.alert",
            "signalType": signalType,
            "anomalyType": data["anomalyType"],
            "severity": data["severity"],
            "message": data["message"],
            "timestamp": data["timestamp"],
            "value": data.get("value"),
            "threshold": data.get("threshold")
        }
        
        await self.broadcast(message)
        self.stats["anomaliesSent"] += 1
        
        self.logger.warning(f"Broadcasted anomaly alert: {signalType} - {data['message']}")

    async def onZmqConnected(self, event):
        """ZeroMQ conectou"""
        await self.broadcast({
            "type": "zmq.connected",
            "timestamp": event.data["timestamp"],
            "socketUrl": event.data["socketUrl"]
        })

    async def onZmqError(self, event):
        """Erro ZeroMQ"""
        await self.broadcast({
            "type": "zmq.error",
            "timestamp": event.data["timestamp"],
            "errorType": event.data["errorType"],
            "message": event.data["message"]
        })

    async def onZmqWarning(self, event):
        """Aviso ZeroMQ"""
        await self.broadcast({
            "type": "zmq.warning",
            "timestamp": event.data["timestamp"],
            "warningType": event.data["warningType"],
            "message": event.data["message"]
        })

    async def onZmqHeartbeat(self, event):
        """Heartbeat ZeroMQ"""
        await self.broadcast({
            "type": "zmq.heartbeat",
            "timestamp": event.data["timestamp"],
            "state": event.data["state"],
            "stats": event.data["stats"]
        })
        
    
    async def broadcast(self, message: Dict[str, Any]):
        """Envia mensagem para todos os clientes conectados"""
        if not self.activeConnections:
            return
        
        # Lista de conexões a remover (se falharem)
        deadConnections = []
        
        # Enviar para cada cliente em paralelo
        tasks = []
        for websocket in self.activeConnections:
            tasks.append(self._sendToClient(websocket, message, deadConnections))
        
        # Executar todos os envios em paralelo
        await asyncio.gather(*tasks, return_exceptions=True)
        
        # Remover conexões mortas
        for websocket in deadConnections:
            await self.disconnect(websocket, "connection_failed")
    
    async def _sendToClient(self, websocket: WebSocket, message: Dict[str, Any], 
                           deadConnections: list = None):
        """Envia mensagem para um cliente específico"""
        try:
            await websocket.send_json(message)
            
            # Atualizar última atividade
            if websocket in self.connectionData:
                self.connectionData[websocket]["lastActivity"] = datetime.now()
                
        except Exception as e:
            self.logger.warning(f"Failed to send message to client: {e}")
            self.stats["errors"] += 1
            if deadConnections is not None:
                deadConnections.append(websocket)
    
    async def sendSignalStatus(self, signalType: str):
        """Envia status detalhado de um sinal específico"""
        try:
            status = signalManager.getSignalStatus(signalType)
            
            if status:
                message = {
                    "type": "signal.status",
                    "signalType": signalType,
                    "timestamp": datetime.now().isoformat(),
                    "status": status
                }
                
                await self.broadcast(message)
            
        except Exception as e:
            self.logger.error(f"Error sending signal status for {signalType}: {e}")
            self.stats["errors"] += 1
    
    def getConnectionStats(self) -> Dict[str, Any]:
        """Retorna estatísticas das conexões incluindo Signal Control"""
        return {
            "activeConnections": len(self.activeConnections),
            "maxConnections": self.maxConnections,
            "totalConnectionsEver": self.connectionCounter,
            "signalControl": {
                "availableSignals": self.getAvailableSignals(),
                "activeSignals": self.getActiveSignals(),
                "componentState": self.getComponentState().value,
                "filteredSignals": [signal for signal in self.availableSignals if signal not in self.activeSignals]
            },
            "filtering": {
                "messagesFiltered": self.stats["messagesFiltered"],
                "anomaliesFiltered": self.stats["anomaliesFiltered"],
                "filteringRate": self.stats["messagesFiltered"] / max(1, self.stats["messagesSent"] + self.stats["messagesFiltered"])
            },
            "connectionData": [
                {
                    "clientId": data["clientId"],
                    "connectedAt": data["connectedAt"].isoformat(),
                    "userAgent": data["userAgent"],
                    "lastActivity": data["lastActivity"].isoformat()
                }
                for data in self.connectionData.values()
            ]
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
                "messagesFiltered": self.stats["messagesFiltered"],
                "anomaliesFiltered": self.stats["anomaliesFiltered"],
                "filteringRate": self.stats["messagesFiltered"] / max(1, 
                    self.stats["messagesSent"] + self.stats["messagesFiltered"]
                )
            },
            "connections": {
                "activeConnections": len(self.activeConnections),
                "maxConnections": self.maxConnections,
                "connectionEvents": self.stats["connectionEvents"]
            },
            "lastUpdate": datetime.now().isoformat()
        }
    
    def getWebSocketStats(self) -> Dict[str, Any]:
        """Estatísticas completas do WebSocketManager"""
        uptime = (datetime.now() - datetime.fromisoformat(self.stats["startTime"])).total_seconds()
        
        return {
            **self.stats,
            "uptime": uptime,
            "averageMessageRate": self.stats["messagesSent"] / max(1, uptime),
            "successRate": 1 - (self.stats["errors"] / max(1, self.stats["messagesSent"])),
            "filterRate": self.stats["messagesFiltered"] / max(1, self.stats["messagesSent"] + self.stats["messagesFiltered"]),
            "signalControl": {
                "availableSignals": self.getAvailableSignals(),
                "activeSignals": self.getActiveSignals(),
                "componentState": self.getComponentState().value,
                "bySignalType": self.stats["bySignalType"]
            },
            "connections": {
                "current": len(self.activeConnections),
                "maximum": self.maxConnections,
                "total": self.connectionCounter
            }
        }
    
    async def cleanup(self):
        """Limpa recursos e fecha todas as conexões"""
        self.logger.info("Cleaning up WebSocket connections...")
        
        # Parar heartbeat
        await self._stopHeartbeat()
        
        # Fechar todas as conexões
        for websocket in list(self.activeConnections):
            try:
                await websocket.close(code=1001, reason="Server shutdown")  
            except Exception:
                pass
        
        self.activeConnections.clear()
        self.connectionData.clear()
        
        self.logger.info("WebSocket cleanup completed")

# Instância global
websocketManager = WebSocketManager()