"""
WebSocketRouter - Endpoints FastAPI para WebSocket

Resumo:
Define os endpoints FastAPI que os browsers usam para se conectar via WebSocket.
Inclui o endpoint principal /ws onde clientes se conectam, fica à escuta de mensagens,
e gere desconexões automáticas. Também inclui endpoints REST para obter estatísticas
das conexões e status do sistema.
# TODO Não esquecer que é api/ws (passei 2 horas a tripar com isto porque esqueci-me), 
# TODO dado ao prefixo mas provavelemnte deivamos tirar o prefixo era da versão antiga devido à estrutura de pastas
"""

from datetime import datetime
import json
import logging
from typing import Dict, Any
from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.responses import JSONResponse

from .webSocketManager import websocketManager
#from ..services.signalManager import signalManager 

# Criar router
router = APIRouter()
logger = logging.getLogger(__name__)

@router.websocket("/ws")
async def websocket_endpoint(websocket: WebSocket):
    """
    Endpoint principal WebSocket onde clientes se conectam
    
    URL: ws://localhost:8000/ws
    """
    clientId = None
    
    try:
        # Conectar cliente
        clientId = await websocketManager.connect(websocket)
        logger.info(f"WebSocket connection established for {clientId}")
        
        # Loop principal - fica à espera de mensagens do cliente
        while True:
            try:
                # Aguardar mensagem do cliente
                data = await websocket.receive_text()
                
                # Processar mensagem do cliente
                await _handleClientMessage(websocket, clientId, data)
                
            except WebSocketDisconnect:
                logger.info(f"Client {clientId} disconnected normally")
                break
                
            except Exception as e:
                logger.error(f"Error in WebSocket loop for {clientId}: {e}")
                # Continuar o loop - não quebrar por erro pontual
                
    except WebSocketDisconnect:
        logger.info(f"Client disconnected during connection setup")
        
    except Exception as e:
        logger.error(f"Error in WebSocket endpoint: {e}")
        
    finally:
        # Limpar conexão
        if clientId:
            await websocketManager.disconnect(websocket, "connection_ended")

async def _handleClientMessage(websocket: WebSocket, clientId: str, message: str):
    """Processa mensagens vindas do cliente"""
    try:
        # Parse da mensagem JSON
        data = json.loads(message)
        messageType = data.get("type", "unknown")
        
        logger.debug(f"Received message from {clientId}: {messageType}")
        
        # Processar diferentes tipos de mensagens
        if messageType == "ping":
            # Cliente quer verificar se conexão está viva
            await websocket.send_json({
                "type": "pong",
                "timestamp": data.get("timestamp"),
                "serverTime": datetime.now().isoformat()
            })
            
        elif messageType == "request.signal_status":
            # Cliente quer status de um sinal específico
            signalType = data.get("signalType")
            if signalType:
                await websocketManager.sendSignalStatus(signalType)
            
        elif messageType == "request.system_status":
            # Cliente quer status geral do sistema
            await websocketManager.sendSystemHeartbeat()
            
        elif messageType == "subscribe":
            # Cliente quer subscrever a tipos específicos de dados
            # Por implementar - permitir clientes escolherem que dados receber
            await websocket.send_json({
                "type": "subscription.confirmed",
                "subscriptions": data.get("signals", [])
            })
        elif messageType == "request.available_signals":
            signals = websocketManager.getActiveSignals()
            await websocket.send_json({
                "type" : "response.available_signals",
                "availableSignals": signals
            })
            
        else:
            # Tipo de mensagem desconhecido
            await websocket.send_json({
                "type": "error",
                "message": f"Unknown message type: {messageType}"
            })
            
    except json.JSONDecodeError:
        # Mensagem não é JSON válido
        await websocket.send_json({
            "type": "error", 
            "message": "Invalid JSON format"
        })
        
    except Exception as e:
        logger.error(f"Error handling client message from {clientId}: {e}")
        await websocket.send_json({
            "type": "error",
            "message": "Internal server error"
        })

@router.get("/ws/connections")
async def get_connection_stats():
    """
    Endpoint REST para obter estatísticas das conexões WebSocket
    
    GET /ws/connections
    """
    try:
        stats = websocketManager.getConnectionStats()
        return JSONResponse(content=stats)
        
    except Exception as e:
        logger.error(f"Error getting connection stats: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.get("/ws/status")
async def get_websocket_status():
    """
    Endpoint REST para status geral do WebSocket
    
    GET /ws/status
    """
    try:
        
        status = {
            "websocket": {
                "activeConnections": len(websocketManager.activeConnections),
                "maxConnections": websocketManager.maxConnections,
                "totalConnectionsEver": websocketManager.connectionCounter
            },
            #"system": signalManager.getSystemHealth(),
            #"signals": signalManager.getAllSignalsStatus()
        }
        
        return JSONResponse(content=status)
        
    except Exception as e:
        logger.error(f"Error getting WebSocket status: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")

@router.post("/ws/broadcast")
async def manual_broadcast(message: Dict[str, Any]):
    """
    Endpoint REST para envio manual de mensagens (útil para testes)
    
    POST /ws/broadcast
    """
    try:
        await websocketManager.broadcast(message)
        
        return JSONResponse(content={
            "success": True,
            "message": "Message broadcasted",
            "recipientCount": len(websocketManager.activeConnections)
        })
        
    except Exception as e:
        logger.error(f"Error broadcasting message: {e}")
        raise HTTPException(status_code=500, detail="Internal server error")