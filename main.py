"""
Main Application - FastAPI com WebSocket integrado (Refactored)

Resumo:
Aplicação principal que junta tudo: FastAPI server, WebSocket endpoints, streaming
contínuo de dados, e API REST. Configura CORS para frontend, inclui todos os routers,
e inicia automaticamente o MockZeroMQController (modo mock) ou ZeroMQListener (modo real).
Remove dependência do DataStreamer, usando diretamente o sistema apropriado.
"""

import asyncio
import logging
from contextlib import asynccontextmanager
from datetime import datetime

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.core import settings
from app.ws.webSocketRouter import router as websocket_router
from app.ws.webSocketManager import websocketManager
from app.services.signalManager import signalManager
from app.services.zeroMQListener import zeroMQListener
from app.ws.signalControlRouter import router as signal_control_router

# Configurar logging
logging.basicConfig(
    level=getattr(logging, settings.logLevel),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Gere ciclo de vida da aplicação"""
    # Startup
    logger.info("[MAIN] Starting Control Room Backend...")
    
    try:
        if settings.useRealSensors:
            # Usar dados reais
            logger.info("[MAIN] Starting ZeroMQ listener for real sensor data...")
            await zeroMQListener.start()
            logger.info("[MAIN] ZeroMQ listener started")
        else:
            # Usar mock data via ZeroMQ 
            logger.info("[MAIN] Starting MockZeroMQ system for simulated sensor data...")
            from tests.mockZeroMQ import mockZeroMQController
            await mockZeroMQController.start()
            logger.info("[MAIN] MockZeroMQ system started")
            logger.info("[MAIN] Starting ZeroMQ listener for fake sensor data...")
            await zeroMQListener.start()
            logger.info("[MAIN] ZeroMQ listener started")
        
        yield  # Aplicação roda aqui
        
    finally:
        # Shutdown
        logger.info("[MAIN] Shutting down Control Room Backend...")
        
        try:
            if settings.useRealSensors:
                await zeroMQListener.stop()
                logger.info("[MAIN] ZeroMQ listener stopped")
            else:
                from tests.mockZeroMQ import mockZeroMQController
                await mockZeroMQController.stop()
                logger.info("[MAIN] MockZeroMQ system stopped")
                await zeroMQListener.stop()
                logger.info("[MAIN] ZeroMQ listener stopped")
            
            # Limpar WebSocket connections
            await websocketManager.cleanup()
            logger.info("[MAIN] WebSocket connections cleaned up")
            
        except Exception as e:
            logger.error(f"[MAIN] Error during shutdown: {e}")

# Criar aplicação FastAPI
app = FastAPI(
    title="Control Room - Automotive Simulator",
    description="Backend para simulador de condução",
    version=settings.version,
    lifespan=lifespan
)

# Configurar CORS para frontend React
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.corsOrigins,  # ["http://localhost:3000"]
    allow_credentials=True,              # Permitir cookies / auth
    allow_methods=["*"],                 # POST , GET etc..
    allow_headers=["*"],                 # Content-Type, Authorization
)

# Incluir routers WebSocket
app.include_router(websocket_router, prefix="/api", tags=["websocket"])
app.include_router(signal_control_router)

# Endpoints REST básicos
@app.get("/")
async def root():
    """Endpoint raiz - informação básica"""
    return {
        "name": settings.projectName,
        "version": settings.version,
        "status": "running",
        "timestamp": datetime.now().isoformat(),
        "websocket_url": "/ws",
        "docs_url": "/docs",
        "dataSource": "real_sensors" if settings.useRealSensors else "mock_data"
    }

@app.get("/api/status")
async def get_system_status():
    """Status completo do sistema"""
    try:
        # Status de todos os componentes
        system_status = {
            "timestamp": datetime.now().isoformat(),
            "server": {
                "name": settings.projectName,
                "version": settings.version,
                "debug_mode": settings.debugMode
            },
            "signals": signalManager.getAllSignalsStatus(),
            "system_health": signalManager.getSystemHealth(),
            "websocket": {
                "endpoint": "/ws",
                "active_connections": len(websocketManager.activeConnections)
            }
        }
        
        # Adicionar status específico da fonte de dados
        if settings.useRealSensors:
            system_status["dataSource"] = {
                "type": "real_sensors",
                "zeroMQListener": zeroMQListener.getStatus()
            }
        else:
            from tests.mockZeroMQ import mockZeroMQController
            system_status["dataSource"] = {
                "type": "mock_data", 
                "mockController": mockZeroMQController.getAllStats()
            }
        
        return JSONResponse(content=system_status)
        
    except Exception as e:
        logger.error(f"Error getting system status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.get("/api/signals/{signal_type}/status")
async def get_signal_status(signal_type: str):
    """Status de um sinal específico"""
    try:
        status = signalManager.getSignalStatus(signal_type)
        
        if status is None:
            return JSONResponse(
                status_code=404,
                content={"error": f"Signal type '{signal_type}' not found"}
            )
        
        return JSONResponse(content={
            "signal_type": signal_type,
            "timestamp": datetime.now().isoformat(),
            "status": status
        })
        
    except Exception as e:
        logger.error(f"Error getting status for {signal_type}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

# Endpoints de controlo do sistema mock (apenas se mock ativo)
@app.post("/api/mock/start")
async def start_mock_system():
    """Inicia sistema mock (apenas se USE_REAL_SENSORS=False)"""
    if settings.useRealSensors:
        return JSONResponse(
            status_code=400,
            content={"error": "Cannot start mock system when USE_REAL_SENSORS=True"}
        )
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        if mockZeroMQController.isRunning():
            return {"status": "already_running", "timestamp": datetime.now().isoformat()}
        
        await mockZeroMQController.start()
        return {"status": "started", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error starting mock system: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/mock/stop")
async def stop_mock_system():
    """Para sistema mock"""
    if settings.useRealSensors:
        return JSONResponse(
            status_code=400,
            content={"error": "Mock system not active when USE_REAL_SENSORS=True"}
        )
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        logger.info(f"Client is trying to stop mock system")
        await mockZeroMQController.stop()
        return {"status": "stopped", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error stopping mock system: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/mock/pause")
async def pause_mock_system():
    """Pausa sistema mock"""
    if settings.useRealSensors:
        return JSONResponse(
            status_code=400,
            content={"error": "Mock system not active when USE_REAL_SENSORS=True"}
        )
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        await mockZeroMQController.pause()
        return {"status": "paused", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error pausing mock system: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/mock/resume")
async def resume_mock_system():
    """Retoma sistema mock"""
    if settings.useRealSensors:
        return JSONResponse(
            status_code=400,
            content={"error": "Mock system not active when USE_REAL_SENSORS=True"}
        )
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        await mockZeroMQController.resume()
        return {"status": "resumed", "timestamp": datetime.now().isoformat()}
    except Exception as e:
        logger.error(f"Error resuming mock system: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.get("/api/mock/status")
async def get_mock_status():
    """Status do sistema mock"""
    if settings.useRealSensors:
        return JSONResponse(content={
            "available": False,
            "reason": "USE_REAL_SENSORS=True"
        })
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        return JSONResponse(content=mockZeroMQController.getAllStats())
    except Exception as e:
        logger.error(f"Error getting mock status: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": "Internal server error"}
        )

@app.post("/api/mock/topics/{topic}/frequency")
async def adjust_topic_frequency(topic: str, frequency: float):
    """Ajusta frequência de um tópico específico"""
    if settings.useRealSensors:
        return JSONResponse(
            status_code=400,
            content={"error": "Mock system not active when USE_REAL_SENSORS=True"}
        )
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        mockZeroMQController.adjustTopicFrequency(topic, frequency)
        return {
            "topic": topic,
            "new_frequency": frequency,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error adjusting frequency for {topic}: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )

@app.post("/api/mock/topics/{topic}/anomaly")
async def inject_topic_anomaly(topic: str, anomaly_type: str, duration: float = 5.0):
    """Injeta anomalia em tópico específico"""
    if settings.useRealSensors:
        return JSONResponse(
            status_code=400,
            content={"error": "Mock system not active when USE_REAL_SENSORS=True"}
        )
    
    try:
        from tests.mockZeroMQ import mockZeroMQController
        mockZeroMQController.forceTopicAnomaly(topic, anomaly_type, duration)
        
        return {
            "injected": True,
            "topic": topic,
            "anomaly_type": anomaly_type,
            "duration": duration,
            "timestamp": datetime.now().isoformat()
        }
        
    except Exception as e:
        logger.error(f"Error injecting anomaly: {e}")
        return JSONResponse(
            status_code=500,
            content={"error": str(e)}
        )


if __name__ == "__main__":
    import uvicorn
    
    # Executar servidor
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=settings.port,
        reload=settings.debugMode,
        log_level=settings.logLevel.lower()
    )