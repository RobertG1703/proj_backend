"""
MockZeroMQController - Coordenador do sistema mock ZeroMQ completo

Resumo:
Orquestra todo o sistema mock simulando perfeitamente o fluxo real:
Publishers + Formatters + Geradores funcionando em conjunto para criar
um stream contínuo de dados que o ZeroMQListener pode consumir como se
fossem sensores reais. Coordena frequências, timing, anomalias e 
permite controlo granular de cada tópico através do sistema Signal Control.
"""

import asyncio
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, Set
from enum import Enum

from app.core import settings, eventManager
from app.core.signalControl import SignalControlInterface, SignalState, ComponentState, signalControlManager
from .zeroMQPublisher import zeroMQPublisher, ZeroMQPublisher
from .zeroMQFormatter import zeroMQFormatter, ZeroMQFormatter
from .generators import (
    cardioWheelEcgGenerator, CardioWheelEcgGenerator,
    cardioWheelAccGenerator, CardioWheelAccGenerator,
    cardioWheelGyrGenerator, CardioWheelGyrGenerator,
    polarPpiGenerator, PolarPpiGenerator,
    brainAccessEegGenerator, BrainAccessEegGenerator,
    cameraFaceLandmarksGenerator, CameraFaceLandmarksGenerator,
    unityAlcoholGenerator, UnityAlcoholGenerator,
    unityCarInfoGenerator, UnityCarInfoGenerator
)

class ControllerState(Enum):
    """Estados possíveis do controller"""
    STOPPED = "stopped"
    STARTING = "starting"
    RUNNING = "running"
    PAUSED = "paused"
    STOPPING = "stopping"
    ERROR = "error"

class MockZeroMQController(SignalControlInterface):
    """Controller principal do sistema mock ZeroMQ com controlo de sinais"""
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações centralizadas
        self.mockConfig = settings.mockZeromq
        
        # Estado do controller
        self.state = ControllerState.STOPPED
        self.startTime: Optional[datetime] = None
        self.pausedTime: Optional[datetime] = None
        self.totalPausedDuration = 0.0
        
        # Componentes do sistema
        self.publisher = zeroMQPublisher
        self.formatter = zeroMQFormatter
        
        # Mapeamento de tópicos para geradores
        self.topicGenerators = {
            "Polar_PPI": polarPpiGenerator,
            "CardioWheel_ECG": cardioWheelEcgGenerator,
            "CardioWheel_ACC": cardioWheelAccGenerator,
            "CardioWheel_GYR": cardioWheelGyrGenerator,
            "BrainAcess_EEG": brainAccessEegGenerator,
            "Camera_FaceLandmarks": cameraFaceLandmarksGenerator,
            "Unity_Alcohol": unityAlcoholGenerator,
            "Unity_CarInfo": unityCarInfoGenerator
        }
        
        # Configurações de frequência por tópico
        self.topicFrequencies = self.mockConfig.topicFrequencies.copy()
        self.topicTasks: Dict[str, asyncio.Task] = {}
        
        # Signal Control properties
        self.availableSignals = list(self.topicGenerators.keys())
        defaultActiveStates = settings.signalControl.defaultActiveStates["publisher"]
        self.activeSignals: Set[str] = {signal for signal, active in defaultActiveStates.items() if active}
        
        # Estatísticas globais
        self.stats = {
            "startTime": None,
            "totalRuntime": 0.0,
            "totalPaused": 0.0,
            "messagesGenerated": 0,
            "messagesSent": 0,
            "messagesRejected": 0,
            "anomaliesInjected": 0,
            "byTopic": {topic: {
                "generated": 0,
                "sent": 0,
                "rejected": 0,
                "anomalies": 0,
                "lastGenerated": None,
                "currentFrequency": freq
            } for topic, freq in self.topicFrequencies.items()},
            "errors": 0
        }
        
        # Configurações de anomalias globais
        self.anomalyInjection = self.mockConfig.anomalyInjection
        self.lastGlobalAnomalyTime = 0.0
        
        # Rate limiting global
        self.maxGlobalRate = self.mockConfig.performanceConfig["maxMessagesPerSecond"]
        self.globalMessageCounter = 0
        self.lastRateResetTime = 0.0
        
        # Registar no manager central de Signal Control
        signalControlManager.registerComponent("publisher", self)
        
        self.logger.info(f"MockZeroMQController initialized for {len(self.topicGenerators)} topics with Signal Control")
    
    # Signal Control Interface Implementation
    
    def getAvailableSignals(self) -> List[str]:
        """Retorna lista de tópicos disponíveis para geração"""
        return self.availableSignals.copy()
    
    def getActiveSignals(self) -> List[str]:
        """Retorna lista de tópicos atualmente ativos"""
        return list(self.activeSignals)
    
    async def enableSignal(self, signal: str) -> bool:
        """Ativa geração de um tópico específico"""
        if signal not in self.availableSignals:
            self.logger.warning(f"Signal Control: Cannot enable unknown signal {signal}")
            return False
        
        if signal in self.activeSignals:
            return True  # Já ativo
        
        self.activeSignals.add(signal)
        
        # Se sistema estiver a rodar, iniciar task do tópico
        if self.state == ControllerState.RUNNING:
            self.topicTasks[signal] = asyncio.create_task(
                self._topicGenerationLoop(signal)
            )
        
        self.logger.info(f"Signal Control: Enabled topic {signal}")
        return True
    
    async def disableSignal(self, signal: str) -> bool:
        """Desativa geração de um tópico específico"""
        if signal not in self.activeSignals:
            return True  # Já inativo
        
        self.activeSignals.remove(signal)
        
        # Parar task se existir
        if signal in self.topicTasks:
            task = self.topicTasks[signal]
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
            del self.topicTasks[signal]
        
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
        if self.state == ControllerState.RUNNING:
            return ComponentState.RUNNING
        elif self.state == ControllerState.STOPPED:
            return ComponentState.STOPPED
        elif self.state == ControllerState.ERROR:
            return ComponentState.ERROR
        else:
            return ComponentState.INITIALIZING
    
    # Controller Core Methods
    
    async def start(self, topics: Optional[List[str]] = None):
        """
        Inicia o sistema mock ZeroMQ completo.
        
        Args:
            topics: Lista de tópicos para ativar (todos se None)
        """
        
        if self.state in [ControllerState.STARTING, ControllerState.RUNNING]:
            self.logger.warning("MockZeroMQController already running")
            return
        
        self.logger.info("Starting MockZeroMQ system...")
        self.state = ControllerState.STARTING
        self.startTime = datetime.now()
        
        try:
            # Determinar tópicos ativos
            if topics is None:
                requestedTopics = set(self.topicGenerators.keys())
            else:
                requestedTopics = set(topics) & set(self.topicGenerators.keys())
                if not requestedTopics:
                    raise ValueError(f"No valid topics in {topics}. Available: {list(self.topicGenerators.keys())}")
            
            # Sincronizar com Signal Control (só ativar os que estão disponíveis)
            self.activeSignals = requestedTopics.copy()
            
            # Iniciar publisher ZeroMQ
            await self.publisher.start()
            
            # Reset geradores
            for topic in self.activeSignals:
                generator = self.topicGenerators[topic]
                generator.reset()
            
            # Iniciar tasks de geração por tópico
            await self._startTopicTasks()
            
            # Atualizar estado
            self.state = ControllerState.RUNNING
            self.stats["startTime"] = datetime.now().isoformat()
            
            # Emitir evento de início
            await eventManager.emit("mock.controller_started", {
                "timestamp": datetime.now().isoformat(),
                "activeTopics": list(self.activeSignals),
                "frequencies": {topic: self.topicFrequencies[topic] for topic in self.activeSignals}
            })
            
            self.logger.info(f"MockZeroMQ system started - Active topics: {list(self.activeSignals)}")
            
        except Exception as e:
            self.state = ControllerState.ERROR
            self.stats["errors"] += 1
            await self._emitError("startup_failed", str(e))
            raise
    
    async def stop(self):
        """Para o sistema mock ZeroMQ completo."""
        
        if self.state == ControllerState.STOPPED:
            return
        
        self.logger.info("Stopping MockZeroMQ system...")
        self.state = ControllerState.STOPPING
        
        try:
            # Parar tasks de tópicos
            await self._stopTopicTasks()
            
            # Parar publisher
            await self.publisher.stop()
            
            # Calcular estatísticas finais
            if self.startTime:
                self.stats["totalRuntime"] = (datetime.now() - self.startTime).total_seconds()
                self.stats["totalPaused"] = self.totalPausedDuration
            
            # Emitir evento de paragem
            await eventManager.emit("mock.controller_stopped", {
                "timestamp": datetime.now().isoformat(),
                "uptime": self.stats["totalRuntime"],
                "finalStats": self.stats.copy()
            })
            
            self.state = ControllerState.STOPPED
            self.logger.info("MockZeroMQ system stopped successfully")
            
        except Exception as e:
            self.logger.error(f"Error stopping MockZeroMQ system: {e}")
            self.stats["errors"] += 1
            self.state = ControllerState.ERROR
    
    async def pause(self):
        """Pausa geração de dados (publisher continua ativo)."""
        
        if self.state != ControllerState.RUNNING:
            return
        
        self.state = ControllerState.PAUSED
        self.pausedTime = datetime.now()
        
        # Pausar tasks de tópicos (cancelar)
        for task in self.topicTasks.values():
            if not task.done():
                task.cancel()
        
        self.logger.info("MockZeroMQ system paused")
        
        await eventManager.emit("mock.controller_paused", {
            "timestamp": datetime.now().isoformat()
        })
    
    async def resume(self):
        """Retoma geração de dados."""
        
        if self.state != ControllerState.PAUSED:
            return
        
        # Calcular tempo pausado
        if self.pausedTime:
            pauseDuration = (datetime.now() - self.pausedTime).total_seconds()
            self.totalPausedDuration += pauseDuration
            self.pausedTime = None
        
        # Reiniciar tasks de tópicos
        await self._startTopicTasks()
        
        self.state = ControllerState.RUNNING
        self.logger.info("MockZeroMQ system resumed")
        
        await eventManager.emit("mock.controller_resumed", {
            "timestamp": datetime.now().isoformat(),
            "pauseDuration": pauseDuration
        })
    
    async def _startTopicTasks(self):
        """Inicia tasks assíncronas para cada tópico ativo."""
        
        self.topicTasks.clear()
        
        for topic in self.activeSignals:
            if topic in self.topicGenerators:
                # Criar task específica para o tópico
                self.topicTasks[topic] = asyncio.create_task(
                    self._topicGenerationLoop(topic)
                )
                self.logger.debug(f"Started generation task for topic: {topic}")
    
    async def _stopTopicTasks(self):
        """Para todas as tasks de geração de tópicos."""
        
        for topic, task in self.topicTasks.items():
            if not task.done():
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass
                self.logger.debug(f"Stopped generation task for topic: {topic}")
        
        self.topicTasks.clear()
    
    async def _topicGenerationLoop(self, topic: str):
        """
        Loop de geração contínua para um tópico específico.
        
        Args:
            topic: Nome do tópico ZeroMQ
        """
        
        generator = self.topicGenerators[topic]
        frequency = self.topicFrequencies[topic]
        interval = 1.0 / frequency
        
        self.logger.debug(f"Starting generation loop for {topic} at {frequency}Hz ({interval:.3f}s interval)")
        
        try:
            while self.state == ControllerState.RUNNING:
                # Verificar se tópico ainda está ativo via Signal Control
                if topic not in self.activeSignals:
                    self.logger.debug(f"Topic {topic} disabled via Signal Control, stopping generation")
                    break
                
                loopStartTime = asyncio.get_event_loop().time()
                
                # Verificar rate limiting global
                if not await self._checkGlobalRateLimit():
                    await asyncio.sleep(0.01)
                    continue
                
                # Gerar dados do tópico
                success = await self._generateAndSendTopicData(topic, generator)
                
                if success:
                    self.stats["byTopic"][topic]["generated"] += 1
                    self.stats["messagesGenerated"] += 1
                
                # Aguardar próximo ciclo
                elapsed = asyncio.get_event_loop().time() - loopStartTime
                sleepTime = max(0, interval - elapsed)
                
                if sleepTime > 0:
                    await asyncio.sleep(sleepTime)
                else:
                    # Log warning se não conseguir manter frequência
                    self.logger.warning(f"Cannot maintain {frequency}Hz for {topic} (took {elapsed:.3f}s)")
                    
        except asyncio.CancelledError:
            self.logger.debug(f"Generation loop cancelled for topic: {topic}")
            
        except Exception as e:
            self.logger.error(f"Error in generation loop for {topic}: {e}")
            self.stats["errors"] += 1
            await self._emitError("generation_loop_failed", f"{topic}: {e}")
    
    async def _generateAndSendTopicData(self, topic: str, generator) -> bool:
        """
        Gera dados de um tópico e envia via publisher.
        
        Args:
            topic: Nome do tópico
            generator: Gerador específico do tópico
            
        Returns:
            True se enviado com sucesso
        """
        
        try:
            # Gerar dados baseado no tipo de gerador
            if topic in ["Polar_PPI", "Unity_Alcohol", "Unity_CarInfo"]:
                rawData = generator.generateEvent()
            elif topic == "Camera_FaceLandmarks": 
                rawData = generator.generateFrame()
            else:
                # ECG, ACC, GYR, EEG usam chunks
                rawData = generator.generateChunk()
            
            # Formatar dados para ZeroMQ
            formattedData = self.formatter.formatTopicData(topic, rawData)
            
            # Enviar via publisher
            success = await self.publisher.publishMessage(topic, formattedData)
            
            if success:
                self.stats["byTopic"][topic]["sent"] += 1
                self.stats["messagesSent"] += 1
                self.stats["byTopic"][topic]["lastGenerated"] = datetime.now().isoformat()
                
                # Verificar se foi injetada anomalia
                if rawData.get("anomalyType", "normal") != "normal":
                    self.stats["byTopic"][topic]["anomalies"] += 1
                    self.stats["anomaliesInjected"] += 1
                
            else:
                self.stats["byTopic"][topic]["rejected"] += 1
                self.stats["messagesRejected"] += 1
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error generating/sending data for {topic}: {e}")
            self.stats["byTopic"][topic]["rejected"] += 1
            self.stats["messagesRejected"] += 1
            self.stats["errors"] += 1
            return False
    
    async def _checkGlobalRateLimit(self) -> bool:
        """
        Verifica rate limiting global.
        
        Returns:
            True se pode enviar mensagem
        """
        
        currentTime = asyncio.get_event_loop().time()
        
        # Reset contador a cada segundo
        if currentTime - self.lastRateResetTime >= 1.0:
            self.globalMessageCounter = 0
            self.lastRateResetTime = currentTime
        
        # Verificar limite
        if self.globalMessageCounter >= self.maxGlobalRate:
            return False
        
        self.globalMessageCounter += 1
        return True
    
    async def _emitError(self, errorType: str, message: str):
        """
        Emite evento de erro.
        
        Args:
            errorType: Tipo do erro
            message: Mensagem de erro
        """
        
        await eventManager.emit("mock.controller_error", {
            "timestamp": datetime.now().isoformat(),
            "errorType": errorType,
            "message": message,
            "state": self.state.value,
            "stats": self.stats.copy()
        })
    
    # Configuration and Control Methods
    
    def adjustTopicFrequency(self, topic: str, newFrequency: float):
        """
        Ajusta frequência de um tópico em tempo real.
        
        Args:
            topic: Nome do tópico
            newFrequency: Nova frequência em Hz
        """
        
        if topic not in self.topicFrequencies:
            raise ValueError(f"Unknown topic: {topic}")
        
        if newFrequency <= 0 or newFrequency > 1000:
            raise ValueError(f"Invalid frequency: {newFrequency} (must be 0-1000 Hz)")
        
        oldFreq = self.topicFrequencies[topic]
        self.topicFrequencies[topic] = newFrequency
        self.stats["byTopic"][topic]["currentFrequency"] = newFrequency
        
        # Reiniciar task do tópico se estiver ativo
        if self.state == ControllerState.RUNNING and topic in self.topicTasks:
            task = self.topicTasks[topic]
            if not task.done():
                task.cancel()
            
            # Criar nova task com nova frequência
            self.topicTasks[topic] = asyncio.create_task(
                self._topicGenerationLoop(topic)
            )
        
        self.logger.info(f"Adjusted {topic} frequency: {oldFreq}Hz → {newFrequency}Hz")
    
    def forceTopicAnomaly(self, topic: str, anomalyType: str, duration: float = 5.0):
        """
        Força anomalia em tópico específico.
        
        Args:
            topic: Nome do tópico
            anomalyType: Tipo de anomalia
            duration: Duração em segundos
        """
        
        if topic not in self.topicGenerators:
            raise ValueError(f"Unknown topic: {topic}")
        
        generator = self.topicGenerators[topic]
        
        if hasattr(generator, 'forceAnomaly'):
            generator.forceAnomaly(anomalyType, duration)
            self.logger.warning(f"Forced anomaly on {topic}: {anomalyType} for {duration}s")
        else:
            raise ValueError(f"Topic {topic} doesn't support forced anomalies")
    
    # Status and Information Methods
    
    def getStatus(self) -> Dict[str, Any]:
        """
        Retorna status completo do controller.
        
        Returns:
            Status detalhado do sistema
        """
        
        # Calcular uptime
        uptime = 0.0
        if self.startTime:
            uptime = (datetime.now() - self.startTime).total_seconds()
        
        # Status dos geradores
        generatorStatus = {}
        for topic, generator in self.topicGenerators.items():
            try:
                generatorStatus[topic] = generator.getStatus()
            except Exception as e:
                generatorStatus[topic] = {"error": str(e)}
        
        return {
            "state": self.state.value,
            "uptime": uptime,
            "totalPausedTime": self.totalPausedDuration,
            "activeTopics": list(self.activeSignals),
            "availableTopics": list(self.topicGenerators.keys()),
            "stats": self.stats.copy(),
            "frequencies": self.topicFrequencies.copy(),
            "publisher": self.publisher.getStatus(),
            "formatter": self.formatter.getStats(),
            "generators": generatorStatus,
            "signalControl": {
                "availableSignals": self.getAvailableSignals(),
                "activeSignals": self.getActiveSignals(),
                "componentState": self.getComponentState().value
            },
            "config": {
                "maxGlobalRate": self.maxGlobalRate,
                "anomalyInjection": self.anomalyInjection,
                "totalGenerators": len(self.topicGenerators)
            }
        }
    
    def getSystemHealth(self) -> Dict[str, Any]:
        """
        Avalia saúde do sistema mock.
        
        Returns:
            Avaliação de saúde detalhada
        """
        
        health = "healthy"
        issues = []
        warnings = []
        
        # Verificar estado do controller
        if self.state == ControllerState.ERROR:
            health = "critical"
            issues.append("Controller in error state")
        elif self.state not in [ControllerState.RUNNING, ControllerState.PAUSED]:
            health = "warning"
            warnings.append(f"Controller not active (state: {self.state.value})")
        
        # Verificar publisher
        publisherHealth = self.publisher.getPublisherHealth()
        if publisherHealth["health"] != "healthy":
            if publisherHealth["health"] == "critical":
                health = "critical"
                issues.extend(publisherHealth["issues"])
            else:
                health = "warning" if health == "healthy" else health
                warnings.extend(publisherHealth["warnings"])
        
        # Verificar tópicos ativos
        if len(self.activeSignals) == 0 and self.state == ControllerState.RUNNING:
            health = "warning" if health == "healthy" else health
            warnings.append("No active topics")
        
        # Verificar taxa de rejeição
        if self.stats["messagesGenerated"] > 0:
            rejectionRate = self.stats["messagesRejected"] / self.stats["messagesGenerated"]
            if rejectionRate > 0.1:  # >10% rejeição
                health = "warning" if health == "healthy" else health
                warnings.append(f"High rejection rate: {rejectionRate:.1%}")
        
        # Verificar erros
        if self.stats["errors"] > 5:
            health = "warning" if health == "healthy" else health
            warnings.append(f"Multiple errors detected: {self.stats['errors']}")
        
        uptime = self.getUptime()
        
        return {
            "health": health,
            "issues": issues,
            "warnings": warnings,
            "lastCheck": datetime.now().isoformat(),
            "metrics": {
                "rejectionRate": self.stats["messagesRejected"] / max(1, self.stats["messagesGenerated"]),
                "successRate": self.stats["messagesSent"] / max(1, self.stats["messagesGenerated"]),
                "activeTopicsCount": len(self.activeSignals),
                "totalTopicsCount": len(self.topicGenerators),
                "anomaliesPerMinute": self.stats["anomaliesInjected"] / max(1, uptime / 60) if uptime > 0 else 0
            },
            "components": {
                "controller": self.state.value,
                "publisher": publisherHealth["health"],
                "formatter": "healthy"  # Formatter é stateless
            }
        }
    
    def reset(self):
        """Reset completo do sistema."""
        
        # Reset estatísticas
        self.stats = {
            "startTime": None,
            "totalRuntime": 0.0,
            "totalPaused": 0.0,
            "messagesGenerated": 0,
            "messagesSent": 0,
            "messagesRejected": 0,
            "anomaliesInjected": 0,
            "byTopic": {topic: {
                "generated": 0,
                "sent": 0,
                "rejected": 0,
                "anomalies": 0,
                "lastGenerated": None,
                "currentFrequency": freq
            } for topic, freq in self.topicFrequencies.items()},
            "errors": 0
        }
        
        # Reset componentes
        self.publisher.reset()
        self.formatter.reset()
        
        for generator in self.topicGenerators.values():
            generator.reset()
        
        # Reset estado
        self.startTime = None
        self.pausedTime = None
        self.totalPausedDuration = 0.0
        self.lastGlobalAnomalyTime = 0.0
        self.globalMessageCounter = 0
        self.lastRateResetTime = 0.0
        
        self.logger.info("MockZeroMQController reset completed")
    
    # Convenience methods para interface mais simples
    
    def isRunning(self) -> bool:
        """Verifica se sistema está rodando."""
        return self.state == ControllerState.RUNNING
    
    def isPaused(self) -> bool:
        """Verifica se sistema está pausado."""
        return self.state == ControllerState.PAUSED
    
    def getUptime(self) -> float:
        """Retorna uptime em segundos."""
        if not self.startTime:
            return 0.0
        return (datetime.now() - self.startTime).total_seconds()
    
    def getAvailableTopics(self) -> List[str]:
        """Lista de todos os tópicos disponíveis."""
        return list(self.topicGenerators.keys())
    
    def getActiveTopics(self) -> List[str]:
        """Lista de tópicos atualmente ativos."""
        return list(self.activeSignals)
    
    def getTopicFrequencies(self) -> Dict[str, float]:
        """Frequências atuais de todos os tópicos."""
        return self.topicFrequencies.copy()
    
    def setTopicFrequencies(self, frequencies: Dict[str, float]):
        """
        Define múltiplas frequências de uma vez.
        
        Args:
            frequencies: Dict {topic: frequency}
        """
        for topic, freq in frequencies.items():
            if topic in self.topicFrequencies:
                self.adjustTopicFrequency(topic, freq)
            else:
                self.logger.warning(f"Unknown topic in frequency update: {topic}")
    
    def getAllStats(self) -> Dict[str, Any]:
        """Estatísticas completas para endpoints REST."""
        return {
            "controller": self.getStatus(),
            "health": self.getSystemHealth(),
            "uptime": self.getUptime(),
            "isRunning": self.isRunning(),
            "isPaused": self.isPaused(),
            "summary": {
                "activeTopics": len(self.activeSignals),
                "totalTopics": len(self.topicGenerators),
                "messagesPerSecond": self.stats["messagesSent"] / max(1, self.getUptime()),
                "anomaliesPerMinute": self.stats["anomaliesInjected"] / max(1, self.getUptime() / 60),
                "successRate": self.stats["messagesSent"] / max(1, self.stats["messagesGenerated"])
            }
        }

# Instância global
mockZeroMQController = MockZeroMQController()