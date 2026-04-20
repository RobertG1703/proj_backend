"""
Sistema de Controlo de Sinais - Interface e Manager Central

Resumo:
Define a interface padrão que todos os componentes devem implementar para permitir
controlo granular de sinais. O SignalControlManager coordena todas as operações
entre componentes e mantém estado consistente em todo o sistema.

Funcionalidades principais:
- Interface padrão para todos os componentes
- Coordenação centralizada de operações de controlo
- Gestão de estado consistente entre componentes
- Operações em lote e validação
- Sistema de eventos para notificações
"""

import asyncio
import logging
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Dict, List, Set, Any, Optional, Union
from enum import Enum

from . import settings, eventManager
from .events import SignalControlEvents, SignalControlEventData
from .signalState import signalStateManager
from .exceptions import (
    SignalControlError, ComponentNotFoundError, SignalNotFoundError,
    OperationTimeoutError, InvalidOperationError, BatchOperationError
)

class SignalState(Enum):
    """Estados possíveis de um sinal"""
    ACTIVE = "active"
    INACTIVE = "inactive"
    UNKNOWN = "unknown"
    ERROR = "error"

class ComponentState(Enum):
    """Estados possíveis de um componente"""
    RUNNING = "running"
    STOPPED = "stopped"
    ERROR = "error"
    INITIALIZING = "initializing"
    UNKNOWN = "unknown"

class OperationType(Enum):
    """Tipos de operações de controlo"""
    ENABLE_SIGNAL = "enable_signal"
    DISABLE_SIGNAL = "disable_signal"
    ENABLE_ALL = "enable_all"
    DISABLE_ALL = "disable_all"
    BATCH_OPERATION = "batch_operation"
    RESET_COMPONENT = "reset_component"

# ================================
# INTERFACE PADRÃO
# ================================

class SignalControlInterface(ABC):
    """
    Interface que todos os componentes devem implementar para controlo de sinais.
    
    Esta interface garante que todos os componentes (ZeroMQPublisher, ZeroMQListener,
    ZeroMQProcessor, SignalManager, WebSocketManager) tenham métodos consistentes para controlo.
    """
    
    @abstractmethod
    def getAvailableSignals(self) -> List[str]:
        """
        Retorna lista de sinais que este componente pode processar.
        
        Returns:
            Lista de nomes de sinais disponíveis
        """
        pass
    
    @abstractmethod
    def getActiveSignals(self) -> List[str]:
        """
        Retorna lista de sinais atualmente ativos neste componente.
        
        Returns:
            Lista de nomes de sinais ativos
        """
        pass
    
    @abstractmethod
    async def enableSignal(self, signal: str) -> bool:
        """
        Ativa processamento de um sinal específico.
        
        Args:
            signal: Nome do sinal para ativar
            
        Returns:
            True se ativado com sucesso, False caso contrário
        """
        pass
    
    @abstractmethod
    async def disableSignal(self, signal: str) -> bool:
        """
        Desativa processamento de um sinal específico.
        
        Args:
            signal: Nome do sinal para desativar
            
        Returns:
            True se desativado com sucesso, False caso contrário
        """
        pass
    
    @abstractmethod
    def getSignalState(self, signal: str) -> SignalState:
        """
        Retorna estado atual de um sinal específico.
        
        Args:
            signal: Nome do sinal
            
        Returns:
            Estado atual do sinal
        """
        pass
    
    @abstractmethod
    def getComponentState(self) -> ComponentState:
        """
        Retorna estado atual do componente.
        
        Returns:
            Estado atual do componente
        """
        pass
    
    # Métodos opcionais com implementação padrão
    
    async def enableAllSignals(self) -> Dict[str, bool]:
        """
        Ativa todos os sinais disponíveis no componente.
        
        Returns:
            Dicionário {signal: success} para cada sinal
        """
        results = {}
        for signal in self.getAvailableSignals():
            results[signal] = await self.enableSignal(signal)
        return results
    
    async def disableAllSignals(self) -> Dict[str, bool]:
        """
        Desativa todos os sinais ativos no componente.
        
        Returns:
            Dicionário {signal: success} para cada sinal
        """
        results = {}
        for signal in self.getActiveSignals():
            results[signal] = await self.disableSignal(signal)
        return results
    
    def getSignalStats(self, signal: str) -> Optional[Dict[str, Any]]:
        """
        Retorna estatísticas de um sinal específico (implementação opcional).
        
        Args:
            signal: Nome do sinal
            
        Returns:
            Estatísticas do sinal ou None se não disponível
        """
        return None
    
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
            "lastUpdate": datetime.now().isoformat()
        }

# ================================
# COORDENADOR CENTRAL
# ================================

class SignalControlManager:
    """
    Coordenador central para controlo de sinais em todos os componentes.
    
    Responsável por:
    - Registar e gerir componentes
    - Coordenar operações entre componentes
    - Manter estado consistente
    - Validar operações
    - Emitir eventos
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações centralizadas
        self.config = settings.signalControl
        print("\n\n\n\n\n\n\n NO INIT DO SIGNAL CONTROL MANAGER")
        
        # Componentes registados
        self.components: Dict[str, SignalControlInterface] = {}
        self.componentStates: Dict[str, ComponentState] = {}
        
        # Estado global de sinais por componente
        self.globalSignalState: Dict[str, Dict[str, SignalState]] = {}
        
        # Controlo de operações
        self.operationTimeout = self.config.operationTimeout
        self.batchOperationTimeout = self.config.batchOperationTimeout
        self.operationInProgress = False
        self.operationLock = asyncio.Lock()
        
        # Estatísticas
        self.stats = {
            "operationsExecuted": 0,
            "batchOperationsExecuted": 0,
            "lastOperation": None,
            "totalComponents": 0,
            "errors": 0,
            "startTime": datetime.now().isoformat()
        }
        
        # Carregar estado 
        self._loadPersistedState()
        
        self.logger.info("SignalControlManager initialized")
        print("\n\n\n\n\n\n\n\n NO FIM DO INIT DE SIGNAL CONTROL MANAGER")
    
    def registerComponent(self, name: str, component: SignalControlInterface) -> None:
        """
        Regista um componente no sistema de controlo.
        
        Args:
            name: Nome único do componente
            component: Instância que implementa SignalControlInterface
        """
        if not isinstance(component, SignalControlInterface):
            raise ValueError(f"Component must implement SignalControlInterface")
        
        self.components[name] = component
        self.componentStates[name] = component.getComponentState()
        
        # Inicializar estado dos sinais do componente
        self.globalSignalState[name] = {}
        for signal in component.getAvailableSignals():
            self.globalSignalState[name][signal] = component.getSignalState(signal)
        
        self.stats["totalComponents"] = len(self.components)
        
        self.logger.info(f"Registered component '{name}' with {len(component.getAvailableSignals())} signals")
    
    def unregisterComponent(self, name: str) -> None:
        """
        Remove componente do sistema de controlo.
        
        Args:
            name: Nome do componente para remover
        """
        if name in self.components:
            del self.components[name]
            del self.componentStates[name]
            del self.globalSignalState[name]
            
            self.stats["totalComponents"] = len(self.components)
            self.logger.info(f"Unregistered component '{name}'")
    
    async def enableSignal(self, signal: str, component: str = None) -> Dict[str, bool]:
        """
        Ativa um sinal num componente específico ou em todos.
        
        Args:
            signal: Nome do sinal
            component: Nome do componente (None para todos)
            
        Returns:
            Dicionário {component: success} 
        """
        async with self.operationLock:
            results = {}
            
            components_to_update = [component] if component else self.components.keys()
            
            for compName in components_to_update:
                if compName not in self.components:
                    results[compName] = False
                    continue
                
                comp = self.components[compName]
                
                # Verificar se sinal existe no componente
                if signal not in comp.getAvailableSignals():
                    results[compName] = False
                    continue
                
                try:
                    # Executar operação com timeout
                    success = await asyncio.wait_for(
                        comp.enableSignal(signal),
                        timeout=self.operationTimeout
                    )
                    
                    results[compName] = success
                    
                    if success:
                        # Atualizar estado global
                        self.globalSignalState[compName][signal] = SignalState.ACTIVE
                        
                        # Emitir evento
                        await eventManager.emit(
                            SignalControlEvents.SIGNAL_ENABLED,
                            SignalControlEventData.signalStateChange(signal, compName, True)
                        )
                    
                except asyncio.TimeoutError:
                    results[compName] = False
                    self.logger.error(f"Timeout enabling {signal} in {compName}")
                    
                except Exception as e:
                    results[compName] = False
                    self.logger.error(f"Error enabling {signal} in {compName}: {e}")
            
            # Atualizar estatísticas
            self.stats["operationsExecuted"] += 1
            self.stats["lastOperation"] = {
                "type": "enable_signal",
                "signal": signal,
                "component": component,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            # Persistir estado se configurado
            await self._saveStateIfConfigured()
            
            return results
    
    async def disableSignal(self, signal: str, component: str = None) -> Dict[str, bool]:
        """
        Desativa um sinal num componente específico ou em todos.
        
        Args:
            signal: Nome do sinal
            component: Nome do componente (None para todos)
            
        Returns:
            Dicionário {component: success}
        """
        async with self.operationLock:
            results = {}
            
            components_to_update = [component] if component else self.components.keys()
            
            for compName in components_to_update:
                if compName not in self.components:
                    results[compName] = False
                    continue
                
                comp = self.components[compName]
                
                # Verificar se sinal existe no componente
                if signal not in comp.getAvailableSignals():
                    results[compName] = False
                    continue
                
                try:
                    # Executar operação com timeout
                    success = await asyncio.wait_for(
                        comp.disableSignal(signal),
                        timeout=self.operationTimeout
                    )
                    
                    results[compName] = success
                    
                    if success:
                        # Atualizar estado global
                        self.globalSignalState[compName][signal] = SignalState.INACTIVE
                        
                        # Emitir evento
                        await eventManager.emit(
                            SignalControlEvents.SIGNAL_DISABLED,
                            SignalControlEventData.signalStateChange(signal, compName, False)
                        )
                    
                except asyncio.TimeoutError:
                    results[compName] = False
                    self.logger.error(f"Timeout disabling {signal} in {compName}")
                    
                except Exception as e:
                    results[compName] = False
                    self.logger.error(f"Error disabling {signal} in {compName}: {e}")
            
            # Atualizar estatísticas
            self.stats["operationsExecuted"] += 1
            self.stats["lastOperation"] = {
                "type": "disable_signal",
                "signal": signal,
                "component": component,
                "timestamp": datetime.now().isoformat(),
                "results": results
            }
            
            # Persistir estado se configurado
            await self._saveStateIfConfigured()
            
            return results
    
    async def enableAllSignals(self) -> Dict[str, Dict[str, bool]]:
        """
        Ativa todos os sinais em todos os componentes.
        
        Returns:
            Dicionário {component: {signal: success}}
        """
        async with self.operationLock:
            results = {}
            
            for compName, comp in self.components.items():
                try:
                    comp_results = await asyncio.wait_for(
                        comp.enableAllSignals(),
                        timeout=self.batchOperationTimeout
                    )
                    results[compName] = comp_results
                    
                    # Atualizar estado global
                    for signal, success in comp_results.items():
                        if success:
                            self.globalSignalState[compName][signal] = SignalState.ACTIVE
                    
                except Exception as e:
                    self.logger.error(f"Error enabling all signals in {compName}: {e}")
                    results[compName] = {}
            
            # Emitir evento global
            await eventManager.emit(
                SignalControlEvents.ALL_SIGNALS_ENABLED,
                {"timestamp": datetime.now().isoformat(), "results": results}
            )
            
            self.stats["batchOperationsExecuted"] += 1
            await self._saveStateIfConfigured()
            
            return results
    
    async def disableAllSignals(self) -> Dict[str, Dict[str, bool]]:
        """
        Desativa todos os sinais em todos os componentes.
        
        Returns:
            Dicionário {component: {signal: success}}
        """
        # Verificar se operação é permitida
        if not self.config.allowEmptyActiveSignals:
            raise InvalidOperationError(
                "disable_all", 
                "Disabling all signals not allowed by configuration"
            )
        
        async with self.operationLock:
            results = {}
            
            for compName, comp in self.components.items():
                try:
                    comp_results = await asyncio.wait_for(
                        comp.disableAllSignals(),
                        timeout=self.batchOperationTimeout
                    )
                    results[compName] = comp_results
                    
                    # Atualizar estado global
                    for signal, success in comp_results.items():
                        if success:
                            self.globalSignalState[compName][signal] = SignalState.INACTIVE
                    
                except Exception as e:
                    self.logger.error(f"Error disabling all signals in {compName}: {e}")
                    results[compName] = {}
            
            # Emitir evento global
            await eventManager.emit(
                SignalControlEvents.ALL_SIGNALS_DISABLED,
                {"timestamp": datetime.now().isoformat(), "results": results}
            )
            
            self.stats["batchOperationsExecuted"] += 1
            await self._saveStateIfConfigured()
            
            return results
    
    async def executeBatchOperation(self, operations: List[Dict[str, str]]) -> Dict[str, Any]:
        """
        Executa múltiplas operações em lote.
        
        Args:
            operations: Lista de operações no formato:
                [{"action": "enable", "signal": "signal_name", "component": "compName"}, ...]
        
        Returns:
            Resultados detalhados da operação em lote
        """
        if len(operations) > self.config.maxBatchOperations:
            raise InvalidOperationError(
                "batch_operation",
                f"Too many operations: {len(operations)} > {self.config.maxBatchOperations}"
            )
        
        async with self.operationLock:
            start_time = datetime.now()
            results = {"operations": [], "summary": {"successful": 0, "failed": 0}}
            
            try:
                await eventManager.emit(
                    SignalControlEvents.BATCH_OPERATION_STARTED,
                    SignalControlEventData.batchOperation("custom_batch", operations, True)
                )
                
                for i, operation in enumerate(operations):
                    op_result = {
                        "index": i,
                        "operation": operation,
                        "success": False,
                        "error": None
                    }
                    
                    try:
                        action = operation.get("action")
                        signal = operation.get("signal")
                        component = operation.get("component")
                        
                        if action == "enable":
                            result = await self.enableSignal(signal, component)
                            op_result["success"] = any(result.values())
                        elif action == "disable":
                            result = await self.disableSignal(signal, component)
                            op_result["success"] = any(result.values())
                        else:
                            op_result["error"] = f"Unknown action: {action}"
                        
                        op_result["details"] = result if 'result' in locals() else None
                        
                    except Exception as e:
                        op_result["error"] = str(e)
                    
                    results["operations"].append(op_result)
                    
                    if op_result["success"]:
                        results["summary"]["successful"] += 1
                    else:
                        results["summary"]["failed"] += 1
                
                # Calcular duração
                duration = (datetime.now() - start_time).total_seconds()
                results["summary"]["duration"] = duration
                results["summary"]["totalOperations"] = len(operations)
                
                # Emitir evento de conclusão
                await eventManager.emit(
                    SignalControlEvents.BATCH_OPERATION_COMPLETED,
                    {
                        **SignalControlEventData.batchOperation("custom_batch", operations, True),
                        "duration": duration,
                        "summary": results["summary"]
                    }
                )
                
                self.stats["batchOperationsExecuted"] += 1
                await self._saveStateIfConfigured()
                
                return results
                
            except Exception as e:
                # Emitir evento de erro
                await eventManager.emit(
                    SignalControlEvents.BATCH_OPERATION_FAILED,
                    SignalControlEventData.batchOperation("custom_batch", operations, False, error=str(e))
                )
                raise
    
    def getGlobalState(self) -> Dict[str, Any]:
        """
        Retorna estado global completo do sistema.
        
        Returns:
            Estado completo de todos os componentes e sinais
        """
        # Atualizar estados atuais
        for compName, comp in self.components.items():
            self.componentStates[compName] = comp.getComponentState()
            
            for signal in comp.getAvailableSignals():
                self.globalSignalState[compName][signal] = comp.getSignalState(signal)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "components": {
                name: {
                    "state": self.componentStates.get(name, ComponentState.UNKNOWN).value,
                    "signals": { 
                        sinal: estado.value  # Converter Enum para string
                        for sinal, estado in self.globalSignalState.get(name, {}).items()
                    },
                    "summary": comp.getControlSummary()
                }
                for name, comp in self.components.items()
            },
            "globalSummary": {
                "totalComponents": len(self.components),
                "totalSignals": sum(len(comp.getAvailableSignals()) for comp in self.components.values()),
                "activeSignals": sum(len(comp.getActiveSignals()) for comp in self.components.values()),
                "componentStates": {name: state.value for name, state in self.componentStates.items()}
            },
            "stats": self.stats.copy()
        }
    
    def getComponentState(self, component: str) -> Optional[Dict[str, Any]]:
        """
        Retorna estado detalhado de um componente específico.
        
        Args:
            component: Nome do componente
            
        Returns:
            Estado detalhado do componente ou None se não encontrado
        """
        if component not in self.components:
            return None
        
        comp = self.components[component]
        
        return {
            "name": component,
            "state": comp.getComponentState().value,
            "availableSignals": comp.getAvailableSignals(),
            "activeSignals": comp.getActiveSignals(),
            "signalStates": {
                signal: comp.getSignalState(signal).value 
                for signal in comp.getAvailableSignals()
            },
            "summary": comp.getControlSummary(),
            "lastUpdate": datetime.now().isoformat()
        }
    
    async def resetComponent(self, component: str) -> bool:
        """
        Reset um componente para estado default.
        
        Args:
            component: Nome do componente
            
        Returns:
            True se reset com sucesso
        """
        if component not in self.components:
            raise ComponentNotFoundError(component, list(self.components.keys()))
        
        try:
            comp = self.components[component]
            
            # Ativar todos os sinais (estado default)
            results = await comp.enableAllSignals()
            success = all(results.values())
            
            if success:
                # Atualizar estado global
                for signal in comp.getAvailableSignals():
                    self.globalSignalState[component][signal] = SignalState.ACTIVE
                
                await eventManager.emit(
                    SignalControlEvents.COMPONENT_STATE_CHANGED,
                    SignalControlEventData.componentStateChange(
                        component, 
                        comp.getActiveSignals(), 
                        len(comp.getAvailableSignals())
                    )
                )
            
            return success
            
        except Exception as e:
            self.logger.error(f"Error resetting component {component}: {e}")
            return False
        
    def _loadPersistedState(self) -> None:
        """Carrega estado persistido se configurado"""
        if not self.config.persistState:
            return
        
        try:
            state = signalStateManager.loadState()
            if state and "components" in state:
                self.logger.warning("Applying persisted signal control state...")
                
                # Aplicar estado a cada componente
                for compName, compState in state["components"].items():
                    if compName in self.components:
                        self._applyStateToComponent(compName, compState)
                
                self.logger.warning("Loaded persisted signal control state")
            else:
                self.logger.warning("No valid persisted state found, using defaults")
        except Exception as e:
            self.logger.warning(f"Could not load persisted state: {e}")

    
    async def _applyStateToComponent(self, compName: str, compState: Dict[str, Any]) -> None:
        """Aplica estado persistido a um componente específico"""
        try:
            comp = self.components[compName]
            signals = compState.get("signals", {})
            
            for signal, stateStr in signals.items():
                try:
                    # Converter string state para boolean
                    isActive = (stateStr == "active")
                    currentState = comp.getSignalState(signal)
                    
                    # Só aplicar se estado for diferente
                    if (isActive and currentState != SignalState.ACTIVE) or \
                    (not isActive and currentState == SignalState.ACTIVE):
                        
                        if isActive:
                            await comp.enableSignal(signal)
                        else:
                            await comp.disableSignal(signal)
                        
                        self.logger.debug(f"Applied state to {compName}.{signal}: {stateStr}")
                        
                except Exception as e:
                    self.logger.warning(f"Failed to apply state to {compName}.{signal}: {e}")
                    
        except Exception as e:
            self.logger.error(f"Failed to apply state to component {compName}: {e}")
        
    async def _saveStateIfConfigured(self) -> None:
        """Salva estado se persistência estiver configurada"""
        if not self.config.persistState:
            return
        
        try:
            currentState = self.getGlobalState()
            signalStateManager.saveState(currentState)
        except Exception as e:
            self.logger.error(f"Error saving state: {e}")

# Instância global
signalControlManager = SignalControlManager()