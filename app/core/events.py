"""
Sistema de eventos 

Resumo:
Permite gerir sistema de eventos de forma assíncrona, fundamental  
quando diferentes partes de um programa precisam comunicar-se entre si 
sem estarem diretamente acoplados, optamos por desacopolar especificamente a parte
de emitir os dados para o ws, para podermos testar se a geração / armazenamento de dados
tava a funcionar bem sem nos preocuparmos com a parte de transmitir para o websocket e enviar etc,
e é ai que entra eventManager 

Funcionalidades principais:
- `subscribe(eventName, callback)`: Permite registar (subscrever) funções que devem ser chamadas 
    quando um evento com um determinado nome for emitido. As funções devem ser assíncronas (async), 
    mas funções normais também são aceites com aviso e executadas numa thread à parte.

- `unsubscribe(eventName, callback)`: Permite remover a subscrição de uma função previamente 
    registada para um evento específico.

- `emit(eventName, data)`: Emite um evento com os dados fornecidos. Todos os 
    "listeners" associados a esse evento são executados em paralelo (de forma assíncrona).

- `getListenerCount()`: Devolve estatísticas básicas, como o número total de listeners 
    registados e os tipos de eventos com listeners.

- `getEventStats()`: Devolve estatísticas detalhadas sobre cada evento, como o número 
    de listeners e se são funções assíncronas ou não.

- `clear()`: Remove todos os listeners registados. Útil para testes ou reinício do sistema.

Notas:
- Os eventos são representados por objetos da classe `Event`, que inclui o nome do evento, 
os dados associados e o timestamp (data/hora).
- É possível utilizar funções síncronas como listeners, mas é feito um aviso nos logs e 
as mesmas são executadas em threads separadas.

"""

import asyncio
import logging
from typing import Dict, List, Callable, Any
from datetime import datetime
from dataclasses import dataclass

@dataclass
class Event:
    name: str               # Nome do evento (e.g., "signal.updated")
    data: Dict[str, Any]    # Dados do evento (e.g., {"ecg": 75.5})
    timestamp: datetime     # Quando o evento foi emitido

class EventManager:
    """Event Manager"""
    
    def __init__(self):
        self._listeners: Dict[str, List[Callable]] = {} # Lista de listeners
        self._logger = logging.getLogger(__name__)
    
    def subscribe(self, eventName: str, callback: Callable) -> None:
        """Subscreve a um evento (callback deve ser async)"""
        if eventName not in self._listeners:
            self._listeners[eventName] = []
        self._listeners[eventName].append(callback)
        self._logger.debug(f"Subscribed to: {eventName}")
    
    def unsubscribe(self, eventName: str, callback: Callable) -> None:
        """Remove subscrição de um evento"""
        if eventName in self._listeners:
            try:
                self._listeners[eventName].remove(callback)
                self._logger.debug(f"Unsubscribed from: {eventName}")
            except ValueError:
                self._logger.warning(f"Callback not found for event: {eventName}")
    
    async def emit(self, eventName: str, data: Dict[str, Any]) -> None:
        """Emite evento async - todos os listeners executam em paralelo"""
        event = Event(
            name=eventName,
            data=data,
            timestamp=datetime.now()
        )
        
        self._logger.debug(f"Emitting event: {eventName} with data: {data}")
        
        # Verificar se há listeners para este evento
        if eventName not in self._listeners:
            self._logger.debug(f"No listeners for event: {eventName}")
            return
        
        # Criar tasks para todos os listeners
        tasks = []
        for callback in self._listeners[eventName]:
            try:
                # Verificar se é async function
                if asyncio.iscoroutinefunction(callback):
                    tasks.append(asyncio.create_task(callback(event)))
                else:
                    # Se não for async, avisar mas permitir (para compatibilidade)
                    self._logger.warning(f"None async callback detected for {eventName}.")
                    # Executar sync callback numa task
                    tasks.append(asyncio.create_task(self._runSyncCallback(callback, event)))
            except Exception as e:
                self._logger.error(f"Error creating task for event {eventName}: {e}")
        
        # Executar todos os listeners em paralelo e guardar os resultados para saber se algum deu erro
        if tasks:
            results = await asyncio.gather(*tasks, return_exceptions=True) # Esperar que todas as tasks acabem (*tasks passa tasks[0], tasks[1]....) e retorna os erros com o retunr_exceptions
            
            # Log erros se houver
            for i, result in enumerate(results):
                if isinstance(result, Exception):
                    self._logger.error(f"Listener {i} for event {eventName} failed: {result}")
    
    async def _runSyncCallback(self, callback: Callable, event: Event):
        """Executa callback sync numa task async (para compatibilidade)"""
        try:
            # Executar callback sync numa thread
            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, callback, event)
        except Exception as e:
            self._logger.error(f"Error in sync callback: {e}")
            raise
    
    def getListenerCount(self) -> Dict[str, int]:
        """Stats para debug"""
        return {
            "totalListeners": sum(len(listeners) for listeners in self._listeners.values()),
            "eventTypes": len(self._listeners),
            "eventsWithListeners": list(self._listeners.keys())
        }
    
    def getEventStats(self) -> Dict[str, Any]:
        """Estatísticas detalhadas para debugging"""
        stats = {}
        for eventName, listeners in self._listeners.items():
            stats[eventName] = {
                "listenerCount": len(listeners),
                "listenerTypes": [
                    "async" if asyncio.iscoroutinefunction(cb) else "sync" 
                    for cb in listeners
                ]
            }
        return stats
    
    def clear(self) -> None:
        """Limpar todos os listeners (útil para testes)"""
        self._listeners.clear()
        self._logger.info("All event listeners cleared")

# ================================
# SIGNAL CONTROL EVENT TYPES
# ================================

class SignalControlEvents:
    """Constantes para eventos de controlo de sinais"""
    
    # Eventos de sinal individual
    SIGNAL_ENABLED = "signal.enabled"
    SIGNAL_DISABLED = "signal.disabled"
    SIGNAL_FILTERED = "signal.filtered"
    
    # Eventos por componente
    COMPONENT_SIGNAL_ACTIVATED = "component.signal_activated"
    COMPONENT_SIGNAL_DEACTIVATED = "component.signal_deactivated"
    COMPONENT_STATE_CHANGED = "component.state_changed"
    
    # Eventos de operações em lote
    BATCH_OPERATION_STARTED = "control.batch_operation_started"
    BATCH_OPERATION_COMPLETED = "control.batch_operation_completed"
    BATCH_OPERATION_FAILED = "control.batch_operation_failed"
    
    # Eventos de controlo global
    ALL_SIGNALS_ENABLED = "control.all_signals_enabled"
    ALL_SIGNALS_DISABLED = "control.all_signals_disabled"
    SYSTEM_STATE_RESET = "control.system_state_reset"
    
    # Eventos de estado e configuração
    STATE_SAVED = "control.state_saved"
    STATE_LOADED = "control.state_loaded"
    PRESET_APPLIED = "control.preset_applied"
    
    # Eventos de erro e aviso
    CONTROL_OPERATION_FAILED = "control.operation_failed"
    INVALID_SIGNAL_REQUEST = "control.invalid_signal_request"
    COMPONENT_NOT_RESPONSIVE = "control.component_not_responsive"

class SignalControlEventData:
    """Estruturas padrão para dados de eventos de controlo"""
    
    @staticmethod
    def signalStateChange(signal: str, component: str, enabled: bool, timestamp: str = None) -> Dict[str, Any]:
        """Dados para eventos de mudança de estado de sinal"""
        return {
            "signal": signal,
            "component": component,
            "enabled": enabled,
            "timestamp": timestamp or datetime.now().isoformat(),
            "previousState": not enabled  # Assumindo toggle
        }
    
    @staticmethod
    def batchOperation(operationType: str, operations: List[Dict], success: bool, 
                      timestamp: str = None, error: str = None) -> Dict[str, Any]:
        """Dados para eventos de operações em lote"""
        return {
            "operationType": operationType,  # "enable_all", "disable_all", "custom_batch"
            "operationCount": len(operations),
            "operations": operations,
            "success": success,
            "error": error,
            "timestamp": timestamp or datetime.now().isoformat(),
            "duration": None  # Será preenchido pelo caller
        }
    
    @staticmethod
    def componentStateChange(component: str, activeSignals: List[str], 
                           totalSignals: int, timestamp: str = None) -> Dict[str, Any]:
        """Dados para eventos de mudança de estado de componente"""
        return {
            "component": component,
            "activeSignals": activeSignals,
            "activeCount": len(activeSignals),
            "totalSignals": totalSignals,
            "timestamp": timestamp or datetime.now().isoformat()
        }
    
    @staticmethod
    def operationError(operation: str, error: str, signal: str = None, 
                      component: str = None, timestamp: str = None) -> Dict[str, Any]:
        """Dados para eventos de erro em operações"""
        return {
            "operation": operation,
            "error": error,
            "signal": signal,
            "component": component,
            "timestamp": timestamp or datetime.now().isoformat()
        }

# Instância global
eventManager = EventManager()