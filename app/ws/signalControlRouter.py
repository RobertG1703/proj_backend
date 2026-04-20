"""
Signal Control Router - API REST para controlo de sinais

Resumo:
Define endpoints REST para controlo granular do sistema Signal Control.
Permite ativar/desativar sinais por componente, obter status detalhado,
e executar operações em lote. Suporta operações globais e específicas
por componente e sinal individual.

Localização: app/ws/signalControlRouter.py
Endpoints disponíveis:
- Operações globais (todos os componentes)
- Operações por componente específico  
- Operações por sinal individual
- Operações em lote
- Status e monitorização
"""

import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from fastapi import APIRouter, HTTPException, Path, Query
from fastapi.responses import JSONResponse

from ..core.signalControl import signalControlManager
from ..core.exceptions import (
    SignalControlError, ComponentNotFoundError, SignalNotFoundError,
    OperationTimeoutError, InvalidOperationError, BatchOperationError
)

# Configurar logging
logger = logging.getLogger(__name__)

# Criar router
router = APIRouter(prefix="/api/signal-control", tags=["Signal Control"])

# ================================
# FUNÇÕES DE VALIDAÇÃO
# ================================

def validate_component(component: str) -> None:
    """Valida se componente existe"""
    if component not in signalControlManager.components:
        available = list(signalControlManager.components.keys())
        raise HTTPException(
            status_code=404,
            detail=f"Component '{component}' not found. Available: {available}"
        )

def validate_signal_in_component(component: str, signal: str) -> None:
    """Valida se sinal existe no componente"""
    validate_component(component)
    comp = signalControlManager.components[component]
    
    if signal not in comp.getAvailableSignals():
        available = comp.getAvailableSignals()
        raise HTTPException(
            status_code=400,
            detail=f"Signal '{signal}' not available in component '{component}'. Available: {available}"
        )

def validate_batch_operations(operations: List[Dict[str, Any]]) -> None:
    """Valida operações em lote"""
    if len(operations) > 10:
        raise HTTPException(
            status_code=400,
            detail=f"Too many operations: {len(operations)} > 10"
        )
    
    if len(operations) == 0:
        raise HTTPException(
            status_code=400,
            detail="No operations provided"
        )
    
    for i, op in enumerate(operations):
        if "action" not in op or "signal" not in op:
            raise HTTPException(
                status_code=400,
                detail=f"Operation {i}: 'action' and 'signal' are required"
            )
        
        if op["action"] not in ["enable", "disable"]:
            raise HTTPException(
                status_code=400,
                detail=f"Operation {i}: Invalid action '{op['action']}'. Must be 'enable' or 'disable'"
            )

def create_operation_response(success: bool, message: str, component: str = None, 
                            signal: str = None, details: Dict[str, Any] = None) -> Dict[str, Any]:
    """Cria resposta padrão para operações"""
    response = {
        "success": success,
        "message": message,
        "timestamp": datetime.now().isoformat()
    }
    
    if component:
        response["component"] = component
    if signal:
        response["signal"] = signal
    if details:
        response["details"] = details
    
    return response

def create_status_response(status: Dict[str, Any], component: str = None, 
                          signal: str = None) -> Dict[str, Any]:
    """Cria resposta padrão para status"""
    response = {
        "timestamp": datetime.now().isoformat(),
        "status": status
    }
    
    if component:
        response["component"] = component
    if signal:
        response["signal"] = signal
    
    return response

# ================================
# ENDPOINTS GLOBAIS
# ================================

@router.get("/status")
async def get_global_status():
    """
    Retorna status completo do sistema Signal Control.
    
    Returns:
        Status de todos os componentes e sinais
    """
    try:
        globalState = signalControlManager.getGlobalState()
        return JSONResponse(content=create_status_response(globalState))
        
    except Exception as e:
        logger.error(f"Error getting global status: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/enable-all")
async def enable_all_signals():
    """
    Ativa todos os sinais em todos os componentes.
    
    Returns:
        Resultado da operação global
    """
    try:
        results = await signalControlManager.enableAllSignals()
        
        # Calcular sucesso global
        totalOperations = sum(len(comp_results) for comp_results in results.values())
        successfulOperations = sum(
            sum(1 for success in comp_results.values() if success) 
            for comp_results in results.values()
        )
        
        overallSuccess = successfulOperations == totalOperations
        
        response = create_operation_response(
            success=overallSuccess,
            message=f"Enabled {successfulOperations}/{totalOperations} signals across all components",
            details={
                "results": results,
                "summary": {
                    "totalOperations": totalOperations,
                    "successful": successfulOperations,
                    "failed": totalOperations - successfulOperations
                }
            }
        )
        
        return JSONResponse(content=response)
        
    except InvalidOperationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error enabling all signals: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/disable-all")
async def disable_all_signals():
    """
    Desativa todos os sinais em todos os componentes.
    
    Returns:
        Resultado da operação global
    """
    try:
        results = await signalControlManager.disableAllSignals()
        
        # Calcular sucesso global
        totalOperations = sum(len(comp_results) for comp_results in results.values())
        successfulOperations = sum(
            sum(1 for success in comp_results.values() if success) 
            for comp_results in results.values()
        )
        
        overallSuccess = successfulOperations == totalOperations
        
        response = create_operation_response(
            success=overallSuccess,
            message=f"Disabled {successfulOperations}/{totalOperations} signals across all components",
            details={
                "results": results,
                "summary": {
                    "totalOperations": totalOperations,
                    "successful": successfulOperations,
                    "failed": totalOperations - successfulOperations
                }
            }
        )
        
        return JSONResponse(content=response)
        
    except InvalidOperationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error disabling all signals: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# ENDPOINTS POR COMPONENTE
# ================================

@router.get("/{component}/status")
async def get_component_status(component: str = Path(..., description="Nome do componente")):
    """
    Retorna status detalhado de um componente específico.
    
    Args:
        component: Nome do componente (publisher, listener, processor, manager, websocket)
    
    Returns:
        Status detalhado do componente
    """
    try:
        validate_component(component)
        componentState = signalControlManager.getComponentState(component)
        
        return JSONResponse(content=create_status_response(componentState, component=component))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting component status for {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/{component}/enable-all")
async def enable_all_component_signals(component: str = Path(..., description="Nome do componente")):
    """
    Ativa todos os sinais de um componente específico.
    
    Args:
        component: Nome do componente
    
    Returns:
        Resultado da operação
    """
    try:
        validate_component(component)
        comp = signalControlManager.components[component]
        results = await comp.enableAllSignals()
        
        successfulOperations = sum(1 for success in results.values() if success)
        totalOperations = len(results)
        overallSuccess = successfulOperations == totalOperations
        
        response = create_operation_response(
            success=overallSuccess,
            message=f"Enabled {successfulOperations}/{totalOperations} signals in component '{component}'",
            component=component,
            details={
                "results": results,
                "availableSignals": comp.getAvailableSignals(),
                "activeSignals": comp.getActiveSignals()
            }
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling all signals for component {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/{component}/disable-all")
async def disable_all_component_signals(component: str = Path(..., description="Nome do componente")):
    """
    Desativa todos os sinais de um componente específico.
    
    Args:
        component: Nome do componente
    
    Returns:
        Resultado da operação
    """
    try:
        validate_component(component)
        comp = signalControlManager.components[component]
        results = await comp.disableAllSignals()
        
        successfulOperations = sum(1 for success in results.values() if success)
        totalOperations = len(results)
        overallSuccess = successfulOperations == totalOperations
        
        response = create_operation_response(
            success=overallSuccess,
            message=f"Disabled {successfulOperations}/{totalOperations} signals in component '{component}'",
            component=component,
            details={
                "results": results,
                "availableSignals": comp.getAvailableSignals(),
                "activeSignals": comp.getActiveSignals()
            }
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling all signals for component {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# ENDPOINTS POR SINAL INDIVIDUAL
# ================================

@router.post("/{component}/signals/{signal}/enable")
async def enable_signal(
    component: str = Path(..., description="Nome do componente"),
    signal: str = Path(..., description="Nome do sinal")
):
    """
    Ativa um sinal específico num componente.
    
    Args:
        component: Nome do componente
        signal: Nome do sinal
    
    Returns:
        Resultado da operação
    """
    try:
        validate_signal_in_component(component, signal)
        results = await signalControlManager.enableSignal(signal, component)
        
        success = results.get(component, False)
        
        response = create_operation_response(
            success=success,
            message=f"{'Successfully enabled' if success else 'Failed to enable'} signal '{signal}' in component '{component}'",
            component=component,
            signal=signal,
            details={"results": results}
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error enabling signal {signal} in component {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.post("/{component}/signals/{signal}/disable")
async def disable_signal(
    component: str = Path(..., description="Nome do componente"),
    signal: str = Path(..., description="Nome do sinal")
):
    """
    Desativa um sinal específico num componente.
    
    Args:
        component: Nome do componente
        signal: Nome do sinal
    
    Returns:
        Resultado da operação
    """
    try:
        validate_signal_in_component(component, signal)
        results = await signalControlManager.disableSignal(signal, component)
        
        success = results.get(component, False)
        
        response = create_operation_response(
            success=success,
            message=f"{'Successfully disabled' if success else 'Failed to disable'} signal '{signal}' in component '{component}'",
            component=component,
            signal=signal,
            details={"results": results}
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error disabling signal {signal} in component {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/{component}/signals/{signal}/status")
async def get_signal_status(
    component: str = Path(..., description="Nome do componente"),
    signal: str = Path(..., description="Nome do sinal")
):
    """
    Retorna status de um sinal específico num componente.
    
    Args:
        component: Nome do componente
        signal: Nome do sinal
    
    Returns:
        Status do sinal
    """
    try:
        validate_signal_in_component(component, signal)
        comp = signalControlManager.components[component]
        
        signalState = comp.getSignalState(signal)
        
        status = {
            "signal": signal,
            "component": component,
            "state": signalState.value,
            "isActive": signal in comp.getActiveSignals(),
            "availableSignals": comp.getAvailableSignals(),
            "activeSignals": comp.getActiveSignals(),
            "componentState": comp.getComponentState().value
        }
        
        # Adicionar estatísticas se disponível
        if hasattr(comp, 'getSignalStats'):
            stats = comp.getSignalStats(signal)
            if stats:
                status["stats"] = stats
        
        return JSONResponse(content=create_status_response(status, component=component, signal=signal))
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error getting signal status for {signal} in {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# ENDPOINTS DE OPERAÇÕES EM LOTE
# ================================

@router.post("/batch")
async def execute_batch_operations(request: Dict[str, Any]):
    """
    Executa múltiplas operações de controlo em lote.
    
    Args:
        request: {"operations": [{"action": "enable/disable", "signal": "name", "component": "name"}], "description": "optional"}
    
    Returns:
        Resultado das operações em lote
    """
    try:
        operations = request.get("operations", [])
        description = request.get("description", "")
        
        validate_batch_operations(operations)
        
        # Executar operações em lote
        results = await signalControlManager.executeBatchOperation(operations)
        
        summary = results.get("summary", {})
        successful = summary.get("successful", 0)
        failed = summary.get("failed", 0)
        total = summary.get("totalOperations", 0)
        duration = summary.get("duration", 0)
        
        overallSuccess = failed == 0
        
        response = create_operation_response(
            success=overallSuccess,
            message=f"Batch operation completed: {successful}/{total} successful operations in {duration:.2f}s",
            details={
                "description": description,
                "results": results,
                "summary": summary
            }
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except BatchOperationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except InvalidOperationError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error executing batch operations: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# ENDPOINTS DE INFORMAÇÃO
# ================================

@router.get("/components")
async def list_components():
    """
    Lista todos os componentes disponíveis e seus sinais.
    
    Returns:
        Lista de componentes com informações detalhadas
    """
    try:
        components = {}
        
        for compName, comp in signalControlManager.components.items():
            components[compName] = {
                "name": compName,
                "state": comp.getComponentState().value,
                "availableSignals": comp.getAvailableSignals(),
                "activeSignals": comp.getActiveSignals(),
                "totalSignals": len(comp.getAvailableSignals()),
                "activeCount": len(comp.getActiveSignals()),
                "summary": comp.getControlSummary() if hasattr(comp, 'getControlSummary') else None
            }
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "totalComponents": len(components),
            "components": components
        })
        
    except Exception as e:
        logger.error(f"Error listing components: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/signals")
async def list_all_signals():
    """
    Lista todos os sinais disponíveis por componente.
    
    Returns:
        Mapeamento completo de sinais por componente
    """
    try:
        signalMapping = {}
        
        for compName, comp in signalControlManager.components.items():
            signals = {}
            for signal in comp.getAvailableSignals():
                signals[signal] = {
                    "name": signal,
                    "state": comp.getSignalState(signal).value,
                    "isActive": signal in comp.getActiveSignals()
                }
            
            signalMapping[compName] = {
                "componentType": compName,
                "totalSignals": len(signals),
                "signals": signals
            }
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "signalMapping": signalMapping
        })
        
    except Exception as e:
        logger.error(f"Error listing signals: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

@router.get("/health")
async def get_signal_control_health():
    """
    Retorna saúde geral do sistema Signal Control.
    
    Returns:
        Status de saúde do sistema
    """
    try:
        globalState = signalControlManager.getGlobalState()
        summary = globalState.get("globalSummary", {})
        
        # Calcular saúde baseada em componentes ativos
        totalComponents = summary.get("totalComponents", 0)
        activeSignals = summary.get("activeSignals", 0)
        totalSignals = summary.get("totalSignals", 0)
        
        health = "healthy"
        issues = []
        warnings = []
        
        if totalComponents == 0:
            health = "critical"
            issues.append("No components registered")
        elif activeSignals == 0:
            health = "warning"
            warnings.append("No active signals")
        elif activeSignals < totalSignals / 2:
            health = "warning"
            warnings.append(f"Low signal activity: {activeSignals}/{totalSignals}")
        
        return JSONResponse(content={
            "timestamp": datetime.now().isoformat(),
            "health": health,
            "issues": issues,
            "warnings": warnings,
            "summary": {
                "totalComponents": totalComponents,
                "totalSignals": totalSignals,
                "activeSignals": activeSignals,
                "signalActivity": f"{activeSignals}/{totalSignals}" if totalSignals > 0 else "0/0"
            },
            "components": summary.get("componentStates", {}),
            "managerStats": signalControlManager.stats
        })
        
    except Exception as e:
        logger.error(f"Error getting signal control health: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")

# ================================
# ENDPOINTS DE RESET
# ================================

@router.post("/{component}/reset")
async def reset_component(component: str = Path(..., description="Nome do componente")):
    """
    Reset um componente para estado default (todos os sinais ativos).
    
    Args:
        component: Nome do componente
    
    Returns:
        Resultado da operação de reset
    """
    try:
        validate_component(component)
        success = await signalControlManager.resetComponent(component)
        
        response = create_operation_response(
            success=success,
            message=f"{'Successfully reset' if success else 'Failed to reset'} component '{component}' to default state",
            component=component,
            details={
                "resetToDefaultState": True,
                "defaultState": "all_signals_active"
            }
        )
        
        return JSONResponse(content=response)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error resetting component {component}: {e}")
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")