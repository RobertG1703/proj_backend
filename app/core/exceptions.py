"""
Exceções 

Classe simples para ter debugging de erros mais clara , obrigando a especifcar para cada tipo de erro
certos detalhes dependendo do casso
"""

from typing import Dict, Any, List

class ControlRoomException(Exception):
    """Exceção base"""
    
    def __init__(self, message: str, details: Dict[str, Any] = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

class SignalValidationError(ControlRoomException):
    """Erro de validação de sinal"""
    
    def __init__(self, signalType: str, value: Any, reason: str = None):
        message = f"Invalid {signalType} signal: {value}"
        if reason:
            message += f" - {reason}"
        
        super().__init__(
            message=message,
            details={
                "signalType": signalType,
                "value": value,
                "reason": reason
            }
        )

class ZeroMQError(ControlRoomException):
    """Erro ZeroMQ"""
    
    def __init__(self, operation: str, reason: str):
        message = f"ZeroMQ {operation} failed: {reason}"
        super().__init__(
            message=message,
            details={"operation": operation, "reason": reason}
        )

class WebSocketError(ControlRoomException):
    """Erro WebSocket"""
    
    def __init__(self, reason: str):
        super().__init__(
            message=f"WebSocket error: {reason}",
            details={"reason": reason}
        )

class ZeroMQProcessingError(ControlRoomException):
    """Erro no processamento de dados ZeroMQ"""
    
    def __init__(self, topic: str, operation: str, reason: str, rawData: Any = None):
        message = f"ZeroMQ processing failed for topic '{topic}' during {operation}: {reason}"
        super().__init__(
            message=message,
            details={
                "topic": topic,
                "operation": operation,
                "reason": reason,
                "rawData": str(rawData)[:200] if rawData else None  # Limitar tamanho
            }
        )

class TopicValidationError(ControlRoomException):
    """Erro de validação específica de tópico"""
    
    def __init__(self, topic: str, field: str, value: Any, expectedRange: tuple = None):
        if expectedRange:
            message = f"Invalid {field} for topic '{topic}': {value} (expected {expectedRange})"
        else:
            message = f"Invalid {field} for topic '{topic}': {value}"
        
        super().__init__(
            message=message,
            details={
                "topic": topic,
                "field": field,
                "value": value,
                "expectedRange": expectedRange
            }
        )

class UnknownTopicError(ControlRoomException):
    """Erro para tópico não reconhecido"""
    
    def __init__(self, topic: str, availableTopics: List[str] = None):
        message = f"Unknown topic: '{topic}'"
        if availableTopics:
            message += f". Available topics: {availableTopics}"
        
        super().__init__(
            message=message,
            details={
                "topic": topic,
                "availableTopics": availableTopics
            }
        )

# ================================
# SIGNAL CONTROL EXCEPTIONS
# ================================

class SignalControlError(ControlRoomException):
    """Exceção base para erros de controlo de sinais"""
    
    def __init__(self, message: str, component: str = None, signal: str = None, details: Dict[str, Any] = None):
        super().__init__(message, details)
        self.component = component
        self.signal = signal

class ComponentNotFoundError(SignalControlError):
    """Componente não encontrado"""
    
    def __init__(self, component: str, availableComponents: List[str] = None):
        message = f"Component '{component}' not found"
        if availableComponents:
            message += f". Available: {availableComponents}"
        
        super().__init__(
            message=message, 
            component=component, 
            details={"availableComponents": availableComponents}
        )

class SignalNotFoundError(SignalControlError):
    """Sinal não encontrado para o componente"""
    
    def __init__(self, signal: str, component: str, availableSignals: List[str] = None):
        message = f"Signal '{signal}' not found in component '{component}'"
        if availableSignals:
            message += f". Available: {availableSignals}"
        
        super().__init__(
            message=message, 
            component=component, 
            signal=signal, 
            details={"availableSignals": availableSignals}
        )

class OperationTimeoutError(SignalControlError):
    """Operação demorou mais que o timeout configurado"""
    
    def __init__(self, operation: str, timeout: float, component: str = None):
        message = f"Operation '{operation}' timed out after {timeout}s"
        
        super().__init__(
            message=message, 
            component=component, 
            details={"timeout": timeout, "operation": operation}
        )

class InvalidOperationError(SignalControlError):
    """Operação inválida ou não permitida"""
    
    def __init__(self, operation: str, reason: str, component: str = None, signal: str = None):
        message = f"Invalid operation '{operation}': {reason}"
        
        super().__init__(
            message=message, 
            component=component, 
            signal=signal, 
            details={"reason": reason, "operation": operation}
        )

class BatchOperationError(SignalControlError):
    """Erro durante operação em lote"""
    
    def __init__(self, failedOperations: List[Dict[str, Any]], totalOperations: int):
        message = f"Batch operation failed: {len(failedOperations)}/{totalOperations} operations failed"
        
        super().__init__(
            message=message,
            details={
                "failedOperations": failedOperations,
                "totalOperations": totalOperations,
                "successfulOperations": totalOperations - len(failedOperations)
            }
        )

class StateValidationError(SignalControlError):
    """Erro de validação de estado persistido"""
    
    def __init__(self, reason: str, stateFile: str = None):
        message = f"State validation failed: {reason}"
        
        super().__init__(
            message=message,
            details={
                "reason": reason,
                "stateFile": stateFile,
                "validationFailure": True
            }
        )

class StatePersistenceError(SignalControlError):
    """Erro de persistência de estado"""
    
    def __init__(self, operation: str, reason: str, filePath: str = None):
        message = f"State persistence {operation} failed: {reason}"
        
        super().__init__(
            message=message,
            details={
                "operation": operation,
                "reason": reason,
                "filePath": filePath
            }
        )