""" Control Room Core """

from .config import settings
from .events import eventManager
from .exceptions import (
    ControlRoomException,
    SignalValidationError,
    ZeroMQError,
    WebSocketError,
    TopicValidationError,
    # Signal Control Exceptions
    SignalControlError,
    ComponentNotFoundError,
    SignalNotFoundError,
    OperationTimeoutError,
    InvalidOperationError,
    BatchOperationError,
    StateValidationError,
    StatePersistenceError
)

# Signal Control System
from .signalControl import (
    SignalControlInterface,
    SignalControlManager,
    signalControlManager,
    SignalState,
    ComponentState,
    OperationType
)

from .signalState import (
    SignalStateManager,
    signalStateManager
)

__all__ = [
    # Core components
    "settings",
    "eventManager",
    
    # Original exceptions
    "ControlRoomException",
    "SignalValidationError",
    "ZeroMQError",
    "WebSocketError",
    "TopicValidationError",
    
    # Signal Control exceptions
    "SignalControlError",
    "ComponentNotFoundError", 
    "SignalNotFoundError",
    "OperationTimeoutError",
    "InvalidOperationError",
    "BatchOperationError",
    "StateValidationError",
    "StatePersistenceError",
    
    # Signal Control system
    "SignalControlInterface",
    "SignalControlManager",
    "signalControlManager",
    "SignalState",
    "ComponentState", 
    "OperationType",
    
    # State management
    "SignalStateManager",
    "signalStateManager"
]

import logging
logging.basicConfig(level=settings.logLevel)  # Por enquanto debug

logger = logging.getLogger(__name__)
logger.info("Control Room - Automotive Simulator")
logger.info(f"Signal Control System: {'Enabled' if settings.signalControl.persistState else 'Basic mode'}")