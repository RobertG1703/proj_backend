"""
Sistema de Estado de Sinais - Persistência e Recovery

Resumo:
Gere a persistência do estado de sinais entre reinícios do sistema,
permitindo recovery automático e manutenção de configurações customizadas.
Thread-safe e com backup automático para prevenir perda de dados.

Funcionalidades principais:
- Persistência em JSON com backup automático
- Recovery automático ao inicializar
- Thread-safe operations
- Validação de integridade dos dados
- Cleanup automático de backups antigos
"""

import json
import logging
import os
import shutil
import threading
from datetime import datetime, timedelta
from typing import Dict, Any, Optional, List
from pathlib import Path

from . import settings

class SignalStateManager:
    """
    Gere persistência e recovery do estado de sinais.
    
    Responsável por:
    - Salvar/carregar estado em JSON
    - Backup automático e rotação
    - Validação de integridade
    - Recovery em caso de corrupção
    - Thread-safe operations
    """
    
    def __init__(self):
        self.logger = logging.getLogger(__name__)
        
        # Configurações
        self.config = settings.signalControl
        self.stateFilePath = Path(self.config.stateFilePath)
        self.backupDir = self.stateFilePath.parent / "signal_state_backups"
        
        # Thread safety
        self.lock = threading.RLock()
        
        # Estado em memória para cache
        self.cachedState: Optional[Dict[str, Any]] = None
        self.lastSaveTime: Optional[datetime] = None
        self.lastLoadTime: Optional[datetime] = None
        
        # Estatísticas
        self.stats = {
            "saveOperations": 0,
            "loadOperations": 0,
            "backupsCreated": 0,
            "recoveryOperations": 0,
            "errors": 0,
            "lastError": None
        }
        
        # Criar diretório de backup se não existir
        self._ensureBackupDirectory()
        
        self.logger.info(f"SignalStateManager initialized - State file: {self.stateFilePath}")
    
    def saveState(self, state: Dict[str, Any]) -> bool:
        """
        Salva estado atual em ficheiro JSON com backup automático.
        
        Args:
            state: Estado completo para persistir
            
        Returns:
            True se salvo com sucesso
        """
        with self.lock:
            try:
                # Validar estado antes de salvar
                if not self._validateState(state):
                    self.logger.error("State validation failed, not saving")
                    return False
                
                # Criar backup do ficheiro atual se existir
                if self.stateFilePath.exists():
                    self._createBackup()
                
                # Preparar dados para salvar
                stateToSave = {
                    "metadata": {
                        "version": settings.version,
                        "timestamp": datetime.now().isoformat(),
                        "saveOperation": self.stats["saveOperations"] + 1
                    },
                    "signalControl": {
                        "globalState": state,
                        "config": {
                            "persistState": self.config.persistState,
                            "allowEmptyActiveSignals": self.config.allowEmptyActiveSignals,
                            "componentMappings": self.config.componentSignalMappings
                        }
                    }
                }
                
                # Salvar com encoding UTF-8 e indentação para legibilidade
                with open(self.stateFilePath, 'w', encoding='utf-8') as f:
                    json.dump(stateToSave, f, indent=2, ensure_ascii=False)
                
                # Atualizar cache e estatísticas
                self.cachedState = state.copy()
                self.lastSaveTime = datetime.now()
                self.stats["saveOperations"] += 1
                
                self.logger.debug(f"State saved successfully to {self.stateFilePath}")
                
                # Cleanup backups antigos
                self._cleanupOldBackups()
                
                return True
                
            except Exception as e:
                self.stats["errors"] += 1
                self.stats["lastError"] = {
                    "operation": "save",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.logger.error(f"Error saving state: {e}")
                return False
    
    def loadState(self) -> Optional[Dict[str, Any]]:
        """
        Carrega estado do ficheiro JSON com recovery automático.
        
        Returns:
            Estado carregado ou None se não disponível/inválido
        """
        with self.lock:
            try:
                # Verificar se ficheiro existe
                if not self.stateFilePath.exists():
                    self.logger.info("No state file found, starting with default state")
                    return None
                
                # Tentar carregar ficheiro principal
                try:
                    with open(self.stateFilePath, 'r', encoding='utf-8') as f:
                        loadedData = json.load(f)
                    
                    # Validar estrutura do ficheiro
                    if not self._validateLoadedData(loadedData):
                        raise ValueError("Invalid state file structure")
                    
                    # Extrair estado dos sinais
                    state = loadedData.get("signalControl", {}).get("globalState")
                    
                    if not state:
                        raise ValueError("No signal control state found in file")
                    
                    # Validar estado dos sinais
                    if not self._validateState(state):
                        raise ValueError("Invalid signal state data")
                    
                    # Atualizar cache e estatísticas
                    self.cachedState = state.copy()
                    self.lastLoadTime = datetime.now()
                    self.stats["loadOperations"] += 1
                    
                    # Log informações sobre o estado carregado
                    metadata = loadedData.get("metadata", {})
                    savedTime = metadata.get("timestamp", "unknown")
                    version = metadata.get("version", "unknown")
                    
                    self.logger.info(f"State loaded successfully - Saved: {savedTime}, Version: {version}")
                    
                    return state
                    
                except (json.JSONDecodeError, ValueError, KeyError) as e:
                    self.logger.warning(f"Main state file corrupted: {e}")
                    
                    # Tentar recovery de backup
                    recoveredState = self._attemptRecoveryFromBackup()
                    if recoveredState:
                        return recoveredState
                    
                    raise
                
            except Exception as e:
                self.stats["errors"] += 1
                self.stats["lastError"] = {
                    "operation": "load",
                    "error": str(e),
                    "timestamp": datetime.now().isoformat()
                }
                self.logger.error(f"Error loading state: {e}")
                return None
    
    def _validateState(self, state: Dict[str, Any]) -> bool:
        """
        Valida integridade do estado dos sinais.
        
        Args:
            state: Estado para validar
            
        Returns:
            True se válido
        """
        try:
            # Verificar estrutura básica
            if not isinstance(state, dict):
                return False
            
            required_fields = ["timestamp", "components", "globalSummary"]
            if not all(field in state for field in required_fields):
                return False
            
            # Validar componentes
            components = state.get("components", {})
            if not isinstance(components, dict):
                return False
            
            # Validar cada componente
            for comp_name, comp_data in components.items():
                if not isinstance(comp_data, dict):
                    return False
                
                comp_required = ["state", "signals", "summary"]
                if not all(field in comp_data for field in comp_required):
                    return False
                
                # Validar sinais do componente
                signals = comp_data.get("signals", {})
                if not isinstance(signals, dict):
                    return False
            
            # Validar timestamp
            try:
                datetime.fromisoformat(state["timestamp"])
            except (ValueError, TypeError):
                return False
            
            return True
            
        except Exception as e:
            self.logger.error(f"State validation error: {e}")
            return False
    
    def _validateLoadedData(self, data: Dict[str, Any]) -> bool:
        """
        Valida estrutura do ficheiro carregado.
        
        Args:
            data: Dados carregados do JSON
            
        Returns:
            True se estrutura válida
        """
        try:
            # Verificar estrutura principal
            if not isinstance(data, dict):
                return False
            
            # Verificar metadados
            metadata = data.get("metadata")
            if not isinstance(metadata, dict):
                return False
            
            # Verificar seção de controlo de sinais
            signalControl = data.get("signalControl")
            if not isinstance(signalControl, dict):
                return False
            
            # Verificar se tem estado global
            globalState = signalControl.get("globalState")
            if not isinstance(globalState, dict):
                return False
            
            return True
            
        except Exception:
            return False
    
    def _createBackup(self) -> bool:
        """
        Cria backup do ficheiro atual antes de sobrescrever.
        
        Returns:
            True se backup criado com sucesso
        """
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backupFileName = f"signal_state_backup_{timestamp}.json"
            backupPath = self.backupDir / backupFileName
            
            # Copiar ficheiro atual para backup
            shutil.copy2(self.stateFilePath, backupPath)
            
            self.stats["backupsCreated"] += 1
            self.logger.debug(f"Backup created: {backupPath}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Error creating backup: {e}")
            return False
    
    def _attemptRecoveryFromBackup(self) -> Optional[Dict[str, Any]]:
        """
        Tenta recovery do estado a partir de backups disponíveis.
        
        Returns:
            Estado recuperado ou None se falhou
        """
        try:
            # Listar backups ordenados por data (mais recente primeiro)
            backupFiles = sorted(
                self.backupDir.glob("signal_state_backup_*.json"),
                key=lambda x: x.stat().st_mtime,
                reverse=True
            )
            
            if not backupFiles:
                self.logger.warning("No backup files found for recovery")
                return None
            
            # Tentar carregar backups em ordem
            for backupFile in backupFiles:
                try:
                    self.logger.info(f"Attempting recovery from backup: {backupFile}")
                    
                    with open(backupFile, 'r', encoding='utf-8') as f:
                        backupData = json.load(f)
                    
                    # Validar backup
                    if not self._validateLoadedData(backupData):
                        continue
                    
                    state = backupData.get("signalControl", {}).get("globalState")
                    if not state or not self._validateState(state):
                        continue
                    
                    # Recovery bem-sucedido
                    self.stats["recoveryOperations"] += 1
                    self.logger.info(f"Successfully recovered state from backup: {backupFile}")
                    
                    # Restaurar ficheiro principal com o backup
                    shutil.copy2(backupFile, self.stateFilePath)
                    
                    return state
                    
                except Exception as e:
                    self.logger.warning(f"Failed to recover from {backupFile}: {e}")
                    continue
            
            self.logger.error("All recovery attempts failed")
            return None
            
        except Exception as e:
            self.logger.error(f"Error during recovery process: {e}")
            return None
    
    def _cleanupOldBackups(self, maxAge: timedelta = timedelta(days=7), maxCount: int = 10) -> None:
        """
        Remove backups antigos para economizar espaço.
        
        Args:
            maxAge: Idade máxima dos backups
            maxCount: Número máximo de backups a manter
        """
        try:
            backupFiles = list(self.backupDir.glob("signal_state_backup_*.json"))
            
            if len(backupFiles) <= maxCount:
                return
            
            # Ordenar por data de modificação (mais recente primeiro)
            backupFiles.sort(key=lambda x: x.stat().st_mtime, reverse=True)
            
            # Manter só os mais recentes
            filesToDelete = backupFiles[maxCount:]
            
            # Também remover baseado na idade
            cutoffTime = datetime.now() - maxAge
            for backupFile in backupFiles:
                fileTime = datetime.fromtimestamp(backupFile.stat().st_mtime)
                if fileTime < cutoffTime and backupFile not in filesToDelete:
                    filesToDelete.append(backupFile)
            
            # Remover ficheiros
            for fileToDelete in filesToDelete:
                try:
                    fileToDelete.unlink()
                    self.logger.debug(f"Deleted old backup: {fileToDelete}")
                except Exception as e:
                    self.logger.warning(f"Failed to delete backup {fileToDelete}: {e}")
            
            if filesToDelete:
                self.logger.info(f"Cleaned up {len(filesToDelete)} old backup files")
                
        except Exception as e:
            self.logger.error(f"Error during backup cleanup: {e}")
    
    def _ensureBackupDirectory(self) -> None:
        """Cria diretório de backup se não existir"""
        try:
            self.backupDir.mkdir(parents=True, exist_ok=True)
        except Exception as e:
            self.logger.error(f"Error creating backup directory: {e}")
    
    def getStats(self) -> Dict[str, Any]:
        """
        Retorna estatísticas do state manager.
        
        Returns:
            Estatísticas detalhadas
        """
        with self.lock:
            backupCount = len(list(self.backupDir.glob("signal_state_backup_*.json")))
            
            return {
                **self.stats.copy(),
                "stateFile": {
                    "path": str(self.stateFilePath),
                    "exists": self.stateFilePath.exists(),
                    "lastModified": (
                        datetime.fromtimestamp(self.stateFilePath.stat().st_mtime).isoformat()
                        if self.stateFilePath.exists() else None
                    ),
                    "size": (
                        self.stateFilePath.stat().st_size
                        if self.stateFilePath.exists() else 0
                    )
                },
                "backup": {
                    "directory": str(self.backupDir),
                    "count": backupCount,
                    "lastSave": self.lastSaveTime.isoformat() if self.lastSaveTime else None,
                    "lastLoad": self.lastLoadTime.isoformat() if self.lastLoadTime else None
                },
                "cache": {
                    "hasCache": self.cachedState is not None,
                    "cacheSize": len(str(self.cachedState)) if self.cachedState else 0
                }
            }
    
    def forceBackup(self) -> bool:
        """
        Força criação de backup manual.
        
        Returns:
            True se backup criado
        """
        with self.lock:
            if not self.stateFilePath.exists():
                self.logger.warning("No state file to backup")
                return False
            
            return self._createBackup()
    
    def reset(self) -> bool:
        """
        Remove ficheiro de estado e backups (reset completo).
        
        Returns:
            True se reset com sucesso
        """
        with self.lock:
            try:
                # Remover ficheiro principal
                if self.stateFilePath.exists():
                    self.stateFilePath.unlink()
                
                # Remover backups
                for backupFile in self.backupDir.glob("signal_state_backup_*.json"):
                    backupFile.unlink()
                
                # Reset cache e estatísticas
                self.cachedState = None
                self.lastSaveTime = None
                self.lastLoadTime = None
                
                # Reset estatísticas (manter algumas para debugging)
                oldStats = self.stats.copy()
                self.stats = {
                    "saveOperations": 0,
                    "loadOperations": 0,
                    "backupsCreated": 0,
                    "recoveryOperations": 0,
                    "errors": 0,
                    "lastError": None,
                    "resetCount": oldStats.get("resetCount", 0) + 1,
                    "lastReset": datetime.now().isoformat()
                }
                
                self.logger.info("Signal state reset completed")
                return True
                
            except Exception as e:
                self.logger.error(f"Error during reset: {e}")
                return False

# Instância global
signalStateManager = SignalStateManager()