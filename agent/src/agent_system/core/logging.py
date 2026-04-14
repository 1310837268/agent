"""
核心日志模块
"""

import json
import logging
import sys
from datetime import datetime
from typing import Any, Dict, Optional

import structlog
from pythonjsonlogger import jsonlogger

from agent_system.core.config import LoggingConfig


class StructuredLogger:
    """结构化日志器"""
    
    _instance: Optional["StructuredLogger"] = None
    _initialized: bool = False
    
    def __new__(cls) -> "StructuredLogger":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self) -> None:
        if self._initialized:
            return
        self._initialized = True
        self._logger: Optional[structlog.BoundLogger] = None
    
    def init(self, config: LoggingConfig) -> None:
        """初始化日志系统"""
        processors = [
            structlog.contextvars.merge_contextvars,
            structlog.processors.add_log_level,
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.TimeStamper(fmt="iso"),
        ]
        
        if config.format == "json":
            processors.append(structlog.processors.JSONRenderer())
        else:
            processors.append(structlog.dev.ConsoleRenderer())
        
        handlers = []
        
        if config.enable_console:
            console_handler = logging.StreamHandler(sys.stdout)
            if config.format == "json":
                console_handler.setFormatter(jsonlogger.JsonFormatter())
            handlers.append(console_handler)
        
        if config.enable_file and config.log_file:
            from logging.handlers import RotatingFileHandler
            file_handler = RotatingFileHandler(
                config.log_file,
                maxBytes=config.max_file_size,
                backupCount=config.backup_count,
            )
            file_handler.setFormatter(jsonlogger.JsonFormatter())
            handlers.append(file_handler)
        
        logging.basicConfig(
            level=getattr(logging, config.level.upper()),
            handlers=handlers,
            force=True,
        )
        
        structlog.configure(
            processors=processors,
            wrapper_class=structlog.make_filtering_bound_logger(
                getattr(logging, config.level.upper())
            ),
            context_class=dict,
            logger_factory=structlog.stdlib.LoggerFactory(),
            cache_logger_on_first_use=True,
        )
        
        self._logger = structlog.get_logger()
    
    def get_logger(self, name: Optional[str] = None) -> structlog.BoundLogger:
        """获取日志器"""
        if self._logger is None:
            self.init(LoggingConfig())
        if name:
            return self._logger.bind(module=name)
        return self._logger
    
    def debug(self, message: str, **kwargs: Any) -> None:
        self.get_logger().debug(message, **kwargs)
    
    def info(self, message: str, **kwargs: Any) -> None:
        self.get_logger().info(message, **kwargs)
    
    def warning(self, message: str, **kwargs: Any) -> None:
        self.get_logger().warning(message, **kwargs)
    
    def error(self, message: str, **kwargs: Any) -> None:
        self.get_logger().error(message, **kwargs)
    
    def critical(self, message: str, **kwargs: Any) -> None:
        self.get_logger().critical(message, **kwargs)
    
    def exception(self, message: str, **kwargs: Any) -> None:
        self.get_logger().exception(message, **kwargs)


logger = StructuredLogger()


def get_logger(name: Optional[str] = None) -> structlog.BoundLogger:
    """获取日志器的便捷函数"""
    return logger.get_logger(name)
