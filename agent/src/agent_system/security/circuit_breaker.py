"""
重试与熔断机制模块
"""

import asyncio
import functools
import time
from abc import ABC, abstractmethod
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Type, Union

from agent_system.core.config import CircuitBreakerConfig, RetryConfig
from agent_system.core.exceptions import CircuitBreakerOpenError
from agent_system.core.logging import get_logger

logger = get_logger(__name__)


class CircuitState(str, Enum):
    """熔断器状态"""
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitBreaker(ABC):
    """熔断器抽象类"""
    
    @abstractmethod
    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """执行函数"""
        pass
    
    @abstractmethod
    async def acall(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """异步执行函数"""
        pass
    
    @abstractmethod
    def get_state(self) -> CircuitState:
        """获取当前状态"""
        pass
    
    @abstractmethod
    def record_success(self) -> None:
        """记录成功"""
        pass
    
    @abstractmethod
    def record_failure(self, exception: Exception) -> None:
        """记录失败"""
        pass


class InMemoryCircuitBreaker(CircuitBreaker):
    """内存熔断器实现"""
    
    def __init__(
        self,
        name: str,
        config: CircuitBreakerConfig,
    ) -> None:
        self.name = name
        self.config = config
        self._state: CircuitState = CircuitState.CLOSED
        self._failure_count: int = 0
        self._last_failure_time: Optional[datetime] = None
        self._open_time: Optional[datetime] = None
    
    def _is_exception_expected(self, exception: Exception) -> bool:
        """检查异常是否在预期列表中"""
        exception_type = f"{type(exception).__module__}.{type(exception).__name__}"
        
        for expected_type in self.config.expected_exception_types:
            if exception_type == expected_type or exception_type.endswith(expected_type):
                return True
        
        return False
    
    def _check_state(self) -> None:
        """检查并更新状态"""
        if self._state == CircuitState.OPEN:
            if self._open_time and (datetime.now() - self._open_time).total_seconds() >= self.config.recovery_timeout:
                self._state = CircuitState.HALF_OPEN
                logger.info(
                    "Circuit breaker transitioning to HALF_OPEN",
                    circuit=self.name,
                )
    
    def _can_execute(self) -> bool:
        """检查是否可以执行"""
        self._check_state()
        
        if self._state == CircuitState.OPEN:
            return False
        
        return True
    
    def call(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """执行函数"""
        if not self._can_execute():
            recovery_time = None
            if self._open_time:
                elapsed = (datetime.now() - self._open_time).total_seconds()
                recovery_time = max(0, int(self.config.recovery_timeout - elapsed))
            
            raise CircuitBreakerOpenError(self.name, recovery_time)
        
        try:
            result = func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            if self._is_exception_expected(e):
                self.record_failure(e)
            raise
    
    async def acall(self, func: Callable, *args: Any, **kwargs: Any) -> Any:
        """异步执行函数"""
        if not self._can_execute():
            recovery_time = None
            if self._open_time:
                elapsed = (datetime.now() - self._open_time).total_seconds()
                recovery_time = max(0, int(self.config.recovery_timeout - elapsed))
            
            raise CircuitBreakerOpenError(self.name, recovery_time)
        
        try:
            result = await func(*args, **kwargs)
            self.record_success()
            return result
        except Exception as e:
            if self._is_exception_expected(e):
                self.record_failure(e)
            raise
    
    def get_state(self) -> CircuitState:
        """获取当前状态"""
        self._check_state()
        return self._state
    
    def record_success(self) -> None:
        """记录成功"""
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.CLOSED
            self._failure_count = 0
            logger.info(
                "Circuit breaker transitioning to CLOSED after success",
                circuit=self.name,
            )
        elif self._state == CircuitState.CLOSED:
            self._failure_count = 0
    
    def record_failure(self, exception: Exception) -> None:
        """记录失败"""
        self._failure_count += 1
        self._last_failure_time = datetime.now()
        
        logger.warning(
            "Circuit breaker recorded failure",
            circuit=self.name,
            failure_count=self._failure_count,
            threshold=self.config.failure_threshold,
            error=str(exception),
        )
        
        if self._state == CircuitState.HALF_OPEN:
            self._state = CircuitState.OPEN
            self._open_time = datetime.now()
            logger.error(
                "Circuit breaker OPENED after HALF_OPEN failure",
                circuit=self.name,
            )
        elif self._failure_count >= self.config.failure_threshold:
            self._state = CircuitState.OPEN
            self._open_time = datetime.now()
            logger.error(
                "Circuit breaker OPENED",
                circuit=self.name,
                failure_count=self._failure_count,
            )


class CircuitBreakerManager:
    """
    熔断器管理器
    
    管理多个熔断器实例
    """
    
    def __init__(self, config: CircuitBreakerConfig) -> None:
        self.config = config
        self._circuits: Dict[str, CircuitBreaker] = {}
    
    def get_circuit(self, name: str) -> CircuitBreaker:
        """获取熔断器"""
        if name not in self._circuits:
            self._circuits[name] = InMemoryCircuitBreaker(name, self.config)
        return self._circuits[name]
    
    def protect(self, name: str) -> Callable:
        """
        熔断器装饰器
        
        Usage:
            @circuit_manager.protect("my_service")
            def my_function():
                pass
        """
        def decorator(func: Callable) -> Callable:
            circuit = self.get_circuit(name)
            
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                return circuit.call(func, *args, **kwargs)
            
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                return await circuit.acall(func, *args, **kwargs)
            
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        
        return decorator
    
    def get_all_states(self) -> Dict[str, CircuitState]:
        """获取所有熔断器状态"""
        return {name: circuit.get_state() for name, circuit in self._circuits.items()}
    
    def reset_circuit(self, name: str) -> bool:
        """重置熔断器"""
        if name in self._circuits:
            del self._circuits[name]
            logger.info("Circuit breaker reset", circuit=name)
            return True
        return False


class RetryManager:
    """
    重试管理器
    
    提供重试装饰器和函数
    """
    
    def __init__(self, config: RetryConfig) -> None:
        self.config = config
    
    def _is_retryable_exception(self, exception: Exception) -> bool:
        """检查异常是否可重试"""
        exception_type = f"{type(exception).__module__}.{type(exception).__name__}"
        
        for retry_type in self.config.retry_on_exceptions:
            if exception_type == retry_type or exception_type.endswith(retry_type):
                return True
        
        return False
    
    def retry(
        self,
        max_attempts: Optional[int] = None,
        wait_exponential_multiplier: Optional[int] = None,
        wait_exponential_max: Optional[int] = None,
    ) -> Callable:
        """
        重试装饰器
        
        Usage:
            @retry_manager.retry()
            def my_function():
                pass
        """
        attempts = max_attempts or self.config.max_attempts
        multiplier = wait_exponential_multiplier or self.config.wait_exponential_multiplier
        max_wait = wait_exponential_max or self.config.wait_exponential_max
        
        def decorator(func: Callable) -> Callable:
            @functools.wraps(func)
            def wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Optional[Exception] = None
                
                for attempt in range(attempts):
                    try:
                        return func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if not self._is_retryable_exception(e):
                            logger.warning(
                                "Exception not retryable, not retrying",
                                function=func.__name__,
                                attempt=attempt + 1,
                                error=str(e),
                            )
                            raise
                        
                        if attempt < attempts - 1:
                            wait_time = min(multiplier * (2 ** attempt), max_wait)
                            logger.warning(
                                "Retrying function",
                                function=func.__name__,
                                attempt=attempt + 1,
                                wait_time=wait_time,
                                error=str(e),
                            )
                            time.sleep(wait_time)
                
                if last_exception:
                    logger.error(
                        "All retry attempts failed",
                        function=func.__name__,
                        attempts=attempts,
                    )
                    raise last_exception
            
            @functools.wraps(func)
            async def async_wrapper(*args: Any, **kwargs: Any) -> Any:
                last_exception: Optional[Exception] = None
                
                for attempt in range(attempts):
                    try:
                        return await func(*args, **kwargs)
                    except Exception as e:
                        last_exception = e
                        
                        if not self._is_retryable_exception(e):
                            logger.warning(
                                "Exception not retryable, not retrying",
                                function=func.__name__,
                                attempt=attempt + 1,
                                error=str(e),
                            )
                            raise
                        
                        if attempt < attempts - 1:
                            wait_time = min(multiplier * (2 ** attempt), max_wait)
                            logger.warning(
                                "Retrying async function",
                                function=func.__name__,
                                attempt=attempt + 1,
                                wait_time=wait_time,
                                error=str(e),
                            )
                            await asyncio.sleep(wait_time)
                
                if last_exception:
                    logger.error(
                        "All async retry attempts failed",
                        function=func.__name__,
                        attempts=attempts,
                    )
                    raise last_exception
            
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        
        return decorator
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args: Any,
        max_attempts: Optional[int] = None,
        **kwargs: Any,
    ) -> Any:
        """
        带重试的执行函数
        """
        attempts = max_attempts or self.config.max_attempts
        multiplier = self.config.wait_exponential_multiplier
        max_wait = self.config.wait_exponential_max
        
        last_exception: Optional[Exception] = None
        
        for attempt in range(attempts):
            try:
                import inspect
                if inspect.iscoroutinefunction(func):
                    return await func(*args, **kwargs)
                else:
                    return func(*args, **kwargs)
            except Exception as e:
                last_exception = e
                
                if not self._is_retryable_exception(e):
                    raise
                
                if attempt < attempts - 1:
                    wait_time = min(multiplier * (2 ** attempt), max_wait)
                    logger.warning(
                        "Retrying execution",
                        attempt=attempt + 1,
                        wait_time=wait_time,
                        error=str(e),
                    )
                    await asyncio.sleep(wait_time)
        
        if last_exception:
            raise last_exception
