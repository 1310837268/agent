"""
核心异常模块
"""

from typing import Any, Dict, Optional


class AgentSystemError(Exception):
    """Agent系统基础异常"""
    
    def __init__(
        self,
        message: str,
        code: str = "UNKNOWN_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message)
        self.message = message
        self.code = code
        self.details = details or {}
    
    def __str__(self) -> str:
        return f"[{self.code}] {self.message}"
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "code": self.code,
            "message": self.message,
            "details": self.details,
        }


class ProviderError(AgentSystemError):
    """Provider相关异常"""
    
    def __init__(
        self,
        message: str,
        provider_name: Optional[str] = None,
        code: str = "PROVIDER_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.provider_name = provider_name


class ProviderNotFoundError(ProviderError):
    """Provider未找到异常"""
    
    def __init__(
        self,
        provider_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Provider '{provider_name}' not found",
            provider_name,
            "PROVIDER_NOT_FOUND",
            details,
        )


class ProviderDisabledError(ProviderError):
    """Provider已禁用异常"""
    
    def __init__(
        self,
        provider_name: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Provider '{provider_name}' is disabled",
            provider_name,
            "PROVIDER_DISABLED",
            details,
        )


class ProviderAuthenticationError(ProviderError):
    """Provider认证异常"""
    
    def __init__(
        self,
        provider_name: str,
        message: str = "Authentication failed",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            provider_name,
            "PROVIDER_AUTHENTICATION_ERROR",
            details,
        )


class ProviderRateLimitError(ProviderError):
    """Provider限流异常"""
    
    def __init__(
        self,
        provider_name: str,
        retry_after: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Rate limit exceeded for provider '{provider_name}'"
        if retry_after:
            message += f". Retry after {retry_after} seconds"
        super().__init__(
            message,
            provider_name,
            "PROVIDER_RATE_LIMIT_ERROR",
            {"retry_after": retry_after, **(details or {})},
        )
        self.retry_after = retry_after


class ProviderTimeoutError(ProviderError):
    """Provider超时异常"""
    
    def __init__(
        self,
        provider_name: str,
        timeout: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Timeout for provider '{provider_name}'"
        if timeout:
            message += f" (timeout={timeout}s)"
        super().__init__(
            message,
            provider_name,
            "PROVIDER_TIMEOUT_ERROR",
            {"timeout": timeout, **(details or {})},
        )


class ProviderConnectionError(ProviderError):
    """Provider连接异常"""
    
    def __init__(
        self,
        provider_name: str,
        message: str = "Connection error",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            provider_name,
            "PROVIDER_CONNECTION_ERROR",
            details,
        )


class ProviderFallbackError(ProviderError):
    """Provider回退异常"""
    
    def __init__(
        self,
        message: str = "All providers failed",
        failed_providers: Optional[Dict[str, str]] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            message,
            None,
            "PROVIDER_FALLBACK_ERROR",
            {"failed_providers": failed_providers or {}, **(details or {})},
        )
        self.failed_providers = failed_providers or {}


class MemoryError(AgentSystemError):
    """记忆系统异常"""
    
    def __init__(
        self,
        message: str,
        memory_type: Optional[str] = None,
        code: str = "MEMORY_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.memory_type = memory_type


class MemoryNotFoundError(MemoryError):
    """记忆未找到异常"""
    
    def __init__(
        self,
        memory_id: str,
        memory_type: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Memory '{memory_id}' not found",
            memory_type,
            "MEMORY_NOT_FOUND",
            {"memory_id": memory_id, **(details or {})},
        )


class MCPError(AgentSystemError):
    """MCP相关异常"""
    
    def __init__(
        self,
        message: str,
        server_name: Optional[str] = None,
        code: str = "MCP_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.server_name = server_name


class MCPToolNotFoundError(MCPError):
    """MCP工具未找到异常"""
    
    def __init__(
        self,
        tool_name: str,
        server_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"MCP tool '{tool_name}' not found",
            server_name,
            "MCP_TOOL_NOT_FOUND",
            {"tool_name": tool_name, **(details or {})},
        )


class PermissionError(AgentSystemError):
    """权限异常"""
    
    def __init__(
        self,
        message: str,
        required_permission: Optional[str] = None,
        user_role: Optional[str] = None,
        code: str = "PERMISSION_DENIED",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.required_permission = required_permission
        self.user_role = user_role


class CircuitBreakerOpenError(AgentSystemError):
    """熔断器打开异常"""
    
    def __init__(
        self,
        resource_name: str,
        recovery_time: Optional[int] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        message = f"Circuit breaker is open for resource '{resource_name}'"
        if recovery_time:
            message += f". Will attempt recovery in {recovery_time} seconds"
        super().__init__(
            message,
            "CIRCUIT_BREAKER_OPEN",
            {"resource_name": resource_name, "recovery_time": recovery_time, **(details or {})},
        )
        self.resource_name = resource_name
        self.recovery_time = recovery_time


class PluginError(AgentSystemError):
    """插件异常"""
    
    def __init__(
        self,
        message: str,
        plugin_name: Optional[str] = None,
        code: str = "PLUGIN_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.plugin_name = plugin_name


class SessionError(AgentSystemError):
    """会话异常"""
    
    def __init__(
        self,
        message: str,
        session_id: Optional[str] = None,
        code: str = "SESSION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.session_id = session_id


class SessionNotFoundError(SessionError):
    """会话未找到异常"""
    
    def __init__(
        self,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Session '{session_id}' not found",
            session_id,
            "SESSION_NOT_FOUND",
            details,
        )


class SessionExpiredError(SessionError):
    """会话过期异常"""
    
    def __init__(
        self,
        session_id: str,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Session '{session_id}' has expired",
            session_id,
            "SESSION_EXPIRED",
            details,
        )


class OrchestrationError(AgentSystemError):
    """编排异常"""
    
    def __init__(
        self,
        message: str,
        agent_name: Optional[str] = None,
        code: str = "ORCHESTRATION_ERROR",
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(message, code, details)
        self.agent_name = agent_name


class MaxIterationsExceededError(OrchestrationError):
    """最大迭代次数超出异常"""
    
    def __init__(
        self,
        max_iterations: int,
        agent_name: Optional[str] = None,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        super().__init__(
            f"Maximum iterations ({max_iterations}) exceeded",
            agent_name,
            "MAX_ITERATIONS_EXCEEDED",
            {"max_iterations": max_iterations, **(details or {})},
        )
        self.max_iterations = max_iterations
