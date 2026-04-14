"""
生产级Agent系统

支持Provider动态切换与故障回退，基于LangChain/LangGraph编排，
具备长期记忆、短期记忆、异步/同步调用、MCP Tools集成、
流式输出、会话状态管理、权限控制、重试熔断、日志监控、插件扩展。

Provider支持:
- 本地模型: Ollama (llama3, mistral, qwen等)
- 国产模型: DeepSeek, Qwen, 智谱, Moonshot, StepFun, MiniMax, 百川等
- 云服务: OpenAI, Anthropic
- 自定义: 任何兼容OpenAI API的服务
"""

__version__ = "0.1.0"

from agent_system.core.config import (
    AgentConfig,
    AgentSystemConfig,
    CircuitBreakerConfig,
    LoggingConfig,
    MCPConfig,
    MemoryConfig,
    PermissionConfig,
    PluginConfig,
    ProviderConfig,
    RetryConfig,
)
from agent_system.core.exceptions import (
    AgentSystemError,
    CircuitBreakerOpenError,
    MaxIterationsExceededError,
    MCPError,
    MCPToolNotFoundError,
    MemoryError,
    MemoryNotFoundError,
    OrchestrationError,
    PermissionError,
    PluginError,
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderDisabledError,
    ProviderError,
    ProviderFallbackError,
    ProviderNotFoundError,
    ProviderRateLimitError,
    ProviderTimeoutError,
    SessionError,
    SessionExpiredError,
    SessionNotFoundError,
)
from agent_system.core.logging import get_logger, logger
from agent_system.memory.manager import MemoryManager
from agent_system.mcp.manager import MCPManager
from agent_system.orchestration.agent import Agent
from agent_system.orchestration.orchestrator import Orchestrator
from agent_system.orchestration.state import AgentContext, AgentState, AgentStep
from agent_system.plugins.manager import Plugin, PluginManager
from agent_system.providers.chinese_models import (
    CHINESE_MODELS,
    ChineseModelProvider,
)
from agent_system.providers.manager import (
    ProviderManager,
    get_registered_provider_types,
    register_provider_type,
)
from agent_system.providers.ollama_provider import OllamaProvider
from agent_system.providers.openai_compatible import OpenAICompatibleProvider
from agent_system.security.circuit_breaker import (
    CircuitBreakerManager,
    CircuitState,
    RetryManager,
)
from agent_system.security.permission import PermissionManager
from agent_system.session.manager import SessionManager, SessionState

__all__ = [
    "Agent",
    "AgentConfig",
    "AgentSystemConfig",
    "CircuitBreakerConfig",
    "LoggingConfig",
    "MCPConfig",
    "MemoryConfig",
    "PermissionConfig",
    "PluginConfig",
    "ProviderConfig",
    "RetryConfig",
    "AgentSystemError",
    "CircuitBreakerOpenError",
    "MaxIterationsExceededError",
    "MCPError",
    "MCPToolNotFoundError",
    "MemoryError",
    "MemoryNotFoundError",
    "OrchestrationError",
    "PermissionError",
    "PluginError",
    "ProviderAuthenticationError",
    "ProviderConnectionError",
    "ProviderDisabledError",
    "ProviderError",
    "ProviderFallbackError",
    "ProviderNotFoundError",
    "ProviderRateLimitError",
    "ProviderTimeoutError",
    "SessionError",
    "SessionExpiredError",
    "SessionNotFoundError",
    "get_logger",
    "logger",
    "MemoryManager",
    "MCPManager",
    "Orchestrator",
    "AgentContext",
    "AgentState",
    "AgentStep",
    "Plugin",
    "PluginManager",
    "ProviderManager",
    "register_provider_type",
    "get_registered_provider_types",
    "ChineseModelProvider",
    "CHINESE_MODELS",
    "OllamaProvider",
    "OpenAICompatibleProvider",
    "CircuitBreakerManager",
    "CircuitState",
    "RetryManager",
    "PermissionManager",
    "SessionManager",
    "SessionState",
]
