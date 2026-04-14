"""
核心配置模块
"""

from typing import Any, Dict, List, Optional
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class ProviderConfig(BaseModel):
    """Provider配置"""
    name: str
    provider_type: str  # openai, anthropic, azure, etc.
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    model: str = "gpt-4"
    max_tokens: int = 4096
    temperature: float = 0.7
    timeout: int = 60
    max_retries: int = 3
    priority: int = 100
    enabled: bool = True
    extra_params: Dict[str, Any] = Field(default_factory=dict)


class MemoryConfig(BaseModel):
    """记忆系统配置"""
    short_term_type: str = "in_memory"  # in_memory, redis
    short_term_ttl: int = 3600  # 1小时
    long_term_type: str = "sqlalchemy"  # sqlalchemy, mongodb
    connection_string: Optional[str] = None
    embedding_model: str = "text-embedding-ada-002"
    embedding_provider: str = "openai"
    max_short_term_messages: int = 100
    max_long_term_memories: int = 1000


class MCPConfig(BaseModel):
    """MCP配置"""
    servers: List[Dict[str, Any]] = Field(default_factory=list)
    auto_discover: bool = True
    discovery_paths: List[str] = Field(default_factory=list)


class CircuitBreakerConfig(BaseModel):
    """熔断器配置"""
    failure_threshold: int = 5
    recovery_timeout: int = 30
    expected_exception_types: List[str] = Field(
        default_factory=lambda: [
            "openai.APIError",
            "openai.APIConnectionError",
            "openai.RateLimitError",
            "anthropic.APIError",
            "aiohttp.ClientError",
            "TimeoutError",
        ]
    )


class RetryConfig(BaseModel):
    """重试配置"""
    max_attempts: int = 3
    wait_exponential_multiplier: int = 1
    wait_exponential_max: int = 10
    retry_on_exceptions: List[str] = Field(
        default_factory=lambda: [
            "openai.APIConnectionError",
            "openai.RateLimitError",
            "anthropic.RateLimitError",
            "aiohttp.ClientError",
            "TimeoutError",
        ]
    )


class PermissionConfig(BaseModel):
    """权限配置"""
    enabled: bool = True
    default_role: str = "user"
    roles: Dict[str, List[str]] = Field(
        default_factory=lambda: {
            "admin": ["*"],
            "user": ["read", "write", "execute_tools"],
            "guest": ["read"],
        }
    )


class LoggingConfig(BaseModel):
    """日志配置"""
    level: str = "INFO"
    format: str = "json"
    log_file: Optional[str] = None
    enable_console: bool = True
    enable_file: bool = False
    max_file_size: int = 10485760  # 10MB
    backup_count: int = 5


class PluginConfig(BaseModel):
    """插件配置"""
    plugin_dirs: List[str] = Field(default_factory=list)
    auto_load: bool = True
    enabled_plugins: List[str] = Field(default_factory=list)
    disabled_plugins: List[str] = Field(default_factory=list)


class AgentSystemConfig(BaseSettings):
    """Agent系统主配置"""
    providers: List[ProviderConfig] = Field(default_factory=list)
    memory: MemoryConfig = Field(default_factory=MemoryConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)
    circuit_breaker: CircuitBreakerConfig = Field(default_factory=CircuitBreakerConfig)
    retry: RetryConfig = Field(default_factory=RetryConfig)
    permission: PermissionConfig = Field(default_factory=PermissionConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)
    plugin: PluginConfig = Field(default_factory=PluginConfig)
    
    default_provider: str = "openai"
    enable_streaming: bool = True
    enable_async: bool = True
    session_ttl: int = 86400  # 24小时
    
    class Config:
        env_prefix = "AGENT_"
        env_nested_delimiter = "__"
        case_sensitive = False


class AgentConfig(BaseModel):
    """单个Agent配置"""
    name: str
    description: str = ""
    system_prompt: str = ""
    provider: Optional[str] = None
    model: Optional[str] = None
    temperature: float = 0.7
    max_tokens: int = 4096
    tools: List[str] = Field(default_factory=list)
    plugins: List[str] = Field(default_factory=list)
    enable_memory: bool = True
    enable_streaming: bool = True
    max_iterations: int = 10
    role: str = "user"
    extra_config: Dict[str, Any] = Field(default_factory=dict)
