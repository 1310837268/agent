"""
生产级Agent系统完整示例

展示系统的所有核心功能：
1. Provider动态切换与故障回退
2. 记忆系统（短期/长期）
3. LangChain/LangGraph编排
4. MCP Tools集成
5. 流式输出
6. 会话状态管理
7. 权限控制
8. 重试熔断
9. 插件扩展
"""

import asyncio
import os
from typing import Any, Dict, List, Optional

from langchain_core.tools import tool

from agent_system.core.config import (
    AgentConfig,
    AgentSystemConfig,
    MemoryConfig,
    MCPConfig,
    PermissionConfig,
    PluginConfig,
    ProviderConfig,
)
from agent_system.core.logging import logger, LoggingConfig
from agent_system.memory.manager import MemoryManager
from agent_system.orchestration.agent import Agent
from agent_system.orchestration.orchestrator import Orchestrator
from agent_system.orchestration.state import AgentState
from agent_system.plugins.manager import PluginManager
from agent_system.providers.manager import ProviderManager
from agent_system.security.circuit_breaker import (
    CircuitBreakerConfig,
    CircuitBreakerManager,
    RetryConfig,
    RetryManager,
)
from agent_system.security.permission import PermissionManager
from agent_system.session.manager import SessionManager


@tool
def web_search(query: str) -> str:
    """
    搜索网络获取信息
    
    Args:
        query: 搜索查询
    """
    return f"[模拟搜索结果] 关于 '{query}' 的相关信息：这是一个模拟的搜索结果。"


@tool
def calculator(expression: str) -> str:
    """
    计算数学表达式
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4"
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果: {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


def create_provider_configs() -> List[ProviderConfig]:
    """创建Provider配置"""
    configs = []
    
    openai_api_key = os.getenv("OPENAI_API_KEY", "sk-xxx")
    configs.append(ProviderConfig(
        name="openai-primary",
        provider_type="openai",
        api_key=openai_api_key,
        model="gpt-4",
        max_tokens=4096,
        temperature=0.7,
        priority=100,
        enabled=True,
    ))
    
    configs.append(ProviderConfig(
        name="openai-secondary",
        provider_type="openai",
        api_key=openai_api_key,
        model="gpt-3.5-turbo",
        max_tokens=4096,
        temperature=0.7,
        priority=90,
        enabled=True,
    ))
    
    anthropic_api_key = os.getenv("ANTHROPIC_API_KEY", "sk-xxx")
    configs.append(ProviderConfig(
        name="anthropic",
        provider_type="anthropic",
        api_key=anthropic_api_key,
        model="claude-3-opus-20240229",
        max_tokens=4096,
        temperature=0.7,
        priority=80,
        enabled=True,
    ))
    
    return configs


def create_system_config() -> AgentSystemConfig:
    """创建系统配置"""
    return AgentSystemConfig(
        providers=create_provider_configs(),
        memory=MemoryConfig(
            short_term_type="in_memory",
            short_term_ttl=3600,
            long_term_type="in_memory",
            max_short_term_messages=100,
            max_long_term_memories=1000,
        ),
        mcp=MCPConfig(
            servers=[],
            auto_discover=False,
        ),
        circuit_breaker=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
        ),
        retry=RetryConfig(
            max_attempts=3,
            wait_exponential_multiplier=1,
            wait_exponential_max=10,
        ),
        permission=PermissionConfig(
            enabled=True,
            default_role="user",
            roles={
                "admin": ["*"],
                "user": ["read", "write", "execute_tools"],
                "guest": ["read"],
            },
        ),
        logging=LoggingConfig(
            level="INFO",
            format="json",
            enable_console=True,
        ),
        plugin=PluginConfig(
            plugin_dirs=[],
            auto_load=False,
        ),
        default_provider="openai-primary",
        enable_streaming=True,
        enable_async=True,
        session_ttl=86400,
    )


class AgentSystem:
    """
    生产级Agent系统
    
    整合所有核心组件
    """
    
    def __init__(self, config: Optional[AgentSystemConfig] = None) -> None:
        self.config = config or create_system_config()
        
        logger.init(self.config.logging)
        
        self.provider_manager = ProviderManager(
            configs=self.config.providers,
            default_provider=self.config.default_provider,
        )
        
        self.memory_manager = MemoryManager(self.config.memory)
        
        self.permission_manager = PermissionManager(self.config.permission)
        
        self.circuit_breaker_manager = CircuitBreakerManager(self.config.circuit_breaker)
        
        self.retry_manager = RetryManager(self.config.retry)
        
        self.plugin_manager = PluginManager(self.config.plugin)
        
        self.session_manager = SessionManager(self.config)
        
        self.orchestrator = Orchestrator(
            config=self.config,
            provider_manager=self.provider_manager,
            memory_manager=self.memory_manager,
        )
        
        self._agents: Dict[str, Agent] = {}
        
        logger.info("AgentSystem initialized")
    
    def create_agent(
        self,
        name: str,
        description: str = "",
        system_prompt: str = "",
        tools: Optional[List[Any]] = None,
        role: str = "user",
        **kwargs: Any,
    ) -> Agent:
        """创建Agent"""
        config = AgentConfig(
            name=name,
            description=description,
            system_prompt=system_prompt,
            role=role,
            **kwargs,
        )
        
        agent = Agent(
            config=config,
            provider_manager=self.provider_manager,
            memory_manager=self.memory_manager,
            tools=tools,
        )
        
        self._agents[name] = agent
        self.orchestrator.register_agent(agent)
        
        logger.info("Agent created", name=name, tools=[t.name for t in (tools or [])])
        
        return agent
    
    def get_agent(self, name: str) -> Optional[Agent]:
        """获取Agent"""
        return self._agents.get(name)
    
    def list_agents(self) -> List[str]:
        """列出所有Agent"""
        return list(self._agents.keys())
    
    async def execute(
        self,
        agent_name: str,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        执行Agent任务
        
        Args:
            agent_name: Agent名称
            user_input: 用户输入
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数
        
        Returns:
            执行结果
        """
        if session_id and not self.session_manager.session_exists(session_id):
            self.session_manager.create_session(user_id=user_id)
        
        context = await self.orchestrator.aexecute(
            user_input=user_input,
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )
        
        if session_id:
            self.session_manager.update_session(
                session_id=session_id,
                increment_message=True,
            )
        
        return {
            "success": context.state == AgentState.FINISHED,
            "state": context.state.value,
            "final_answer": context.final_answer,
            "session_id": context.session_id,
            "iterations": context.current_iteration,
            "tools_used": context.tools_used,
            "error": context.error,
        }
    
    async def stream_execute(
        self,
        agent_name: str,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ):
        """
        流式执行Agent任务
        
        Yields:
            流式事件
        """
        async for event in self.orchestrator.astream(
            user_input=user_input,
            agent_name=agent_name,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        ):
            yield event
    
    def add_long_term_memory(
        self,
        user_id: str,
        content: str,
        importance: float = 0.5,
    ) -> str:
        """添加长期记忆"""
        return self.memory_manager.add_long_term_memory(
            user_id=user_id,
            content=content,
            importance=importance,
        )
    
    def search_long_term_memory(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
    ) -> List[Dict[str, Any]]:
        """搜索长期记忆"""
        memories = self.memory_manager.search_long_term_memories(
            user_id=user_id,
            query=query,
            limit=limit,
        )
        return [m.model_dump() for m in memories]
    
    def check_permission(
        self,
        permission: str,
        user_role: Optional[str] = None,
    ) -> bool:
        """检查权限"""
        try:
            return self.permission_manager.check_permission(
                permission=permission,
                user_role=user_role,
                raise_error=True,
            )
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """获取系统状态"""
        return {
            "providers": self.provider_manager.get_provider_info(),
            "agents": self.list_agents(),
            "active_sessions": self.session_manager.get_session_count(),
            "permission_enabled": self.permission_manager.is_enabled(),
            "plugin_count": len(self.plugin_manager.get_all_plugins()),
        }


async def main():
    """主函数 - 演示系统功能"""
    print("=" * 60)
    print("生产级Agent系统演示")
    print("=" * 60)
    
    system = AgentSystem()
    
    print("\n1. 系统状态:")
    status = system.get_status()
    print(f"   - Providers: {status['providers']['default_provider']}")
    print(f"   - 活跃会话: {status['active_sessions']}")
    
    print("\n2. 创建Agent:")
    assistant_agent = system.create_agent(
        name="assistant",
        description="通用助手Agent",
        system_prompt="你是一个有帮助的AI助手。请用简洁明了的语言回答用户的问题。",
        tools=[web_search, calculator],
        role="user",
    )
    print(f"   - 创建了Agent: {assistant_agent.name}")
    print(f"   - 可用工具: {[t.name for t in assistant_agent.get_tools()]}")
    
    print("\n3. 权限控制演示:")
    print(f"   - admin角色 execute_tools 权限: {system.check_permission('execute_tools', 'admin')}")
    print(f"   - user角色 execute_tools 权限: {system.check_permission('execute_tools', 'user')}")
    print(f"   - guest角色 execute_tools 权限: {system.check_permission('execute_tools', 'guest')}")
    
    print("\n4. 记忆系统演示:")
    memory_id = system.add_long_term_memory(
        user_id="user_001",
        content="用户喜欢喝咖啡，特别是拿铁。",
        importance=0.8,
    )
    print(f"   - 添加长期记忆: {memory_id}")
    
    memories = system.search_long_term_memory(
        user_id="user_001",
        query="咖啡",
    )
    print(f"   - 搜索到的记忆: {len(memories)} 条")
    
    print("\n5. 会话管理演示:")
    session = system.session_manager.create_session(user_id="user_001")
    print(f"   - 创建会话: {session.session_id}")
    print(f"   - 活跃会话数: {system.session_manager.get_session_count()}")
    
    print("\n" + "=" * 60)
    print("系统演示完成！")
    print("=" * 60)
    
    print("\n系统架构概览:")
    print("""
┌─────────────────────────────────────────────────────────────┐
│                      Agent System                             │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │ Orchestrator│  │   Agent(s)  │  │  LangGraph Flow │    │
│  └──────┬──────┘  └──────┬──────┘  └────────┬────────┘    │
├─────────┼─────────────────┼───────────────────┼─────────────┤
│         │                 │                   │              │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼───────┐      │
│  │  Provider   │  │   Memory    │  │  MCP Tools    │      │
│  │   Manager   │  │   Manager   │  │   Manager     │      │
│  └──────┬──────┘  └──────┬──────┘  └───────┬───────┘      │
├─────────┼─────────────────┼───────────────────┼─────────────┤
│         │                 │                   │              │
│  ┌──────▼──────┐  ┌──────▼──────┐  ┌───────▼───────┐      │
│  │  OpenAI     │  │ Short Term  │  │  MCP Server   │      │
│  │  Anthropic  │  │ Long Term   │  │  (Filesystem, │      │
│  │  Azure...   │  │  (Vector DB)│  │   SQLite...)  │      │
│  └─────────────┘  └─────────────┘  └───────────────┘      │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────────┐    │
│  │  Security   │  │   Session   │  │    Plugins      │    │
│  │  (RBAC, CB) │  │   Manager   │  │   (Extensible)  │    │
│  └─────────────┘  └─────────────┘  └─────────────────┘    │
└─────────────────────────────────────────────────────────────┘
    """)


if __name__ == "__main__":
    asyncio.run(main())
