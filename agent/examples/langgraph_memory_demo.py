"""
LangGraph记忆系统完整示例

展示如何使用LangGraph原生的记忆机制：
1. MemorySaver - 状态持久化
2. Checkpointer - 检查点机制
3. StateGraph - 状态图管理
4. 短期记忆和长期记忆的实现
"""

import asyncio
import os
from typing import List, Optional

from langchain_core.tools import tool

from agent_system.core.config import AgentConfig, ProviderConfig
from agent_system.memory.langgraph_memory import (
    InMemoryLangGraphMemory,
    SqliteLangGraphMemory,
    create_langgraph_memory,
)
from agent_system.orchestration.langgraph_agent import LangGraphAgent
from agent_system.providers.manager import ProviderManager


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


def create_sample_provider_manager() -> ProviderManager:
    """创建示例Provider管理器"""
    configs = [
        ProviderConfig(
            name="openai",
            provider_type="openai",
            api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
            model="gpt-3.5-turbo",
            priority=100,
            enabled=True,
        ),
    ]
    return ProviderManager(configs=configs)


async def demonstrate_in_memory_memory():
    """
    演示内存记忆
    
    使用MemorySaver进行状态持久化
    适合开发和测试环境
    """
    print("\n" + "=" * 60)
    print("1. 内存记忆演示 (MemorySaver)")
    print("=" * 60)
    
    provider_manager = create_sample_provider_manager()
    
    config = AgentConfig(
        name="memory-agent",
        description="具有记忆能力的Agent",
        system_prompt="你是一个有帮助的AI助手。请记住用户的偏好和历史对话。",
        max_iterations=10,
    )
    
    agent = LangGraphAgent(
        config=config,
        provider_manager=provider_manager,
        tools=[web_search, calculator],
        memory_type="memory",
    )
    
    print("\n创建了LangGraphAgent，使用内存记忆")
    print(f"Agent名称: {agent.name}")
    print(f"可用工具: {[t.name for t in agent.get_tools()]}")
    
    thread_id = "conversation-001"
    
    print(f"\n使用线程ID: {thread_id}")
    print("-" * 40)
    
    print("\n第一次对话:")
    print("用户: 你好，我叫张三，我喜欢喝咖啡。")
    
    result1 = agent.invoke(
        user_input="你好，我叫张三，我喜欢喝咖啡。",
        thread_id=thread_id,
    )
    print(f"AI: {result1['final_answer'][:100]}...")
    
    print("\n第二次对话 (使用相同的thread_id，AI应该记得用户信息):")
    print("用户: 你还记得我叫什么名字吗？我喜欢喝什么？")
    
    result2 = agent.invoke(
        user_input="你还记得我叫什么名字吗？我喜欢喝什么？",
        thread_id=thread_id,
    )
    print(f"AI: {result2['final_answer'][:100]}...")
    
    print("\n获取线程历史:")
    history = agent.get_thread_history(thread_id)
    if history:
        print(f"  消息数量: {len(history['messages'])}")
        print(f"  执行步骤: {history['current_step']}")
    
    print("\n列出所有线程:")
    threads = agent.list_threads()
    print(f"  线程数量: {len(threads)}")
    
    print("\n删除线程:")
    deleted = agent.delete_thread(thread_id)
    print(f"  删除成功: {deleted}")


async def demonstrate_sqlite_memory():
    """
    演示SQLite记忆
    
    使用SqliteSaver进行状态持久化
    适合生产环境的单机部署
    """
    print("\n" + "=" * 60)
    print("2. SQLite记忆演示 (SqliteSaver)")
    print("=" * 60)
    
    provider_manager = create_sample_provider_manager()
    
    db_path = "example_checkpoints.sqlite"
    
    print(f"\n使用SQLite数据库: {db_path}")
    
    memory = SqliteLangGraphMemory(db_path=db_path)
    
    config = AgentConfig(
        name="sqlite-agent",
        description="使用SQLite持久化记忆的Agent",
        system_prompt="你是一个有帮助的AI助手。",
        max_iterations=10,
    )
    
    agent = LangGraphAgent(
        config=config,
        provider_manager=provider_manager,
        memory=memory,
    )
    
    print("\n创建了使用SQLite记忆的LangGraphAgent")
    
    thread_id = "persistent-conversation-001"
    
    print(f"\n使用线程ID: {thread_id}")
    print("-" * 40)
    
    print("\n第一次对话:")
    print("用户: 请记住，我的邮箱是 test@example.com")
    
    result1 = agent.invoke(
        user_input="请记住，我的邮箱是 test@example.com",
        thread_id=thread_id,
    )
    print(f"AI: {result1['final_answer'][:100]}...")
    
    print("\n第二次对话 (重启程序后仍然可以访问):")
    print("用户: 我的邮箱是什么？")
    
    result2 = agent.invoke(
        user_input="我的邮箱是什么？",
        thread_id=thread_id,
    )
    print(f"AI: {result2['final_answer'][:100]}...")
    
    print("\nSQLite记忆的优势:")
    print("  1. 数据持久化 - 程序重启后仍然存在")
    print("  2. 单机部署 - 适合单服务器场景")
    print("  3. 易于备份 - 直接复制数据库文件")
    
    if os.path.exists(db_path):
        os.remove(db_path)
        print(f"\n已清理示例数据库: {db_path}")


async def demonstrate_memory_types():
    """
    演示不同类型的记忆
    """
    print("\n" + "=" * 60)
    print("3. 记忆类型汇总")
    print("=" * 60)
    
    print("""
LangGraph支持的记忆类型:

1. 内存记忆 (InMemoryLangGraphMemory)
   - 使用: MemorySaver
   - 适用: 开发、测试、临时会话
   - 特点: 快速，程序重启后丢失

2. SQLite记忆 (SqliteLangGraphMemory)
   - 使用: SqliteSaver
   - 适用: 生产环境单机部署
   - 特点: 持久化，易于备份
   - 安装: pip install langgraph-checkpoint-sqlite

3. Redis记忆 (RedisLangGraphMemory)
   - 使用: RedisSaver
   - 适用: 生产环境分布式部署
   - 特点: 高性能，支持多实例
   - 安装: pip install langgraph-checkpoint-redis

4. PostgreSQL记忆 (PostgresLangGraphMemory)
   - 使用: PostgresSaver
   - 适用: 生产环境大规模部署
   - 特点: 企业级，支持复杂查询
   - 安装: pip install langgraph-checkpoint-postgres
""")
    
    print("使用工厂函数创建记忆:")
    print("""
   # 内存记忆
   memory = create_langgraph_memory("memory")
   
   # SQLite记忆
   memory = create_langgraph_memory("sqlite", db_path="checkpoints.sqlite")
   
   # Redis记忆
   memory = create_langgraph_memory("redis", redis_url="redis://localhost:6379")
   
   # PostgreSQL记忆
   memory = create_langgraph_memory("postgres", postgres_url="postgresql://user:pass@localhost/db")
""")


async def demonstrate_langgraph_vs_traditional():
    """
    对比LangGraph记忆和传统记忆
    """
    print("\n" + "=" * 60)
    print("4. LangGraph记忆 vs 传统记忆")
    print("=" * 60)
    
    print("""
传统记忆系统 (之前实现的):
┌─────────────────────────────────────────────────────────────┐
│  ShortTermMemory (短期记忆)                                  │
│  - 存储当前会话的消息历史                                      │
│  - 实现: 内存或Redis                                          │
│  - 特点: 手动管理，需要自己实现持久化逻辑                        │
├─────────────────────────────────────────────────────────────┤
│  LongTermMemory (长期记忆)                                   │
│  - 存储用户偏好、历史知识                                      │
│  - 实现: SQL或向量数据库                                       │
│  - 特点: 需要自己实现检索和存储逻辑                             │
└─────────────────────────────────────────────────────────────┘

LangGraph记忆系统 (新实现的):
┌─────────────────────────────────────────────────────────────┐
│  Checkpointer (检查点)                                       │
│  - 自动持久化整个状态图                                        │
│  - 实现: MemorySaver, SqliteSaver, RedisSaver, PostgresSaver│
│  - 特点: LangGraph原生支持，自动管理状态                        │
├─────────────────────────────────────────────────────────────┤
│  StateGraph (状态图)                                         │
│  - 定义状态结构和流转逻辑                                      │
│  - 包含: messages, metadata, current_step等                   │
│  - 特点: 结构化，支持复杂工作流                                  │
├─────────────────────────────────────────────────────────────┤
│  Thread (线程)                                                │
│  - 相当于会话ID                                                │
│  - 用于隔离不同的对话/用户                                      │
│  - 特点: 自动关联状态历史                                        │
└─────────────────────────────────────────────────────────────┘

LangGraph记忆的优势:
1. 原生集成 - LangGraph自动管理状态持久化
2. 灵活选择 - 支持多种后端存储
3. 状态完整 - 保存整个状态图，不只是消息
4. 可恢复 - 支持从检查点恢复执行
5. 可调试 - 可以查看和分析历史状态
""")


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("LangGraph记忆系统完整示例")
    print("=" * 60)
    
    await demonstrate_in_memory_memory()
    await demonstrate_sqlite_memory()
    await demonstrate_memory_types()
    await demonstrate_langgraph_vs_traditional()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    
    print("""
快速开始:

1. 使用内存记忆 (开发测试):
   ```python
   agent = LangGraphAgent(
       config=config,
       provider_manager=provider_manager,
       memory_type="memory",
   )
   ```

2. 使用SQLite记忆 (生产单机):
   ```python
   agent = LangGraphAgent(
       config=config,
       provider_manager=provider_manager,
       memory_type="sqlite",
       db_path="checkpoints.sqlite",
   )
   ```

3. 使用相同的thread_id保持对话记忆:
   ```python
   # 第一次对话
   result1 = agent.invoke(
       user_input="我叫张三",
       thread_id="conversation-001",
   )
   
   # 第二次对话 (AI会记得用户信息)
   result2 = agent.invoke(
       user_input="我叫什么名字？",
       thread_id="conversation-001",
   )
   ```

4. 获取历史记录:
   ```python
   history = agent.get_thread_history("conversation-001")
   ```
""")


if __name__ == "__main__":
    asyncio.run(main())
