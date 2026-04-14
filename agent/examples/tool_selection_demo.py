"""
工具选择与MCP集成完整示例

展示：
1. 智能工具选择（ToolSelector）
2. MCP工具集成
3. 工具调用决策闭环
4. LangGraph状态图执行
"""

import asyncio
import os
from typing import List, Optional

from langchain_core.tools import tool

from agent_system.core.config import AgentConfig, ProviderConfig
from agent_system.orchestration.langgraph_agent_enhanced import EnhancedLangGraphAgent
from agent_system.providers.manager import ProviderManager
from agent_system.tools.selector import (
    HybridToolSelector,
    ToolSelectionResult,
    create_tool_selector,
)


@tool
def web_search(query: str) -> str:
    """
    搜索网络获取最新信息
    
    当用户询问实时信息、新闻、天气、或需要外部知识时使用此工具。
    
    Args:
        query: 搜索查询，应该是简洁的关键词或问题
    """
    return f"[搜索结果] 关于 '{query}' 的最新信息：这是一个模拟的搜索结果，包含了相关的网页摘要和链接。"


@tool
def calculator(expression: str) -> str:
    """
    执行数学计算
    
    当用户需要进行数学运算时使用此工具。
    支持基本运算：加(+)、减(-)、乘(*)、除(/)、幂(**)等。
    
    Args:
        expression: 数学表达式，如 "2 + 3 * 4" 或 "(10 + 5) / 3"
    """
    try:
        result = eval(expression, {"__builtins__": {}})
        return f"计算结果: {expression} = {result}"
    except Exception as e:
        return f"计算错误: {str(e)}"


@tool
def get_current_time() -> str:
    """
    获取当前时间和日期
    
    当用户询问现在几点、今天是几号、星期几等时间相关问题时使用此工具。
    """
    from datetime import datetime
    now = datetime.now()
    return f"当前时间: {now.strftime('%Y-%m-%d %H:%M:%S')}，星期{['一','二','三','四','五','六','日'][now.weekday()]}"


@tool
def weather_forecast(location: str, days: int = 1) -> str:
    """
    获取天气预报
    
    当用户询问天气情况时使用此工具。
    
    Args:
        location: 城市名称，如 "北京"、"上海"、"New York"
        days: 预报天数，默认为1天（今天），最多支持7天
    """
    return f"[天气预报] {location} 未来{days}天的天气预报：\n- 今天：晴，25°C\n- 明天：多云，23°C\n- 后天：小雨，20°C"


@tool
def translate_text(text: str, target_language: str) -> str:
    """
    翻译文本
    
    当用户需要翻译文字时使用此工具。
    
    Args:
        text: 需要翻译的文本
        target_language: 目标语言，如 "中文"、"英文"、"日语"、"法语"
    """
    translations = {
        "中文": "这是翻译后的中文文本",
        "英文": "This is the translated English text",
        "日语": "これは翻訳された日本語のテキストです",
    }
    return f"翻译结果 ({target_language}): {translations.get(target_language, f'[{target_language}] {text}')}"


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


async def demonstrate_tool_selection():
    """
    演示工具选择功能
    """
    print("\n" + "=" * 60)
    print("1. 工具选择演示")
    print("=" * 60)
    
    provider_manager = create_sample_provider_manager()
    
    tools = [web_search, calculator, get_current_time, weather_forecast, translate_text]
    
    print("\n可用工具:")
    for tool in tools:
        print(f"  - {tool.name}: {tool.description[:50]}...")
    
    selector = HybridToolSelector(
        provider_manager=provider_manager,
    )
    
    test_cases = [
        ("2 + 3 * 4 等于多少？", "应该选择 calculator"),
        ("现在几点了？", "应该选择 get_current_time"),
        ("北京明天的天气怎么样？", "应该选择 weather_forecast"),
        ("最新的AI新闻有哪些？", "应该选择 web_search"),
        ("你好", "不应该选择任何工具"),
    ]
    
    print("\n工具选择测试:")
    print("-" * 40)
    
    for user_input, expected in test_cases:
        print(f"\n用户输入: {user_input}")
        print(f"预期: {expected}")
        
        result = selector.select_tools(
            user_input=user_input,
            available_tools=tools,
        )
        
        print(f"选择结果:")
        print(f"  - 应该调用工具: {result.should_call_tools}")
        print(f"  - 置信度: {result.confidence}")
        print(f"  - 推理: {result.reasoning[:100]}...")
        
        if result.selected_tools:
            print(f"  - 选中的工具:")
            for tool in result.selected_tools:
                print(f"    * {tool.get('name')}: {tool.get('reason')}")


async def demonstrate_enhanced_agent():
    """
    演示增强版Agent
    """
    print("\n" + "=" * 60)
    print("2. 增强版Agent演示")
    print("=" * 60)
    
    provider_manager = create_sample_provider_manager()
    
    tools = [web_search, calculator, get_current_time, weather_forecast, translate_text]
    
    config = AgentConfig(
        name="tool-agent",
        description="具有工具选择能力的智能Agent",
        system_prompt="你是一个有帮助的AI助手。当用户需要外部信息或计算时，使用合适的工具。",
        max_iterations=10,
    )
    
    agent = EnhancedLangGraphAgent(
        config=config,
        provider_manager=provider_manager,
        tools=tools,
        memory_type="memory",
    )
    
    print(f"\n创建了增强版Agent: {agent.name}")
    print(f"可用工具: {agent.get_tool_names()}")
    print(f"工具数量: {agent.get_tool_count()}")
    
    print("\nAgent状态图结构:")
    print("""
    START -> tool_selection -> agent -> tool_execution -> agent -> ... -> END
                    |                                    |
                    +---------- should_continue ---------+
    """)
    
    print("\n工具选择节点 (tool_selection):")
    print("  - 分析用户意图")
    print("  - 匹配可用工具")
    print("  - 输出选择结果和推理")
    
    print("\nAgent节点 (agent):")
    print("  - 接收工具选择建议")
    print("  - 调用LLM生成响应")
    print("  - 决定是否调用工具")
    
    print("\n工具执行节点 (tool_execution):")
    print("  - 执行工具调用")
    print("  - 处理工具结果")
    print("  - 返回结果给Agent")


async def demonstrate_tool_decision_cycle():
    """
    演示工具决策闭环
    """
    print("\n" + "=" * 60)
    print("3. 工具决策闭环演示")
    print("=" * 60)
    
    print("""
工具决策闭环流程:

┌─────────────────────────────────────────────────────────────┐
│                    用户输入                                  │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              1. 工具选择 (ToolSelector)                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  a. 分析用户意图                                      │    │
│  │  b. 匹配工具描述                                      │    │
│  │  c. 评估是否需要工具                                   │    │
│  │  d. 输出: 选中的工具 + 推理 + 置信度                    │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
                          ▼
┌─────────────────────────────────────────────────────────────┐
│              2. LLM决策 (Agent Node)                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │  a. 接收工具选择建议                                   │    │
│  │  b. 结合对话历史                                       │    │
│  │  c. 生成响应或工具调用                                  │    │
│  │  d. 输出: 消息内容 + 工具调用(可选)                     │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────┬───────────────────────────────────┘
                          │
              ┌───────────┴───────────┐
              │                       │
              ▼                       ▼
    ┌─────────────────┐    ┌─────────────────┐
    │  有工具调用?      │    │   无工具调用?    │
    └────────┬────────┘    └────────┬────────┘
             │                       │
             ▼                       ▼
┌─────────────────────────┐  ┌─────────────────────────┐
│  3. 工具执行             │  │  4. 直接回答            │
│  ┌───────────────────┐  │  │  - 返回最终答案        │
│  │  a. 解析工具调用    │  │  │  - 结束流程           │
│  │  b. 执行工具        │  │  └───────────────────────┘
│  │  c. 处理结果        │  │
│  │  d. 输出: 工具结果   │  │
│  └───────────────────┘  │
└───────────┬─────────────┘
            │
            ▼
    ┌─────────────────┐
    │  返回Agent节点   │
    │  (循环继续)      │
    └─────────────────┘
""")
    
    print("\n关键决策点:")
    print("  1. 是否需要调用工具？")
    print("     - 是: 进入工具执行流程")
    print("     - 否: 直接回答用户")
    print("")
    print("  2. 选择哪个工具？")
    print("     - 基于用户意图匹配")
    print("     - 基于工具描述匹配")
    print("     - 基于对话历史匹配")
    print("")
    print("  3. 如何处理工具结果？")
    print("     - 将结果返回给LLM")
    print("     - LLM根据结果生成最终回答")
    print("     - 或继续调用其他工具")


async def demonstrate_mcp_integration_concept():
    """
    演示MCP工具集成概念
    """
    print("\n" + "=" * 60)
    print("4. MCP工具集成概念")
    print("=" * 60)
    
    print("""
MCP (Model Context Protocol) 工具集成:

┌─────────────────────────────────────────────────────────────┐
│                    MCP架构                                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌──────────────┐         ┌──────────────┐               │
│  │   Agent      │         │  MCP Client  │               │
│  │  (本系统)     │◄───────►│  (MCPManager)│               │
│  └──────────────┘         └──────────────┘               │
│                                  │                          │
│                                  ▼                          │
│                         ┌──────────────┐                  │
│                         │ MCP Server   │                  │
│                         │  (外部服务)   │                  │
│                         └──────────────┘                  │
│                                  │                          │
│                    ┌─────────────┼─────────────┐          │
│                    ▼             ▼             ▼          │
│              ┌──────────┐ ┌──────────┐ ┌──────────┐      │
│              │ Filesystem│ │  SQLite  │ │  Web API │      │
│              │   Server  │ │  Server  │ │  Server  │      │
│              └──────────┘ └──────────┘ └──────────┘      │
│                                                             │
└─────────────────────────────────────────────────────────────┘

MCP工具集成流程:

1. 配置MCP服务器
   ```python
   mcp_config = MCPConfig(
       servers=[
           {
               "name": "filesystem",
               "command": "npx",
               "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/dir"],
           },
           {
               "name": "sqlite",
               "command": "npx",
               "args": ["-y", "@modelcontextprotocol/server-sqlite", "mydb.db"],
           },
       ]
   )
   ```

2. 连接MCP服务器
   ```python
   mcp_manager = MCPManager(mcp_config)
   await mcp_manager.connect_all()
   ```

3. 获取MCP工具
   ```python
   mcp_tools = mcp_manager.get_tools()
   # 这些工具会自动绑定到Agent
   ```

4. 工具选择与执行
   - MCP工具与普通工具一样参与工具选择
   - ToolSelector会根据用户意图选择合适的工具
   - ToolBinder会执行MCP工具调用
""")


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("工具选择与MCP集成完整示例")
    print("=" * 60)
    
    await demonstrate_tool_selection()
    await demonstrate_enhanced_agent()
    await demonstrate_tool_decision_cycle()
    await demonstrate_mcp_integration_concept()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    
    print("""
快速开始:

1. 创建增强版Agent:
   ```python
   agent = EnhancedLangGraphAgent(
       config=AgentConfig(name="my-agent"),
       provider_manager=provider_manager,
       tools=[web_search, calculator],
       mcp_manager=mcp_manager,  # 可选，MCP工具
   )
   ```

2. 执行Agent:
   ```python
   result = await agent.ainvoke(
       user_input="2 + 3 * 4 等于多少？",
       thread_id="conversation-001",
   )
   ```

3. 查看结果:
   ```python
   print(result["final_answer"])
   print(result["tool_results"])  # 工具执行结果
   print(result["tool_selection"])  # 工具选择结果
   ```

4. 关键组件:
   - ToolSelector: 智能选择工具
   - ToolBinder: 绑定和执行工具
   - EnhancedLangGraphAgent: 完整的决策闭环
""")


if __name__ == "__main__":
    asyncio.run(main())
