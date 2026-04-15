"""
基于LangGraph的Agent实现（增强版）

使用LangGraph的StateGraph和Checkpointer来实现：
1. 状态管理（短期记忆）
2. 检查点持久化（长期记忆）
3. 智能工具选择
4. MCP工具集成
5. 工具调用决策闭环
6. 流式输出
"""

import asyncio
import json
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Union, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode

from agent_system.core.config import AgentConfig
from agent_system.core.exceptions import MaxIterationsExceededError, OrchestrationError
from agent_system.core.logging import get_logger
from agent_system.memory.langgraph_memory import (
    GraphState,
    LangGraphMemory,
    create_langgraph_memory,
)
from agent_system.mcp.manager import MCPManager
from agent_system.providers.manager import ProviderManager
from agent_system.tools.binder import ToolBinder
from agent_system.tools.selector import (
    HybridToolSelector,
    ToolSelectionResult,
    ToolSelector,
    create_tool_selector,
)

logger = get_logger(__name__)


class AgentState(TypedDict):
    """
    Agent状态定义（增强版）
    
    使用TypedDict定义LangGraph的状态
    - messages: 消息历史（使用add_messages进行合并）
    - metadata: 元数据
    - current_step: 当前步骤
    - max_steps: 最大步骤数
    - tool_selection_result: 工具选择结果
    - tool_results: 工具执行结果
    """
    messages: Annotated[List[BaseMessage], add_messages]
    metadata: Dict[str, Any]
    current_step: int
    max_steps: int
    tool_selection_result: Optional[ToolSelectionResult]
    tool_results: List[Dict[str, Any]]


class EnhancedLangGraphAgent:
    """
    基于LangGraph的增强版Agent
    
    核心特性：
    1. 使用LangGraph StateGraph进行状态管理
    2. 使用Checkpointer进行状态持久化（记忆）
    3. 智能工具选择（ToolSelector）
    4. MCP工具集成
    5. 工具调用决策闭环
    6. 支持同步/异步调用
    7. 支持流式输出
    """
    
    def __init__(
        self,
        config: AgentConfig,
        provider_manager: ProviderManager,
        tools: Optional[List[BaseTool]] = None,
        tool_selector: Optional[ToolSelector] = None,
        mcp_manager: Optional[MCPManager] = None,
        memory: Optional[LangGraphMemory] = None,
        memory_type: str = "memory",
        **memory_kwargs: Any,
    ) -> None:
        """
        初始化增强版LangGraph Agent
        
        Args:
            config: Agent配置
            provider_manager: Provider管理器
            tools: 工具列表
            tool_selector: 工具选择器（可选，默认使用HybridToolSelector）
            mcp_manager: MCP管理器（可选，用于MCP工具集成）
            memory: LangGraph记忆实例（可选）
            memory_type: 记忆类型（如果没有提供memory）
            **memory_kwargs: 记忆初始化参数
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        self._provider_manager = provider_manager
        self._mcp_manager = mcp_manager
        
        if tool_selector:
            self._tool_selector = tool_selector
        else:
            self._tool_selector = HybridToolSelector(
                provider_manager=provider_manager,
                model=config.model,
                provider=config.provider,
            )
        
        self._tool_binder = ToolBinder(
            tools=tools,
            tool_selector=self._tool_selector,
            mcp_manager=mcp_manager,
            provider_manager=provider_manager,
        )
        
        if memory:
            self._memory = memory
        else:
            self._memory = create_langgraph_memory(memory_type, **memory_kwargs)
        
        self._system_prompt = self._build_system_prompt()
        
        self._graph = self._build_graph()
        self._compiled_graph = self._compile_graph()
        
        logger.info(
            "EnhancedLangGraphAgent initialized",
            name=self.name,
            tools=self._tool_binder.get_tool_names(),
            has_mcp=bool(mcp_manager),
            memory_type=type(self._memory).__name__,
        )
    
    def _build_system_prompt(self) -> str:
        """
        构建系统提示词（增强版）
        
        包含：
        1. Agent角色定义
        2. 工具描述
        3. 工具使用规则
        4. 输出格式
        """
        parts = []
        
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)
        
        if self._tool_binder.has_tools():
            tool_prompt = self._tool_binder.build_tool_system_prompt()
            parts.append(tool_prompt)
        
        return "\n\n".join(parts)
    
    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph状态图（增强版）
        
        图结构：
        START -> tool_selection -> agent_node -> tool_execution -> agent_node -> ... -> END
        
        决策逻辑：
        1. tool_selection: 智能选择工具
        2. agent_node: 调用LLM生成响应
        3. tool_execution: 执行工具（如果需要）
        4. 循环直到完成或达到最大步骤
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("tool_selection", self._tool_selection_node)
        workflow.add_node("agent", self._agent_node)
        
        if self._tool_binder.has_tools():
            workflow.add_node("tool_execution", self._tool_execution_node)
            
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "tools": "tool_execution",
                    "end": END,
                },
            )
            
            workflow.add_edge("tool_execution", "agent")
        else:
            workflow.add_edge("agent", END)
        
        workflow.add_edge(START, "tool_selection")
        workflow.add_edge("tool_selection", "agent")
        
        return workflow
    
    def _compile_graph(self) -> Any:
        """编译图，启用Checkpointer"""
        return self._graph.compile(
            checkpointer=self._memory.get_checkpointer(),
        )
    
    def _tool_selection_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具选择节点
        
        智能分析用户意图，选择合适的工具
        """
        messages = state["messages"]
        
        if not messages:
            return {
                "tool_selection_result": ToolSelectionResult(
                    selected_tools=[],
                    reasoning="没有消息",
                    should_call_tools=False,
                    confidence=0.0,
                ),
            }
        
        last_message = messages[-1]
        user_input = last_message.content if hasattr(last_message, 'content') else str(last_message)
        
        if not self._tool_binder.has_tools():
            return {
                "tool_selection_result": ToolSelectionResult(
                    selected_tools=[],
                    reasoning="没有可用的工具",
                    should_call_tools=False,
                    confidence=1.0,
                ),
            }
        
        conversation_history = messages[:-1] if len(messages) > 1 else None
        
        selection_result = self._tool_binder.select_tools(
            user_input=user_input,
            conversation_history=conversation_history,
        )
        
        logger.info(
            "Tool selection completed",
            agent=self.name,
            should_call_tools=selection_result.should_call_tools,
            selected_tools=[t.get("name") for t in selection_result.selected_tools],
            confidence=selection_result.confidence,
        )
        
        return {
            "tool_selection_result": selection_result,
        }
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Agent节点（增强版）
        
        调用LLM生成响应，考虑工具选择结果
        """
        messages = state["messages"]
        tool_selection_result = state.get("tool_selection_result")
        
        enhanced_messages = []
        
        if self._system_prompt:
            enhanced_messages.append(SystemMessage(content=self._system_prompt))
        
        if tool_selection_result and tool_selection_result.should_call_tools:
            tool_info = "\n## 工具选择建议\n"
            tool_info += f"推理: {tool_selection_result.reasoning}\n"
            tool_info += f"置信度: {tool_selection_result.confidence}\n"
            
            if tool_selection_result.selected_tools:
                tool_info += "\n建议调用的工具:\n"
                for tool in tool_selection_result.selected_tools:
                    tool_info += f"- {tool.get('name')}: {tool.get('reason')}\n"
            
            enhanced_messages.append(SystemMessage(content=tool_info))
        
        enhanced_messages.extend(list(messages))
        
        try:
            formatted_tools = self._tool_binder.format_tools_for_llm()
            
            if formatted_tools:
                result, provider_name = self._provider_manager.generate_with_fallback(
                    enhanced_messages,
                    preferred_provider=self.config.provider,
                    tools=formatted_tools,
                )
            else:
                result, provider_name = self._provider_manager.generate_with_fallback(
                    enhanced_messages,
                    preferred_provider=self.config.provider,
                )
            
            ai_message = result.generations[0].message
            
            logger.debug(
                "Agent node executed",
                agent=self.name,
                provider=provider_name,
                has_tool_calls=bool(hasattr(ai_message, 'tool_calls') and ai_message.tool_calls),
                message_length=len(ai_message.content) if ai_message.content else 0,
            )
            
            return {
                "messages": [ai_message],
                "current_step": state.get("current_step", 0) + 1,
            }
            
        except Exception as e:
            logger.error(
                "Agent node failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"Agent '{self.name}' failed: {str(e)}",
                self.name,
            ) from e
    
    async def _tool_execution_node_async(self, state: AgentState) -> Dict[str, Any]:
        """
        工具执行节点（异步）
        
        执行LLM请求的工具调用
        """
        messages = state["messages"]
        last_message = messages[-1]
        
        tool_calls = self._tool_binder.parse_tool_calls_from_message(last_message)
        
        if not tool_calls:
            logger.debug("No tool calls to execute")
            return {"tool_results": []}
        
        logger.info(
            "Executing tool calls",
            agent=self.name,
            tool_count=len(tool_calls),
            tools=[tc.get("name") for tc in tool_calls],
        )
        
        tool_results = await self._tool_binder.execute_tool_calls(tool_calls)
        
        tool_messages = []
        for result in tool_results:
            tool_message = ToolMessage(
                content=result["result"],
                tool_call_id=result.get("tool_call_id", result.get("tool_name", "")),
                name=result["tool_name"],
            )
            tool_messages.append(tool_message)
        
        return {
            "messages": tool_messages,
            "tool_results": tool_results,
        }
    
    def _tool_execution_node(self, state: AgentState) -> Dict[str, Any]:
        """
        工具执行节点（同步包装器）
        """
        return asyncio.run(self._tool_execution_node_async(state))
    
    def _should_continue(self, state: AgentState) -> str:
        """
        决定是否继续执行（增强版）
        
        Returns:
            "tools" - 继续执行工具
            "end" - 结束执行
        """
        messages = state["messages"]
        current_step = state.get("current_step", 0)
        max_steps = state.get("max_steps", self.config.max_iterations)
        
        if current_step >= max_steps:
            logger.warning(
                "Max steps reached",
                agent=self.name,
                steps=current_step,
                max_steps=max_steps,
            )
            return "end"
        
        last_message = messages[-1]
        
        if hasattr(last_message, 'tool_calls') and last_message.tool_calls:
            logger.debug(
                "Tool calls detected, continuing to tool execution",
                agent=self.name,
                tool_calls=[tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None) 
                           for tc in last_message.tool_calls],
            )
            return "tools"
        
        tool_selection_result = state.get("tool_selection_result")
        if tool_selection_result and tool_selection_result.should_call_tools:
            logger.debug(
                "Tool selection suggests calling tools, but no tool calls in message",
                agent=self.name,
            )
        
        return "end"
    
    def add_tool(self, tool: BaseTool) -> None:
        """添加工具"""
        self._tool_binder.add_tool(tool)
        self._graph = self._build_graph()
        self._compiled_graph = self._compile_graph()
        logger.info("Tool added to EnhancedLangGraphAgent", agent=self.name, tool=tool.name)
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        result = self._tool_binder.remove_tool(tool_name)
        if result:
            self._graph = self._build_graph()
            self._compiled_graph = self._compile_graph()
            logger.info("Tool removed from EnhancedLangGraphAgent", agent=self.name, tool=tool_name)
        return result
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return self._tool_binder.get_tools()
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return self._tool_binder.get_tool_names()
    
    def _create_thread_config(
        self,
        thread_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """创建线程配置"""
        import uuid
        
        if thread_id is None:
            thread_id = str(uuid.uuid4())
        
        return self._memory.create_thread_config(thread_id, **kwargs)
    
    def invoke(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        同步执行Agent
        
        Args:
            user_input: 用户输入
            thread_id: 线程ID（用于记忆持久化）
            metadata: 元数据
            **kwargs: 其他参数
        
        Returns:
            执行结果
        """
        config = self._create_thread_config(thread_id)
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "metadata": metadata or {},
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "tool_selection_result": None,
            "tool_results": [],
        }
        
        try:
            result = self._compiled_graph.invoke(initial_state, config=config)
            
            final_message = result["messages"][-1]
            final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            logger.info(
                "EnhancedLangGraphAgent invoked successfully",
                agent=self.name,
                thread_id=config["configurable"]["thread_id"],
                steps=result.get("current_step", 0),
                tool_count=len(result.get("tool_results", [])),
            )
            
            return {
                "success": True,
                "final_answer": final_answer,
                "thread_id": config["configurable"]["thread_id"],
                "steps": result.get("current_step", 0),
                "tool_results": result.get("tool_results", []),
                "tool_selection": result.get("tool_selection_result"),
                "messages": [m.model_dump() for m in result["messages"]],
            }
            
        except Exception as e:
            logger.error(
                "EnhancedLangGraphAgent invocation failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"EnhancedLangGraphAgent '{self.name}' invocation failed: {str(e)}",
                self.name,
            ) from e
    
    async def ainvoke(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """
        异步执行Agent
        """
        config = self._create_thread_config(thread_id)
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "metadata": metadata or {},
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "tool_selection_result": None,
            "tool_results": [],
        }
        
        try:
            result = await self._compiled_graph.ainvoke(initial_state, config=config)
            
            final_message = result["messages"][-1]
            final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            logger.info(
                "EnhancedLangGraphAgent async invoked successfully",
                agent=self.name,
                thread_id=config["configurable"]["thread_id"],
                steps=result.get("current_step", 0),
                tool_count=len(result.get("tool_results", [])),
            )
            
            return {
                "success": True,
                "final_answer": final_answer,
                "thread_id": config["configurable"]["thread_id"],
                "steps": result.get("current_step", 0),
                "tool_results": result.get("tool_results", []),
                "tool_selection": result.get("tool_selection_result"),
                "messages": [m.model_dump() for m in result["messages"]],
            }
            
        except Exception as e:
            logger.error(
                "EnhancedLangGraphAgent async invocation failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"EnhancedLangGraphAgent '{self.name}' async invocation failed: {str(e)}",
                self.name,
            ) from e
    
    def stream(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        流式执行Agent
        """
        config = self._create_thread_config(thread_id)
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "metadata": metadata or {},
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "tool_selection_result": None,
            "tool_results": [],
        }
        
        return self._compiled_graph.stream(initial_state, config=config)
    
    async def astream(
        self,
        user_input: str,
        thread_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        **kwargs: Any,
    ) -> Any:
        """
        异步流式执行Agent
        """
        config = self._create_thread_config(thread_id)
        
        initial_state: AgentState = {
            "messages": [HumanMessage(content=user_input)],
            "metadata": metadata or {},
            "current_step": 0,
            "max_steps": self.config.max_iterations,
            "tool_selection_result": None,
            "tool_results": [],
        }
        
        return self._compiled_graph.astream(initial_state, config=config)
    
    def get_thread_history(
        self,
        thread_id: str,
    ) -> Optional[Dict[str, Any]]:
        """
        获取线程历史（记忆）
        
        Args:
            thread_id: 线程ID
        
        Returns:
            线程状态历史
        """
        config = self._create_thread_config(thread_id)
        
        try:
            state = self._compiled_graph.get_state(config)
            if state.values:
                return {
                    "thread_id": thread_id,
                    "messages": [m.model_dump() for m in state.values.get("messages", [])],
                    "current_step": state.values.get("current_step", 0),
                    "metadata": state.values.get("metadata", {}),
                    "tool_results": state.values.get("tool_results", []),
                }
            return None
        except Exception as e:
            logger.warning(
                "Failed to get thread history",
                thread_id=thread_id,
                error=str(e),
            )
            return None
    
    def list_threads(self) -> List[str]:
        """列出所有线程"""
        return self._memory.list_threads()
    
    def delete_thread(self, thread_id: str) -> bool:
        """删除线程（清除记忆）"""
        return self._memory.delete_thread(thread_id)
    
    def get_graph_image(self) -> Optional[bytes]:
        """
        获取图的可视化图像
        
        需要安装: pip install pygraphviz
        """
        try:
            return self._compiled_graph.get_graph().draw_mermaid_png()
        except Exception as e:
            logger.warning("Failed to get graph image", error=str(e))
            return None
    
    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return self._tool_binder.has_tools()
    
    def get_tool_count(self) -> int:
        """获取工具数量"""
        return self._tool_binder.get_tool_count()
