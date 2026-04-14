"""
基于LangGraph的Agent实现

使用LangGraph的StateGraph和Checkpointer来实现：
1. 状态管理（短期记忆）
2. 检查点持久化（长期记忆）
3. 工具调用循环
4. 流式输出
"""

import asyncio
from typing import Any, AsyncIterator, Dict, List, Optional, TypedDict, Union, Annotated

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
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
from agent_system.providers.manager import ProviderManager

logger = get_logger(__name__)


class AgentState(TypedDict):
    """
    Agent状态定义
    
    使用TypedDict定义LangGraph的状态
    - messages: 消息历史（使用add_messages进行合并）
    - metadata: 元数据
    - current_step: 当前步骤
    - max_steps: 最大步骤数
    """
    messages: Annotated[List[BaseMessage], add_messages]
    metadata: Dict[str, Any]
    current_step: int
    max_steps: int


class LangGraphAgent:
    """
    基于LangGraph的Agent
    
    核心特性：
    1. 使用LangGraph StateGraph进行状态管理
    2. 使用Checkpointer进行状态持久化（记忆）
    3. 支持工具调用循环
    4. 支持同步/异步调用
    5. 支持流式输出
    """
    
    def __init__(
        self,
        config: AgentConfig,
        provider_manager: ProviderManager,
        tools: Optional[List[BaseTool]] = None,
        memory: Optional[LangGraphMemory] = None,
        memory_type: str = "memory",
        **memory_kwargs: Any,
    ) -> None:
        """
        初始化LangGraph Agent
        
        Args:
            config: Agent配置
            provider_manager: Provider管理器
            tools: 工具列表
            memory: LangGraph记忆实例（可选）
            memory_type: 记忆类型（如果没有提供memory）
            **memory_kwargs: 记忆初始化参数
        """
        self.config = config
        self.name = config.name
        self.description = config.description
        self._provider_manager = provider_manager
        self._tools: Dict[str, BaseTool] = {}
        self._system_prompt = self._build_system_prompt()
        
        if tools:
            for tool in tools:
                self._tools[tool.name] = tool
        
        if memory:
            self._memory = memory
        else:
            self._memory = create_langgraph_memory(memory_type, **memory_kwargs)
        
        self._graph = self._build_graph()
        self._compiled_graph = self._compile_graph()
        
        logger.info(
            "LangGraphAgent initialized",
            name=self.name,
            tools=list(self._tools.keys()),
            memory_type=type(self._memory).__name__,
        )
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        parts = []
        
        if self.config.system_prompt:
            parts.append(self.config.system_prompt)
        
        if self._tools:
            parts.append("\nYou have access to the following tools:")
            for name, tool in self._tools.items():
                parts.append(f"- {name}: {tool.description}")
            parts.append(
                "\nWhen you need to use a tool, make sure to call it properly. "
                "The system will handle the tool execution for you."
            )
        
        return "\n".join(parts)
    
    def _build_graph(self) -> StateGraph:
        """
        构建LangGraph状态图
        
        图结构：
        START -> agent_node -> tool_node (如果需要) -> agent_node -> ... -> END
        """
        workflow = StateGraph(AgentState)
        
        workflow.add_node("agent", self._agent_node)
        
        if self._tools:
            tool_node = ToolNode(list(self._tools.values()))
            workflow.add_node("tools", tool_node)
            
            workflow.add_conditional_edges(
                "agent",
                self._should_continue,
                {
                    "continue": "tools",
                    "end": END,
                },
            )
            
            workflow.add_edge("tools", "agent")
        else:
            workflow.add_edge("agent", END)
        
        workflow.add_edge(START, "agent")
        
        return workflow
    
    def _compile_graph(self) -> Any:
        """编译图，启用Checkpointer"""
        return self._graph.compile(
            checkpointer=self._memory.get_checkpointer(),
        )
    
    def _agent_node(self, state: AgentState) -> Dict[str, Any]:
        """
        Agent节点
        
        调用LLM生成响应
        """
        messages = state["messages"]
        
        if self._system_prompt:
            messages = [SystemMessage(content=self._system_prompt)] + list(messages)
        
        try:
            result, provider_name = self._provider_manager.generate_with_fallback(
                messages,
                preferred_provider=self.config.provider,
            )
            
            ai_message = result.generations[0].message
            
            logger.debug(
                "Agent node executed",
                agent=self.name,
                provider=provider_name,
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
    
    def _should_continue(self, state: AgentState) -> str:
        """
        决定是否继续执行
        
        Returns:
            "continue" - 继续执行工具
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
                "Tool calls detected, continuing",
                agent=self.name,
                tool_calls=[tc["name"] for tc in last_message.tool_calls],
            )
            return "continue"
        
        return "end"
    
    def add_tool(self, tool: BaseTool) -> None:
        """添加工具"""
        self._tools[tool.name] = tool
        self._graph = self._build_graph()
        self._compiled_graph = self._compile_graph()
        logger.info("Tool added to LangGraphAgent", agent=self.name, tool=tool.name)
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            self._graph = self._build_graph()
            self._compiled_graph = self._compile_graph()
            logger.info("Tool removed from LangGraphAgent", agent=self.name, tool=tool_name)
            return True
        return False
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
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
        }
        
        try:
            result = self._compiled_graph.invoke(initial_state, config=config)
            
            final_message = result["messages"][-1]
            final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            logger.info(
                "LangGraphAgent invoked successfully",
                agent=self.name,
                thread_id=config["configurable"]["thread_id"],
                steps=result.get("current_step", 0),
            )
            
            return {
                "success": True,
                "final_answer": final_answer,
                "thread_id": config["configurable"]["thread_id"],
                "steps": result.get("current_step", 0),
                "messages": [m.model_dump() for m in result["messages"]],
            }
            
        except Exception as e:
            logger.error(
                "LangGraphAgent invocation failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"LangGraphAgent '{self.name}' invocation failed: {str(e)}",
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
        }
        
        try:
            result = await self._compiled_graph.ainvoke(initial_state, config=config)
            
            final_message = result["messages"][-1]
            final_answer = final_message.content if hasattr(final_message, 'content') else str(final_message)
            
            logger.info(
                "LangGraphAgent async invoked successfully",
                agent=self.name,
                thread_id=config["configurable"]["thread_id"],
                steps=result.get("current_step", 0),
            )
            
            return {
                "success": True,
                "final_answer": final_answer,
                "thread_id": config["configurable"]["thread_id"],
                "steps": result.get("current_step", 0),
                "messages": [m.model_dump() for m in result["messages"]],
            }
            
        except Exception as e:
            logger.error(
                "LangGraphAgent async invocation failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"LangGraphAgent '{self.name}' async invocation failed: {str(e)}",
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
