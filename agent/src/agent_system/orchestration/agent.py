"""
核心Agent实现
"""

import asyncio
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_core.tools import BaseTool

from agent_system.core.config import AgentConfig
from agent_system.core.exceptions import MaxIterationsExceededError, OrchestrationError
from agent_system.core.logging import get_logger
from agent_system.memory.manager import MemoryManager
from agent_system.orchestration.state import AgentContext, AgentState, AgentStep
from agent_system.providers.manager import ProviderManager

logger = get_logger(__name__)


class Agent:
    """
    核心Agent类
    
    基于LangChain/LangGraph的Agent实现
    """
    
    def __init__(
        self,
        config: AgentConfig,
        provider_manager: ProviderManager,
        memory_manager: Optional[MemoryManager] = None,
        tools: Optional[List[BaseTool]] = None,
    ) -> None:
        self.config = config
        self.name = config.name
        self.description = config.description
        self._provider_manager = provider_manager
        self._memory_manager = memory_manager
        self._tools: Dict[str, BaseTool] = {}
        
        if tools:
            for tool in tools:
                self._tools[tool.name] = tool
        
        self._system_prompt = self._build_system_prompt()
        
        logger.info(
            "Agent initialized",
            name=self.name,
            tools=list(self._tools.keys()),
            enable_memory=config.enable_memory,
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
                "\nWhen you need to use a tool, respond with a JSON object "
                "containing 'tool_name' and 'tool_input' fields."
            )
        
        return "\n".join(parts)
    
    def add_tool(self, tool: BaseTool) -> None:
        """添加工具"""
        self._tools[tool.name] = tool
        logger.info("Tool added to agent", agent=self.name, tool=tool.name)
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info("Tool removed from agent", agent=self.name, tool=tool_name)
            return True
        return False
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def _create_context(
        self,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> AgentContext:
        """创建执行上下文"""
        return AgentContext(
            session_id=session_id or str(uuid.uuid4()),
            user_id=user_id,
            max_iterations=self.config.max_iterations,
            metadata={
                "agent_name": self.name,
                "agent_config": self.config.model_dump(),
            },
        )
    
    def _build_messages(
        self,
        context: AgentContext,
        user_input: str,
    ) -> List[BaseMessage]:
        """构建消息列表"""
        messages = []
        
        if self._system_prompt:
            messages.append(SystemMessage(content=self._system_prompt))
        
        if self._memory_manager and context.session_id:
            history = self._memory_manager.get_messages(context.session_id)
            messages.extend(history)
            
            if context.user_id:
                context_data = self._memory_manager.build_context(
                    session_id=context.session_id,
                    user_id=context.user_id,
                    query=user_input,
                )
                if context_data["context_prompt"]:
                    messages.insert(
                        1,
                        SystemMessage(content=context_data["context_prompt"]),
                    )
        
        messages.append(HumanMessage(content=user_input))
        
        return messages
    
    def _parse_tool_call(self, content: str) -> Optional[Dict[str, Any]]:
        """解析工具调用"""
        import json
        import re
        
        json_match = re.search(r'\{[\s\S]*"tool_name"[\s\S]*\}', content)
        if json_match:
            try:
                tool_call = json.loads(json_match.group())
                if "tool_name" in tool_call:
                    return tool_call
            except json.JSONDecodeError:
                pass
        
        return None
    
    def _execute_tool(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """执行工具"""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self._tools[tool_name]
        try:
            result = tool.invoke(tool_input)
            return str(result)
        except Exception as e:
            logger.error(
                "Tool execution failed",
                tool=tool_name,
                error=str(e),
            )
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    async def _execute_tool_async(self, tool_name: str, tool_input: Dict[str, Any]) -> str:
        """异步执行工具"""
        if tool_name not in self._tools:
            return f"Error: Tool '{tool_name}' not found"
        
        tool = self._tools[tool_name]
        try:
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(tool_input)
            else:
                result = await asyncio.to_thread(tool.invoke, tool_input)
            return str(result)
        except Exception as e:
            logger.error(
                "Tool execution failed",
                tool=tool_name,
                error=str(e),
            )
            return f"Error executing tool '{tool_name}': {str(e)}"
    
    def run(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentContext:
        """
        同步执行Agent
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数
        
        Returns:
            AgentContext: 执行上下文
        """
        context = self._create_context(session_id, user_id)
        context.state = AgentState.THINKING
        
        try:
            messages = self._build_messages(context, user_input)
            
            if self._memory_manager and context.session_id:
                self._memory_manager.add_message(
                    context.session_id,
                    HumanMessage(content=user_input),
                )
            
            while True:
                if context.increment_iteration():
                    raise MaxIterationsExceededError(
                        context.max_iterations,
                        self.name,
                    )
                
                context.add_step(AgentStep(
                    step_id=str(uuid.uuid4()),
                    step_type="thinking",
                    content=f"Iteration {context.current_iteration}",
                ))
                
                result, provider_name = self._provider_manager.generate_with_fallback(
                    messages,
                    preferred_provider=self.config.provider,
                    **kwargs,
                )
                
                ai_message = result.generations[0].message
                response_content = ai_message.content
                
                context.add_step(AgentStep(
                    step_id=str(uuid.uuid4()),
                    step_type="thinking",
                    content=response_content,
                    metadata={"provider": provider_name},
                ))
                
                tool_call = self._parse_tool_call(response_content)
                
                if tool_call and self._tools:
                    context.state = AgentState.TOOL_CALLING
                    tool_name = tool_call["tool_name"]
                    tool_input = tool_call.get("tool_input", {})
                    
                    context.add_step(AgentStep(
                        step_id=str(uuid.uuid4()),
                        step_type="tool_call",
                        content=f"Calling tool: {tool_name}",
                        metadata={"tool_name": tool_name, "tool_input": tool_input},
                    ))
                    
                    context.tools_used.append(tool_name)
                    
                    tool_result = self._execute_tool(tool_name, tool_input)
                    
                    context.add_step(AgentStep(
                        step_id=str(uuid.uuid4()),
                        step_type="tool_result",
                        content=tool_result,
                        metadata={"tool_name": tool_name},
                    ))
                    
                    messages.append(AIMessage(content=response_content))
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_name,
                    ))
                    
                    if self._memory_manager and context.session_id:
                        self._memory_manager.add_message(
                            context.session_id,
                            AIMessage(content=response_content),
                        )
                else:
                    context.state = AgentState.FINISHED
                    context.final_answer = response_content
                    
                    context.add_step(AgentStep(
                        step_id=str(uuid.uuid4()),
                        step_type="final_answer",
                        content=response_content,
                    ))
                    
                    if self._memory_manager and context.session_id:
                        self._memory_manager.add_message(
                            context.session_id,
                            AIMessage(content=response_content),
                        )
                    
                    break
        
        except Exception as e:
            context.state = AgentState.ERROR
            context.error = str(e)
            logger.error(
                "Agent execution failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"Agent '{self.name}' execution failed: {str(e)}",
                self.name,
            ) from e
        
        return context
    
    async def arun(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentContext:
        """
        异步执行Agent
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数
        
        Returns:
            AgentContext: 执行上下文
        """
        context = self._create_context(session_id, user_id)
        context.state = AgentState.THINKING
        
        try:
            messages = self._build_messages(context, user_input)
            
            if self._memory_manager and context.session_id:
                self._memory_manager.add_message(
                    context.session_id,
                    HumanMessage(content=user_input),
                )
            
            while True:
                if context.increment_iteration():
                    raise MaxIterationsExceededError(
                        context.max_iterations,
                        self.name,
                    )
                
                context.add_step(AgentStep(
                    step_id=str(uuid.uuid4()),
                    step_type="thinking",
                    content=f"Iteration {context.current_iteration}",
                ))
                
                result, provider_name = await self._provider_manager.agenerate_with_fallback(
                    messages,
                    preferred_provider=self.config.provider,
                    **kwargs,
                )
                
                ai_message = result.generations[0].message
                response_content = ai_message.content
                
                context.add_step(AgentStep(
                    step_id=str(uuid.uuid4()),
                    step_type="thinking",
                    content=response_content,
                    metadata={"provider": provider_name},
                ))
                
                tool_call = self._parse_tool_call(response_content)
                
                if tool_call and self._tools:
                    context.state = AgentState.TOOL_CALLING
                    tool_name = tool_call["tool_name"]
                    tool_input = tool_call.get("tool_input", {})
                    
                    context.add_step(AgentStep(
                        step_id=str(uuid.uuid4()),
                        step_type="tool_call",
                        content=f"Calling tool: {tool_name}",
                        metadata={"tool_name": tool_name, "tool_input": tool_input},
                    ))
                    
                    context.tools_used.append(tool_name)
                    
                    tool_result = await self._execute_tool_async(tool_name, tool_input)
                    
                    context.add_step(AgentStep(
                        step_id=str(uuid.uuid4()),
                        step_type="tool_result",
                        content=tool_result,
                        metadata={"tool_name": tool_name},
                    ))
                    
                    messages.append(AIMessage(content=response_content))
                    messages.append(ToolMessage(
                        content=tool_result,
                        tool_call_id=tool_name,
                    ))
                    
                    if self._memory_manager and context.session_id:
                        self._memory_manager.add_message(
                            context.session_id,
                            AIMessage(content=response_content),
                        )
                else:
                    context.state = AgentState.FINISHED
                    context.final_answer = response_content
                    
                    context.add_step(AgentStep(
                        step_id=str(uuid.uuid4()),
                        step_type="final_answer",
                        content=response_content,
                    ))
                    
                    if self._memory_manager and context.session_id:
                        self._memory_manager.add_message(
                            context.session_id,
                            AIMessage(content=response_content),
                        )
                    
                    break
        
        except Exception as e:
            context.state = AgentState.ERROR
            context.error = str(e)
            logger.error(
                "Agent async execution failed",
                agent=self.name,
                error=str(e),
            )
            raise OrchestrationError(
                f"Agent '{self.name}' async execution failed: {str(e)}",
                self.name,
            ) from e
        
        return context
    
    def stream(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式执行Agent
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数
        
        Yields:
            Dict[str, Any]: 流式事件
        """
        context = self._create_context(session_id, user_id)
        
        messages = self._build_messages(context, user_input)
        
        if self._memory_manager and context.session_id:
            self._memory_manager.add_message(
                context.session_id,
                HumanMessage(content=user_input),
            )
        
        yield {
            "type": "start",
            "session_id": context.session_id,
            "agent": self.name,
        }
        
        full_response = ""
        for chunk, provider_name in self._provider_manager.stream_with_fallback(
            messages,
            preferred_provider=self.config.provider,
            **kwargs,
        ):
            content = chunk.message.content if hasattr(chunk.message, 'content') else ""
            if content:
                full_response += content
                yield {
                    "type": "content",
                    "content": content,
                    "provider": provider_name,
                }
        
        if self._memory_manager and context.session_id:
            self._memory_manager.add_message(
                context.session_id,
                AIMessage(content=full_response),
            )
        
        yield {
            "type": "end",
            "content": full_response,
            "session_id": context.session_id,
        }
    
    async def astream(
        self,
        user_input: str,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        异步流式执行Agent
        
        Args:
            user_input: 用户输入
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数
        
        Yields:
            Dict[str, Any]: 流式事件
        """
        context = self._create_context(session_id, user_id)
        
        messages = self._build_messages(context, user_input)
        
        if self._memory_manager and context.session_id:
            self._memory_manager.add_message(
                context.session_id,
                HumanMessage(content=user_input),
            )
        
        yield {
            "type": "start",
            "session_id": context.session_id,
            "agent": self.name,
        }
        
        full_response = ""
        async for chunk, provider_name in self._provider_manager.astream_with_fallback(
            messages,
            preferred_provider=self.config.provider,
            **kwargs,
        ):
            content = chunk.message.content if hasattr(chunk.message, 'content') else ""
            if content:
                full_response += content
                yield {
                    "type": "content",
                    "content": content,
                    "provider": provider_name,
                }
        
        if self._memory_manager and context.session_id:
            self._memory_manager.add_message(
                context.session_id,
                AIMessage(content=full_response),
            )
        
        yield {
            "type": "end",
            "content": full_response,
            "session_id": context.session_id,
        }
