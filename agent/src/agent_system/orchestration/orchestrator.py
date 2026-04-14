"""
编排器 - 管理多个Agent的协作
"""

import asyncio
import uuid
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from agent_system.core.config import AgentConfig, AgentSystemConfig
from agent_system.core.exceptions import OrchestrationError
from agent_system.core.logging import get_logger
from agent_system.memory.manager import MemoryManager
from agent_system.orchestration.agent import Agent
from agent_system.orchestration.state import AgentContext, AgentState
from agent_system.providers.manager import ProviderManager

logger = get_logger(__name__)


class Orchestrator:
    """
    编排器
    
    管理多个Agent的协作，支持：
    1. Agent注册与管理
    2. 任务路由与分发
    3. Agent间协作
    4. 工作流编排
    """
    
    def __init__(
        self,
        config: AgentSystemConfig,
        provider_manager: ProviderManager,
        memory_manager: Optional[MemoryManager] = None,
    ) -> None:
        self.config = config
        self._provider_manager = provider_manager
        self._memory_manager = memory_manager
        self._agents: Dict[str, Agent] = {}
        self._workflows: Dict[str, Any] = {}
        
        logger.info("Orchestrator initialized")
    
    def register_agent(self, agent: Agent) -> None:
        """注册Agent"""
        self._agents[agent.name] = agent
        logger.info("Agent registered", agent=agent.name)
    
    def unregister_agent(self, agent_name: str) -> bool:
        """注销Agent"""
        if agent_name in self._agents:
            del self._agents[agent_name]
            logger.info("Agent unregistered", agent=agent_name)
            return True
        return False
    
    def get_agent(self, agent_name: str) -> Optional[Agent]:
        """获取Agent"""
        return self._agents.get(agent_name)
    
    def list_agents(self) -> List[str]:
        """列出所有注册的Agent"""
        return list(self._agents.keys())
    
    def create_agent(
        self,
        config: AgentConfig,
        tools: Optional[List[Any]] = None,
    ) -> Agent:
        """创建并注册Agent"""
        agent = Agent(
            config=config,
            provider_manager=self._provider_manager,
            memory_manager=self._memory_manager,
            tools=tools,
        )
        self.register_agent(agent)
        return agent
    
    def route_task(
        self,
        user_input: str,
        context: Optional[Dict[str, Any]] = None,
    ) -> str:
        """
        任务路由
        
        根据用户输入选择最合适的Agent
        """
        if not self._agents:
            raise OrchestrationError("No agents registered")
        
        if len(self._agents) == 1:
            return list(self._agents.keys())[0]
        
        agent_names = list(self._agents.keys())
        descriptions = [
            f"{name}: {self._agents[name].description}"
            for name in agent_names
        ]
        
        routing_prompt = f"""
Given the user input: "{user_input}"

Available agents:
{chr(10).join(descriptions)}

Which agent is most suitable for this task? Respond with only the agent name.
"""
        
        try:
            from langchain_core.messages import HumanMessage
            
            messages = [HumanMessage(content=routing_prompt)]
            result, _ = self._provider_manager.generate_with_fallback(messages)
            selected_agent = result.generations[0].message.content.strip()
            
            if selected_agent in self._agents:
                logger.info(
                    "Task routed",
                    input=user_input[:50],
                    agent=selected_agent,
                )
                return selected_agent
        except Exception as e:
            logger.warning("Task routing failed, using first agent", error=str(e))
        
        return agent_names[0]
    
    def execute(
        self,
        user_input: str,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentContext:
        """
        执行任务
        
        Args:
            user_input: 用户输入
            agent_name: 指定Agent名称，不指定则自动路由
            session_id: 会话ID
            user_id: 用户ID
            **kwargs: 其他参数
        
        Returns:
            AgentContext: 执行上下文
        """
        if not agent_name:
            agent_name = self.route_task(user_input)
        
        agent = self.get_agent(agent_name)
        if not agent:
            raise OrchestrationError(f"Agent '{agent_name}' not found")
        
        logger.info(
            "Executing task",
            agent=agent_name,
            session_id=session_id,
        )
        
        return agent.run(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )
    
    async def aexecute(
        self,
        user_input: str,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AgentContext:
        """
        异步执行任务
        """
        if not agent_name:
            agent_name = self.route_task(user_input)
        
        agent = self.get_agent(agent_name)
        if not agent:
            raise OrchestrationError(f"Agent '{agent_name}' not found")
        
        logger.info(
            "Executing async task",
            agent=agent_name,
            session_id=session_id,
        )
        
        return await agent.arun(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )
    
    def execute_parallel(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[AgentContext]:
        """
        并行执行多个任务
        
        Args:
            tasks: 任务列表，每个任务包含:
                - user_input: 用户输入
                - agent_name: Agent名称（可选）
                - session_id: 会话ID（可选）
                - user_id: 用户ID（可选）
        
        Returns:
            List[AgentContext]: 执行结果列表
        """
        import concurrent.futures
        
        results = []
        
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = {
                executor.submit(
                    self.execute,
                    user_input=task["user_input"],
                    agent_name=task.get("agent_name"),
                    session_id=task.get("session_id"),
                    user_id=task.get("user_id"),
                ): i
                for i, task in enumerate(tasks)
            }
            
            for future in concurrent.futures.as_completed(futures):
                try:
                    result = future.result()
                    results.append(result)
                except Exception as e:
                    logger.error("Parallel task execution failed", error=str(e))
                    results.append(None)
        
        return results
    
    async def aexecute_parallel(
        self,
        tasks: List[Dict[str, Any]],
    ) -> List[AgentContext]:
        """
        异步并行执行多个任务
        """
        async def run_task(task: Dict[str, Any]) -> AgentContext:
            return await self.aexecute(
                user_input=task["user_input"],
                agent_name=task.get("agent_name"),
                session_id=task.get("session_id"),
                user_id=task.get("user_id"),
            )
        
        coroutines = [run_task(task) for task in tasks]
        results = await asyncio.gather(*coroutines, return_exceptions=True)
        
        return [r if not isinstance(r, Exception) else None for r in results]
    
    def execute_sequential(
        self,
        tasks: List[Dict[str, Any]],
        pass_output: bool = False,
    ) -> List[AgentContext]:
        """
        顺序执行多个任务
        
        Args:
            tasks: 任务列表
            pass_output: 是否将前一个任务的输出传递给下一个任务
        
        Returns:
            List[AgentContext]: 执行结果列表
        """
        results = []
        previous_output = None
        
        for task in tasks:
            user_input = task["user_input"]
            
            if pass_output and previous_output:
                user_input = f"{user_input}\n\nPrevious context: {previous_output}"
            
            result = self.execute(
                user_input=user_input,
                agent_name=task.get("agent_name"),
                session_id=task.get("session_id"),
                user_id=task.get("user_id"),
            )
            
            results.append(result)
            
            if result.final_answer:
                previous_output = result.final_answer
        
        return results
    
    async def aexecute_sequential(
        self,
        tasks: List[Dict[str, Any]],
        pass_output: bool = False,
    ) -> List[AgentContext]:
        """
        异步顺序执行多个任务
        """
        results = []
        previous_output = None
        
        for task in tasks:
            user_input = task["user_input"]
            
            if pass_output and previous_output:
                user_input = f"{user_input}\n\nPrevious context: {previous_output}"
            
            result = await self.aexecute(
                user_input=user_input,
                agent_name=task.get("agent_name"),
                session_id=task.get("session_id"),
                user_id=task.get("user_id"),
            )
            
            results.append(result)
            
            if result.final_answer:
                previous_output = result.final_answer
        
        return results
    
    def stream(
        self,
        user_input: str,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[Dict[str, Any]]:
        """
        流式执行任务
        """
        if not agent_name:
            agent_name = self.route_task(user_input)
        
        agent = self.get_agent(agent_name)
        if not agent:
            raise OrchestrationError(f"Agent '{agent_name}' not found")
        
        yield from agent.stream(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        )
    
    async def astream(
        self,
        user_input: str,
        agent_name: Optional[str] = None,
        session_id: Optional[str] = None,
        user_id: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Dict[str, Any]]:
        """
        异步流式执行任务
        """
        if not agent_name:
            agent_name = self.route_task(user_input)
        
        agent = self.get_agent(agent_name)
        if not agent:
            raise OrchestrationError(f"Agent '{agent_name}' not found")
        
        async for event in agent.astream(
            user_input=user_input,
            session_id=session_id,
            user_id=user_id,
            **kwargs,
        ):
            yield event
