"""
Agent状态定义
"""

from datetime import datetime
from enum import Enum
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class AgentState(str, Enum):
    """Agent状态枚举"""
    IDLE = "idle"
    THINKING = "thinking"
    TOOL_CALLING = "tool_calling"
    WAITING = "waiting"
    FINISHED = "finished"
    ERROR = "error"


class AgentStep(BaseModel):
    """Agent执行步骤"""
    step_id: str
    step_type: str  # thinking, tool_call, tool_result, final_answer
    content: str
    timestamp: datetime = Field(default_factory=datetime.now)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    
    class Config:
        arbitrary_types_allowed = True


class AgentContext(BaseModel):
    """Agent执行上下文"""
    session_id: str
    user_id: Optional[str] = None
    messages: List[BaseMessage] = Field(default_factory=list)
    steps: List[AgentStep] = Field(default_factory=list)
    state: AgentState = AgentState.IDLE
    current_iteration: int = 0
    max_iterations: int = 10
    tools_used: List[str] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    error: Optional[str] = None
    final_answer: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True
    
    def add_message(self, message: BaseMessage) -> None:
        """添加消息"""
        self.messages.append(message)
    
    def add_step(self, step: AgentStep) -> None:
        """添加执行步骤"""
        self.steps.append(step)
    
    def increment_iteration(self) -> bool:
        """增加迭代计数，返回是否超过最大迭代次数"""
        self.current_iteration += 1
        return self.current_iteration >= self.max_iterations
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "state": self.state.value,
            "current_iteration": self.current_iteration,
            "max_iterations": self.max_iterations,
            "tools_used": self.tools_used,
            "metadata": self.metadata,
            "error": self.error,
            "final_answer": self.final_answer,
            "step_count": len(self.steps),
        }
