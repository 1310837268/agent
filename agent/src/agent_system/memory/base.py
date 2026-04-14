"""
记忆系统基础接口
"""

from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field


class MemoryItem(BaseModel):
    """记忆项"""
    id: str
    content: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)
    importance: float = 0.5
    embedding: Optional[List[float]] = None
    
    class Config:
        arbitrary_types_allowed = True


class ShortTermMemory(ABC):
    """短期记忆抽象类"""
    
    @abstractmethod
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """添加消息到短期记忆"""
        pass
    
    @abstractmethod
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """获取短期记忆中的消息"""
        pass
    
    @abstractmethod
    def clear(self, session_id: str) -> None:
        """清除短期记忆"""
        pass
    
    @abstractmethod
    def get_session_ids(self) -> List[str]:
        """获取所有会话ID"""
        pass


class LongTermMemory(ABC):
    """长期记忆抽象类"""
    
    @abstractmethod
    def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """添加长期记忆"""
        pass
    
    @abstractmethod
    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """搜索相关记忆"""
        pass
    
    @abstractmethod
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取指定记忆"""
        pass
    
    @abstractmethod
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """更新记忆"""
        pass
    
    @abstractmethod
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        pass
    
    @abstractmethod
    def get_all_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """获取用户所有记忆"""
        pass
