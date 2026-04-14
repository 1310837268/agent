"""
记忆管理器
"""

from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from agent_system.core.config import MemoryConfig
from agent_system.core.logging import get_logger
from agent_system.memory.base import LongTermMemory, MemoryItem, ShortTermMemory
from agent_system.memory.long_term import (
    InMemoryLongTermMemory,
    SQLAlchemyLongTermMemory,
)
from agent_system.memory.short_term import (
    InMemoryShortTermMemory,
    RedisShortTermMemory,
)

logger = get_logger(__name__)


class MemoryManager:
    """
    记忆管理器
    
    统一管理短期记忆和长期记忆
    """
    
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._short_term: ShortTermMemory = self._init_short_term()
        self._long_term: LongTermMemory = self._init_long_term()
        
        logger.info(
            "MemoryManager initialized",
            short_term_type=config.short_term_type,
            long_term_type=config.long_term_type,
        )
    
    def _init_short_term(self) -> ShortTermMemory:
        """初始化短期记忆"""
        if self.config.short_term_type == "redis":
            return RedisShortTermMemory(self.config)
        elif self.config.short_term_type == "in_memory":
            return InMemoryShortTermMemory(self.config)
        else:
            logger.warning(
                "Unknown short term memory type, using in_memory",
                type=self.config.short_term_type,
            )
            return InMemoryShortTermMemory(self.config)
    
    def _init_long_term(self) -> LongTermMemory:
        """初始化长期记忆"""
        if self.config.long_term_type == "sqlalchemy":
            return SQLAlchemyLongTermMemory(self.config)
        elif self.config.long_term_type == "in_memory":
            return InMemoryLongTermMemory(self.config)
        else:
            logger.warning(
                "Unknown long term memory type, using in_memory",
                type=self.config.long_term_type,
            )
            return InMemoryLongTermMemory(self.config)
    
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """添加消息到短期记忆"""
        self._short_term.add_message(session_id, message)
    
    def get_messages(
        self,
        session_id: str,
        limit: Optional[int] = None,
    ) -> List[BaseMessage]:
        """获取短期记忆中的消息"""
        return self._short_term.get_messages(session_id, limit)
    
    def clear_short_term(self, session_id: str) -> None:
        """清除短期记忆"""
        self._short_term.clear(session_id)
    
    def get_session_ids(self) -> List[str]:
        """获取所有会话ID"""
        return self._short_term.get_session_ids()
    
    def add_long_term_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """添加长期记忆"""
        return self._long_term.add_memory(user_id, content, metadata, importance)
    
    def search_long_term_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """搜索相关长期记忆"""
        return self._long_term.search_memories(user_id, query, limit, min_importance)
    
    def get_long_term_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取指定长期记忆"""
        return self._long_term.get_memory(memory_id)
    
    def update_long_term_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """更新长期记忆"""
        return self._long_term.update_memory(memory_id, content, metadata, importance)
    
    def delete_long_term_memory(self, memory_id: str) -> bool:
        """删除长期记忆"""
        return self._long_term.delete_memory(memory_id)
    
    def get_all_long_term_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """获取用户所有长期记忆"""
        return self._long_term.get_all_memories(user_id, limit)
    
    def build_context(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        query: Optional[str] = None,
        short_term_limit: Optional[int] = None,
        long_term_limit: int = 5,
    ) -> Dict[str, Any]:
        """
        构建上下文
        
        整合短期记忆和相关长期记忆
        """
        context = {
            "short_term_messages": [],
            "long_term_memories": [],
            "context_prompt": "",
        }
        
        short_term_messages = self.get_messages(session_id, short_term_limit)
        context["short_term_messages"] = short_term_messages
        
        if user_id and query:
            long_term_memories = self.search_long_term_memories(
                user_id=user_id,
                query=query,
                limit=long_term_limit,
            )
            context["long_term_memories"] = long_term_memories
        
        context_parts = []
        
        if context["long_term_memories"]:
            context_parts.append("Relevant memories from previous conversations:")
            for i, memory in enumerate(context["long_term_memories"], 1):
                context_parts.append(f"{i}. {memory.content}")
            context_parts.append("")
        
        context["context_prompt"] = "\n".join(context_parts)
        
        return context
    
    def save_conversation_summary(
        self,
        session_id: str,
        user_id: str,
        summary: str,
        importance: float = 0.5,
    ) -> str:
        """
        保存对话摘要到长期记忆
        
        在对话结束时调用，将重要信息保存到长期记忆
        """
        messages = self.get_messages(session_id)
        
        metadata = {
            "session_id": session_id,
            "message_count": len(messages),
            "type": "conversation_summary",
        }
        
        memory_id = self.add_long_term_memory(
            user_id=user_id,
            content=summary,
            metadata=metadata,
            importance=importance,
        )
        
        logger.info(
            "Conversation summary saved to long term memory",
            memory_id=memory_id,
            user_id=user_id,
            session_id=session_id,
        )
        
        return memory_id
