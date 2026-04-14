"""
长期记忆实现
"""

import json
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from agent_system.core.config import MemoryConfig
from agent_system.core.logging import get_logger
from agent_system.memory.base import LongTermMemory, MemoryItem

logger = get_logger(__name__)


try:
    from sqlalchemy import (
        JSON,
        Column,
        DateTime,
        Float,
        String,
        Text,
        create_engine,
    )
    from sqlalchemy.ext.declarative import declarative_base
    from sqlalchemy.orm import sessionmaker

    SQLALCHEMY_AVAILABLE = True
except ImportError:
    SQLALCHEMY_AVAILABLE = False


if SQLALCHEMY_AVAILABLE:
    Base = declarative_base()
    
    class MemoryModel(Base):
        """记忆数据模型"""
        __tablename__ = "long_term_memories"
        
        id = Column(String(36), primary_key=True)
        user_id = Column(String(255), index=True, nullable=False)
        content = Column(Text, nullable=False)
        metadata = Column(JSON, default=dict)
        created_at = Column(DateTime, default=datetime.now)
        updated_at = Column(DateTime, default=datetime.now, onupdate=datetime.now)
        importance = Column(Float, default=0.5)
        embedding = Column(JSON, nullable=True)


class SQLAlchemyLongTermMemory(LongTermMemory):
    """SQLAlchemy长期记忆实现"""
    
    def __init__(self, config: MemoryConfig) -> None:
        if not SQLALCHEMY_AVAILABLE:
            raise ImportError(
                "SQLAlchemy is not installed. Install with: pip install sqlalchemy"
            )
        
        self.config = config
        self._max_memories = config.max_long_term_memories
        
        if not config.connection_string:
            raise ValueError("Database connection string is required")
        
        self._engine = create_engine(config.connection_string)
        self._Session = sessionmaker(bind=self._engine)
        
        Base.metadata.create_all(self._engine)
        
        logger.info("SQLAlchemy long term memory initialized")
    
    def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """添加长期记忆"""
        memory_id = str(uuid.uuid4())
        
        session = self._Session()
        try:
            memory = MemoryModel(
                id=memory_id,
                user_id=user_id,
                content=content,
                metadata=metadata or {},
                importance=importance,
            )
            session.add(memory)
            session.commit()
            
            self._enforce_max_memories(user_id, session)
            
            logger.debug(
                "Long term memory added",
                memory_id=memory_id,
                user_id=user_id,
            )
            return memory_id
        finally:
            session.close()
    
    def _enforce_max_memories(self, user_id: str, session: Any) -> None:
        """强制限制最大记忆数量"""
        count = session.query(MemoryModel).filter(
            MemoryModel.user_id == user_id
        ).count()
        
        if count > self._max_memories:
            excess = count - self._max_memories
            old_memories = (
                session.query(MemoryModel)
                .filter(MemoryModel.user_id == user_id)
                .order_by(MemoryModel.importance.asc(), MemoryModel.created_at.asc())
                .limit(excess)
                .all()
            )
            
            for memory in old_memories:
                session.delete(memory)
            
            session.commit()
            logger.debug(
                "Old memories cleaned up",
                user_id=user_id,
                count=excess,
            )
    
    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """搜索相关记忆"""
        session = self._Session()
        try:
            query_obj = (
                session.query(MemoryModel)
                .filter(
                    MemoryModel.user_id == user_id,
                    MemoryModel.importance >= min_importance,
                )
            )
            
            query_obj = query_obj.filter(
                MemoryModel.content.ilike(f"%{query}%")
            )
            
            memories = (
                query_obj.order_by(
                    MemoryModel.importance.desc(),
                    MemoryModel.updated_at.desc(),
                )
                .limit(limit)
                .all()
            )
            
            return [
                MemoryItem(
                    id=m.id,
                    content=m.content,
                    metadata=m.metadata or {},
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    importance=m.importance,
                    embedding=m.embedding,
                )
                for m in memories
            ]
        finally:
            session.close()
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取指定记忆"""
        session = self._Session()
        try:
            memory = session.query(MemoryModel).filter(
                MemoryModel.id == memory_id
            ).first()
            
            if not memory:
                return None
            
            return MemoryItem(
                id=memory.id,
                content=memory.content,
                metadata=memory.metadata or {},
                created_at=memory.created_at,
                updated_at=memory.updated_at,
                importance=memory.importance,
                embedding=memory.embedding,
            )
        finally:
            session.close()
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """更新记忆"""
        session = self._Session()
        try:
            memory = session.query(MemoryModel).filter(
                MemoryModel.id == memory_id
            ).first()
            
            if not memory:
                return False
            
            if content is not None:
                memory.content = content
            if metadata is not None:
                memory.metadata = metadata
            if importance is not None:
                memory.importance = importance
            
            memory.updated_at = datetime.now()
            session.commit()
            
            logger.debug("Long term memory updated", memory_id=memory_id)
            return True
        finally:
            session.close()
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        session = self._Session()
        try:
            memory = session.query(MemoryModel).filter(
                MemoryModel.id == memory_id
            ).first()
            
            if not memory:
                return False
            
            session.delete(memory)
            session.commit()
            
            logger.debug("Long term memory deleted", memory_id=memory_id)
            return True
        finally:
            session.close()
    
    def get_all_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """获取用户所有记忆"""
        session = self._Session()
        try:
            query_obj = (
                session.query(MemoryModel)
                .filter(MemoryModel.user_id == user_id)
                .order_by(
                    MemoryModel.importance.desc(),
                    MemoryModel.updated_at.desc(),
                )
            )
            
            if limit:
                query_obj = query_obj.limit(limit)
            
            memories = query_obj.all()
            
            return [
                MemoryItem(
                    id=m.id,
                    content=m.content,
                    metadata=m.metadata or {},
                    created_at=m.created_at,
                    updated_at=m.updated_at,
                    importance=m.importance,
                    embedding=m.embedding,
                )
                for m in memories
            ]
        finally:
            session.close()


class InMemoryLongTermMemory(LongTermMemory):
    """内存长期记忆实现（用于测试）"""
    
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._max_memories = config.max_long_term_memories
        self._memories: Dict[str, Dict[str, MemoryItem]] = {}
    
    def add_memory(
        self,
        user_id: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
        importance: float = 0.5,
    ) -> str:
        """添加长期记忆"""
        memory_id = str(uuid.uuid4())
        
        if user_id not in self._memories:
            self._memories[user_id] = {}
        
        memory = MemoryItem(
            id=memory_id,
            content=content,
            metadata=metadata or {},
            importance=importance,
        )
        
        self._memories[user_id][memory_id] = memory
        
        if len(self._memories[user_id]) > self._max_memories:
            sorted_memories = sorted(
                self._memories[user_id].values(),
                key=lambda m: (m.importance, m.created_at),
            )
            excess = len(self._memories[user_id]) - self._max_memories
            for m in sorted_memories[:excess]:
                del self._memories[user_id][m.id]
        
        logger.debug(
            "In-memory long term memory added",
            memory_id=memory_id,
            user_id=user_id,
        )
        return memory_id
    
    def search_memories(
        self,
        user_id: str,
        query: str,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryItem]:
        """搜索相关记忆"""
        if user_id not in self._memories:
            return []
        
        query_lower = query.lower()
        memories = [
            m
            for m in self._memories[user_id].values()
            if m.importance >= min_importance and query_lower in m.content.lower()
        ]
        
        memories.sort(
            key=lambda m: (m.importance, m.updated_at),
            reverse=True,
        )
        
        return memories[:limit]
    
    def get_memory(self, memory_id: str) -> Optional[MemoryItem]:
        """获取指定记忆"""
        for user_memories in self._memories.values():
            if memory_id in user_memories:
                return user_memories[memory_id]
        return None
    
    def update_memory(
        self,
        memory_id: str,
        content: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
        importance: Optional[float] = None,
    ) -> bool:
        """更新记忆"""
        memory = self.get_memory(memory_id)
        if not memory:
            return False
        
        if content is not None:
            memory.content = content
        if metadata is not None:
            memory.metadata = metadata
        if importance is not None:
            memory.importance = importance
        
        memory.updated_at = datetime.now()
        
        logger.debug("In-memory long term memory updated", memory_id=memory_id)
        return True
    
    def delete_memory(self, memory_id: str) -> bool:
        """删除记忆"""
        for user_memories in self._memories.values():
            if memory_id in user_memories:
                del user_memories[memory_id]
                logger.debug("In-memory long term memory deleted", memory_id=memory_id)
                return True
        return False
    
    def get_all_memories(
        self,
        user_id: str,
        limit: Optional[int] = None,
    ) -> List[MemoryItem]:
        """获取用户所有记忆"""
        if user_id not in self._memories:
            return []
        
        memories = sorted(
            self._memories[user_id].values(),
            key=lambda m: (m.importance, m.updated_at),
            reverse=True,
        )
        
        if limit:
            memories = memories[:limit]
        
        return memories
