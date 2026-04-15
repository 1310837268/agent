"""
会话状态管理模块
"""

import json
import uuid
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from agent_system.core.config import AgentSystemConfig
from agent_system.core.exceptions import SessionExpiredError, SessionNotFoundError
from agent_system.core.logging import get_logger

logger = get_logger(__name__)


class SessionState:
    """会话状态"""
    
    def __init__(
        self,
        session_id: str,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.session_id = session_id
        self.user_id = user_id
        self.created_at = datetime.now()
        self.updated_at = datetime.now()
        self.metadata = metadata or {}
        self.data: Dict[str, Any] = {}
        self.message_count = 0
        self.is_active = True
    
    def update(self) -> None:
        """更新会话时间"""
        self.updated_at = datetime.now()
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "session_id": self.session_id,
            "user_id": self.user_id,
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "metadata": self.metadata,
            "data": self.data,
            "message_count": self.message_count,
            "is_active": self.is_active,
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "SessionState":
        """从字典创建"""
        session = cls(
            session_id=data["session_id"],
            user_id=data.get("user_id"),
            metadata=data.get("metadata"),
        )
        session.created_at = datetime.fromisoformat(data["created_at"])
        session.updated_at = datetime.fromisoformat(data["updated_at"])
        session.data = data.get("data", {})
        session.message_count = data.get("message_count", 0)
        session.is_active = data.get("is_active", True)
        return session


class SessionManager:
    """
    会话管理器
    
    管理会话的创建、存储、过期和清理
    """
    
    def __init__(self, config: AgentSystemConfig) -> None:
        self.config = config
        self._session_ttl = config.session_ttl
        self._sessions: Dict[str, SessionState] = {}
        self._last_cleanup = datetime.now()
    
    def _cleanup_expired(self) -> None:
        """清理过期会话"""
        now = datetime.now()
        
        if (now - self._last_cleanup).total_seconds() < 60:
            return
        
        expired_sessions = [
            session_id
            for session_id, session in self._sessions.items()
            if (now - session.updated_at).total_seconds() > self._session_ttl
        ]
        
        for session_id in expired_sessions:
            del self._sessions[session_id]
            logger.debug("Session expired and removed", session_id=session_id)
        
        self._last_cleanup = now
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """创建新会话"""
        self._cleanup_expired()
        
        session_id = str(uuid.uuid4())
        session = SessionState(
            session_id=session_id,
            user_id=user_id,
            metadata=metadata,
        )
        
        self._sessions[session_id] = session
        
        logger.info(
            "Session created",
            session_id=session_id,
            user_id=user_id,
        )
        
        return session
    
    def get_session(self, session_id: str) -> SessionState:
        """获取会话"""
        self._cleanup_expired()
        
        if session_id not in self._sessions:
            raise SessionNotFoundError(session_id)
        
        session = self._sessions[session_id]
        
        if (datetime.now() - session.updated_at).total_seconds() > self._session_ttl:
            del self._sessions[session_id]
            raise SessionExpiredError(session_id)
        
        session.update()
        return session
    
    def update_session(
        self,
        session_id: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        increment_message: bool = False,
    ) -> SessionState:
        """更新会话"""
        session = self.get_session(session_id)
        
        if data:
            session.data.update(data)
        
        if metadata:
            session.metadata.update(metadata)
        
        if increment_message:
            session.message_count += 1
        
        session.update()
        
        logger.debug(
            "Session updated",
            session_id=session_id,
            message_count=session.message_count,
        )
        
        return session
    
    def end_session(self, session_id: str) -> bool:
        """结束会话"""
        if session_id in self._sessions:
            session = self._sessions[session_id]
            session.is_active = False
            del self._sessions[session_id]
            
            logger.info(
                "Session ended",
                session_id=session_id,
                message_count=session.message_count,
            )
            return True
        
        return False
    
    def get_all_sessions(self) -> List[SessionState]:
        """获取所有活跃会话"""
        self._cleanup_expired()
        return list(self._sessions.values())
    
    def get_user_sessions(self, user_id: str) -> List[SessionState]:
        """获取用户的所有活跃会话"""
        self._cleanup_expired()
        return [
            session
            for session in self._sessions.values()
            if session.user_id == user_id
        ]
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        self._cleanup_expired()
        return session_id in self._sessions
    
    def get_session_count(self) -> int:
        """获取活跃会话数量"""
        self._cleanup_expired()
        return len(self._sessions)


try:
    import redis
    from redis.asyncio import Redis as AsyncRedis

    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisSessionManager(SessionManager):
    """
    Redis会话管理器
    
    使用Redis存储会话，支持分布式部署
    """
    
    def __init__(
        self,
        config: AgentSystemConfig,
        redis_url: str,
    ) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")
        
        super().__init__(config)
        self._redis = redis.from_url(redis_url)
        self._key_prefix = "session:"
    
    def _get_key(self, session_id: str) -> str:
        """获取Redis键"""
        return f"{self._key_prefix}{session_id}"
    
    def create_session(
        self,
        user_id: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> SessionState:
        """创建新会话"""
        session = super().create_session(user_id, metadata)
        
        key = self._get_key(session.session_id)
        self._redis.setex(
            key,
            self._session_ttl,
            json.dumps(session.to_dict()),
        )
        
        return session
    
    def get_session(self, session_id: str) -> SessionState:
        """获取会话"""
        key = self._get_key(session_id)
        data = self._redis.get(key)
        
        if not data:
            raise SessionNotFoundError(session_id)
        
        session = SessionState.from_dict(json.loads(data))
        
        if (datetime.now() - session.updated_at).total_seconds() > self._session_ttl:
            self._redis.delete(key)
            raise SessionExpiredError(session_id)
        
        session.update()
        self._redis.setex(
            key,
            self._session_ttl,
            json.dumps(session.to_dict()),
        )
        
        return session
    
    def update_session(
        self,
        session_id: str,
        data: Optional[Dict[str, Any]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        increment_message: bool = False,
    ) -> SessionState:
        """更新会话"""
        session = self.get_session(session_id)
        
        if data:
            session.data.update(data)
        
        if metadata:
            session.metadata.update(metadata)
        
        if increment_message:
            session.message_count += 1
        
        session.update()
        
        key = self._get_key(session_id)
        self._redis.setex(
            key,
            self._session_ttl,
            json.dumps(session.to_dict()),
        )
        
        logger.debug(
            "Session updated in Redis",
            session_id=session_id,
            message_count=session.message_count,
        )
        
        return session
    
    def end_session(self, session_id: str) -> bool:
        """结束会话"""
        key = self._get_key(session_id)
        result = self._redis.delete(key) > 0
        
        if session_id in self._sessions:
            del self._sessions[session_id]
        
        if result:
            logger.info("Session ended in Redis", session_id=session_id)
        
        return result
    
    def session_exists(self, session_id: str) -> bool:
        """检查会话是否存在"""
        key = self._get_key(session_id)
        return self._redis.exists(key) > 0
    
    def get_session_count(self) -> int:
        """获取活跃会话数量"""
        keys = self._redis.keys(f"{self._key_prefix}*")
        return len(keys)
