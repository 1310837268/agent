"""
短期记忆实现
"""

from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

from langchain_core.messages import BaseMessage

from agent_system.core.config import MemoryConfig
from agent_system.core.logging import get_logger
from agent_system.memory.base import ShortTermMemory

logger = get_logger(__name__)


class InMemoryShortTermMemory(ShortTermMemory):
    """内存短期记忆实现"""
    
    def __init__(self, config: MemoryConfig) -> None:
        self.config = config
        self._sessions: Dict[str, OrderedDict] = {}
        self._ttl = config.short_term_ttl
        self._max_messages = config.max_short_term_messages
        self._last_access: Dict[str, datetime] = {}
    
    def _cleanup_expired(self) -> None:
        """清理过期会话"""
        now = datetime.now()
        expired_sessions = [
            session_id
            for session_id, last_access in self._last_access.items()
            if now - last_access > timedelta(seconds=self._ttl)
        ]
        
        for session_id in expired_sessions:
            self.clear(session_id)
            logger.debug("Session expired and cleared", session_id=session_id)
    
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """添加消息到短期记忆"""
        self._cleanup_expired()
        
        if session_id not in self._sessions:
            self._sessions[session_id] = OrderedDict()
        
        messages = self._sessions[session_id]
        message_id = str(len(messages))
        messages[message_id] = message
        
        while len(messages) > self._max_messages:
            messages.popitem(last=False)
        
        self._last_access[session_id] = datetime.now()
        logger.debug(
            "Message added to short term memory",
            session_id=session_id,
            message_type=message.type,
        )
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """获取短期记忆中的消息"""
        self._cleanup_expired()
        
        if session_id not in self._sessions:
            return []
        
        messages = list(self._sessions[session_id].values())
        self._last_access[session_id] = datetime.now()
        
        if limit:
            messages = messages[-limit:]
        
        return messages
    
    def clear(self, session_id: str) -> None:
        """清除短期记忆"""
        if session_id in self._sessions:
            del self._sessions[session_id]
        if session_id in self._last_access:
            del self._last_access[session_id]
        logger.debug("Short term memory cleared", session_id=session_id)
    
    def get_session_ids(self) -> List[str]:
        """获取所有会话ID"""
        self._cleanup_expired()
        return list(self._sessions.keys())


try:
    import redis
    from redis.asyncio import Redis as AsyncRedis
    
    REDIS_AVAILABLE = True
except ImportError:
    REDIS_AVAILABLE = False


class RedisShortTermMemory(ShortTermMemory):
    """Redis短期记忆实现"""
    
    def __init__(self, config: MemoryConfig) -> None:
        if not REDIS_AVAILABLE:
            raise ImportError("Redis is not installed. Install with: pip install redis")
        
        self.config = config
        self._ttl = config.short_term_ttl
        self._max_messages = config.max_short_term_messages
        
        if not config.connection_string:
            raise ValueError("Redis connection string is required")
        
        self._redis = redis.from_url(config.connection_string)
        self._async_redis = AsyncRedis.from_url(config.connection_string)
        
        logger.info("Redis short term memory initialized")
    
    def _get_key(self, session_id: str) -> str:
        """获取Redis键"""
        return f"stm:{session_id}"
    
    def add_message(self, session_id: str, message: BaseMessage) -> None:
        """添加消息到短期记忆"""
        import json
        
        key = self._get_key(session_id)
        message_dict = {
            "type": message.type,
            "content": message.content,
            "additional_kwargs": message.additional_kwargs,
        }
        message_json = json.dumps(message_dict)
        
        pipe = self._redis.pipeline()
        pipe.rpush(key, message_json)
        pipe.ltrim(key, -self._max_messages, -1)
        pipe.expire(key, self._ttl)
        pipe.execute()
        
        logger.debug(
            "Message added to Redis short term memory",
            session_id=session_id,
            message_type=message.type,
        )
    
    def get_messages(self, session_id: str, limit: Optional[int] = None) -> List[BaseMessage]:
        """获取短期记忆中的消息"""
        import json
        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        
        key = self._get_key(session_id)
        
        if limit:
            messages_json = self._redis.lrange(key, -limit, -1)
        else:
            messages_json = self._redis.lrange(key, 0, -1)
        
        messages = []
        for msg_json in messages_json:
            msg_dict = json.loads(msg_json)
            msg_type = msg_dict["type"]
            
            if msg_type == "human":
                msg = HumanMessage(
                    content=msg_dict["content"],
                    additional_kwargs=msg_dict.get("additional_kwargs", {}),
                )
            elif msg_type == "ai":
                msg = AIMessage(
                    content=msg_dict["content"],
                    additional_kwargs=msg_dict.get("additional_kwargs", {}),
                )
            elif msg_type == "system":
                msg = SystemMessage(
                    content=msg_dict["content"],
                    additional_kwargs=msg_dict.get("additional_kwargs", {}),
                )
            else:
                continue
            
            messages.append(msg)
        
        self._redis.expire(key, self._ttl)
        
        return messages
    
    def clear(self, session_id: str) -> None:
        """清除短期记忆"""
        key = self._get_key(session_id)
        self._redis.delete(key)
        logger.debug("Redis short term memory cleared", session_id=session_id)
    
    def get_session_ids(self) -> List[str]:
        """获取所有会话ID"""
        keys = self._redis.keys("stm:*")
        return [key.decode("utf-8").split(":", 1)[1] for key in keys]
