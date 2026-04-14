"""
基于LangGraph的记忆系统

使用LangGraph原生的记忆机制：
1. MemorySaver - 状态持久化
2. Checkpointer - 检查点机制
3. StateGraph - 状态图管理
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Sequence, Type, Union

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.store.base import BaseStore
from pydantic import BaseModel, Field

from agent_system.core.logging import get_logger

logger = get_logger(__name__)


try:
    from langgraph.checkpoint.sqlite import SqliteSaver

    SQLITE_SAVER_AVAILABLE = True
except ImportError:
    SQLITE_SAVER_AVAILABLE = False


try:
    from langgraph.checkpoint.postgres import PostgresSaver

    POSTGRES_SAVER_AVAILABLE = True
except ImportError:
    POSTGRES_SAVER_AVAILABLE = False


try:
    from langgraph.checkpoint.redis import RedisSaver

    REDIS_SAVER_AVAILABLE = True
except ImportError:
    REDIS_SAVER_AVAILABLE = False


class GraphState(BaseModel):
    """
    LangGraph状态定义
    
    这是Agent执行的核心状态，包含：
    - messages: 对话消息历史（短期记忆）
    - memories: 长期记忆引用
    - metadata: 元数据
    """
    messages: List[BaseMessage] = Field(default_factory=list)
    memories: List[Dict[str, Any]] = Field(default_factory=list)
    metadata: Dict[str, Any] = Field(default_factory=dict)
    current_step: int = 0
    max_steps: int = 10
    
    class Config:
        arbitrary_types_allowed = True


class LangGraphMemory(ABC):
    """
    LangGraph记忆抽象基类
    
    封装LangGraph的Checkpointer和Store
    """
    
    @abstractmethod
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取Checkpointer"""
        pass
    
    @abstractmethod
    def get_store(self) -> Optional[BaseStore]:
        """获取Store（用于长期记忆）"""
        pass
    
    @abstractmethod
    def create_thread_config(self, thread_id: str, **kwargs: Any) -> Dict[str, Any]:
        """创建线程配置"""
        pass
    
    @abstractmethod
    def list_threads(self) -> List[str]:
        """列出所有线程"""
        pass
    
    @abstractmethod
    def delete_thread(self, thread_id: str) -> bool:
        """删除线程"""
        pass


class InMemoryLangGraphMemory(LangGraphMemory):
    """
    内存LangGraph记忆
    
    使用MemorySaver进行状态持久化
    适合开发和测试环境
    """
    
    def __init__(self) -> None:
        self._checkpointer = MemorySaver()
        self._store: Optional[BaseStore] = None
        
        logger.info("InMemoryLangGraphMemory initialized")
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取Checkpointer"""
        return self._checkpointer
    
    def get_store(self) -> Optional[BaseStore]:
        """获取Store"""
        return self._store
    
    def create_thread_config(self, thread_id: str, **kwargs: Any) -> Dict[str, Any]:
        """
        创建线程配置
        
        LangGraph使用thread_id来区分不同的对话/会话
        """
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        if kwargs:
            config["configurable"].update(kwargs)
        
        return config
    
    def list_threads(self) -> List[str]:
        """列出所有线程"""
        try:
            return list(self._checkpointer.list())
        except Exception as e:
            logger.warning("Failed to list threads", error=str(e))
            return []
    
    def delete_thread(self, thread_id: str) -> bool:
        """删除线程"""
        try:
            config = self.create_thread_config(thread_id)
            self._checkpointer.delete(config)
            logger.info("Thread deleted", thread_id=thread_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete thread", thread_id=thread_id, error=str(e))
            return False


class SqliteLangGraphMemory(LangGraphMemory):
    """
    SQLite LangGraph记忆
    
    使用SqliteSaver进行状态持久化
    适合生产环境的单机部署
    """
    
    def __init__(self, db_path: str = "checkpoints.sqlite") -> None:
        if not SQLITE_SAVER_AVAILABLE:
            raise ImportError(
                "SqliteSaver is not available. "
                "Install with: pip install langgraph-checkpoint-sqlite"
            )
        
        self._db_path = db_path
        self._checkpointer = SqliteSaver.from_conn_string(db_path)
        self._store: Optional[BaseStore] = None
        
        logger.info("SqliteLangGraphMemory initialized", db_path=db_path)
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取Checkpointer"""
        return self._checkpointer
    
    def get_store(self) -> Optional[BaseStore]:
        """获取Store"""
        return self._store
    
    def create_thread_config(self, thread_id: str, **kwargs: Any) -> Dict[str, Any]:
        """创建线程配置"""
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        if kwargs:
            config["configurable"].update(kwargs)
        
        return config
    
    def list_threads(self) -> List[str]:
        """列出所有线程"""
        try:
            return list(self._checkpointer.list())
        except Exception as e:
            logger.warning("Failed to list threads", error=str(e))
            return []
    
    def delete_thread(self, thread_id: str) -> bool:
        """删除线程"""
        try:
            config = self.create_thread_config(thread_id)
            self._checkpointer.delete(config)
            logger.info("Thread deleted", thread_id=thread_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete thread", thread_id=thread_id, error=str(e))
            return False


class RedisLangGraphMemory(LangGraphMemory):
    """
    Redis LangGraph记忆
    
    使用RedisSaver进行状态持久化
    适合生产环境的分布式部署
    """
    
    def __init__(self, redis_url: str) -> None:
        if not REDIS_SAVER_AVAILABLE:
            raise ImportError(
                "RedisSaver is not available. "
                "Install with: pip install langgraph-checkpoint-redis"
            )
        
        self._redis_url = redis_url
        self._checkpointer = RedisSaver.from_conn_string(redis_url)
        self._store: Optional[BaseStore] = None
        
        logger.info("RedisLangGraphMemory initialized")
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取Checkpointer"""
        return self._checkpointer
    
    def get_store(self) -> Optional[BaseStore]:
        """获取Store"""
        return self._store
    
    def create_thread_config(self, thread_id: str, **kwargs: Any) -> Dict[str, Any]:
        """创建线程配置"""
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        if kwargs:
            config["configurable"].update(kwargs)
        
        return config
    
    def list_threads(self) -> List[str]:
        """列出所有线程"""
        try:
            return list(self._checkpointer.list())
        except Exception as e:
            logger.warning("Failed to list threads", error=str(e))
            return []
    
    def delete_thread(self, thread_id: str) -> bool:
        """删除线程"""
        try:
            config = self.create_thread_config(thread_id)
            self._checkpointer.delete(config)
            logger.info("Thread deleted", thread_id=thread_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete thread", thread_id=thread_id, error=str(e))
            return False


class PostgresLangGraphMemory(LangGraphMemory):
    """
    PostgreSQL LangGraph记忆
    
    使用PostgresSaver进行状态持久化
    适合生产环境的大规模部署
    """
    
    def __init__(self, postgres_url: str) -> None:
        if not POSTGRES_SAVER_AVAILABLE:
            raise ImportError(
                "PostgresSaver is not available. "
                "Install with: pip install langgraph-checkpoint-postgres"
            )
        
        self._postgres_url = postgres_url
        self._checkpointer = PostgresSaver.from_conn_string(postgres_url)
        self._store: Optional[BaseStore] = None
        
        logger.info("PostgresLangGraphMemory initialized")
    
    def get_checkpointer(self) -> BaseCheckpointSaver:
        """获取Checkpointer"""
        return self._checkpointer
    
    def get_store(self) -> Optional[BaseStore]:
        """获取Store"""
        return self._store
    
    def create_thread_config(self, thread_id: str, **kwargs: Any) -> Dict[str, Any]:
        """创建线程配置"""
        config = {
            "configurable": {
                "thread_id": thread_id,
            }
        }
        
        if kwargs:
            config["configurable"].update(kwargs)
        
        return config
    
    def list_threads(self) -> List[str]:
        """列出所有线程"""
        try:
            return list(self._checkpointer.list())
        except Exception as e:
            logger.warning("Failed to list threads", error=str(e))
            return []
    
    def delete_thread(self, thread_id: str) -> bool:
        """删除线程"""
        try:
            config = self.create_thread_config(thread_id)
            self._checkpointer.delete(config)
            logger.info("Thread deleted", thread_id=thread_id)
            return True
        except Exception as e:
            logger.warning("Failed to delete thread", thread_id=thread_id, error=str(e))
            return False


def create_langgraph_memory(
    memory_type: str = "memory",
    **kwargs: Any,
) -> LangGraphMemory:
    """
    工厂函数：创建LangGraph记忆实例
    
    Args:
        memory_type: 记忆类型
            - "memory": 内存记忆（开发测试）
            - "sqlite": SQLite记忆（生产单机）
            - "redis": Redis记忆（生产分布式）
            - "postgres": PostgreSQL记忆（生产大规模）
        **kwargs: 其他参数
    
    Returns:
        LangGraphMemory实例
    """
    if memory_type == "memory":
        return InMemoryLangGraphMemory()
    elif memory_type == "sqlite":
        db_path = kwargs.get("db_path", "checkpoints.sqlite")
        return SqliteLangGraphMemory(db_path)
    elif memory_type == "redis":
        redis_url = kwargs.get("redis_url")
        if not redis_url:
            raise ValueError("redis_url is required for RedisLangGraphMemory")
        return RedisLangGraphMemory(redis_url)
    elif memory_type == "postgres":
        postgres_url = kwargs.get("postgres_url")
        if not postgres_url:
            raise ValueError("postgres_url is required for PostgresLangGraphMemory")
        return PostgresLangGraphMemory(postgres_url)
    else:
        raise ValueError(f"Unknown memory type: {memory_type}")
