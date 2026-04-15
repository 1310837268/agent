"""
资源管理模块

提供生产级的资源管理能力：
1. 上下文管理器
2. 连接池管理
3. 资源清理钩子
4. 资源泄漏检测
5. 优雅关闭机制
"""

import asyncio
import atexit
import gc
import inspect
import weakref
from abc import ABC, abstractmethod
from contextlib import asynccontextmanager, contextmanager
from datetime import datetime
from typing import Any, AsyncIterator, Callable, Dict, List, Optional, Set

from agent_system.core.logging import get_logger

logger = get_logger(__name__)


class ResourceState:
    """资源状态"""
    CREATED = "created"
    ACQUIRED = "acquired"
    RELEASED = "released"
    CLOSED = "closed"
    ERROR = "error"


class ResourceLeakError(Exception):
    """资源泄漏错误"""
    pass


class ManagedResource(ABC):
    """
    受管资源抽象基类
    
    所有需要管理的资源都应该继承此类
    """
    
    def __init__(self, name: str) -> None:
        self._name = name
        self._state = ResourceState.CREATED
        self._created_at = datetime.now()
        self._acquired_at: Optional[datetime] = None
        self._released_at: Optional[datetime] = None
        self._closed_at: Optional[datetime] = None
        self._error: Optional[Exception] = None
        
        ResourceTracker.register(self)
    
    @property
    def name(self) -> str:
        """资源名称"""
        return self._name
    
    @property
    def state(self) -> str:
        """资源状态"""
        return self._state
    
    @property
    def is_acquired(self) -> bool:
        """是否已获取"""
        return self._state == ResourceState.ACQUIRED
    
    @property
    def is_closed(self) -> bool:
        """是否已关闭"""
        return self._state in (ResourceState.CLOSED, ResourceState.ERROR)
    
    @abstractmethod
    async def _acquire(self) -> None:
        """获取资源（子类实现）"""
        pass
    
    @abstractmethod
    async def _release(self) -> None:
        """释放资源（子类实现）"""
        pass
    
    @abstractmethod
    async def _close(self) -> None:
        """关闭资源（子类实现）"""
        pass
    
    async def acquire(self) -> None:
        """获取资源"""
        if self._state == ResourceState.ACQUIRED:
            logger.warning(f"Resource already acquired: {self._name}")
            return
        
        if self.is_closed:
            raise RuntimeError(f"Resource already closed: {self._name}")
        
        try:
            await self._acquire()
            self._state = ResourceState.ACQUIRED
            self._acquired_at = datetime.now()
            logger.debug(f"Resource acquired: {self._name}")
        except Exception as e:
            self._state = ResourceState.ERROR
            self._error = e
            logger.error(f"Failed to acquire resource: {self._name}", error=str(e))
            raise
    
    async def release(self) -> None:
        """释放资源"""
        if self._state != ResourceState.ACQUIRED:
            logger.warning(f"Resource not acquired: {self._name}")
            return
        
        try:
            await self._release()
            self._state = ResourceState.RELEASED
            self._released_at = datetime.now()
            logger.debug(f"Resource released: {self._name}")
        except Exception as e:
            self._state = ResourceState.ERROR
            self._error = e
            logger.error(f"Failed to release resource: {self._name}", error=str(e))
    
    async def close(self) -> None:
        """关闭资源"""
        if self.is_closed:
            return
        
        try:
            if self._state == ResourceState.ACQUIRED:
                await self.release()
            
            await self._close()
            self._state = ResourceState.CLOSED
            self._closed_at = datetime.now()
            logger.info(f"Resource closed: {self._name}")
        except Exception as e:
            self._state = ResourceState.ERROR
            self._error = e
            logger.error(f"Failed to close resource: {self._name}", error=str(e))
    
    def get_stats(self) -> Dict[str, Any]:
        """获取资源统计"""
        return {
            "name": self._name,
            "state": self._state,
            "created_at": self._created_at.isoformat() if self._created_at else None,
            "acquired_at": self._acquired_at.isoformat() if self._acquired_at else None,
            "released_at": self._released_at.isoformat() if self._released_at else None,
            "closed_at": self._closed_at.isoformat() if self._closed_at else None,
            "error": str(self._error) if self._error else None,
        }
    
    def __del__(self) -> None:
        """析构函数 - 检测资源泄漏"""
        if self._state == ResourceState.ACQUIRED:
            logger.warning(
                f"Resource leak detected: {self._name} was acquired but never released",
                stack=inspect.stack(),
            )
        
        if self._state not in (ResourceState.CLOSED, ResourceState.ERROR):
            logger.warning(
                f"Resource not properly closed: {self._name}",
                stack=inspect.stack(),
            )
    
    @asynccontextmanager
    async def managed(self) -> AsyncIterator["ManagedResource"]:
        """
        异步上下文管理器
        
        Usage:
            async with resource.managed() as r:
                # 使用资源
                pass
        """
        try:
            await self.acquire()
            yield self
        except Exception as e:
            logger.error(f"Error in managed resource: {self._name}", error=str(e))
            raise
        finally:
            if self._state == ResourceState.ACQUIRED:
                await self.release()


class ResourceTracker:
    """
    资源追踪器
    
    追踪所有受管资源，提供泄漏检测和优雅关闭
    """
    
    _resources: weakref.WeakSet = weakref.WeakSet()
    _finalizers: List[Callable] = []
    _initialized: bool = False
    
    @classmethod
    def register(cls, resource: ManagedResource) -> None:
        """注册资源"""
        cls._resources.add(resource)
        logger.debug(f"Resource registered: {resource.name}")
        
        if not cls._initialized:
            cls._init_atexit()
    
    @classmethod
    def _init_atexit(cls) -> None:
        """初始化退出钩子"""
        if cls._initialized:
            return
        
        def cleanup():
            logger.info("Running resource cleanup on exit...")
            cls.cleanup_all()
        
        atexit.register(cleanup)
        cls._initialized = True
    
    @classmethod
    def get_all_resources(cls) -> List[ManagedResource]:
        """获取所有资源"""
        return list(cls._resources)
    
    @classmethod
    def get_active_resources(cls) -> List[ManagedResource]:
        """获取活跃资源"""
        return [r for r in cls._resources if r.is_acquired]
    
    @classmethod
    def get_leaked_resources(cls) -> List[ManagedResource]:
        """获取泄漏的资源"""
        return [r for r in cls._resources if r.is_acquired]
    
    @classmethod
    async def cleanup_all(cls) -> None:
        """清理所有资源"""
        resources = list(cls._resources)
        
        for resource in resources:
            try:
                if not resource.is_closed:
                    await resource.close()
            except Exception as e:
                logger.error(f"Failed to cleanup resource: {resource.name}", error=str(e))
        
        logger.info(f"Cleaned up {len(resources)} resources")
    
    @classmethod
    def get_stats(cls) -> Dict[str, Any]:
        """获取统计信息"""
        all_resources = list(cls._resources)
        active = [r for r in all_resources if r.is_acquired]
        closed = [r for r in all_resources if r.is_closed]
        
        return {
            "total": len(all_resources),
            "active": len(active),
            "closed": len(closed),
            "leaked": len([r for r in all_resources if r.is_acquired]),
            "resources": [r.get_stats() for r in all_resources],
        }


class ConnectionPool(ABC):
    """
    连接池抽象基类
    """
    
    def __init__(
        self,
        max_size: int = 10,
        min_size: int = 1,
        max_idle_time: float = 300.0,
    ) -> None:
        self._max_size = max_size
        self._min_size = min_size
        self._max_idle_time = max_idle_time
        self._connections: Dict[Any, datetime] = {}
        self._lock = asyncio.Lock()
        self._semaphore = asyncio.Semaphore(max_size)
    
    @abstractmethod
    async def _create_connection(self) -> Any:
        """创建连接（子类实现）"""
        pass
    
    @abstractmethod
    async def _destroy_connection(self, conn: Any) -> None:
        """销毁连接（子类实现）"""
        pass
    
    @abstractmethod
    async def _validate_connection(self, conn: Any) -> bool:
        """验证连接（子类实现）"""
        pass
    
    async def acquire(self) -> Any:
        """获取连接"""
        await self._semaphore.acquire()
        
        async with self._lock:
            for conn, last_used in list(self._connections.items()):
                idle_time = (datetime.now() - last_used).total_seconds()
                
                if idle_time > self._max_idle_time:
                    await self._destroy_connection(conn)
                    del self._connections[conn]
                    continue
                
                if await self._validate_connection(conn):
                    self._connections[conn] = datetime.now()
                    return conn
            
            conn = await self._create_connection()
            self._connections[conn] = datetime.now()
            logger.debug(f"Created new connection, pool size: {len(self._connections)}")
            return conn
    
    async def release(self, conn: Any) -> None:
        """释放连接"""
        async with self._lock:
            if conn in self._connections:
                self._connections[conn] = datetime.now()
        
        self._semaphore.release()
        logger.debug("Connection released")
    
    async def close(self) -> None:
        """关闭连接池"""
        async with self._lock:
            for conn in list(self._connections.keys()):
                try:
                    await self._destroy_connection(conn)
                except Exception as e:
                    logger.error("Failed to destroy connection", error=str(e))
            
            self._connections.clear()
        
        logger.info("Connection pool closed")


@contextmanager
def managed_sync_resource(
    acquire_func: Callable,
    release_func: Callable,
    resource_name: str = "resource",
) -> Any:
    """
    同步资源上下文管理器
    
    Usage:
        with managed_sync_resource(acquire, release, "my_resource") as resource:
            # 使用资源
            pass
    """
    resource = None
    try:
        resource = acquire_func()
        logger.debug(f"Acquired sync resource: {resource_name}")
        yield resource
    except Exception as e:
        logger.error(f"Error in sync resource: {resource_name}", error=str(e))
        raise
    finally:
        if resource is not None:
            try:
                release_func(resource)
                logger.debug(f"Released sync resource: {resource_name}")
            except Exception as e:
                logger.error(f"Failed to release sync resource: {resource_name}", error=str(e))


@asynccontextmanager
async def managed_async_resource(
    acquire_func: Callable,
    release_func: Callable,
    resource_name: str = "resource",
) -> Any:
    """
    异步资源上下文管理器
    
    Usage:
        async with managed_async_resource(acquire, release, "my_resource") as resource:
            # 使用资源
            pass
    """
    resource = None
    try:
        resource = await acquire_func()
        logger.debug(f"Acquired async resource: {resource_name}")
        yield resource
    except Exception as e:
        logger.error(f"Error in async resource: {resource_name}", error=str(e))
        raise
    finally:
        if resource is not None:
            try:
                await release_func(resource)
                logger.debug(f"Released async resource: {resource_name}")
            except Exception as e:
                logger.error(f"Failed to release async resource: {resource_name}", error=str(e))


async def graceful_shutdown() -> None:
    """
    优雅关闭
    
    清理所有资源
    """
    logger.info("Starting graceful shutdown...")
    
    await ResourceTracker.cleanup_all()
    
    gc.collect()
    
    logger.info("Graceful shutdown completed")


def get_resource_stats() -> Dict[str, Any]:
    """获取资源统计"""
    return ResourceTracker.get_stats()
