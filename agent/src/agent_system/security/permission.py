"""
权限控制模块
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Set

from agent_system.core.config import PermissionConfig
from agent_system.core.exceptions import PermissionError
from agent_system.core.logging import get_logger

logger = get_logger(__name__)


class PermissionProvider(ABC):
    """权限提供者抽象类"""
    
    @abstractmethod
    def has_permission(self, user_role: str, permission: str) -> bool:
        """检查用户角色是否有指定权限"""
        pass
    
    @abstractmethod
    def get_role_permissions(self, user_role: str) -> Set[str]:
        """获取角色的所有权限"""
        pass
    
    @abstractmethod
    def add_role_permission(self, role: str, permission: str) -> None:
        """添加角色权限"""
        pass
    
    @abstractmethod
    def remove_role_permission(self, role: str, permission: str) -> None:
        """移除角色权限"""
        pass


class InMemoryPermissionProvider(PermissionProvider):
    """内存权限提供者"""
    
    def __init__(self, config: PermissionConfig) -> None:
        self._role_permissions: Dict[str, Set[str]] = {}
        
        for role, permissions in config.roles.items():
            self._role_permissions[role] = set(permissions)
        
        self._default_role = config.default_role
    
    def has_permission(self, user_role: str, permission: str) -> bool:
        """检查用户角色是否有指定权限"""
        role = user_role or self._default_role
        
        if role not in self._role_permissions:
            return False
        
        permissions = self._role_permissions[role]
        
        if "*" in permissions:
            return True
        
        if permission in permissions:
            return True
        
        for perm in permissions:
            if perm.endswith("*"):
                prefix = perm[:-1]
                if permission.startswith(prefix):
                    return True
        
        return False
    
    def get_role_permissions(self, user_role: str) -> Set[str]:
        """获取角色的所有权限"""
        return self._role_permissions.get(user_role, set())
    
    def add_role_permission(self, role: str, permission: str) -> None:
        """添加角色权限"""
        if role not in self._role_permissions:
            self._role_permissions[role] = set()
        self._role_permissions[role].add(permission)
        logger.info("Permission added", role=role, permission=permission)
    
    def remove_role_permission(self, role: str, permission: str) -> None:
        """移除角色权限"""
        if role in self._role_permissions:
            self._role_permissions[role].discard(permission)
            logger.info("Permission removed", role=role, permission=permission)


class PermissionManager:
    """
    权限管理器
    
    管理权限检查、角色分配等
    """
    
    def __init__(self, config: PermissionConfig) -> None:
        self.config = config
        self._enabled = config.enabled
        self._provider: PermissionProvider = InMemoryPermissionProvider(config)
        
        logger.info(
            "PermissionManager initialized",
            enabled=self._enabled,
            default_role=config.default_role,
        )
    
    def set_provider(self, provider: PermissionProvider) -> None:
        """设置权限提供者"""
        self._provider = provider
    
    def check_permission(
        self,
        permission: str,
        user_role: Optional[str] = None,
        raise_error: bool = True,
    ) -> bool:
        """
        检查权限
        
        Args:
            permission: 权限名称
            user_role: 用户角色
            raise_error: 是否抛出异常
        
        Returns:
            bool: 是否有权限
        """
        if not self._enabled:
            return True
        
        has_perm = self._provider.has_permission(user_role or self.config.default_role, permission)
        
        if not has_perm and raise_error:
            raise PermissionError(
                f"Permission denied: {permission}",
                required_permission=permission,
                user_role=user_role,
            )
        
        return has_perm
    
    def require_permission(self, permission: str):
        """
        权限检查装饰器
        
        Usage:
            @permission_manager.require_permission("execute_tools")
            def my_function():
                pass
        """
        def decorator(func):
            def wrapper(*args, **kwargs):
                user_role = kwargs.get('user_role')
                if len(args) > 0 and hasattr(args[0], 'config'):
                    user_role = getattr(args[0].config, 'role', None)
                
                self.check_permission(permission, user_role)
                return func(*args, **kwargs)
            
            async def async_wrapper(*args, **kwargs):
                user_role = kwargs.get('user_role')
                if len(args) > 0 and hasattr(args[0], 'config'):
                    user_role = getattr(args[0].config, 'role', None)
                
                self.check_permission(permission, user_role)
                return await func(*args, **kwargs)
            
            import inspect
            if inspect.iscoroutinefunction(func):
                return async_wrapper
            return wrapper
        
        return decorator
    
    def get_role_permissions(self, user_role: str) -> Set[str]:
        """获取角色的所有权限"""
        return self._provider.get_role_permissions(user_role)
    
    def add_role_permission(self, role: str, permission: str) -> None:
        """添加角色权限"""
        self._provider.add_role_permission(role, permission)
    
    def remove_role_permission(self, role: str, permission: str) -> None:
        """移除角色权限"""
        self._provider.remove_role_permission(role, permission)
    
    def is_enabled(self) -> bool:
        """检查权限控制是否启用"""
        return self._enabled
    
    def enable(self) -> None:
        """启用权限控制"""
        self._enabled = True
        logger.info("Permission control enabled")
    
    def disable(self) -> None:
        """禁用权限控制"""
        self._enabled = False
        logger.info("Permission control disabled")
