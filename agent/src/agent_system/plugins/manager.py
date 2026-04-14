"""
插件扩展系统
"""

import importlib
import importlib.util
import inspect
import os
import sys
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Dict, List, Optional, Type

from agent_system.core.config import PluginConfig
from agent_system.core.exceptions import PluginError
from agent_system.core.logging import get_logger

logger = get_logger(__name__)


class Plugin(ABC):
    """插件基类"""
    
    name: str
    version: str
    description: str
    author: str = ""
    
    def __init__(self) -> None:
        self._loaded: bool = False
        self._enabled: bool = False
        self._metadata: Dict[str, Any] = {}
    
    @abstractmethod
    def on_load(self) -> None:
        """插件加载时调用"""
        pass
    
    @abstractmethod
    def on_unload(self) -> None:
        """插件卸载时调用"""
        pass
    
    def on_enable(self) -> None:
        """插件启用时调用"""
        pass
    
    def on_disable(self) -> None:
        """插件禁用时调用"""
        pass
    
    def get_hooks(self) -> Dict[str, Any]:
        """
        获取插件提供的钩子
        
        返回格式:
        {
            "hook_name": callback_function,
            ...
        }
        """
        return {}
    
    def get_tools(self) -> List[Any]:
        """
        获取插件提供的工具
        
        返回LangChain兼容的工具列表
        """
        return []
    
    def get_agents(self) -> List[Type]:
        """
        获取插件提供的Agent类
        """
        return []
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "version": self.version,
            "description": self.description,
            "author": self.author,
            "loaded": self._loaded,
            "enabled": self._enabled,
            "metadata": self._metadata,
        }


class HookManager:
    """钩子管理器"""
    
    def __init__(self) -> None:
        self._hooks: Dict[str, List[Any]] = {}
    
    def register_hook(self, hook_name: str, callback: Any) -> None:
        """注册钩子"""
        if hook_name not in self._hooks:
            self._hooks[hook_name] = []
        self._hooks[hook_name].append(callback)
        logger.debug("Hook registered", hook=hook_name)
    
    def unregister_hook(self, hook_name: str, callback: Any) -> bool:
        """注销钩子"""
        if hook_name in self._hooks:
            try:
                self._hooks[hook_name].remove(callback)
                logger.debug("Hook unregistered", hook=hook_name)
                return True
            except ValueError:
                pass
        return False
    
    def execute_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """执行钩子"""
        results = []
        
        if hook_name not in self._hooks:
            return results
        
        for callback in self._hooks[hook_name]:
            try:
                import inspect
                if inspect.iscoroutinefunction(callback):
                    import asyncio
                    result = asyncio.run(callback(*args, **kwargs))
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
                logger.debug(
                    "Hook executed",
                    hook=hook_name,
                    callback=callback.__name__,
                )
            except Exception as e:
                logger.error(
                    "Hook execution failed",
                    hook=hook_name,
                    error=str(e),
                )
        
        return results
    
    async def aexecute_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """异步执行钩子"""
        results = []
        
        if hook_name not in self._hooks:
            return results
        
        for callback in self._hooks[hook_name]:
            try:
                import inspect
                if inspect.iscoroutinefunction(callback):
                    result = await callback(*args, **kwargs)
                else:
                    result = callback(*args, **kwargs)
                results.append(result)
                logger.debug(
                    "Async hook executed",
                    hook=hook_name,
                    callback=callback.__name__,
                )
            except Exception as e:
                logger.error(
                    "Async hook execution failed",
                    hook=hook_name,
                    error=str(e),
                )
        
        return results
    
    def get_hook_names(self) -> List[str]:
        """获取所有钩子名称"""
        return list(self._hooks.keys())


class PluginManager:
    """
    插件管理器
    
    管理插件的加载、启用、禁用和卸载
    """
    
    def __init__(self, config: PluginConfig) -> None:
        self.config = config
        self._plugins: Dict[str, Plugin] = {}
        self._hook_manager = HookManager()
        self._plugin_dirs: List[str] = list(config.plugin_dirs)
    
    def _load_plugin_from_file(self, file_path: str) -> Optional[Plugin]:
        """从文件加载插件"""
        try:
            module_name = os.path.splitext(os.path.basename(file_path))[0]
            
            spec = importlib.util.spec_from_file_location(module_name, file_path)
            if spec is None or spec.loader is None:
                logger.warning("Could not load plugin spec", file=file_path)
                return None
            
            module = importlib.util.module_from_spec(spec)
            sys.modules[module_name] = module
            spec.loader.exec_module(module)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj is not Plugin:
                    plugin = obj()
                    return plugin
            
            logger.warning("No Plugin subclass found in file", file=file_path)
            return None
            
        except Exception as e:
            logger.error(
                "Failed to load plugin from file",
                file=file_path,
                error=str(e),
            )
            return None
    
    def _load_plugin_from_module(self, module_name: str) -> Optional[Plugin]:
        """从模块加载插件"""
        try:
            module = importlib.import_module(module_name)
            
            for name, obj in inspect.getmembers(module, inspect.isclass):
                if issubclass(obj, Plugin) and obj is not Plugin:
                    plugin = obj()
                    return plugin
            
            logger.warning("No Plugin subclass found in module", module=module_name)
            return None
            
        except Exception as e:
            logger.error(
                "Failed to load plugin from module",
                module=module_name,
                error=str(e),
            )
            return None
    
    def load_plugin(self, plugin_path: str) -> Optional[Plugin]:
        """
        加载插件
        
        Args:
            plugin_path: 插件路径，可以是文件路径或模块名
        """
        if os.path.isfile(plugin_path):
            plugin = self._load_plugin_from_file(plugin_path)
        else:
            plugin = self._load_plugin_from_module(plugin_path)
        
        if plugin is None:
            return None
        
        if plugin.name in self._plugins:
            logger.warning("Plugin already loaded", name=plugin.name)
            return self._plugins[plugin.name]
        
        try:
            plugin.on_load()
            plugin._loaded = True
            
            self._plugins[plugin.name] = plugin
            
            hooks = plugin.get_hooks()
            for hook_name, callback in hooks.items():
                self._hook_manager.register_hook(hook_name, callback)
            
            logger.info(
                "Plugin loaded",
                name=plugin.name,
                version=plugin.version,
            )
            
            return plugin
            
        except Exception as e:
            logger.error(
                "Failed to initialize plugin",
                name=plugin.name,
                error=str(e),
            )
            raise PluginError(
                f"Failed to initialize plugin '{plugin.name}': {str(e)}",
                plugin.name,
            ) from e
    
    def load_plugins_from_dir(self, directory: str) -> List[Plugin]:
        """从目录加载所有插件"""
        if not os.path.isdir(directory):
            logger.warning("Plugin directory not found", directory=directory)
            return []
        
        loaded_plugins = []
        
        for filename in os.listdir(directory):
            if filename.startswith("_") or filename.startswith("."):
                continue
            
            file_path = os.path.join(directory, filename)
            
            if os.path.isfile(file_path) and filename.endswith(".py"):
                plugin = self.load_plugin(file_path)
                if plugin:
                    loaded_plugins.append(plugin)
            elif os.path.isdir(file_path):
                init_file = os.path.join(file_path, "__init__.py")
                if os.path.isfile(init_file):
                    plugin = self.load_plugin(file_path)
                    if plugin:
                        loaded_plugins.append(plugin)
        
        return loaded_plugins
    
    def enable_plugin(self, plugin_name: str) -> bool:
        """启用插件"""
        if plugin_name not in self._plugins:
            logger.warning("Plugin not found", name=plugin_name)
            return False
        
        plugin = self._plugins[plugin_name]
        
        if plugin._enabled:
            logger.warning("Plugin already enabled", name=plugin_name)
            return True
        
        try:
            plugin.on_enable()
            plugin._enabled = True
            logger.info("Plugin enabled", name=plugin_name)
            return True
        except Exception as e:
            logger.error(
                "Failed to enable plugin",
                name=plugin_name,
                error=str(e),
            )
            return False
    
    def disable_plugin(self, plugin_name: str) -> bool:
        """禁用插件"""
        if plugin_name not in self._plugins:
            logger.warning("Plugin not found", name=plugin_name)
            return False
        
        plugin = self._plugins[plugin_name]
        
        if not plugin._enabled:
            logger.warning("Plugin already disabled", name=plugin_name)
            return True
        
        try:
            plugin.on_disable()
            plugin._enabled = False
            logger.info("Plugin disabled", name=plugin_name)
            return True
        except Exception as e:
            logger.error(
                "Failed to disable plugin",
                name=plugin_name,
                error=str(e),
            )
            return False
    
    def unload_plugin(self, plugin_name: str) -> bool:
        """卸载插件"""
        if plugin_name not in self._plugins:
            logger.warning("Plugin not found", name=plugin_name)
            return False
        
        plugin = self._plugins[plugin_name]
        
        try:
            if plugin._enabled:
                plugin.on_disable()
            
            hooks = plugin.get_hooks()
            for hook_name, callback in hooks.items():
                self._hook_manager.unregister_hook(hook_name, callback)
            
            plugin.on_unload()
            plugin._loaded = False
            
            del self._plugins[plugin_name]
            
            logger.info("Plugin unloaded", name=plugin_name)
            return True
            
        except Exception as e:
            logger.error(
                "Failed to unload plugin",
                name=plugin_name,
                error=str(e),
            )
            return False
    
    def get_plugin(self, plugin_name: str) -> Optional[Plugin]:
        """获取插件"""
        return self._plugins.get(plugin_name)
    
    def get_all_plugins(self) -> List[Plugin]:
        """获取所有插件"""
        return list(self._plugins.values())
    
    def get_enabled_plugins(self) -> List[Plugin]:
        """获取所有启用的插件"""
        return [p for p in self._plugins.values() if p._enabled]
    
    def get_plugin_tools(self) -> List[Any]:
        """获取所有插件提供的工具"""
        tools = []
        for plugin in self.get_enabled_plugins():
            tools.extend(plugin.get_tools())
        return tools
    
    def execute_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """执行钩子"""
        return self._hook_manager.execute_hook(hook_name, *args, **kwargs)
    
    async def aexecute_hook(self, hook_name: str, *args: Any, **kwargs: Any) -> List[Any]:
        """异步执行钩子"""
        return await self._hook_manager.aexecute_hook(hook_name, *args, **kwargs)
    
    def auto_load(self) -> None:
        """自动加载配置的插件"""
        if not self.config.auto_load:
            return
        
        for plugin_dir in self.config.plugin_dirs:
            self.load_plugins_from_dir(plugin_dir)
        
        for plugin_name in self.config.enabled_plugins:
            if plugin_name not in self._plugins:
                self.load_plugin(plugin_name)
            
            self.enable_plugin(plugin_name)
        
        for plugin_name in self.config.disabled_plugins:
            if plugin_name in self._plugins:
                self.disable_plugin(plugin_name)
