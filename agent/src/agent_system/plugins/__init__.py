"""
Plugins模块初始化
"""

from agent_system.plugins.example_plugin import ExamplePlugin
from agent_system.plugins.manager import HookManager, Plugin, PluginManager

__all__ = [
    "Plugin",
    "HookManager",
    "PluginManager",
    "ExamplePlugin",
]
