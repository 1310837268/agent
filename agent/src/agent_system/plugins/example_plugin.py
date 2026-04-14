"""
示例插件
"""

from typing import Any, Dict, List

from langchain_core.tools import tool

from agent_system.plugins.manager import Plugin


class ExamplePlugin(Plugin):
    """示例插件"""
    
    name = "example_plugin"
    version = "1.0.0"
    description = "示例插件，展示插件系统的基本功能"
    author = "Agent System"
    
    def on_load(self) -> None:
        """插件加载时调用"""
        print(f"Plugin {self.name} loaded")
    
    def on_unload(self) -> None:
        """插件卸载时调用"""
        print(f"Plugin {self.name} unloaded")
    
    def on_enable(self) -> None:
        """插件启用时调用"""
        print(f"Plugin {self.name} enabled")
    
    def on_disable(self) -> None:
        """插件禁用时调用"""
        print(f"Plugin {self.name} disabled")
    
    def get_hooks(self) -> Dict[str, Any]:
        """获取插件提供的钩子"""
        return {
            "before_agent_execute": self._before_agent_execute,
            "after_agent_execute": self._after_agent_execute,
        }
    
    def get_tools(self) -> List[Any]:
        """获取插件提供的工具"""
        
        @tool
        def get_current_time() -> str:
            """获取当前时间"""
            from datetime import datetime
            return datetime.now().isoformat()
        
        @tool
        def calculate(expression: str) -> str:
            """
            计算数学表达式
            
            Args:
                expression: 数学表达式，如 "2 + 3 * 4"
            """
            try:
                result = eval(expression, {"__builtins__": {}})
                return str(result)
            except Exception as e:
                return f"Error: {str(e)}"
        
        return [get_current_time, calculate]
    
    def _before_agent_execute(self, agent_name: str, user_input: str) -> Dict[str, Any]:
        """Agent执行前的钩子"""
        return {
            "plugin": self.name,
            "action": "before_execute",
            "agent": agent_name,
            "input_preview": user_input[:50] if user_input else "",
        }
    
    def _after_agent_execute(self, agent_name: str, result: Any) -> Dict[str, Any]:
        """Agent执行后的钩子"""
        return {
            "plugin": self.name,
            "action": "after_execute",
            "agent": agent_name,
            "success": True,
        }
