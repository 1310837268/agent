"""
工具绑定模块

实现工具与LLM的绑定：
1. 工具描述增强
2. OpenAI函数调用格式转换
3. MCP工具集成
4. 工具调用决策
"""

import json
from typing import Any, Dict, List, Optional, Tuple, Union

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import BaseTool
from langchain_core.utils.function_calling import convert_to_openai_tool
from pydantic import BaseModel, Field

from agent_system.core.logging import get_logger
from agent_system.mcp.manager import MCPManager, MCPToolWrapper
from agent_system.tools.selector import (
    HybridToolSelector,
    ToolSelectionResult,
    ToolSelector,
    create_tool_selector,
)

logger = get_logger(__name__)


class ToolBindingResult(BaseModel):
    """工具绑定结果"""
    tool_calls: List[Dict[str, Any]] = Field(default_factory=list)
    tool_results: List[Dict[str, Any]] = Field(default_factory=list)
    should_continue: bool = False
    final_answer: Optional[str] = None
    
    class Config:
        arbitrary_types_allowed = True


class ToolBinder:
    """
    工具绑定器
    
    负责：
    1. 工具描述增强
    2. 工具选择
    3. 工具调用执行
    4. 结果处理
    """
    
    def __init__(
        self,
        tools: Optional[List[BaseTool]] = None,
        tool_selector: Optional[ToolSelector] = None,
        mcp_manager: Optional[MCPManager] = None,
        provider_manager: Optional[Any] = None,
    ) -> None:
        """
        初始化工具绑定器
        
        Args:
            tools: 工具列表
            tool_selector: 工具选择器
            mcp_manager: MCP管理器
            provider_manager: Provider管理器
        """
        self._tools: Dict[str, BaseTool] = {}
        self._tool_selector = tool_selector
        self._mcp_manager = mcp_manager
        self._provider_manager = provider_manager
        
        if tools:
            for tool in tools:
                self._tools[tool.name] = tool
        
        if mcp_manager:
            mcp_tools = mcp_manager.get_tools()
            for tool in mcp_tools:
                self._tools[tool.name] = tool
        
        logger.info(
            "ToolBinder initialized",
            tools=list(self._tools.keys()),
            has_mcp=bool(mcp_manager),
        )
    
    def add_tool(self, tool: BaseTool) -> None:
        """添加工具"""
        self._tools[tool.name] = tool
        logger.info("Tool added", tool=tool.name)
    
    def remove_tool(self, tool_name: str) -> bool:
        """移除工具"""
        if tool_name in self._tools:
            del self._tools[tool_name]
            logger.info("Tool removed", tool=tool_name)
            return True
        return False
    
    def get_tools(self) -> List[BaseTool]:
        """获取所有工具"""
        return list(self._tools.values())
    
    def get_tool_names(self) -> List[str]:
        """获取所有工具名称"""
        return list(self._tools.keys())
    
    def get_tool(self, tool_name: str) -> Optional[BaseTool]:
        """获取指定工具"""
        return self._tools.get(tool_name)
    
    def format_tools_for_llm(self) -> List[Dict[str, Any]]:
        """
        将工具格式化为LLM可用的格式
        
        使用OpenAI函数调用格式
        """
        formatted_tools = []
        
        for tool_name, tool in self._tools.items():
            try:
                formatted = convert_to_openai_tool(tool)
                formatted_tools.append(formatted)
            except Exception as e:
                logger.warning(
                    "Failed to format tool",
                    tool=tool_name,
                    error=str(e),
                )
        
        return formatted_tools
    
    def build_tool_system_prompt(self) -> str:
        """
        构建工具系统提示词
        
        让LLM了解可用的工具和如何使用它们
        """
        if not self._tools:
            return ""
        
        parts = [
            "## 可用工具",
            "",
            "你可以使用以下工具来帮助回答用户的问题：",
            "",
        ]
        
        for tool_name, tool in self._tools.items():
            parts.append(f"### {tool_name}")
            parts.append(f"描述: {tool.description}")
            
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                if properties:
                    parts.append("参数:")
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        is_required = param_name in required
                        parts.append(
                            f"  - {param_name} ({param_type}, {'必需' if is_required else '可选'}): {param_desc}"
                        )
            
            parts.append("")
        
        parts.extend([
            "## 工具使用规则",
            "",
            "1. 仔细分析用户的问题，判断是否需要使用工具",
            "2. 如果需要使用工具，选择最合适的工具",
            "3. 确保提供正确的参数",
            "4. 工具执行后，根据结果继续处理",
            "5. 如果不需要工具，直接回答用户的问题",
            "",
            "## 输出格式",
            "",
            "如果你决定调用工具，请使用以下JSON格式：",
            "```json",
            "{",
            '  "tool_calls": [',
            "    {",
            '      "name": "工具名称",',
            '      "arguments": {',
            '        "参数名": "参数值"',
            "      }",
            "    }",
            "  ]",
            "}",
            "```",
            "",
            "如果你决定不调用工具，直接回答用户的问题。",
        ])
        
        return "\n".join(parts)
    
    def select_tools(
        self,
        user_input: str,
        conversation_history: Optional[List[BaseMessage]] = None,
        **kwargs: Any,
    ) -> ToolSelectionResult:
        """
        选择工具
        
        Args:
            user_input: 用户输入
            conversation_history: 对话历史
            **kwargs: 其他参数
        
        Returns:
            ToolSelectionResult: 工具选择结果
        """
        if not self._tools:
            return ToolSelectionResult(
                selected_tools=[],
                reasoning="没有可用的工具",
                should_call_tools=False,
                confidence=1.0,
            )
        
        if self._tool_selector:
            return self._tool_selector.select_tools(
                user_input=user_input,
                available_tools=list(self._tools.values()),
                conversation_history=conversation_history,
                **kwargs,
            )
        
        return ToolSelectionResult(
            selected_tools=[],
            reasoning="没有配置工具选择器",
            should_call_tools=False,
            confidence=0.0,
        )
    
    async def execute_tool(
        self,
        tool_name: str,
        arguments: Dict[str, Any],
    ) -> Tuple[str, bool]:
        """
        执行工具
        
        Args:
            tool_name: 工具名称
            arguments: 工具参数
        
        Returns:
            Tuple[str, bool]: (结果, 是否成功)
        """
        tool = self._tools.get(tool_name)
        
        if not tool:
            error_msg = f"工具 '{tool_name}' 不存在"
            logger.error(error_msg)
            return error_msg, False
        
        try:
            logger.info(
                "Executing tool",
                tool=tool_name,
                arguments=arguments,
            )
            
            if isinstance(tool, MCPToolWrapper):
                if self._mcp_manager:
                    result = await self._mcp_manager.execute_tool(
                        tool_name=tool.tool_name,
                        arguments=arguments,
                    )
                    return result, True
                else:
                    return "MCP管理器未配置", False
            
            if hasattr(tool, 'ainvoke'):
                result = await tool.ainvoke(arguments)
            else:
                import asyncio
                result = await asyncio.to_thread(tool.invoke, arguments)
            
            logger.debug(
                "Tool executed successfully",
                tool=tool_name,
                result_length=len(str(result)) if result else 0,
            )
            
            return str(result), True
            
        except Exception as e:
            error_msg = f"执行工具 '{tool_name}' 时出错: {str(e)}"
            logger.error(error_msg, exc_info=True)
            return error_msg, False
    
    async def execute_tool_calls(
        self,
        tool_calls: List[Dict[str, Any]],
    ) -> List[Dict[str, Any]]:
        """
        执行多个工具调用
        
        Args:
            tool_calls: 工具调用列表
        
        Returns:
            List[Dict[str, Any]]: 工具执行结果列表
        """
        results = []
        
        for tool_call in tool_calls:
            tool_name = tool_call.get("name") or tool_call.get("tool_name")
            arguments = tool_call.get("arguments") or tool_call.get("tool_input", {})
            
            if not tool_name:
                results.append({
                    "tool_name": None,
                    "success": False,
                    "result": "工具名称未指定",
                })
                continue
            
            result, success = await self.execute_tool(tool_name, arguments)
            
            results.append({
                "tool_name": tool_name,
                "arguments": arguments,
                "success": success,
                "result": result,
            })
        
        return results
    
    def parse_tool_calls_from_message(
        self,
        message: BaseMessage,
    ) -> List[Dict[str, Any]]:
        """
        从消息中解析工具调用
        
        Args:
            message: 消息
        
        Returns:
            List[Dict[str, Any]]: 工具调用列表
        """
        tool_calls = []
        
        if hasattr(message, 'tool_calls') and message.tool_calls:
            for tc in message.tool_calls:
                tool_calls.append({
                    "name": tc.get("name") if isinstance(tc, dict) else getattr(tc, 'name', None),
                    "arguments": tc.get("args") if isinstance(tc, dict) else getattr(tc, 'args', {}),
                    "id": tc.get("id") if isinstance(tc, dict) else getattr(tc, 'id', None),
                })
        
        if not tool_calls and hasattr(message, 'content') and message.content:
            content = message.content
            if isinstance(content, str):
                tool_calls = self._parse_tool_calls_from_text(content)
        
        return tool_calls
    
    def _parse_tool_calls_from_text(self, text: str) -> List[Dict[str, Any]]:
        """
        从文本中解析工具调用
        
        尝试从JSON或文本中提取工具调用
        """
        tool_calls = []
        
        json_match = self._extract_json(text)
        if json_match:
            try:
                parsed = json.loads(json_match)
                
                if "tool_calls" in parsed:
                    for tc in parsed["tool_calls"]:
                        tool_calls.append({
                            "name": tc.get("name") or tc.get("tool_name"),
                            "arguments": tc.get("arguments") or tc.get("tool_input", {}),
                        })
                elif "name" in parsed:
                    tool_calls.append({
                        "name": parsed.get("name") or parsed.get("tool_name"),
                        "arguments": parsed.get("arguments") or parsed.get("tool_input", {}),
                    })
            except json.JSONDecodeError:
                pass
        
        return tool_calls
    
    def _extract_json(self, text: str) -> Optional[str]:
        """从文本中提取JSON"""
        start = text.find('{')
        if start == -1:
            return None
        
        brace_count = 0
        in_string = False
        escape_next = False
        
        for i in range(start, len(text)):
            char = text[i]
            
            if escape_next:
                escape_next = False
                continue
            
            if char == '\\':
                escape_next = True
                continue
            
            if char == '"' and not escape_next:
                in_string = not in_string
                continue
            
            if not in_string:
                if char == '{':
                    brace_count += 1
                elif char == '}':
                    brace_count -= 1
                    if brace_count == 0:
                        return text[start:i+1]
        
        return None
    
    def has_tools(self) -> bool:
        """检查是否有可用工具"""
        return len(self._tools) > 0
    
    def get_tool_count(self) -> int:
        """获取工具数量"""
        return len(self._tools)
