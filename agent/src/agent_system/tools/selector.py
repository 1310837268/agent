"""
工具选择器模块

实现智能工具选择逻辑：
1. 基于用户意图匹配工具
2. 工具描述增强
3. 工具调用决策
4. MCP工具集成
"""

import json
from abc import ABC, abstractmethod
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

logger = get_logger(__name__)


class ToolSelectionResult(BaseModel):
    """工具选择结果"""
    selected_tools: List[Dict[str, Any]] = Field(default_factory=list)
    reasoning: str = ""
    should_call_tools: bool = False
    confidence: float = 0.0
    
    class Config:
        arbitrary_types_allowed = True


class ToolSelector(ABC):
    """工具选择器抽象基类"""
    
    @abstractmethod
    def select_tools(
        self,
        user_input: str,
        available_tools: List[BaseTool],
        conversation_history: Optional[List[BaseMessage]] = None,
        **kwargs: Any,
    ) -> ToolSelectionResult:
        """
        选择合适的工具
        
        Args:
            user_input: 用户输入
            available_tools: 可用工具列表
            conversation_history: 对话历史
            **kwargs: 其他参数
        
        Returns:
            ToolSelectionResult: 工具选择结果
        """
        pass


class LLMBasedToolSelector(ToolSelector):
    """
    基于LLM的工具选择器
    
    使用LLM来理解用户意图并选择合适的工具
    """
    
    def __init__(
        self,
        provider_manager: Any,
        model: Optional[str] = None,
        provider: Optional[str] = None,
    ) -> None:
        """
        初始化LLM工具选择器
        
        Args:
            provider_manager: Provider管理器
            model: 模型名称
            provider: Provider名称
        """
        self._provider_manager = provider_manager
        self._model = model
        self._provider = provider
    
    def _build_tool_descriptions(self, tools: List[BaseTool]) -> str:
        """构建工具描述"""
        descriptions = []
        
        for i, tool in enumerate(tools, 1):
            desc = f"{i}. {tool.name}: {tool.description}"
            
            if hasattr(tool, 'args_schema') and tool.args_schema:
                schema = tool.args_schema.model_json_schema()
                properties = schema.get('properties', {})
                required = schema.get('required', [])
                
                if properties:
                    params = []
                    for param_name, param_info in properties.items():
                        param_type = param_info.get('type', 'any')
                        param_desc = param_info.get('description', '')
                        is_required = param_name in required
                        params.append(
                            f"  - {param_name} ({param_type}, {'required' if is_required else 'optional'}): {param_desc}"
                        )
                    desc += "\n    参数:\n" + "\n".join(params)
            
            descriptions.append(desc)
        
        return "\n\n".join(descriptions)
    
    def _build_system_prompt(self) -> str:
        """构建系统提示词"""
        return """你是一个工具选择专家。你的任务是分析用户的问题，决定是否需要调用工具，以及调用哪些工具。

## 工具选择规则：

1. **仔细分析用户意图**：理解用户真正想要什么
2. **匹配工具能力**：检查哪个工具最适合解决用户的问题
3. **评估必要性**：
   - 如果用户的问题可以直接回答，不需要调用工具
   - 如果需要外部信息、计算或操作，需要调用工具
4. **考虑上下文**：参考对话历史，理解用户的完整需求

## 输出格式：

以JSON格式输出，包含以下字段：
- "should_call_tools": boolean，是否需要调用工具
- "selected_tools": array，选中的工具列表，每个工具包含：
  - "name": 工具名称
  - "reason": 选择这个工具的原因
  - "arguments": object，工具参数（如果已知）
- "reasoning": string，整体推理过程
- "confidence": float，0-1之间的置信度

## 示例：

用户输入："今天北京的天气怎么样？"
输出：
{
  "should_call_tools": true,
  "selected_tools": [
    {
      "name": "weather_search",
      "reason": "用户需要查询天气信息，这需要调用天气搜索工具",
      "arguments": {"location": "北京", "date": "今天"}
    }
  ],
  "reasoning": "用户询问北京今天的天气，这是一个需要外部信息的问题，应该调用天气搜索工具。",
  "confidence": 0.95
}

用户输入："你好"
输出：
{
  "should_call_tools": false,
  "selected_tools": [],
  "reasoning": "用户只是打招呼，不需要调用任何工具，可以直接回答。",
  "confidence": 0.99
}
"""
    
    def select_tools(
        self,
        user_input: str,
        available_tools: List[BaseTool],
        conversation_history: Optional[List[BaseMessage]] = None,
        **kwargs: Any,
    ) -> ToolSelectionResult:
        """
        选择合适的工具
        
        Args:
            user_input: 用户输入
            available_tools: 可用工具列表
            conversation_history: 对话历史
            **kwargs: 其他参数
        
        Returns:
            ToolSelectionResult: 工具选择结果
        """
        if not available_tools:
            return ToolSelectionResult(
                selected_tools=[],
                reasoning="没有可用的工具",
                should_call_tools=False,
                confidence=1.0,
            )
        
        tool_descriptions = self._build_tool_descriptions(available_tools)
        
        user_prompt = f"""## 可用工具：

{tool_descriptions}

## 用户输入：
{user_input}

## 对话历史（如果有）：
"""
        
        if conversation_history:
            for msg in conversation_history[-5:]:
                if isinstance(msg, HumanMessage):
                    user_prompt += f"用户: {msg.content}\n"
                elif isinstance(msg, AIMessage):
                    user_prompt += f"助手: {msg.content}\n"
        
        user_prompt += """
## 请分析并输出JSON结果：
"""
        
        messages = [
            SystemMessage(content=self._build_system_prompt()),
            HumanMessage(content=user_prompt),
        ]
        
        try:
            result, _ = self._provider_manager.generate_with_fallback(
                messages,
                preferred_provider=self._provider,
                **kwargs,
            )
            
            response_content = result.generations[0].message.content
            
            json_match = self._extract_json(response_content)
            
            if json_match:
                try:
                    parsed = json.loads(json_match)
                    return ToolSelectionResult(
                        selected_tools=parsed.get("selected_tools", []),
                        reasoning=parsed.get("reasoning", ""),
                        should_call_tools=parsed.get("should_call_tools", False),
                        confidence=parsed.get("confidence", 0.5),
                    )
                except json.JSONDecodeError:
                    pass
            
            return ToolSelectionResult(
                selected_tools=[],
                reasoning=f"无法解析LLM响应: {response_content[:100]}",
                should_call_tools=False,
                confidence=0.0,
            )
            
        except Exception as e:
            logger.error("Tool selection failed", error=str(e))
            return ToolSelectionResult(
                selected_tools=[],
                reasoning=f"工具选择失败: {str(e)}",
                should_call_tools=False,
                confidence=0.0,
            )
    
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


class KeywordBasedToolSelector(ToolSelector):
    """
    基于关键词的工具选择器
    
    简单快速的工具选择，适合不需要复杂推理的场景
    """
    
    def __init__(self) -> None:
        self._tool_keywords: Dict[str, List[str]] = {}
    
    def register_tool_keywords(self, tool_name: str, keywords: List[str]) -> None:
        """注册工具关键词"""
        self._tool_keywords[tool_name] = keywords
    
    def select_tools(
        self,
        user_input: str,
        available_tools: List[BaseTool],
        conversation_history: Optional[List[BaseMessage]] = None,
        **kwargs: Any,
    ) -> ToolSelectionResult:
        """
        基于关键词选择工具
        
        Args:
            user_input: 用户输入
            available_tools: 可用工具列表
            conversation_history: 对话历史
            **kwargs: 其他参数
        
        Returns:
            ToolSelectionResult: 工具选择结果
        """
        if not available_tools:
            return ToolSelectionResult(
                selected_tools=[],
                reasoning="没有可用的工具",
                should_call_tools=False,
                confidence=1.0,
            )
        
        user_input_lower = user_input.lower()
        selected_tools = []
        reasons = []
        
        for tool in available_tools:
            tool_name = tool.name
            tool_name_lower = tool_name.lower()
            
            if tool_name_lower in user_input_lower:
                selected_tools.append({
                    "name": tool_name,
                    "reason": f"用户输入中提到了工具名称 '{tool_name}'",
                    "arguments": {},
                })
                reasons.append(f"检测到工具名称: {tool_name}")
                continue
            
            keywords = self._tool_keywords.get(tool_name, [])
            for keyword in keywords:
                if keyword.lower() in user_input_lower:
                    selected_tools.append({
                        "name": tool_name,
                        "reason": f"用户输入中包含关键词 '{keyword}'",
                        "arguments": {},
                    })
                    reasons.append(f"检测到关键词: {keyword}")
                    break
            
            tool_desc_lower = tool.description.lower()
            for word in user_input_lower.split():
                if len(word) > 2 and word in tool_desc_lower:
                    if tool_name not in [t["name"] for t in selected_tools]:
                        selected_tools.append({
                            "name": tool_name,
                            "reason": f"工具描述与用户输入相关",
                            "arguments": {},
                        })
                        reasons.append(f"工具描述匹配: {tool_name}")
                    break
        
        if selected_tools:
            return ToolSelectionResult(
                selected_tools=selected_tools,
                reasoning="; ".join(reasons),
                should_call_tools=True,
                confidence=0.7,
            )
        
        return ToolSelectionResult(
            selected_tools=[],
            reasoning="没有找到匹配的工具",
            should_call_tools=False,
            confidence=0.5,
        )


class HybridToolSelector(ToolSelector):
    """
    混合工具选择器
    
    结合关键词匹配和LLM推理：
    1. 先用关键词快速筛选
    2. 再用LLM进行精确推理
    """
    
    def __init__(
        self,
        provider_manager: Any,
        model: Optional[str] = None,
        provider: Optional[str] = None,
        keyword_threshold: int = 1,
    ) -> None:
        """
        初始化混合工具选择器
        
        Args:
            provider_manager: Provider管理器
            model: 模型名称
            provider: Provider名称
            keyword_threshold: 关键词匹配阈值
        """
        self._llm_selector = LLMBasedToolSelector(provider_manager, model, provider)
        self._keyword_selector = KeywordBasedToolSelector()
        self._keyword_threshold = keyword_threshold
    
    def register_tool_keywords(self, tool_name: str, keywords: List[str]) -> None:
        """注册工具关键词"""
        self._keyword_selector.register_tool_keywords(tool_name, keywords)
    
    def select_tools(
        self,
        user_input: str,
        available_tools: List[BaseTool],
        conversation_history: Optional[List[BaseMessage]] = None,
        **kwargs: Any,
    ) -> ToolSelectionResult:
        """
        混合选择工具
        
        Args:
            user_input: 用户输入
            available_tools: 可用工具列表
            conversation_history: 对话历史
            **kwargs: 其他参数
        
        Returns:
            ToolSelectionResult: 工具选择结果
        """
        if not available_tools:
            return ToolSelectionResult(
                selected_tools=[],
                reasoning="没有可用的工具",
                should_call_tools=False,
                confidence=1.0,
            )
        
        keyword_result = self._keyword_selector.select_tools(
            user_input,
            available_tools,
            conversation_history,
            **kwargs,
        )
        
        if keyword_result.should_call_tools and len(keyword_result.selected_tools) >= self._keyword_threshold:
            logger.debug(
                "Keyword matching found tools, using LLM for verification",
                tools=[t["name"] for t in keyword_result.selected_tools],
            )
            
            matched_tool_names = [t["name"] for t in keyword_result.selected_tools]
            filtered_tools = [
                t for t in available_tools if t.name in matched_tool_names
            ]
            
            llm_result = self._llm_selector.select_tools(
                user_input,
                filtered_tools,
                conversation_history,
                **kwargs,
            )
            
            if llm_result.should_call_tools:
                return llm_result
            else:
                return ToolSelectionResult(
                    selected_tools=[],
                    reasoning=f"关键词匹配到工具，但LLM认为不需要调用: {llm_result.reasoning}",
                    should_call_tools=False,
                    confidence=0.6,
                )
        
        return self._llm_selector.select_tools(
            user_input,
            available_tools,
            conversation_history,
            **kwargs,
        )


def create_tool_selector(
    selector_type: str = "hybrid",
    **kwargs: Any,
) -> ToolSelector:
    """
    工厂函数：创建工具选择器
    
    Args:
        selector_type: 选择器类型
            - "llm": 基于LLM的选择器
            - "keyword": 基于关键词的选择器
            - "hybrid": 混合选择器（推荐）
        **kwargs: 其他参数
    
    Returns:
        ToolSelector: 工具选择器实例
    """
    if selector_type == "llm":
        return LLMBasedToolSelector(**kwargs)
    elif selector_type == "keyword":
        return KeywordBasedToolSelector()
    elif selector_type == "hybrid":
        return HybridToolSelector(**kwargs)
    else:
        raise ValueError(f"Unknown selector type: {selector_type}")
