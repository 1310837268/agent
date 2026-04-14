"""
Providers模块初始化

支持的Provider类型：
- openai: OpenAI官方API
- anthropic: Anthropic Claude API
- openai_compatible: 任何兼容OpenAI API的服务（国产模型、本地部署等）
- ollama: 本地Ollama模型

国产模型支持（通过openai_compatible）：
- DeepSeek: deepseek-chat, deepseek-coder
- Qwen (通义千问): qwen-turbo, qwen-plus, qwen-max
- 智谱清言: glm-4, glm-3-turbo
- Moonshot: moonshot-v1-8k/32k/128k
- StepFun: step-1-8k/32k/128k
- MiniMax: abab6.5s-chat
- 百川智能: Baichuan3-Turbo

本地模型支持：
- Ollama: llama3, mistral, qwen, deepseek-coder等
"""

from agent_system.providers.anthropic_provider import AnthropicProvider
from agent_system.providers.base import BaseProvider
from agent_system.providers.chinese_models import (
    CHINESE_MODELS,
    ChineseModelProvider,
)
from agent_system.providers.manager import (
    ProviderManager,
    get_registered_provider_types,
    register_provider_type,
)
from agent_system.providers.ollama_provider import OllamaProvider
from agent_system.providers.openai_compatible import OpenAICompatibleProvider
from agent_system.providers.openai_provider import OpenAIProvider

PROVIDER_REGISTRY = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openai_compatible": OpenAICompatibleProvider,
    "ollama": OllamaProvider,
}

__all__ = [
    "BaseProvider",
    "OpenAIProvider",
    "AnthropicProvider",
    "OpenAICompatibleProvider",
    "OllamaProvider",
    "ProviderManager",
    "register_provider_type",
    "get_registered_provider_types",
    "ChineseModelProvider",
    "CHINESE_MODELS",
    "PROVIDER_REGISTRY",
]
