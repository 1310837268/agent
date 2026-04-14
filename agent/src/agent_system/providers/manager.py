"""
Provider管理器 - 支持动态切换与故障回退

增强版：
1. 支持动态注册新的Provider类型
2. 支持本地模型（Ollama）
3. 支持国产模型（DeepSeek、Qwen、智谱等）
4. 支持任何OpenAI兼容接口
"""

import asyncio
from typing import Any, AsyncIterator, Callable, Dict, Iterator, List, Optional, Tuple, Type

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_system.core.config import ProviderConfig
from agent_system.core.exceptions import (
    ProviderDisabledError,
    ProviderFallbackError,
    ProviderNotFoundError,
)
from agent_system.core.logging import get_logger
from agent_system.providers.anthropic_provider import AnthropicProvider
from agent_system.providers.base import BaseProvider
from agent_system.providers.openai_compatible import OpenAICompatibleProvider
from agent_system.providers.openai_provider import OpenAIProvider

logger = get_logger(__name__)


PROVIDER_REGISTRY: Dict[str, Type[BaseProvider]] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
    "openai_compatible": OpenAICompatibleProvider,
}


def register_provider_type(
    provider_type: str,
    provider_class: Type[BaseProvider],
    override: bool = False,
) -> None:
    """
    注册新的Provider类型
    
    这是一个全局函数，可以在任何地方调用以注册新的Provider类型。
    
    Args:
        provider_type: Provider类型名称
        provider_class: Provider类（必须继承自BaseProvider）
        override: 是否覆盖已存在的类型
    
    Example:
        ```python
        from agent_system.providers.base import BaseProvider
        from agent_system.providers.manager import register_provider_type
        
        class MyCustomProvider(BaseProvider):
            # 实现自定义Provider
            pass
        
        register_provider_type("my_custom", MyCustomProvider)
        ```
    """
    global PROVIDER_REGISTRY
    
    if not issubclass(provider_class, BaseProvider):
        raise ValueError(
            f"Provider class must inherit from BaseProvider, got {provider_class}"
        )
    
    if provider_type in PROVIDER_REGISTRY and not override:
        raise ValueError(
            f"Provider type '{provider_type}' already registered. "
            f"Use override=True to replace it."
        )
    
    PROVIDER_REGISTRY[provider_type] = provider_class
    logger.info("New provider type registered", provider_type=provider_type)


def get_registered_provider_types() -> List[str]:
    """获取所有已注册的Provider类型"""
    return list(PROVIDER_REGISTRY.keys())


class ProviderManager:
    """
    Provider管理器
    
    功能：
    1. 管理多个Provider实例
    2. 支持动态切换Provider
    3. 支持故障回退机制
    4. 支持优先级排序
    5. 支持动态注册新的Provider类型
    6. 支持本地模型（Ollama）
    7. 支持国产模型（DeepSeek、Qwen、智谱等）
    """
    
    def __init__(
        self,
        configs: List[ProviderConfig],
        default_provider: Optional[str] = None,
    ) -> None:
        self._configs: Dict[str, ProviderConfig] = {}
        self._providers: Dict[str, BaseProvider] = {}
        self._default_provider = default_provider
        self._local_registry: Dict[str, Type[BaseProvider]] = {}
        
        for config in configs:
            self._configs[config.name] = config
            if config.enabled:
                self._providers[config.name] = self._create_provider(config)
        
        if not self._default_provider and self._providers:
            sorted_providers = sorted(
                self._providers.values(),
                key=lambda p: p.get_priority(),
                reverse=True,
            )
            self._default_provider = sorted_providers[0].name
        
        logger.info(
            "ProviderManager initialized",
            default_provider=self._default_provider,
            enabled_providers=list(self._providers.keys()),
            registered_types=self.get_registered_types(),
        )
    
    def _get_provider_class(self, provider_type: str) -> Type[BaseProvider]:
        """获取Provider类"""
        if provider_type in self._local_registry:
            return self._local_registry[provider_type]
        
        if provider_type in PROVIDER_REGISTRY:
            return PROVIDER_REGISTRY[provider_type]
        
        raise ValueError(
            f"Unknown provider type: {provider_type}. "
            f"Registered types: {self.get_registered_types()}"
        )
    
    def _create_provider(self, config: ProviderConfig) -> BaseProvider:
        """创建Provider实例"""
        provider_class = self._get_provider_class(config.provider_type)
        return provider_class(config)
    
    def register_provider_type(
        self,
        provider_type: str,
        provider_class: Type[BaseProvider],
        override: bool = False,
    ) -> None:
        """
        注册新的Provider类型（实例级别）
        
        这个方法只影响当前ProviderManager实例，不影响全局注册表。
        
        Args:
            provider_type: Provider类型名称
            provider_class: Provider类
            override: 是否覆盖已存在的类型
        """
        if not issubclass(provider_class, BaseProvider):
            raise ValueError(
                f"Provider class must inherit from BaseProvider, got {provider_class}"
            )
        
        if provider_type in self._local_registry and not override:
            raise ValueError(
                f"Provider type '{provider_type}' already registered in this instance. "
                f"Use override=True to replace it."
            )
        
        self._local_registry[provider_type] = provider_class
        logger.info(
            "New provider type registered in instance",
            provider_type=provider_type,
        )
    
    def get_registered_types(self) -> List[str]:
        """获取所有已注册的Provider类型"""
        types = set(PROVIDER_REGISTRY.keys())
        types.update(self._local_registry.keys())
        return sorted(list(types))
    
    def get_provider(self, name: Optional[str] = None) -> BaseProvider:
        """获取Provider实例"""
        provider_name = name or self._default_provider
        
        if not provider_name:
            raise ProviderNotFoundError("No default provider configured")
        
        if provider_name not in self._providers:
            if provider_name in self._configs:
                config = self._configs[provider_name]
                if not config.enabled:
                    raise ProviderDisabledError(provider_name)
            raise ProviderNotFoundError(provider_name)
        
        return self._providers[provider_name]
    
    def get_available_providers(self) -> List[BaseProvider]:
        """获取所有可用的Provider（按优先级排序）"""
        return sorted(
            self._providers.values(),
            key=lambda p: p.get_priority(),
            reverse=True,
        )
    
    def register_provider(self, config: ProviderConfig) -> None:
        """注册新的Provider"""
        self._configs[config.name] = config
        if config.enabled:
            self._providers[config.name] = self._create_provider(config)
            logger.info("Provider registered", provider=config.name)
    
    def register_provider_from_dict(self, config_dict: Dict[str, Any]) -> None:
        """
        从字典注册Provider
        
        便捷方法，用于动态创建Provider配置。
        
        Example:
            ```python
            manager.register_provider_from_dict({
                "name": "my-ollama",
                "provider_type": "ollama",
                "model": "llama3",
                "base_url": "http://localhost:11434",
            })
            ```
        """
        config = ProviderConfig(**config_dict)
        self.register_provider(config)
    
    def unregister_provider(self, name: str) -> None:
        """注销Provider"""
        if name in self._providers:
            del self._providers[name]
        if name in self._configs:
            del self._configs[name]
        logger.info("Provider unregistered", provider=name)
    
    def enable_provider(self, name: str) -> None:
        """启用Provider"""
        if name not in self._configs:
            raise ProviderNotFoundError(name)
        
        config = self._configs[name]
        config.enabled = True
        
        if name not in self._providers:
            self._providers[name] = self._create_provider(config)
        
        logger.info("Provider enabled", provider=name)
    
    def disable_provider(self, name: str) -> None:
        """禁用Provider"""
        if name not in self._configs:
            raise ProviderNotFoundError(name)
        
        self._configs[name].enabled = False
        
        if name in self._providers:
            del self._providers[name]
        
        logger.info("Provider disabled", provider=name)
    
    def set_default_provider(self, name: str) -> None:
        """设置默认Provider"""
        if name not in self._providers:
            raise ProviderNotFoundError(name)
        self._default_provider = name
        logger.info("Default provider set", provider=name)
    
    def _get_fallback_chain(self, preferred: Optional[str] = None) -> List[BaseProvider]:
        """获取回退链"""
        providers = self.get_available_providers()
        
        if preferred and preferred in self._providers:
            preferred_provider = self._providers[preferred]
            providers = [p for p in providers if p.name != preferred]
            providers.insert(0, preferred_provider)
        
        return providers
    
    def generate_with_fallback(
        self,
        messages: List[BaseMessage],
        preferred_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[ChatResult, str]:
        """
        带故障回退的同步生成
        
        按优先级顺序尝试Provider，直到成功或全部失败
        """
        fallback_chain = self._get_fallback_chain(preferred_provider)
        failed_providers: Dict[str, str] = {}
        
        for provider in fallback_chain:
            try:
                logger.debug(
                    "Attempting generation with provider",
                    provider=provider.name,
                )
                result = provider.generate(messages, **kwargs)
                logger.info(
                    "Generation succeeded",
                    provider=provider.name,
                )
                return result, provider.name
            except Exception as e:
                failed_providers[provider.name] = str(e)
                logger.warning(
                    "Provider failed, trying next",
                    provider=provider.name,
                    error=str(e),
                )
                continue
        
        raise ProviderFallbackError(
            "All providers failed to generate response",
            failed_providers=failed_providers,
        )
    
    async def agenerate_with_fallback(
        self,
        messages: List[BaseMessage],
        preferred_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Tuple[ChatResult, str]:
        """
        带故障回退的异步生成
        
        按优先级顺序尝试Provider，直到成功或全部失败
        """
        fallback_chain = self._get_fallback_chain(preferred_provider)
        failed_providers: Dict[str, str] = {}
        
        for provider in fallback_chain:
            try:
                logger.debug(
                    "Attempting async generation with provider",
                    provider=provider.name,
                )
                result = await provider.agenerate(messages, **kwargs)
                logger.info(
                    "Async generation succeeded",
                    provider=provider.name,
                )
                return result, provider.name
            except Exception as e:
                failed_providers[provider.name] = str(e)
                logger.warning(
                    "Provider failed, trying next",
                    provider=provider.name,
                    error=str(e),
                )
                continue
        
        raise ProviderFallbackError(
            "All providers failed to generate response",
            failed_providers=failed_providers,
        )
    
    def stream_with_fallback(
        self,
        messages: List[BaseMessage],
        preferred_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> Iterator[Tuple[ChatGeneration, str]]:
        """
        带故障回退的流式生成
        
        按优先级顺序尝试Provider，直到成功或全部失败
        """
        fallback_chain = self._get_fallback_chain(preferred_provider)
        failed_providers: Dict[str, str] = {}
        
        for provider in fallback_chain:
            try:
                logger.debug(
                    "Attempting streaming with provider",
                    provider=provider.name,
                )
                for chunk in provider.stream(messages, **kwargs):
                    yield chunk, provider.name
                logger.info(
                    "Streaming succeeded",
                    provider=provider.name,
                )
                return
            except Exception as e:
                failed_providers[provider.name] = str(e)
                logger.warning(
                    "Provider streaming failed, trying next",
                    provider=provider.name,
                    error=str(e),
                )
                continue
        
        raise ProviderFallbackError(
            "All providers failed to stream response",
            failed_providers=failed_providers,
        )
    
    async def astream_with_fallback(
        self,
        messages: List[BaseMessage],
        preferred_provider: Optional[str] = None,
        **kwargs: Any,
    ) -> AsyncIterator[Tuple[ChatGeneration, str]]:
        """
        带故障回退的异步流式生成
        
        按优先级顺序尝试Provider，直到成功或全部失败
        """
        fallback_chain = self._get_fallback_chain(preferred_provider)
        failed_providers: Dict[str, str] = {}
        
        for provider in fallback_chain:
            try:
                logger.debug(
                    "Attempting async streaming with provider",
                    provider=provider.name,
                )
                async for chunk in provider.astream(messages, **kwargs):
                    yield chunk, provider.name
                logger.info(
                    "Async streaming succeeded",
                    provider=provider.name,
                )
                return
            except Exception as e:
                failed_providers[provider.name] = str(e)
                logger.warning(
                    "Provider async streaming failed, trying next",
                    provider=provider.name,
                    error=str(e),
                )
                continue
        
        raise ProviderFallbackError(
            "All providers failed to stream response",
            failed_providers=failed_providers,
        )
    
    def get_provider_info(self, name: Optional[str] = None) -> Dict[str, Any]:
        """获取Provider信息"""
        if name:
            provider = self.get_provider(name)
            return provider.to_dict()
        
        return {
            "default_provider": self._default_provider,
            "registered_types": self.get_registered_types(),
            "providers": [p.to_dict() for p in self.get_available_providers()],
        }
    
    def switch_provider(self, name: str) -> Dict[str, Any]:
        """
        切换默认Provider
        
        便捷方法，切换并返回新的默认Provider信息。
        
        Args:
            name: Provider名称
        
        Returns:
            切换后的Provider信息
        """
        self.set_default_provider(name)
        return self.get_provider_info(name)
    
    def rotate_provider(self) -> Dict[str, Any]:
        """
        轮换到下一个Provider
        
        当当前Provider失败时，可以调用此方法切换到下一个优先级的Provider。
        
        Returns:
            切换后的Provider信息
        """
        available = self.get_available_providers()
        
        if not available:
            raise ProviderFallbackError("No available providers")
        
        if len(available) == 1:
            return self.get_provider_info(available[0].name)
        
        current_index = 0
        if self._default_provider:
            for i, p in enumerate(available):
                if p.name == self._default_provider:
                    current_index = i
                    break
        
        next_index = (current_index + 1) % len(available)
        next_provider = available[next_index]
        
        return self.switch_provider(next_provider.name)
