"""
Provider管理器 - 支持动态切换与故障回退
"""

import asyncio
from typing import Any, AsyncIterator, Dict, Iterator, List, Optional, Tuple

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
from agent_system.providers.openai_provider import OpenAIProvider

logger = get_logger(__name__)


PROVIDER_REGISTRY: Dict[str, type] = {
    "openai": OpenAIProvider,
    "anthropic": AnthropicProvider,
}


class ProviderManager:
    """
    Provider管理器
    
    功能：
    1. 管理多个Provider实例
    2. 支持动态切换Provider
    3. 支持故障回退机制
    4. 支持优先级排序
    """
    
    def __init__(
        self,
        configs: List[ProviderConfig],
        default_provider: Optional[str] = None,
    ) -> None:
        self._configs: Dict[str, ProviderConfig] = {}
        self._providers: Dict[str, BaseProvider] = {}
        self._default_provider = default_provider
        
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
        )
    
    def _create_provider(self, config: ProviderConfig) -> BaseProvider:
        """创建Provider实例"""
        provider_class = PROVIDER_REGISTRY.get(config.provider_type)
        if not provider_class:
            raise ValueError(f"Unknown provider type: {config.provider_type}")
        return provider_class(config)
    
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
            "providers": [p.to_dict() for p in self.get_available_providers()],
        }
