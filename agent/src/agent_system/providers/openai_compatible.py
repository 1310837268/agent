"""
通用OpenAI兼容Provider

支持所有兼容OpenAI API的服务：
- 国产模型：DeepSeek、Qwen、通义千问、文心一言、星火等
- 本地部署：vLLM、Ollama OpenAI兼容接口、LM Studio等
- 其他云服务：Together AI、Anyscale等
"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult
from langchain_openai import ChatOpenAI

from agent_system.core.config import ProviderConfig
from agent_system.core.exceptions import (
    ProviderAuthenticationError,
    ProviderConnectionError,
    ProviderRateLimitError,
    ProviderTimeoutError,
)
from agent_system.providers.base import BaseProvider


class OpenAICompatibleProvider(BaseProvider):
    """
    通用OpenAI兼容Provider
    
    这是一个通用的Provider，可以连接到任何兼容OpenAI API的服务。
    只需要配置base_url和api_key即可。
    
    支持的服务示例：
    - DeepSeek: base_url="https://api.deepseek.com/v1"
    - Qwen (阿里云): base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    - 通义千问: base_url="https://dashscope.aliyuncs.com/compatible-mode/v1"
    - Ollama (OpenAI兼容): base_url="http://localhost:11434/v1"
    - vLLM: base_url="http://localhost:8000/v1"
    - LM Studio: base_url="http://localhost:1234/v1"
    - Together AI: base_url="https://api.together.xyz/v1"
    """
    
    def __init__(self, config: ProviderConfig) -> None:
        super().__init__(config)
        self._api_key = config.api_key or "dummy"
        self._base_url = config.base_url
        
        if not self._base_url:
            raise ValueError(
                "OpenAICompatibleProvider requires base_url to be specified. "
                "Please set the base_url in ProviderConfig."
            )
    
    def _init_model(self) -> ChatOpenAI:
        """初始化同步模型"""
        return ChatOpenAI(
            model=self.config.model,
            api_key=self._api_key,
            base_url=self._base_url,
            max_tokens=self.config.max_tokens,
            temperature=self.config.temperature,
            timeout=self.config.timeout,
            max_retries=self.config.max_retries,
            **self.config.extra_params,
        )
    
    def _init_async_model(self) -> ChatOpenAI:
        """初始化异步模型"""
        return self._init_model()
    
    def _wrap_exception(self, exc: Exception) -> Exception:
        """包装异常"""
        import openai
        
        if isinstance(exc, openai.AuthenticationError):
            return ProviderAuthenticationError(
                self.name,
                str(exc),
                {"original_error": str(exc), "base_url": self._base_url},
            )
        elif isinstance(exc, openai.RateLimitError):
            retry_after = None
            if hasattr(exc, 'response') and exc.response:
                retry_after = exc.response.headers.get('retry-after')
            return ProviderRateLimitError(
                self.name,
                int(retry_after) if retry_after else None,
                {"original_error": str(exc), "base_url": self._base_url},
            )
        elif isinstance(exc, openai.APIConnectionError):
            return ProviderConnectionError(
                self.name,
                str(exc),
                {"original_error": str(exc), "base_url": self._base_url},
            )
        elif isinstance(exc, openai.APITimeoutError):
            return ProviderTimeoutError(
                self.name,
                self.config.timeout,
                {"original_error": str(exc), "base_url": self._base_url},
            )
        return exc
    
    def generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成响应"""
        try:
            return self.model.invoke(messages, **kwargs)
        except Exception as e:
            raise self._wrap_exception(e)
    
    async def agenerate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成响应"""
        try:
            return await self.async_model.ainvoke(messages, **kwargs)
        except Exception as e:
            raise self._wrap_exception(e)
    
    def stream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """流式生成响应"""
        try:
            for chunk in self.model.stream(messages, **kwargs):
                yield ChatGeneration(message=chunk)
        except Exception as e:
            raise self._wrap_exception(e)
    
    async def astream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        """异步流式生成响应"""
        try:
            async for chunk in self.async_model.astream(messages, **kwargs):
                yield ChatGeneration(message=chunk)
        except Exception as e:
            raise self._wrap_exception(e)
