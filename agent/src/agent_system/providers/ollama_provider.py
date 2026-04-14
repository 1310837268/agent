"""
Ollama Provider

支持本地运行的Ollama模型：
- Llama 3, Llama 2
- Mistral, Mixtral
- Qwen, DeepSeek
- 等等...
"""

from typing import Any, AsyncIterator, Dict, Iterator, List, Optional

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_system.core.config import ProviderConfig
from agent_system.core.exceptions import (
    ProviderConnectionError,
    ProviderTimeoutError,
)
from agent_system.core.logging import get_logger
from agent_system.providers.base import BaseProvider

logger = get_logger(__name__)


try:
    from langchain_ollama import ChatOllama

    OLLAMA_AVAILABLE = True
except ImportError:
    OLLAMA_AVAILABLE = False


class OllamaProvider(BaseProvider):
    """
    Ollama Provider
    
    连接到本地运行的Ollama服务，支持各种开源模型。
    
    使用前需要：
    1. 安装Ollama: https://ollama.ai
    2. 拉取模型: ollama pull llama3
    3. 启动服务: ollama serve (默认端口11434)
    
    支持的模型示例：
    - llama3: Meta的Llama 3模型
    - llama2: Meta的Llama 2模型
    - mistral: Mistral AI的Mistral模型
    - mixtral: Mistral AI的Mixtral 8x7B模型
    - qwen: 阿里云的Qwen模型
    - deepseek-coder: DeepSeek的代码模型
    - codellama: Meta的代码Llama模型
    - phi: Microsoft的Phi模型
    - gemma: Google的Gemma模型
    """
    
    def __init__(self, config: ProviderConfig) -> None:
        if not OLLAMA_AVAILABLE:
            raise ImportError(
                "Ollama integration is not installed. "
                "Install with: pip install langchain-ollama"
            )
        
        super().__init__(config)
        self._base_url = config.base_url or "http://localhost:11434"
        self._model = config.model
        
        logger.info(
            "OllamaProvider initialized",
            name=self.name,
            model=self._model,
            base_url=self._base_url,
        )
    
    def _init_model(self) -> Any:
        """初始化同步模型"""
        return ChatOllama(
            model=self._model,
            base_url=self._base_url,
            temperature=self.config.temperature,
            num_predict=self.config.max_tokens,
            timeout=self.config.timeout,
            **self.config.extra_params,
        )
    
    def _init_async_model(self) -> Any:
        """初始化异步模型"""
        return self._init_model()
    
    def _wrap_exception(self, exc: Exception) -> Exception:
        """包装异常"""
        error_msg = str(exc).lower()
        
        if "connection" in error_msg or "refused" in error_msg:
            return ProviderConnectionError(
                self.name,
                f"Could not connect to Ollama at {self._base_url}. "
                f"Make sure Ollama is running: 'ollama serve'",
                {"original_error": str(exc)},
            )
        elif "timeout" in error_msg:
            return ProviderTimeoutError(
                self.name,
                self.config.timeout,
                {"original_error": str(exc)},
            )
        elif "model" in error_msg and "not found" in error_msg:
            return ProviderConnectionError(
                self.name,
                f"Model '{self._model}' not found. "
                f"Pull it first: 'ollama pull {self._model}'",
                {"original_error": str(exc)},
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
    
    @staticmethod
    def list_local_models(base_url: str = "http://localhost:11434") -> List[Dict[str, Any]]:
        """
        列出本地Ollama可用的模型
        
        Args:
            base_url: Ollama服务地址
        
        Returns:
            模型列表
        """
        import requests
        
        try:
            response = requests.get(f"{base_url}/api/tags", timeout=10)
            response.raise_for_status()
            data = response.json()
            return data.get("models", [])
        except Exception as e:
            logger.error("Failed to list Ollama models", error=str(e))
            return []
    
    @staticmethod
    def pull_model(model_name: str, base_url: str = "http://localhost:11434") -> bool:
        """
        拉取模型
        
        Args:
            model_name: 模型名称
            base_url: Ollama服务地址
        
        Returns:
            是否成功
        """
        import requests
        
        try:
            response = requests.post(
                f"{base_url}/api/pull",
                json={"name": model_name},
                timeout=300,
            )
            return response.status_code == 200
        except Exception as e:
            logger.error(
                "Failed to pull Ollama model",
                model=model_name,
                error=str(e),
            )
            return False
