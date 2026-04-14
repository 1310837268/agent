"""
Provider基础接口
"""

from abc import ABC, abstractmethod
from typing import Any, AsyncIterator, Dict, List, Optional, Union

from langchain_core.messages import BaseMessage
from langchain_core.outputs import ChatGeneration, ChatResult

from agent_system.core.config import ProviderConfig


class BaseProvider(ABC):
    """Provider基础抽象类"""
    
    def __init__(self, config: ProviderConfig) -> None:
        self.config = config
        self.name = config.name
        self.provider_type = config.provider_type
        self._model = None
        self._async_model = None
    
    @abstractmethod
    def _init_model(self) -> Any:
        """初始化同步模型"""
        pass
    
    @abstractmethod
    def _init_async_model(self) -> Any:
        """初始化异步模型"""
        pass
    
    @property
    def model(self) -> Any:
        """获取同步模型实例"""
        if self._model is None:
            self._model = self._init_model()
        return self._model
    
    @property
    def async_model(self) -> Any:
        """获取异步模型实例"""
        if self._async_model is None:
            self._async_model = self._init_async_model()
        return self._async_model
    
    @abstractmethod
    def generate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> ChatResult:
        """同步生成响应"""
        pass
    
    @abstractmethod
    async def agenerate(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> ChatResult:
        """异步生成响应"""
        pass
    
    @abstractmethod
    def stream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> Iterator[ChatGeneration]:
        """流式生成响应"""
        pass
    
    @abstractmethod
    async def astream(
        self,
        messages: List[BaseMessage],
        **kwargs: Any,
    ) -> AsyncIterator[ChatGeneration]:
        """异步流式生成响应"""
        pass
    
    def get_model_name(self) -> str:
        """获取模型名称"""
        return self.config.model
    
    def get_max_tokens(self) -> int:
        """获取最大token数"""
        return self.config.max_tokens
    
    def get_temperature(self) -> float:
        """获取温度参数"""
        return self.config.temperature
    
    def is_enabled(self) -> bool:
        """检查Provider是否启用"""
        return self.config.enabled
    
    def get_priority(self) -> int:
        """获取优先级"""
        return self.config.priority
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "name": self.name,
            "provider_type": self.provider_type,
            "model": self.config.model,
            "enabled": self.config.enabled,
            "priority": self.config.priority,
        }
