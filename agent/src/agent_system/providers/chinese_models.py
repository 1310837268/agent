"""
国产模型Provider工厂

提供便捷的方法创建国产模型的Provider配置：
- DeepSeek (深度求索)
- Qwen (通义千问, 阿里云)
- 文心一言 (百度)
- 星火 (科大讯飞)
- 智谱清言 (智谱AI)
- 等等...
"""

from typing import Any, Dict, Optional

from agent_system.core.config import ProviderConfig


class ChineseModelProvider:
    """
    国产模型Provider工厂
    
    提供便捷的静态方法创建各种国产模型的Provider配置。
    所有国产模型都使用OpenAI兼容接口，通过OpenAICompatibleProvider连接。
    """
    
    @staticmethod
    def deepseek(
        name: str = "deepseek",
        api_key: Optional[str] = None,
        model: str = "deepseek-chat",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建DeepSeek模型配置
        
        DeepSeek API文档: https://platform.deepseek.com/
        
        支持的模型:
        - deepseek-chat: DeepSeek Chat模型
        - deepseek-coder: DeepSeek代码模型
        
        Args:
            name: Provider名称
            api_key: DeepSeek API Key (从 https://platform.deepseek.com/ 获取)
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://api.deepseek.com/v1",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def qwen(
        name: str = "qwen",
        api_key: Optional[str] = None,
        model: str = "qwen-plus",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建阿里云Qwen（通义千问）模型配置
        
        阿里云DashScope文档: https://help.aliyun.com/zh/dashscope/
        
        支持的模型:
        - qwen-turbo: 通义千问超快速版
        - qwen-plus: 通义千问升级版
        - qwen-max: 通义千问最大版
        - qwen-long: 通义千问长上下文版
        
        Args:
            name: Provider名称
            api_key: 阿里云DashScope API Key
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://dashscope.aliyuncs.com/compatible-mode/v1",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def zhipu(
        name: str = "zhipu",
        api_key: Optional[str] = None,
        model: str = "glm-4",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建智谱AI（智谱清言）模型配置
        
        智谱AI文档: https://open.bigmodel.cn/dev/api
        
        支持的模型:
        - glm-4: GLM-4模型
        - glm-4v: GLM-4视觉模型
        - glm-3-turbo: GLM-3 Turbo模型
        
        Args:
            name: Provider名称
            api_key: 智谱AI API Key
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://open.bigmodel.cn/api/paas/v4",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def moonshot(
        name: str = "moonshot",
        api_key: Optional[str] = None,
        model: str = "moonshot-v1-8k",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建月之暗面（Moonshot）模型配置
        
        Moonshot文档: https://platform.moonshot.cn/docs
        
        支持的模型:
        - moonshot-v1-8k: 8K上下文
        - moonshot-v1-32k: 32K上下文
        - moonshot-v1-128k: 128K上下文
        
        Args:
            name: Provider名称
            api_key: Moonshot API Key
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://api.moonshot.cn/v1",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def stepfun(
        name: str = "stepfun",
        api_key: Optional[str] = None,
        model: str = "step-1-8k",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建阶跃星辰（StepFun）模型配置
        
        StepFun文档: https://platform.stepfun.com/docs
        
        支持的模型:
        - step-1-8k: Step 1 8K
        - step-1-32k: Step 1 32K
        - step-1-128k: Step 1 128K
        - step-2-16k: Step 2 16K
        
        Args:
            name: Provider名称
            api_key: StepFun API Key
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://api.stepfun.com/v1",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def minimax(
        name: str = "minimax",
        api_key: Optional[str] = None,
        model: str = "abab6.5s-chat",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建MiniMax模型配置
        
        MiniMax文档: https://www.minimax.cn/documentation
        
        支持的模型:
        - abab6.5s-chat: 最新模型
        - abab6.5-chat: 通用模型
        - abab5.5-chat: 轻量模型
        
        Args:
            name: Provider名称
            api_key: MiniMax API Key (需要同时配置group_id)
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://api.minimax.chat/v1",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def baichuan(
        name: str = "baichuan",
        api_key: Optional[str] = None,
        model: str = "Baichuan3-Turbo",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建百川智能模型配置
        
        百川文档: https://platform.baichuan-ai.com/docs
        
        支持的模型:
        - Baichuan3-Turbo: 百川3 Turbo
        - Baichuan3: 百川3
        - Baichuan2-Turbo: 百川2 Turbo
        
        Args:
            name: Provider名称
            api_key: 百川API Key
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key,
            base_url="https://api.baichuan-ai.com/v1",
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )
    
    @staticmethod
    def custom_openai_compatible(
        name: str,
        base_url: str,
        api_key: Optional[str] = None,
        model: str = "default",
        max_tokens: int = 4096,
        temperature: float = 0.7,
        priority: int = 80,
        **kwargs: Any,
    ) -> ProviderConfig:
        """
        创建自定义OpenAI兼容模型配置
        
        用于连接任何兼容OpenAI API的服务，如：
        - 本地部署的vLLM
        - LM Studio
        - Ollama OpenAI兼容接口
        - 其他云服务
        
        Args:
            name: Provider名称
            base_url: API基础URL
            api_key: API Key (可选，某些本地服务不需要)
            model: 模型名称
            max_tokens: 最大token数
            temperature: 温度参数
            priority: 优先级
            **kwargs: 其他参数
        
        Returns:
            ProviderConfig: Provider配置
        """
        return ProviderConfig(
            name=name,
            provider_type="openai_compatible",
            api_key=api_key or "dummy",
            base_url=base_url,
            model=model,
            max_tokens=max_tokens,
            temperature=temperature,
            priority=priority,
            enabled=True,
            extra_params=kwargs,
        )


# 常用模型的快捷配置
CHINESE_MODELS: Dict[str, Dict[str, Any]] = {
    "deepseek": {
        "base_url": "https://api.deepseek.com/v1",
        "models": ["deepseek-chat", "deepseek-coder"],
        "factory": ChineseModelProvider.deepseek,
    },
    "qwen": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "models": ["qwen-turbo", "qwen-plus", "qwen-max", "qwen-long"],
        "factory": ChineseModelProvider.qwen,
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "models": ["glm-4", "glm-4v", "glm-3-turbo"],
        "factory": ChineseModelProvider.zhipu,
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "factory": ChineseModelProvider.moonshot,
    },
    "stepfun": {
        "base_url": "https://api.stepfun.com/v1",
        "models": ["step-1-8k", "step-1-32k", "step-1-128k", "step-2-16k"],
        "factory": ChineseModelProvider.stepfun,
    },
    "minimax": {
        "base_url": "https://api.minimax.chat/v1",
        "models": ["abab6.5s-chat", "abab6.5-chat", "abab5.5-chat"],
        "factory": ChineseModelProvider.minimax,
    },
    "baichuan": {
        "base_url": "https://api.baichuan-ai.com/v1",
        "models": ["Baichuan3-Turbo", "Baichuan3", "Baichuan2-Turbo"],
        "factory": ChineseModelProvider.baichuan,
    },
}
