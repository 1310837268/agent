"""
Provider动态切换完整示例

展示如何使用各种Provider：
1. 本地模型（Ollama）
2. 国产模型（DeepSeek、Qwen、智谱等）
3. 自定义OpenAI兼容接口
4. 动态切换和故障回退
"""

import asyncio
import os
from typing import List, Optional

from agent_system.core.config import ProviderConfig
from agent_system.providers.base import BaseProvider
from agent_system.providers.chinese_models import ChineseModelProvider
from agent_system.providers.manager import (
    ProviderManager,
    register_provider_type,
)


def create_ollama_config(
    model: str = "llama3",
    name: str = "ollama-local",
    base_url: str = "http://localhost:11434",
    priority: int = 70,
) -> ProviderConfig:
    """
    创建Ollama本地模型配置
    
    使用前需要：
    1. 安装Ollama: https://ollama.ai
    2. 拉取模型: ollama pull llama3
    3. 启动服务: ollama serve (默认端口11434)
    
    支持的模型：
    - llama3: Meta的Llama 3
    - llama2: Meta的Llama 2
    - mistral: Mistral AI的Mistral
    - mixtral: Mistral AI的Mixtral 8x7B
    - qwen: 阿里云的Qwen
    - deepseek-coder: DeepSeek的代码模型
    - codellama: Meta的代码Llama
    - phi: Microsoft的Phi
    - gemma: Google的Gemma
    """
    return ProviderConfig(
        name=name,
        provider_type="ollama",
        base_url=base_url,
        model=model,
        max_tokens=4096,
        temperature=0.7,
        priority=priority,
        enabled=True,
    )


def create_deepseek_config(
    api_key: Optional[str] = None,
    model: str = "deepseek-chat",
    name: str = "deepseek",
    priority: int = 85,
) -> ProviderConfig:
    """
    创建DeepSeek配置
    
    API Key获取: https://platform.deepseek.com/
    
    支持的模型：
    - deepseek-chat: DeepSeek Chat模型
    - deepseek-coder: DeepSeek代码模型
    """
    return ChineseModelProvider.deepseek(
        name=name,
        api_key=api_key or os.getenv("DEEPSEEK_API_KEY"),
        model=model,
        priority=priority,
    )


def create_qwen_config(
    api_key: Optional[str] = None,
    model: str = "qwen-plus",
    name: str = "qwen",
    priority: int = 85,
) -> ProviderConfig:
    """
    创建阿里云Qwen（通义千问）配置
    
    API Key获取: https://dashscope.console.aliyun.com/
    
    支持的模型：
    - qwen-turbo: 通义千问超快速版
    - qwen-plus: 通义千问升级版
    - qwen-max: 通义千问最大版
    - qwen-long: 通义千问长上下文版
    """
    return ChineseModelProvider.qwen(
        name=name,
        api_key=api_key or os.getenv("DASHSCOPE_API_KEY"),
        model=model,
        priority=priority,
    )


def create_zhipu_config(
    api_key: Optional[str] = None,
    model: str = "glm-4",
    name: str = "zhipu",
    priority: int = 85,
) -> ProviderConfig:
    """
    创建智谱AI（智谱清言）配置
    
    API Key获取: https://open.bigmodel.cn/
    
    支持的模型：
    - glm-4: GLM-4模型
    - glm-4v: GLM-4视觉模型
    - glm-3-turbo: GLM-3 Turbo模型
    """
    return ChineseModelProvider.zhipu(
        name=name,
        api_key=api_key or os.getenv("ZHIPU_API_KEY"),
        model=model,
        priority=priority,
    )


def create_moonshot_config(
    api_key: Optional[str] = None,
    model: str = "moonshot-v1-8k",
    name: str = "moonshot",
    priority: int = 85,
) -> ProviderConfig:
    """
    创建月之暗面（Moonshot）配置
    
    API Key获取: https://platform.moonshot.cn/
    
    支持的模型：
    - moonshot-v1-8k: 8K上下文
    - moonshot-v1-32k: 32K上下文
    - moonshot-v1-128k: 128K上下文
    """
    return ChineseModelProvider.moonshot(
        name=name,
        api_key=api_key or os.getenv("MOONSHOT_API_KEY"),
        model=model,
        priority=priority,
    )


def create_custom_openai_compatible_config(
    name: str,
    base_url: str,
    model: str = "default",
    api_key: Optional[str] = None,
    priority: int = 80,
) -> ProviderConfig:
    """
    创建自定义OpenAI兼容接口配置
    
    适用于：
    - 本地部署的vLLM
    - LM Studio
    - Ollama OpenAI兼容接口 (http://localhost:11434/v1)
    - 其他云服务
    
    Args:
        name: Provider名称
        base_url: API基础URL
        model: 模型名称
        api_key: API Key (可选，某些本地服务不需要)
        priority: 优先级
    """
    return ChineseModelProvider.custom_openai_compatible(
        name=name,
        base_url=base_url,
        api_key=api_key,
        model=model,
        priority=priority,
    )


def create_multi_provider_manager() -> ProviderManager:
    """
    创建一个包含多种Provider的Manager示例
    
    展示如何配置多个Provider并设置优先级
    """
    configs: List[ProviderConfig] = []
    
    configs.append(ProviderConfig(
        name="openai-primary",
        provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        model="gpt-4",
        priority=100,
        enabled=True,
    ))
    
    configs.append(ProviderConfig(
        name="openai-secondary",
        provider_type="openai",
        api_key=os.getenv("OPENAI_API_KEY", "sk-xxx"),
        model="gpt-3.5-turbo",
        priority=90,
        enabled=True,
    ))
    
    if os.getenv("DEEPSEEK_API_KEY"):
        configs.append(create_deepseek_config())
    
    if os.getenv("DASHSCOPE_API_KEY"):
        configs.append(create_qwen_config())
    
    configs.append(create_ollama_config(
        model="llama3",
        name="ollama-llama3",
        priority=70,
    ))
    
    return ProviderManager(
        configs=configs,
        default_provider="openai-primary",
    )


async def demonstrate_dynamic_switching():
    """
    演示动态切换Provider
    """
    print("=" * 60)
    print("Provider动态切换演示")
    print("=" * 60)
    
    manager = create_multi_provider_manager()
    
    print("\n1. 已注册的Provider类型:")
    for provider_type in manager.get_registered_types():
        print(f"   - {provider_type}")
    
    print("\n2. 已配置的Provider实例:")
    info = manager.get_provider_info()
    print(f"   默认Provider: {info['default_provider']}")
    for provider in info['providers']:
        print(f"   - {provider['name']} (model: {provider['model']}, priority: {provider['priority']})")
    
    print("\n3. 动态切换Provider:")
    print(f"   当前默认: {manager.get_provider_info()['default_provider']}")
    
    new_info = manager.switch_provider("openai-secondary")
    print(f"   切换后: {new_info['name']}")
    
    print("\n4. 轮换Provider (自动切换到下一个):")
    rotated_info = manager.rotate_provider()
    print(f"   轮换后: {rotated_info['name']}")
    
    print("\n5. 运行时注册新Provider:")
    new_config = create_custom_openai_compatible_config(
        name="my-custom-api",
        base_url="http://localhost:8000/v1",
        model="my-custom-model",
        priority=85,
    )
    manager.register_provider(new_config)
    print(f"   已注册: my-custom-api")
    
    print("\n6. 禁用/启用Provider:")
    manager.disable_provider("ollama-llama3")
    print(f"   已禁用: ollama-llama3")
    
    manager.enable_provider("ollama-llama3")
    print(f"   已启用: ollama-llama3")
    
    print("\n" + "=" * 60)
    print("动态切换演示完成！")
    print("=" * 60)


async def demonstrate_fallback_mechanism():
    """
    演示故障回退机制
    """
    print("\n" + "=" * 60)
    print("故障回退机制演示")
    print("=" * 60)
    
    configs = [
        ProviderConfig(
            name="primary",
            provider_type="openai_compatible",
            base_url="http://localhost:9999/v1",
            model="test",
            priority=100,
            enabled=True,
        ),
        ProviderConfig(
            name="secondary",
            provider_type="openai_compatible",
            base_url="http://localhost:9998/v1",
            model="test",
            priority=90,
            enabled=True,
        ),
        ProviderConfig(
            name="fallback",
            provider_type="openai_compatible",
            base_url="http://localhost:9997/v1",
            model="test",
            priority=80,
            enabled=True,
        ),
    ]
    
    manager = ProviderManager(configs=configs, default_provider="primary")
    
    print("\n配置的Provider (按优先级):")
    for provider in manager.get_available_providers():
        print(f"   - {provider.name} (priority: {provider.get_priority()})")
    
    print("\n故障回退链:")
    print("   当 primary 失败 -> 尝试 secondary -> 尝试 fallback")
    print("\n注意: 这些Provider指向不存在的地址，用于演示回退机制")
    print("      在实际使用中，系统会自动尝试下一个Provider")
    
    print("\n" + "=" * 60)
    print("故障回退机制说明:")
    print("=" * 60)
    print("""
1. 同步调用: generate_with_fallback()
   - 按优先级顺序尝试每个Provider
   - 遇到可恢复错误时自动切换到下一个
   - 全部失败时抛出 ProviderFallbackError

2. 异步调用: agenerate_with_fallback()
   - 同上，但支持异步操作

3. 流式调用: stream_with_fallback() / astream_with_fallback()
   - 支持流式输出的故障回退

4. 触发回退的异常类型:
   - 连接错误 (ConnectionError)
   - 超时错误 (TimeoutError)
   - 限流错误 (RateLimitError)
   - 认证错误 (AuthenticationError)
   - 其他可恢复的API错误
    """)


def demonstrate_custom_provider_registration():
    """
    演示如何注册自定义Provider类型
    """
    print("\n" + "=" * 60)
    print("自定义Provider类型注册演示")
    print("=" * 60)
    
    print("\n1. 全局注册 (影响所有ProviderManager实例):")
    print("""
   from agent_system.providers.base import BaseProvider
   from agent_system.providers.manager import register_provider_type
   
   class MyCustomProvider(BaseProvider):
       def _init_model(self):
           # 初始化你的模型
           pass
       
       def generate(self, messages, **kwargs):
           # 实现生成逻辑
           pass
       
       # 实现其他抽象方法...
   
   # 全局注册
   register_provider_type("my_custom", MyCustomProvider)
   
   # 现在可以在配置中使用
   config = ProviderConfig(
       name="my-provider",
       provider_type="my_custom",  # 使用新注册的类型
       model="my-model",
       ...
   )
    """)
    
    print("\n2. 实例级注册 (只影响当前实例):")
    print("""
   manager = ProviderManager(configs=[])
   
   # 只在这个manager实例中注册
   manager.register_provider_type("my_custom", MyCustomProvider)
   
   # 其他manager实例不受影响
    """)
    
    print("\n3. 支持的Provider类型汇总:")
    print("""
   内置类型:
   - openai: OpenAI官方API
   - anthropic: Anthropic Claude API
   - openai_compatible: 任何兼容OpenAI API的服务
     - 国产模型: DeepSeek、Qwen、智谱、Moonshot等
     - 本地部署: vLLM、LM Studio、Ollama OpenAI兼容接口
   - ollama: 本地Ollama服务 (专用实现)
   
   可扩展:
   - 通过 register_provider_type() 注册自定义类型
    """)


async def main():
    """主函数"""
    print("\n" + "=" * 60)
    print("生产级Agent系统 - Provider动态切换完整示例")
    print("=" * 60)
    
    await demonstrate_dynamic_switching()
    await demonstrate_fallback_mechanism()
    demonstrate_custom_provider_registration()
    
    print("\n" + "=" * 60)
    print("示例运行完成！")
    print("=" * 60)
    
    print("""
快速开始指南:

1. 使用本地Ollama模型:
   ```python
   config = create_ollama_config(model="llama3")
   manager = ProviderManager(configs=[config])
   ```

2. 使用国产模型 (DeepSeek):
   ```python
   config = create_deepseek_config(api_key="your-key")
   manager = ProviderManager(configs=[config])
   ```

3. 使用多个Provider + 故障回退:
   ```python
   manager = create_multi_provider_manager()
   # 系统会自动按优先级尝试，失败时切换
   ```

4. 动态切换:
   ```python
   manager.switch_provider("ollama-llama3")  # 切换到指定Provider
   manager.rotate_provider()                  # 轮换到下一个
   ```
""")


if __name__ == "__main__":
    asyncio.run(main())
