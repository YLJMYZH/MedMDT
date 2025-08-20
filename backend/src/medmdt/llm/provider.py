# src/medmdt/llm/provider.py
"""LLM provider factory — unified interface for multiple providers."""

from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic


@dataclass
class LLMConfig:
    """Configuration for an LLM provider instance."""

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


_OPENAI_COMPAT_URLS: dict[str, str] = {
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
    "deepseek": "https://api.deepseek.com/v1",
}

def _create_openai(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def _create_anthropic(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatAnthropic(model=model, temperature=temperature, **kwargs)


def _create_openai_compat(
    provider: str, model: str, temperature: float, **kwargs
) -> BaseChatModel:
    base_url = kwargs.pop("base_url", _OPENAI_COMPAT_URLS[provider])
    api_key = kwargs.pop("api_key", None)
    params: dict = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        **kwargs,
    }
    if api_key:
        params["api_key"] = api_key
    return ChatOpenAI(**params)


def _create_custom(model: str, temperature: float, **kwargs) -> BaseChatModel:
    base_url = kwargs.pop("base_url", None)
    api_key = kwargs.pop("api_key", None)
    if not base_url:
        raise ValueError("Custom provider requires base_url")
    params: dict = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        **kwargs,
    }
    if api_key:
        params["api_key"] = api_key
    return ChatOpenAI(**params)


PROVIDER_REGISTRY: dict[str, Callable] = {
    "openai": lambda m, t, **kw: _create_openai(m, t, **kw),
    "anthropic": lambda m, t, **kw: _create_anthropic(m, t, **kw),
    "qwen": lambda m, t, **kw: _create_openai_compat("qwen", m, t, **kw),
    "zhipu": lambda m, t, **kw: _create_openai_compat("zhipu", m, t, **kw),
    "moonshot": lambda m, t, **kw: _create_openai_compat("moonshot", m, t, **kw),
    "deepseek": lambda m, t, **kw: _create_openai_compat("deepseek", m, t, **kw),
    "custom": lambda m, t, **kw: _create_custom(m, t, **kw),
}


def create_chat_model(
    provider: str, model: str, temperature: float = 0.0, **kwargs
) -> BaseChatModel:
    """Create a chat model instance for the given provider.

    Args:
        provider: Provider name (one of PROVIDER_REGISTRY keys).
        model: Model identifier (e.g. "gpt-4o", "qwen-plus").
        temperature: Sampling temperature, defaults to 0.0.
        **kwargs: Additional provider-specific arguments.

    Returns:
        A LangChain BaseChatModel instance.

    Raises:
        ValueError: If the provider is not in PROVIDER_REGISTRY.
    """
    factory = PROVIDER_REGISTRY.get(provider)
    if not factory:
        raise ValueError(
            f"Unknown provider: {provider}. "
            f"Available: {list(PROVIDER_REGISTRY.keys())}"
        )
    return factory(model, temperature, **kwargs)
