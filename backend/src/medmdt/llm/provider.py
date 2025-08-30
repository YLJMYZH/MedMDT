# src/medmdt/llm/provider.py
"""LLM provider factory — unified interface for multiple providers.

Providers with a dedicated (vendor-native or LangChain partner) integration use
it directly; the rest fall back to the OpenAI-compatible ``ChatOpenAI``:

    openai      -> ChatOpenAI        (langchain-openai)
    anthropic   -> ChatAnthropic     (langchain-anthropic)
    deepseek    -> ChatDeepSeek      (langchain-deepseek, official partner)
    qwen        -> ChatTongyi        (langchain-community, dashscope SDK)
    zhipu       -> ChatZhipuAI       (langchain-community, zhipuai SDK)
    moonshot    -> ChatOpenAI + base_url   (no vendor SDK)
    custom      -> ChatOpenAI + base_url

All return a ``BaseChatModel`` so callers use the same ``.invoke()/.stream()``.
"""

from dataclasses import dataclass
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models.zhipuai import ChatZhipuAI


@dataclass
class LLMConfig:
    """Configuration for an LLM provider instance."""

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


# Only providers still served through the OpenAI-compatible ChatOpenAI path.
_OPENAI_COMPAT_URLS: dict[str, str] = {
    "moonshot": "https://api.moonshot.cn/v1",
}


def _clamp_open_unit(t: float) -> float:
    """Clamp temperature into the open interval (0, 1) required by GLM."""
    return min(max(t, 0.01), 0.99)


def _create_openai(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatOpenAI(model=model, temperature=temperature, **kwargs)


def _create_anthropic(model: str, temperature: float, **kwargs) -> BaseChatModel:
    return ChatAnthropic(model=model, temperature=temperature, **kwargs)


def _create_deepseek(model: str, temperature: float, **kwargs) -> BaseChatModel:
    api_key = kwargs.pop("api_key", None)
    base_url = kwargs.pop("base_url", None)
    params: dict = {"model": model, "temperature": temperature, **kwargs}
    if api_key:
        params["api_key"] = api_key
    if base_url:
        params["api_base"] = base_url
    return ChatDeepSeek(**params)


def _create_tongyi(model: str, temperature: float, **kwargs) -> BaseChatModel:
    api_key = kwargs.pop("api_key", None)
    kwargs.pop("base_url", None)  # ChatTongyi has no base_url; DashScope endpoint is fixed
    model_kwargs = kwargs.pop("model_kwargs", {})
    model_kwargs.setdefault("temperature", temperature)
    params: dict = {"model": model, "model_kwargs": model_kwargs, "streaming": True, **kwargs}
    if api_key:
        params["dashscope_api_key"] = api_key
    return ChatTongyi(**params)


def _create_zhipu(model: str, temperature: float, **kwargs) -> BaseChatModel:
    api_key = kwargs.pop("api_key", None)
    kwargs.pop("base_url", None)  # ChatZhipuAI uses the fixed BigModel endpoint
    params: dict = {"model": model, "temperature": _clamp_open_unit(temperature), **kwargs}
    if api_key:
        params["api_key"] = api_key
    return ChatZhipuAI(**params)


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
    "deepseek": lambda m, t, **kw: _create_deepseek(m, t, **kw),
    "qwen": lambda m, t, **kw: _create_tongyi(m, t, **kw),
    "zhipu": lambda m, t, **kw: _create_zhipu(m, t, **kw),
    "moonshot": lambda m, t, **kw: _create_openai_compat("moonshot", m, t, **kw),
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
        **kwargs: Additional provider-specific arguments (e.g. api_key, base_url).

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
