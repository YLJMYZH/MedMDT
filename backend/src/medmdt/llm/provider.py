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
from typing import Callable, Literal

from langchain_core.language_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_deepseek import ChatDeepSeek
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_community.chat_models.zhipuai import ChatZhipuAI

from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.http_clients import (
    get_shared_async_http_client,
    get_shared_http_client,
)

ModelFactory = Callable[..., BaseChatModel]
VisionStatus = Literal["supported", "unavailable", "unknown"]
EmbeddingStatus = Literal["supported", "unavailable"]


@dataclass
class LLMConfig:
    """Configuration for an LLM provider instance."""

    provider: str
    model: str
    base_url: str | None = None
    api_key_env: str | None = None
    temperature: float = 0.0
    max_tokens: int | None = None


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    chat_factory: ModelFactory
    vision_factory: ModelFactory | None
    vision_status: VisionStatus
    api_base_url: str | None = None
    needs_key: bool = True
    needs_base_url: bool = False
    embedding_status: EmbeddingStatus = "unavailable"
    embedding_base_url: str | None = None


_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
}


def _clamp_open_unit(t: float) -> float:
    """Clamp temperature into the open interval (0, 1) required by GLM."""
    return min(max(t, 0.01), 0.99)


def _with_redirect_safe_openai_clients(kwargs: dict) -> dict:
    params = dict(kwargs)
    client_getters = (
        ("http_client", get_shared_http_client),
        ("http_async_client", get_shared_async_http_client),
    )
    for key, _ in client_getters:
        client = params.get(key)
        if client is not None and getattr(client, "follow_redirects", None) is not False:
            raise ValueError(f"{key} must disable redirects")
    for key, get_client in client_getters:
        if params.get(key) is None:
            params[key] = get_client()
    return params


def _create_openai(model: str, temperature: float, **kwargs) -> BaseChatModel:
    params = _with_redirect_safe_openai_clients(kwargs)
    return ChatOpenAI(model=model, temperature=temperature, **params)


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
    base_url = kwargs.pop("base_url", _PROVIDER_BASE_URLS[provider])
    api_key = kwargs.pop("api_key", None)
    params: dict = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        **kwargs,
    }
    if api_key:
        params["api_key"] = api_key
    return ChatOpenAI(**_with_redirect_safe_openai_clients(params))


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
    return ChatOpenAI(**_with_redirect_safe_openai_clients(params))


def _create_compatible_vision(
    provider: str, model: str, temperature: float, **kwargs
) -> BaseChatModel:
    base_url = kwargs.pop("base_url", None) or _PROVIDER_BASE_URLS.get(provider)
    if not base_url:
        raise ValueError(f"{provider} vision provider requires base_url")
    api_key = kwargs.pop("api_key", None)
    params = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        **kwargs,
        "use_responses_api": False,
    }
    if api_key:
        params["api_key"] = api_key
    return ChatOpenAI(**_with_redirect_safe_openai_clients(params))


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai",
        "OpenAI",
        _create_openai,
        _create_openai,
        "supported",
        api_base_url=_PROVIDER_BASE_URLS["openai"],
        embedding_status="supported",
        embedding_base_url=_PROVIDER_BASE_URLS["openai"],
    ),
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic", _create_anthropic, _create_anthropic, "supported"
    ),
    "deepseek": ProviderSpec(
        "deepseek",
        "DeepSeek",
        _create_deepseek,
        None,
        "unavailable",
        api_base_url=_PROVIDER_BASE_URLS["deepseek"],
    ),
    "qwen": ProviderSpec(
        "qwen",
        "Qwen (通义千问)",
        _create_tongyi,
        lambda m, t, **kw: _create_compatible_vision("qwen", m, t, **kw),
        "supported",
        api_base_url=_PROVIDER_BASE_URLS["qwen"],
        embedding_status="supported",
        embedding_base_url=_PROVIDER_BASE_URLS["qwen"],
    ),
    "zhipu": ProviderSpec(
        "zhipu",
        "ZhiPu (智谱)",
        _create_zhipu,
        lambda m, t, **kw: _create_compatible_vision("zhipu", m, t, **kw),
        "supported",
        api_base_url=_PROVIDER_BASE_URLS["zhipu"],
        embedding_status="supported",
        embedding_base_url=_PROVIDER_BASE_URLS["zhipu"],
    ),
    "moonshot": ProviderSpec(
        "moonshot",
        "Moonshot (月之暗面)",
        lambda m, t, **kw: _create_openai_compat("moonshot", m, t, **kw),
        lambda m, t, **kw: _create_compatible_vision("moonshot", m, t, **kw),
        "supported",
        api_base_url=_PROVIDER_BASE_URLS["moonshot"],
    ),
    "custom": ProviderSpec(
        "custom",
        "自定义 (OpenAI 兼容)",
        _create_custom,
        lambda m, t, **kw: _create_compatible_vision("custom", m, t, **kw),
        "unknown",
        needs_base_url=True,
        embedding_status="supported",
    ),
}


def _get_provider(provider: str) -> ProviderSpec:
    spec = PROVIDER_REGISTRY.get(provider)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(PROVIDER_REGISTRY)}"
        )
    return spec


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
    return _get_provider(provider).chat_factory(model, temperature, **kwargs)


def create_vision_model(
    provider: str, model: str, temperature: float = 0.0, **kwargs
) -> BaseChatModel:
    spec = _get_provider(provider)
    if spec.vision_factory is None:
        raise VisionProviderNotSupported(
            f"{spec.label} 官方 API 暂不支持图像理解，请选择其他视觉模型"
        )
    return spec.vision_factory(model, temperature, **kwargs)


def get_provider_base_url(provider: str) -> str | None:
    return _get_provider(provider).api_base_url


def get_embedding_base_url(provider: str, base_url: str | None = None) -> str:
    """Return the explicit effective target for a supported embedding provider."""
    spec = _get_provider(provider)
    if spec.embedding_status != "supported":
        raise ValueError(f"embedding provider {provider} is unavailable")
    if provider == "custom":
        if not base_url:
            raise ValueError("Custom embedding provider requires base_url")
        return base_url
    if base_url:
        return base_url
    if spec.embedding_base_url is None:
        raise ValueError(f"embedding provider {provider} has no trusted Base URL")
    return spec.embedding_base_url


def list_provider_metadata() -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "needs_key": spec.needs_key,
            "needs_base_url": spec.needs_base_url,
            "vision_status": spec.vision_status,
            "embedding_status": spec.embedding_status,
        }
        for spec in PROVIDER_REGISTRY.values()
    ]
