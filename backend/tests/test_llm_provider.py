# tests/test_llm_provider.py
from unittest.mock import MagicMock, patch

import pytest

from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.provider import (
    LLMConfig,
    PROVIDER_REGISTRY,
    create_chat_model,
    create_vision_model,
    get_provider_base_url,
    list_provider_metadata,
)


def test_llm_config_defaults():
    cfg = LLMConfig(provider="openai", model="gpt-4o")
    assert cfg.temperature == 0.0
    assert cfg.max_tokens is None


def test_provider_registry_has_all_providers():
    expected = {"openai", "anthropic", "qwen", "zhipu", "moonshot", "deepseek", "custom"}
    assert expected == set(PROVIDER_REGISTRY.keys())


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_openai_model(mock_cls):
    mock_cls.return_value = MagicMock()
    model = create_chat_model("openai", "gpt-4o")
    mock_cls.assert_called_once_with(model="gpt-4o", temperature=0.0)
    assert model is mock_cls.return_value


@patch("medmdt.llm.provider.ChatAnthropic")
def test_create_anthropic_model(mock_cls):
    mock_cls.return_value = MagicMock()
    model = create_chat_model("anthropic", "claude-sonnet-4-6")
    mock_cls.assert_called_once_with(model="claude-sonnet-4-6", temperature=0.0)
    assert model is mock_cls.return_value


@patch("medmdt.llm.provider.ChatTongyi")
def test_create_qwen_chat_uses_native_tongyi(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("qwen", "qwen-plus", api_key="qwen-key")
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["dashscope_api_key"] == "qwen-key"


@patch("medmdt.llm.provider.ChatDeepSeek")
def test_create_deepseek_chat_uses_partner_integration(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("deepseek", "deepseek-chat", api_key="deepseek-key")
    mock_cls.assert_called_once()


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_custom_requires_base_url(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("custom", "my-model", base_url="https://my.api/v1", api_key="sk-test")
    call_kwargs = mock_cls.call_args[1]
    assert call_kwargs["base_url"] == "https://my.api/v1"
    assert call_kwargs["api_key"] == "sk-test"


def test_create_unknown_provider_raises():
    with pytest.raises(ValueError, match="Unknown provider"):
        create_chat_model("nonexistent", "model")


@pytest.mark.parametrize(
    ("provider", "expected_base_url"),
    [
        ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("zhipu", "https://open.bigmodel.cn/api/paas/v4"),
        ("moonshot", "https://api.moonshot.cn/v1"),
    ],
)
@patch("medmdt.llm.provider.ChatOpenAI")
def test_compatible_vision_routes_use_chat_completions(
    mock_cls, provider, expected_base_url
):
    mock_cls.return_value = MagicMock()
    create_vision_model(provider, "vision-model", api_key="vision-key")
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["base_url"] == expected_base_url
    assert kwargs["use_responses_api"] is False


@patch("medmdt.llm.provider.ChatOpenAI")
def test_compatible_vision_cannot_override_chat_completions_mode(mock_cls):
    mock_cls.return_value = MagicMock()
    create_vision_model(
        "qwen",
        "vision-model",
        api_key="vision-key",
        use_responses_api=True,
    )
    assert mock_cls.call_args.kwargs["use_responses_api"] is False


@patch("medmdt.llm.provider.ChatAnthropic")
def test_anthropic_vision_uses_native_integration(mock_cls):
    mock_cls.return_value = MagicMock()
    create_vision_model("anthropic", "claude-vision", api_key="anthropic-key")
    mock_cls.assert_called_once()


def test_deepseek_vision_is_registered_but_unavailable():
    assert PROVIDER_REGISTRY["deepseek"].vision_status == "unavailable"
    with pytest.raises(VisionProviderNotSupported, match="DeepSeek"):
        create_vision_model("deepseek", "deepseek-chat", api_key="key")


def test_custom_vision_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        create_vision_model("custom", "local-vlm", api_key="key")


def test_provider_metadata_and_base_urls_come_from_registry():
    metadata = {item["key"]: item for item in list_provider_metadata()}
    assert metadata["deepseek"]["vision_status"] == "unavailable"
    assert metadata["custom"]["vision_status"] == "unknown"
    assert get_provider_base_url("qwen") == "https://dashscope.aliyuncs.com/compatible-mode/v1"
