# tests/test_llm_provider.py
from unittest.mock import patch, MagicMock
import pytest
from medmdt.llm.provider import create_chat_model, LLMConfig, PROVIDER_REGISTRY


def test_llm_config_defaults():
    cfg = LLMConfig(provider="openai", model="gpt-4o")
    assert cfg.temperature == 0.0
    assert cfg.max_tokens is None


def test_provider_registry_has_all_providers():
    expected = {"openai", "anthropic", "qwen", "zhipu", "moonshot", "deepseek", "ollama", "custom"}
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


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_qwen_uses_dashscope_base_url(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("qwen", "qwen-plus")
    call_kwargs = mock_cls.call_args[1]
    assert "dashscope" in call_kwargs["base_url"]
    assert call_kwargs["model"] == "qwen-plus"


@patch("medmdt.llm.provider.ChatOpenAI")
def test_create_deepseek(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("deepseek", "deepseek-chat")
    call_kwargs = mock_cls.call_args[1]
    assert "deepseek" in call_kwargs["base_url"]


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
