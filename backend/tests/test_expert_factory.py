# tests/test_expert_factory.py
from unittest.mock import patch, MagicMock
import json
import pytest
import yaml
from medmdt.mdt.experts.factory import (
    ExpertConfig, load_expert_configs, create_expert, create_all_experts,
    select_experts,
)
from medmdt.mdt.experts.base import BaseExpert
from medmdt.config.runtime import LLMEndpoint, LLMSettings


@pytest.fixture
def experts_yaml(tmp_path):
    config = {
        "defaults": {"llm": {"provider": "openai", "model": "gpt-4o"}},
        "experts": {
            "internist": {
                "name": "内科专家",
                "knowledge_domains": ["internal_medicine"],
                "system_prompt": "你是一位内科专家。",
                "llm": {"provider": "openai", "model": "gpt-4o"},
            },
            "surgeon": {
                "name": "外科专家",
                "knowledge_domains": ["surgery"],
                "system_prompt": "你是一位外科专家。",
            },
        },
    }
    yaml_file = tmp_path / "experts.yaml"
    yaml_file.write_text(yaml.dump(config, allow_unicode=True))
    return str(yaml_file)


def test_load_expert_configs(experts_yaml):
    configs = load_expert_configs(experts_yaml)
    assert len(configs) == 2
    internist = next(c for c in configs if c.expert_id == "internist")
    assert internist.name == "内科专家"
    assert internist.llm_provider == "openai"


def test_load_expert_configs_inherits_defaults(experts_yaml):
    configs = load_expert_configs(experts_yaml)
    surgeon = next(c for c in configs if c.expert_id == "surgeon")
    assert surgeon.llm_provider == "openai"
    assert surgeon.llm_model == "gpt-4o"


@patch("medmdt.mdt.experts.factory.create_chat_model")
def test_create_expert(mock_create):
    mock_create.return_value = MagicMock()
    config = ExpertConfig(
        expert_id="internist", name="内科", system_prompt="test",
        knowledge_domains=["internal_medicine"],
        llm_provider="openai", llm_model="gpt-4o",
    )
    expert = create_expert(config)
    assert isinstance(expert, BaseExpert)
    assert expert.expert_id == "internist"
    mock_create.assert_called_once_with("openai", "gpt-4o")


@patch("medmdt.mdt.experts.factory.create_chat_model")
def test_create_all_experts(mock_create, experts_yaml):
    mock_create.return_value = MagicMock()
    experts = create_all_experts(experts_yaml)
    assert len(experts) == 2
    assert "internist" in experts
    assert "surgeon" in experts
    assert isinstance(experts["internist"], BaseExpert)


@patch("medmdt.mdt.experts.factory.create_chat_model")
def test_create_all_experts_revalidates_resolved_runtime_endpoint(
    mock_create, experts_yaml, monkeypatch
):
    monkeypatch.delenv("MEDMDT_ALLOWED_BASE_URL_HOSTS", raising=False)
    runtime = LLMSettings(
        consultation=LLMEndpoint(
            provider="custom",
            model="legacy-model",
            api_key="legacy-key",
            base_url="https://removed-expert.example/v1",
        )
    )

    with patch(
        "medmdt.config.runtime.load_llm_settings", return_value=runtime
    ):
        with pytest.raises(ValueError, match="not allowed"):
            create_all_experts(experts_yaml)

    mock_create.assert_not_called()


def test_select_experts():
    mock_llm = MagicMock()
    response = MagicMock()
    response.content = '{"selected_experts": ["internist", "cardiologist"], "reasoning": "心血管相关"}'
    mock_llm.invoke.return_value = response

    all_experts = {
        "internist": MagicMock(spec=BaseExpert, name="内科"),
        "cardiologist": MagicMock(spec=BaseExpert, name="心内科"),
        "surgeon": MagicMock(spec=BaseExpert, name="外科"),
    }
    selected = select_experts(
        all_experts,
        patient_info={"chief_complaint": "胸闷气短"},
        llm=mock_llm,
    )
    assert "internist" in selected
    assert "cardiologist" in selected


def test_select_experts_returns_all_on_parse_failure():
    mock_llm = MagicMock()
    response = MagicMock()
    response.content = "invalid json"
    mock_llm.invoke.return_value = response

    all_experts = {"a": MagicMock(), "b": MagicMock()}
    selected = select_experts(all_experts, patient_info={}, llm=mock_llm)
    assert set(selected) == {"a", "b"}
