# tests/test_expert_configs.py
import pytest
from pathlib import Path
from medmdt.mdt.experts.factory import load_expert_configs


EXPERTS_YAML = str(Path(__file__).parent.parent / "config" / "experts.yaml")


def test_load_all_experts():
    configs = load_expert_configs(EXPERTS_YAML)
    assert len(configs) == 7  # internist, radiologist, surgeon + 4 new


def test_expert_ids():
    configs = load_expert_configs(EXPERTS_YAML)
    ids = {c.expert_id for c in configs}
    expected = {"internist", "radiologist", "surgeon", "cardiologist", "neurologist", "oncologist", "pathologist"}
    assert ids == expected


def test_all_experts_have_system_prompt():
    configs = load_expert_configs(EXPERTS_YAML)
    for cfg in configs:
        assert len(cfg.system_prompt) > 20, f"{cfg.expert_id} has insufficient system_prompt"


def test_all_experts_have_knowledge_domains():
    configs = load_expert_configs(EXPERTS_YAML)
    for cfg in configs:
        assert len(cfg.knowledge_domains) >= 1, f"{cfg.expert_id} has no knowledge_domains"


def test_cardiologist_config():
    configs = load_expert_configs(EXPERTS_YAML)
    cardio = next(c for c in configs if c.expert_id == "cardiologist")
    assert "cardiology" in cardio.knowledge_domains
    assert "心" in cardio.name or "心" in cardio.system_prompt


def test_oncologist_config():
    configs = load_expert_configs(EXPERTS_YAML)
    onco = next(c for c in configs if c.expert_id == "oncologist")
    assert "oncology" in onco.knowledge_domains
