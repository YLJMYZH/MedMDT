# tests/test_expert_base.py
from unittest.mock import MagicMock
import json
import pytest
from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.state import ExpertOpinion


@pytest.fixture
def mock_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = json.dumps({
        "analysis": "患者血压持续偏高",
        "diagnosis": "原发性高血压",
        "recommendation": "建议ACEI类降压药",
        "confidence": 0.85,
        "reasoning": "血压反复>140/90, 排除继发因素",
        "references": ["高血压诊疗指南"],
    })
    llm.invoke.return_value = response
    return llm


@pytest.fixture
def expert(mock_llm):
    return BaseExpert(
        expert_id="cardiologist",
        name="心内科专家",
        system_prompt="你是一位资深心内科主任医师。",
        knowledge_domains=["cardiology"],
        llm=mock_llm,
    )


def test_expert_init(expert):
    assert expert.expert_id == "cardiologist"
    assert expert.name == "心内科专家"
    assert expert.knowledge_domains == ["cardiology"]


def test_expert_analyze_returns_opinion(expert, mock_llm):
    result = expert.analyze(
        patient_info={"chief_complaint": "头痛头晕1月"},
        medical_records=[{"type": "vital", "content": "BP 160/100"}],
        knowledge_context=[{"text": "高血压诊断标准..."}],
    )
    assert isinstance(result, ExpertOpinion)
    assert result.expert_id == "cardiologist"
    assert result.expert_name == "心内科专家"
    assert result.confidence == 0.85
    mock_llm.invoke.assert_called_once()


def test_expert_analyze_includes_system_prompt(expert, mock_llm):
    expert.analyze(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        knowledge_context=[],
    )
    call_args = mock_llm.invoke.call_args[0][0]
    system_msg = call_args[0]
    assert "心内科主任医师" in system_msg.content


def test_expert_analyze_with_previous_rounds(expert, mock_llm):
    previous = [{"round_num": 1, "opinions": [], "summary": "首轮意见", "divergences": ["诊断分歧"]}]
    expert.analyze(
        patient_info={"chief_complaint": "test"},
        medical_records=[],
        knowledge_context=[],
        previous_rounds=previous,
    )
    call_args = mock_llm.invoke.call_args[0][0]
    user_msg_content = call_args[1].content
    assert "首轮意见" in user_msg_content or "诊断分歧" in user_msg_content


def test_expert_analyze_handles_json_in_markdown_fence(mock_llm):
    response = MagicMock()
    response.content = '```json\n{"analysis":"a","diagnosis":"d","recommendation":"r","confidence":0.7,"reasoning":"r","references":[]}\n```'
    mock_llm.invoke.return_value = response
    expert = BaseExpert(
        expert_id="test", name="Test", system_prompt="test",
        knowledge_domains=[], llm=mock_llm,
    )
    result = expert.analyze(
        patient_info={}, medical_records=[], knowledge_context=[],
    )
    assert isinstance(result, ExpertOpinion)
    assert result.confidence == 0.7
