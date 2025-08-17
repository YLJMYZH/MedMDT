# tests/test_mdt_state.py
import pytest
from medmdt.mdt.state import MDTState, ExpertOpinion, DiscussionRound


def test_expert_opinion():
    op = ExpertOpinion(
        expert_id="internist",
        expert_name="内科专家",
        analysis="患者症状符合2型糖尿病表现",
        diagnosis="2型糖尿病",
        recommendation="建议口服二甲双胍",
        confidence=0.9,
        reasoning="空腹血糖>7.0, HbA1c>6.5%",
        references=["2型糖尿病诊疗指南2024"],
    )
    assert op.confidence == 0.9
    assert op.expert_id == "internist"


def test_expert_opinion_confidence_range():
    with pytest.raises(Exception):
        ExpertOpinion(
            expert_id="x", expert_name="x", analysis="x", diagnosis="x",
            recommendation="x", confidence=1.5, reasoning="x", references=[],
        )


def test_expert_opinion_defaults():
    op = ExpertOpinion(
        expert_id="test", expert_name="Test", analysis="a",
        diagnosis="d", recommendation="r", confidence=0.8, reasoning="r",
    )
    assert op.references == []


def test_discussion_round():
    opinion = ExpertOpinion(
        expert_id="internist", expert_name="内科", analysis="a",
        diagnosis="d", recommendation="r", confidence=0.8, reasoning="r",
    )
    rd = DiscussionRound(
        round_num=1,
        opinions=[opinion],
        summary="各专家意见一致",
        divergences=[],
    )
    assert rd.round_num == 1
    assert len(rd.opinions) == 1


def test_discussion_round_serialization():
    opinion = ExpertOpinion(
        expert_id="internist", expert_name="内科", analysis="a",
        diagnosis="d", recommendation="r", confidence=0.8, reasoning="r",
    )
    rd = DiscussionRound(round_num=1, opinions=[opinion])
    d = rd.model_dump()
    restored = DiscussionRound.model_validate(d)
    assert restored.round_num == rd.round_num
    assert restored.opinions[0].expert_id == "internist"


def test_mdt_state_is_typeddict():
    state: MDTState = {
        "patient_info": {"name": "张三", "chief_complaint": "反复头痛1月"},
        "medical_records": [{"type": "lab", "content": "血压150/95mmHg"}],
        "knowledge_context": [],
        "experts": ["internist", "cardiologist"],
        "discussion_rounds": [],
        "consensus": None,
        "divergences": [],
        "final_report": None,
        "current_round": 0,
        "max_rounds": 3,
    }
    assert state["max_rounds"] == 3
    assert state["consensus"] is None
