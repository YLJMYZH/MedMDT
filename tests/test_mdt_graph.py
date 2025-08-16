# tests/test_mdt_graph.py
from unittest.mock import MagicMock, patch
import json
import pytest
from medmdt.mdt.graph import build_mdt_graph, run_consultation
from medmdt.mdt.state import MDTState, ExpertOpinion, DiscussionRound
from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.moderator import Moderator


@pytest.fixture
def mock_expert():
    def make(eid, name, diagnosis, confidence=0.9):
        expert = MagicMock(spec=BaseExpert)
        expert.expert_id = eid
        expert.name = name
        expert.knowledge_domains = ["test"]
        expert.analyze.return_value = ExpertOpinion(
            expert_id=eid, expert_name=name, analysis="分析",
            diagnosis=diagnosis, recommendation="建议",
            confidence=confidence, reasoning="推理",
        )
        return expert
    return make


@pytest.fixture
def mock_moderator():
    mod = MagicMock(spec=Moderator)
    mod.summarize_round.return_value = DiscussionRound(
        round_num=1, opinions=[], summary="意见一致", divergences=[],
    )
    mod.check_consensus.return_value = True
    mod.generate_report.return_value = "# MDT会诊报告\n\n共识诊断：高血压"
    return mod


@pytest.fixture
def mock_retriever():
    retriever = MagicMock()
    retriever.retrieve.return_value = [
        MagicMock(text="高血压诊断标准", score=0.9, source="vector", metadata={}),
    ]
    return retriever


@pytest.fixture
def mock_selector_llm():
    llm = MagicMock()
    response = MagicMock()
    response.content = '{"selected_experts": ["internist", "cardiologist"], "reasoning": "心血管"}'
    llm.invoke.return_value = response
    return llm


def test_build_mdt_graph(mock_expert, mock_moderator, mock_retriever, mock_selector_llm):
    experts = {
        "internist": mock_expert("internist", "内科", "高血压"),
        "cardiologist": mock_expert("cardiologist", "心内科", "高血压"),
    }
    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    assert graph is not None


def test_run_consultation_reaches_consensus(
    mock_expert, mock_moderator, mock_retriever, mock_selector_llm
):
    experts = {
        "internist": mock_expert("internist", "内科", "高血压"),
        "cardiologist": mock_expert("cardiologist", "心内科", "高血压"),
    }
    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    result = run_consultation(
        graph,
        patient_info={"chief_complaint": "头痛头晕"},
        medical_records=[{"type": "vital", "content": "BP 160/100"}],
    )
    assert result["final_report"] is not None
    assert "会诊报告" in result["final_report"]
    assert result["current_round"] >= 1


def test_run_consultation_max_rounds(
    mock_expert, mock_moderator, mock_retriever, mock_selector_llm
):
    experts = {
        "internist": mock_expert("internist", "内科", "良性", 0.6),
        "surgeon": mock_expert("surgeon", "外科", "恶性", 0.7),
    }
    mock_moderator.check_consensus.return_value = False
    mock_moderator.summarize_round.return_value = DiscussionRound(
        round_num=1, opinions=[], summary="分歧", divergences=["诊断分歧"],
    )

    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    result = run_consultation(
        graph,
        patient_info={"chief_complaint": "腹部肿块"},
        medical_records=[],
        max_rounds=2,
    )
    assert result["final_report"] is not None
    assert result["current_round"] <= 2


def test_run_consultation_stores_discussion_rounds(
    mock_expert, mock_moderator, mock_retriever, mock_selector_llm
):
    experts = {
        "internist": mock_expert("internist", "内科", "高血压"),
    }
    graph = build_mdt_graph(experts, mock_moderator, mock_retriever, mock_selector_llm)
    result = run_consultation(
        graph,
        patient_info={"chief_complaint": "头痛"},
        medical_records=[],
    )
    assert len(result["discussion_rounds"]) >= 1
