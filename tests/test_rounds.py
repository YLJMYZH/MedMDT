# tests/test_rounds.py
from unittest.mock import MagicMock
import pytest
from medmdt.mdt.rounds import run_expert_round, should_continue
from medmdt.mdt.state import ExpertOpinion
from medmdt.mdt.experts.base import BaseExpert


@pytest.fixture
def mock_experts():
    def make_expert(expert_id, name, diagnosis, confidence):
        expert = MagicMock(spec=BaseExpert)
        expert.expert_id = expert_id
        expert.name = name
        expert.analyze.return_value = ExpertOpinion(
            expert_id=expert_id, expert_name=name,
            analysis="analysis", diagnosis=diagnosis,
            recommendation="rec", confidence=confidence,
            reasoning="reason",
        )
        return expert

    return {
        "internist": make_expert("internist", "内科", "高血压", 0.9),
        "cardiologist": make_expert("cardiologist", "心内科", "高血压", 0.85),
        "surgeon": make_expert("surgeon", "外科", "观察", 0.7),
    }


def test_run_expert_round_all_experts(mock_experts):
    opinions = run_expert_round(
        experts=mock_experts,
        expert_ids=["internist", "cardiologist"],
        patient_info={"complaint": "头痛"},
        medical_records=[],
        knowledge_context=[],
    )
    assert len(opinions) == 2
    assert all(isinstance(op, ExpertOpinion) for op in opinions)
    mock_experts["internist"].analyze.assert_called_once()
    mock_experts["cardiologist"].analyze.assert_called_once()
    mock_experts["surgeon"].analyze.assert_not_called()


def test_run_expert_round_passes_previous_rounds(mock_experts):
    previous = [{"round_num": 1, "summary": "test"}]
    run_expert_round(
        experts=mock_experts,
        expert_ids=["internist"],
        patient_info={},
        medical_records=[],
        knowledge_context=[],
        previous_rounds=previous,
    )
    call_kwargs = mock_experts["internist"].analyze.call_args[1]
    assert call_kwargs["previous_rounds"] == previous


def test_run_expert_round_first_round_no_previous(mock_experts):
    run_expert_round(
        experts=mock_experts,
        expert_ids=["internist"],
        patient_info={},
        medical_records=[],
        knowledge_context=[],
    )
    call_kwargs = mock_experts["internist"].analyze.call_args[1]
    assert call_kwargs["previous_rounds"] is None


def test_should_continue_first_round():
    assert should_continue(current_round=0, max_rounds=3, has_consensus=False) is True


def test_should_continue_no_consensus():
    assert should_continue(current_round=2, max_rounds=3, has_consensus=False) is True


def test_should_continue_consensus_reached():
    assert should_continue(current_round=1, max_rounds=3, has_consensus=True) is False


def test_should_continue_max_rounds_reached():
    assert should_continue(current_round=3, max_rounds=3, has_consensus=False) is False


def test_should_continue_past_max_rounds():
    assert should_continue(current_round=5, max_rounds=3, has_consensus=False) is False
