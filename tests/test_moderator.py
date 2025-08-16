# tests/test_moderator.py
from unittest.mock import MagicMock
import json
import pytest
from medmdt.mdt.moderator import Moderator
from medmdt.mdt.state import ExpertOpinion, DiscussionRound


@pytest.fixture
def mock_llm():
    return MagicMock()


@pytest.fixture
def moderator(mock_llm):
    return Moderator(llm=mock_llm, consensus_threshold=0.8)


@pytest.fixture
def agreeing_opinions():
    return [
        ExpertOpinion(
            expert_id="internist", expert_name="内科", analysis="高血压",
            diagnosis="原发性高血压", recommendation="ACEI降压",
            confidence=0.9, reasoning="BP>140/90",
        ),
        ExpertOpinion(
            expert_id="cardiologist", expert_name="心内科", analysis="高血压",
            diagnosis="原发性高血压", recommendation="ARB降压",
            confidence=0.85, reasoning="反复血压升高",
        ),
    ]


@pytest.fixture
def diverging_opinions():
    return [
        ExpertOpinion(
            expert_id="internist", expert_name="内科", analysis="肿块",
            diagnosis="良性肿瘤", recommendation="观察随访",
            confidence=0.6, reasoning="影像学特征偏良性",
        ),
        ExpertOpinion(
            expert_id="surgeon", expert_name="外科", analysis="肿块",
            diagnosis="疑似恶性肿瘤", recommendation="手术切除",
            confidence=0.7, reasoning="不排除恶性可能",
        ),
    ]


def test_moderator_init(moderator):
    assert moderator._consensus_threshold == 0.8


def test_summarize_round(moderator, mock_llm, agreeing_opinions):
    response = MagicMock()
    response.content = json.dumps({
        "summary": "两位专家均诊断为原发性高血压",
        "divergences": ["降压药物选择存在分歧：ACEI vs ARB"],
    })
    mock_llm.invoke.return_value = response

    result = moderator.summarize_round(agreeing_opinions)
    assert isinstance(result, DiscussionRound)
    assert "高血压" in result.summary
    assert len(result.divergences) == 1


def test_check_consensus_true(moderator, agreeing_opinions):
    rd = DiscussionRound(round_num=1, opinions=agreeing_opinions, divergences=[])
    assert moderator.check_consensus(rd, agreeing_opinions) is True


def test_check_consensus_false_low_confidence(moderator, diverging_opinions):
    rd = DiscussionRound(
        round_num=1, opinions=diverging_opinions,
        divergences=["诊断分歧"],
    )
    assert moderator.check_consensus(rd, diverging_opinions) is False


def test_check_consensus_false_with_divergences(moderator, agreeing_opinions):
    rd = DiscussionRound(
        round_num=1, opinions=agreeing_opinions,
        divergences=["存在分歧"],
    )
    assert moderator.check_consensus(rd, agreeing_opinions) is False


def test_generate_report(moderator, mock_llm, agreeing_opinions):
    response = MagicMock()
    response.content = "# MDT会诊报告\n\n## 诊断意见\n原发性高血压\n\n## 治疗方案\nACEI降压"
    mock_llm.invoke.return_value = response

    rd = DiscussionRound(round_num=1, opinions=agreeing_opinions, summary="共识")
    report = moderator.generate_report(
        patient_info={"chief_complaint": "头痛"},
        rounds=[rd],
        consensus={"diagnosis": "原发性高血压"},
        divergences=[],
    )
    assert isinstance(report, str)
    assert len(report) > 0
    mock_llm.invoke.assert_called_once()


def test_generate_report_with_divergences(moderator, mock_llm, diverging_opinions):
    response = MagicMock()
    response.content = "# 报告\n\n有分歧"
    mock_llm.invoke.return_value = response

    rd = DiscussionRound(round_num=1, opinions=diverging_opinions, summary="有分歧")
    report = moderator.generate_report(
        patient_info={},
        rounds=[rd],
        consensus=None,
        divergences=["诊断分歧：良性 vs 恶性"],
    )
    assert isinstance(report, str)
