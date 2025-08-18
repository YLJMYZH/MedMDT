# src/medmdt/mdt/moderator.py
import json
from collections.abc import Callable
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from medmdt.mdt.state import ExpertOpinion, DiscussionRound
from medmdt.mdt.utils import parse_llm_json
from medmdt.llm.prompts.mdt.moderator_summary import MODERATOR_SUMMARY_PROMPT
from medmdt.llm.prompts.mdt.report_generation import REPORT_GENERATION_PROMPT


class Moderator:
    def __init__(self, llm: BaseChatModel, consensus_threshold: float = 0.8):
        self._llm = llm
        self._consensus_threshold = consensus_threshold

    def summarize_round(self, opinions: list[ExpertOpinion]) -> DiscussionRound:
        opinions_text = "\n\n".join(
            f"### {op.expert_name} ({op.expert_id})\n"
            f"- 诊断：{op.diagnosis}\n"
            f"- 分析：{op.analysis}\n"
            f"- 建议：{op.recommendation}\n"
            f"- 置信度：{op.confidence}\n"
            f"- 推理：{op.reasoning}"
            for op in opinions
        )
        prompt = MODERATOR_SUMMARY_PROMPT.format(opinions_text=opinions_text)
        response = self._llm.invoke([
            SystemMessage(content="你是MDT会诊主持人，负责汇总和协调各专家意见。"),
            HumanMessage(content=prompt),
        ])
        data = parse_llm_json(response.content)
        return DiscussionRound(
            round_num=0,
            opinions=opinions,
            summary=data.get("summary", ""),
            divergences=data.get("divergences", []),
        )

    def check_consensus(self, round_result: DiscussionRound, all_opinions: list[ExpertOpinion]) -> bool:
        if round_result.divergences:
            return False
        return all(op.confidence >= self._consensus_threshold for op in all_opinions)

    def generate_report(
        self,
        patient_info: dict,
        rounds: list[DiscussionRound],
        consensus: dict | None,
        divergences: list[str],
    ) -> str:
        rounds_text = ""
        for rd in rounds:
            rounds_text += f"\n### 第{rd.round_num}轮\n"
            if rd.summary:
                rounds_text += f"汇总：{rd.summary}\n"
            for op in rd.opinions:
                rounds_text += f"- {op.expert_name}：{op.diagnosis}（置信度{op.confidence}）\n"
            if rd.divergences:
                rounds_text += f"分歧：{', '.join(rd.divergences)}\n"

        if consensus:
            conclusion = f"达成共识：{json.dumps(consensus, ensure_ascii=False)}"
        else:
            conclusion = f"未达成完全共识。分歧点：\n" + "\n".join(f"- {d}" for d in divergences)

        prompt = REPORT_GENERATION_PROMPT.format(
            patient_info=json.dumps(patient_info, ensure_ascii=False, indent=2),
            rounds_text=rounds_text,
            conclusion=conclusion,
        )
        response = self._llm.invoke([
            SystemMessage(content="你是一位专业的医学报告撰写者。"),
            HumanMessage(content=prompt),
        ])
        return response.content

    def stream_generate_report(
        self,
        patient_info: dict,
        rounds: list[DiscussionRound],
        consensus: dict | None,
        divergences: list[str],
        on_token: Callable[[str], None] | None = None,
    ) -> str:
        rounds_text = ""
        for rd in rounds:
            rounds_text += f"\n### 第{rd.round_num}轮\n"
            if rd.summary:
                rounds_text += f"汇总：{rd.summary}\n"
            for op in rd.opinions:
                rounds_text += f"- {op.expert_name}：{op.diagnosis}（置信度{op.confidence}）\n"
            if rd.divergences:
                rounds_text += f"分歧：{', '.join(rd.divergences)}\n"

        if consensus:
            conclusion = f"达成共识：{json.dumps(consensus, ensure_ascii=False)}"
        else:
            conclusion = f"未达成完全共识。分歧点：\n" + "\n".join(f"- {d}" for d in divergences)

        prompt = REPORT_GENERATION_PROMPT.format(
            patient_info=json.dumps(patient_info, ensure_ascii=False, indent=2),
            rounds_text=rounds_text,
            conclusion=conclusion,
        )

        full_text = ""
        for chunk in self._llm.stream([
            SystemMessage(content="你是一位专业的医学报告撰写者。"),
            HumanMessage(content=prompt),
        ]):
            token = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token and on_token:
                on_token(token)
            full_text += token

        return full_text

