# src/medmdt/mdt/experts/base.py
import json
import re
from langchain_core.language_models import BaseChatModel
from langchain_core.messages import SystemMessage, HumanMessage
from medmdt.mdt.state import ExpertOpinion
from medmdt.llm.prompts.mdt.expert_analysis import (
    EXPERT_ANALYSIS_PROMPT, PREVIOUS_ROUNDS_SECTION, NO_PREVIOUS_ROUNDS,
)


class BaseExpert:
    def __init__(
        self,
        expert_id: str,
        name: str,
        system_prompt: str,
        knowledge_domains: list[str],
        llm: BaseChatModel,
    ):
        self.expert_id = expert_id
        self.name = name
        self.system_prompt = system_prompt
        self.knowledge_domains = knowledge_domains
        self._llm = llm

    def analyze(
        self,
        patient_info: dict,
        medical_records: list[dict],
        knowledge_context: list[dict],
        previous_rounds: list[dict] | None = None,
    ) -> ExpertOpinion:
        previous_section = self._format_previous_rounds(previous_rounds)

        prompt = EXPERT_ANALYSIS_PROMPT.format(
            patient_info=json.dumps(patient_info, ensure_ascii=False, indent=2),
            medical_records=json.dumps(medical_records, ensure_ascii=False, indent=2),
            knowledge_context=json.dumps(knowledge_context, ensure_ascii=False, indent=2),
            previous_rounds_section=previous_section,
        )

        messages = [
            SystemMessage(content=self.system_prompt),
            HumanMessage(content=prompt),
        ]
        response = self._llm.invoke(messages)
        data = self._parse_json(response.content)

        return ExpertOpinion(
            expert_id=self.expert_id,
            expert_name=self.name,
            **data,
        )

    def _format_previous_rounds(self, rounds: list[dict] | None) -> str:
        if not rounds:
            return NO_PREVIOUS_ROUNDS
        rounds_text = json.dumps(rounds, ensure_ascii=False, indent=2)
        all_divergences = []
        for rd in rounds:
            all_divergences.extend(rd.get("divergences", []))
        divergences_text = "\n".join(f"- {d}" for d in all_divergences) if all_divergences else "无明确分歧"
        return PREVIOUS_ROUNDS_SECTION.format(
            rounds_text=rounds_text,
            divergences_text=divergences_text,
        )

    @staticmethod
    def _parse_json(text: str) -> dict:
        match = re.search(r"```(?:json)?\s*\n?(.*?)\n?\s*```", text, re.DOTALL)
        if match:
            text = match.group(1)
        return json.loads(text.strip())
