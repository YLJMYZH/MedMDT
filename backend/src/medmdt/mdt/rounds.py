# src/medmdt/mdt/rounds.py
from medmdt.mdt.state import ExpertOpinion
from medmdt.mdt.experts.base import BaseExpert


def run_expert_round(
    experts: dict[str, BaseExpert],
    expert_ids: list[str],
    patient_info: dict,
    medical_records: list[dict],
    knowledge_context: list[dict],
    previous_rounds: list[dict] | None = None,
) -> list[ExpertOpinion]:
    opinions = []
    for eid in expert_ids:
        expert = experts[eid]
        opinion = expert.analyze(
            patient_info=patient_info,
            medical_records=medical_records,
            knowledge_context=knowledge_context,
            previous_rounds=previous_rounds,
        )
        opinions.append(opinion)
    return opinions


def should_continue(current_round: int, max_rounds: int, has_consensus: bool) -> bool:
    if has_consensus:
        return False
    return current_round < max_rounds
