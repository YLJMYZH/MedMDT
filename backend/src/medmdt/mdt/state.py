# src/medmdt/mdt/state.py
from typing import TypedDict
from pydantic import BaseModel, Field


class ExpertOpinion(BaseModel):
    expert_id: str
    expert_name: str
    analysis: str
    diagnosis: str
    recommendation: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasoning: str
    references: list[str] = Field(default_factory=list)


class DiscussionRound(BaseModel):
    round_num: int
    opinions: list[ExpertOpinion] = Field(default_factory=list)
    summary: str | None = None
    divergences: list[str] = Field(default_factory=list)


class MDTState(TypedDict):
    patient_info: dict
    medical_records: list[dict]
    knowledge_context: list[dict]
    experts: list[str]
    discussion_rounds: list[dict]
    consensus: dict | None
    divergences: list[str]
    final_report: str | None
    current_round: int
    max_rounds: int
