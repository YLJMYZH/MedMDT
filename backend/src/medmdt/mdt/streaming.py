import logging
from langchain_core.language_models import BaseChatModel

from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.experts.factory import select_experts
from medmdt.mdt.moderator import Moderator
from medmdt.mdt.state import MDTState, DiscussionRound
from medmdt.mdt.rounds import should_continue
from medmdt.knowledge.retriever import FusionRetriever

logger = logging.getLogger(__name__)


class StreamingOrchestrator:
    def __init__(
        self,
        experts: dict[str, BaseExpert],
        moderator: Moderator,
        retriever: FusionRetriever,
        selector_llm: BaseChatModel,
        event_bus,
        consultation_id: str,
    ):
        self._experts = experts
        self._moderator = moderator
        self._retriever = retriever
        self._selector_llm = selector_llm
        self._bus = event_bus
        self._cid = consultation_id

    def _emit(self, event: dict):
        event.setdefault("type", "status")
        self._bus.publish(self._cid, event)

    def run(
        self,
        patient_info: dict,
        medical_records: list[dict],
        max_rounds: int = 3,
    ) -> MDTState:
        # Phase 1: Retrieve knowledge
        self._emit({"type": "phase_start", "phase": "retrieve_knowledge"})
        query = patient_info.get("chief_complaint", "")
        if medical_records:
            query += " " + " ".join(r.get("content", "") for r in medical_records)
        results = self._retriever.retrieve(query, top_k=10)
        knowledge_context = [
            {"text": r.text, "score": r.score, "source": r.source}
            for r in results
        ]

        # Phase 2: Select experts
        self._emit({"type": "phase_start", "phase": "select_experts"})
        selected_ids = select_experts(self._experts, patient_info, self._selector_llm)
        if not selected_ids:
            selected_ids = list(self._experts.keys())
        expert_names = {eid: self._experts[eid].name for eid in selected_ids}
        self._emit({
            "type": "experts_selected",
            "expert_ids": selected_ids,
            "expert_names": expert_names,
        })

        # Phase 3: Discussion rounds
        self._emit({"type": "phase_start", "phase": "discussion"})
        discussion_rounds: list[dict] = []
        consensus = None
        divergences: list[str] = []
        current_round = 0

        while True:
            round_num = current_round + 1
            self._emit({
                "type": "round_start",
                "round_num": round_num,
                "total_rounds": max_rounds,
            })

            previous = discussion_rounds if current_round > 0 else None
            opinions = []

            for idx, eid in enumerate(selected_ids):
                expert = self._experts[eid]
                self._emit({
                    "type": "expert_start",
                    "round_num": round_num,
                    "expert_id": eid,
                    "expert_name": expert.name,
                    "expert_index": idx,
                    "total_experts": len(selected_ids),
                })

                def make_token_cb(expert_id):
                    def on_token(token: str):
                        self._emit({
                            "type": "expert_token",
                            "expert_id": expert_id,
                            "token": token,
                        })
                    return on_token

                opinion = expert.stream_analyze(
                    patient_info=patient_info,
                    medical_records=medical_records,
                    knowledge_context=knowledge_context,
                    previous_rounds=previous,
                    on_token=make_token_cb(eid),
                )
                opinions.append(opinion)

                self._emit({
                    "type": "expert_end",
                    "round_num": round_num,
                    "expert_id": eid,
                    "opinion": opinion.model_dump(),
                })

            round_result = self._moderator.summarize_round(opinions)
            round_result.round_num = round_num
            has_consensus = self._moderator.check_consensus(round_result, opinions)

            self._emit({
                "type": "round_summary",
                "round_num": round_num,
                "summary": round_result.summary or "",
                "divergences": round_result.divergences,
                "has_consensus": has_consensus,
            })

            round_dict = round_result.model_dump()
            discussion_rounds.append(round_dict)
            divergences = round_result.divergences
            current_round += 1

            if has_consensus:
                diagnoses = [op.diagnosis for op in opinions]
                consensus = {
                    "diagnosis": diagnoses[0] if diagnoses else "",
                    "confidence": sum(op.confidence for op in opinions) / len(opinions),
                    "experts_agreed": [op.expert_id for op in opinions],
                }

            if not should_continue(current_round, max_rounds, has_consensus):
                break

        # Phase 4: Generate report
        self._emit({"type": "phase_start", "phase": "generate_report"})
        rounds_objs = [DiscussionRound.model_validate(rd) for rd in discussion_rounds]

        def on_report_token(token: str):
            self._emit({"type": "report_token", "token": token})

        final_report = self._moderator.stream_generate_report(
            patient_info=patient_info,
            rounds=rounds_objs,
            consensus=consensus,
            divergences=divergences,
            on_token=on_report_token,
        )

        return MDTState(
            patient_info=patient_info,
            medical_records=medical_records,
            knowledge_context=knowledge_context,
            experts=selected_ids,
            discussion_rounds=discussion_rounds,
            consensus=consensus,
            divergences=divergences,
            final_report=final_report,
            current_round=current_round,
            max_rounds=max_rounds,
        )
