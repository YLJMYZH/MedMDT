# src/medmdt/mdt/graph.py
from langgraph.graph import StateGraph, END
from langchain_core.language_models import BaseChatModel

from medmdt.mdt.state import MDTState, DiscussionRound
from medmdt.mdt.experts.base import BaseExpert
from medmdt.mdt.experts.factory import select_experts
from medmdt.mdt.moderator import Moderator
from medmdt.mdt.rounds import run_expert_round, should_continue
from medmdt.knowledge.retriever import FusionRetriever


def build_mdt_graph(
    experts: dict[str, BaseExpert],
    moderator: Moderator,
    retriever: FusionRetriever,
    selector_llm: BaseChatModel,
):
    def retrieve_knowledge(state: MDTState) -> dict:
        query = state["patient_info"].get("chief_complaint", "")
        if state["medical_records"]:
            query += " " + " ".join(
                r.get("content", "") for r in state["medical_records"]
            )
        results = retriever.retrieve(query, top_k=10)
        return {
            "knowledge_context": [
                {"text": r.text, "score": r.score, "source": r.source}
                for r in results
            ]
        }

    def select_expert_ids(state: MDTState) -> dict:
        selected = select_experts(experts, state["patient_info"], selector_llm)
        if not selected:
            selected = list(experts.keys())
        return {"experts": selected}

    def run_discussion_round(state: MDTState) -> dict:
        current = state["current_round"]
        previous = state["discussion_rounds"] if current > 0 else None

        opinions = run_expert_round(
            experts=experts,
            expert_ids=state["experts"],
            patient_info=state["patient_info"],
            medical_records=state["medical_records"],
            knowledge_context=state["knowledge_context"],
            previous_rounds=previous,
        )

        round_result = moderator.summarize_round(opinions)
        round_result.round_num = current + 1

        has_consensus = moderator.check_consensus(round_result, opinions)

        round_dict = round_result.model_dump()
        new_rounds = state["discussion_rounds"] + [round_dict]

        update = {
            "discussion_rounds": new_rounds,
            "current_round": current + 1,
            "divergences": round_result.divergences,
        }

        if has_consensus:
            diagnoses = [op.diagnosis for op in opinions]
            update["consensus"] = {
                "diagnosis": diagnoses[0] if diagnoses else "",
                "confidence": sum(op.confidence for op in opinions) / len(opinions),
                "experts_agreed": [op.expert_id for op in opinions],
            }
        else:
            update["consensus"] = None

        return update

    def generate_final_report(state: MDTState) -> dict:
        rounds = [DiscussionRound.model_validate(rd) for rd in state["discussion_rounds"]]
        report = moderator.generate_report(
            patient_info=state["patient_info"],
            rounds=rounds,
            consensus=state["consensus"],
            divergences=state["divergences"],
        )
        return {"final_report": report}

    def should_discuss_more(state: MDTState) -> str:
        has_consensus = state["consensus"] is not None
        if should_continue(state["current_round"], state["max_rounds"], has_consensus):
            return "continue"
        return "report"

    graph = StateGraph(MDTState)

    graph.add_node("retrieve_knowledge", retrieve_knowledge)
    graph.add_node("select_experts", select_expert_ids)
    graph.add_node("discussion_round", run_discussion_round)
    graph.add_node("generate_report", generate_final_report)

    graph.set_entry_point("retrieve_knowledge")
    graph.add_edge("retrieve_knowledge", "select_experts")
    graph.add_edge("select_experts", "discussion_round")
    graph.add_conditional_edges(
        "discussion_round",
        should_discuss_more,
        {"continue": "discussion_round", "report": "generate_report"},
    )
    graph.add_edge("generate_report", END)

    return graph.compile()


def run_consultation(
    graph,
    patient_info: dict,
    medical_records: list[dict],
    max_rounds: int = 3,
) -> MDTState:
    initial_state: MDTState = {
        "patient_info": patient_info,
        "medical_records": medical_records,
        "knowledge_context": [],
        "experts": [],
        "discussion_rounds": [],
        "consensus": None,
        "divergences": [],
        "final_report": None,
        "current_round": 0,
        "max_rounds": max_rounds,
    }
    result = graph.invoke(initial_state)
    return result
