import logging
from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from medmdt.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    ConsultationResult,
    ConsultationStatus,
    ConsultationStore,
)
from medmdt.api.deps import get_consultation_store

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/consultation", tags=["consultation"])


def run_consultation_task(consultation_id: str, store: ConsultationStore) -> None:
    from medmdt.config.settings import get_settings
    from medmdt.llm.provider import create_chat_model
    from medmdt.mdt.experts.factory import create_all_experts
    from medmdt.mdt.moderator import Moderator
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.knowledge.retriever import FusionRetriever
    from medmdt.mdt.graph import build_mdt_graph, run_consultation

    entry = store.get(consultation_id)
    if not entry:
        return

    store.update(consultation_id, status=ConsultationStatus.RUNNING)

    try:
        settings = get_settings()
        llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)

        experts = create_all_experts("config/experts.yaml")
        moderator = Moderator(llm=llm, consensus_threshold=settings.mdt_consensus_threshold)

        graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
        vector_store = VectorStore(
            settings.milvus_host, settings.milvus_port, "medmdt_chunks", settings.embedding_dim,
        )
        keyword_store = KeywordStore(settings.elasticsearch_url)

        def embed_fn(texts):
            from langchain_openai import OpenAIEmbeddings
            embeddings = OpenAIEmbeddings(model=settings.embedding_model)
            return embeddings.embed_documents(texts)

        retriever = FusionRetriever(graph_store, vector_store, keyword_store, llm, embed_fn)

        graph = build_mdt_graph(experts, moderator, retriever, llm)
        result = run_consultation(
            graph,
            patient_info=entry["patient_info"],
            medical_records=entry["medical_records"],
            max_rounds=entry["max_rounds"],
        )

        store.update(
            consultation_id,
            status=ConsultationStatus.COMPLETED,
            final_report=result["final_report"],
            discussion_rounds=result["discussion_rounds"],
            consensus=result["consensus"],
            divergences=result["divergences"],
        )
    except Exception as e:
        logger.exception("Consultation %s failed", consultation_id)
        store.update(consultation_id, status=ConsultationStatus.FAILED)


@router.post("", status_code=201, response_model=ConsultationResponse)
def create_consultation(
    request: ConsultationRequest,
    background_tasks: BackgroundTasks,
    store: ConsultationStore = Depends(get_consultation_store),
):
    cid = store.create(request)
    entry = store.get(cid)
    background_tasks.add_task(run_consultation_task, cid, store)
    return ConsultationResponse(
        id=cid, status=entry["status"], created_at=entry["created_at"],
    )


@router.get("", response_model=list[ConsultationResult])
def list_consultations(
    store: ConsultationStore = Depends(get_consultation_store),
):
    items = store.list_all()
    return [
        ConsultationResult(
            id=item["id"],
            status=item["status"],
            final_report=item.get("final_report"),
            discussion_rounds=item.get("discussion_rounds", []),
            consensus=item.get("consensus"),
            divergences=item.get("divergences", []),
        )
        for item in items
    ]


@router.get("/{consultation_id}", response_model=ConsultationResult)
def get_consultation(
    consultation_id: str,
    store: ConsultationStore = Depends(get_consultation_store),
):
    entry = store.get(consultation_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return ConsultationResult(
        id=entry["id"],
        status=entry["status"],
        final_report=entry.get("final_report"),
        discussion_rounds=entry.get("discussion_rounds", []),
        consensus=entry.get("consensus"),
        divergences=entry.get("divergences", []),
    )
