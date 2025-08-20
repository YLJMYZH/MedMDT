from functools import lru_cache
from medmdt.api.models import ConsultationStore


@lru_cache
def get_consultation_store() -> ConsultationStore:
    return ConsultationStore()


def build_infrastructure() -> dict:
    """Build shared infrastructure components (settings, LLM, stores, embed_fn, retriever).

    Uses lazy imports to avoid import-time side effects. Returns a dict with keys:
    settings, llm, vision_llm, graph_store, vector_store, keyword_store, embed_fn, retriever.
    """
    from medmdt.config.settings import get_settings
    from medmdt.llm.provider import create_chat_model
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.knowledge.retriever import FusionRetriever

    from medmdt.config.runtime import load_llm_settings, get_llm_kwargs

    settings = get_settings()
    runtime = load_llm_settings()

    knowledge_ep = runtime.knowledge
    llm = create_chat_model(
        knowledge_ep.provider,
        knowledge_ep.model,
        **get_llm_kwargs(knowledge_ep),
    )

    vision_ep = runtime.vision
    vision_llm = create_chat_model(
        vision_ep.provider,
        vision_ep.model,
        **get_llm_kwargs(vision_ep),
    )

    graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = VectorStore(
        settings.milvus_host, settings.milvus_port, "medmdt_chunks", embed_ep.dim,
    )
    keyword_store = KeywordStore(settings.elasticsearch_url)

    embed_ep = runtime.embedding
    embed_kwargs = {}
    if embed_ep.api_key:
        embed_kwargs["api_key"] = embed_ep.api_key
    if embed_ep.base_url:
        embed_kwargs["openai_api_base"] = embed_ep.base_url

    def embed_fn(texts):
        from langchain_openai import OpenAIEmbeddings
        embeddings = OpenAIEmbeddings(model=embed_ep.model, **embed_kwargs)
        return embeddings.embed_documents(texts)

    retriever = FusionRetriever(graph_store, vector_store, keyword_store, llm, embed_fn)

    return {
        "settings": settings,
        "llm": llm,
        "vision_llm": vision_llm,
        "graph_store": graph_store,
        "vector_store": vector_store,
        "keyword_store": keyword_store,
        "embed_fn": embed_fn,
        "retriever": retriever,
    }
