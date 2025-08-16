from functools import lru_cache
from medmdt.api.models import ConsultationStore


@lru_cache
def get_consultation_store() -> ConsultationStore:
    return ConsultationStore()


def build_infrastructure() -> dict:
    """Build shared infrastructure components (settings, LLM, stores, embed_fn, retriever).

    Uses lazy imports to avoid import-time side effects. Returns a dict with keys:
    settings, llm, graph_store, vector_store, keyword_store, embed_fn, retriever.
    """
    from medmdt.config.settings import get_settings
    from medmdt.llm.provider import create_chat_model
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.knowledge.retriever import FusionRetriever

    settings = get_settings()
    llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)

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

    return {
        "settings": settings,
        "llm": llm,
        "graph_store": graph_store,
        "vector_store": vector_store,
        "keyword_store": keyword_store,
        "embed_fn": embed_fn,
        "retriever": retriever,
    }
