from functools import lru_cache
from medmdt.api.models import ConsultationStore


@lru_cache
def get_consultation_store() -> ConsultationStore:
    return ConsultationStore()


def build_infrastructure() -> dict:
    """Build shared infrastructure components (settings, LLM, stores, embed_fn, retriever).

    Uses lazy imports to avoid import-time side effects. Returns a dict with keys:
    settings, llm, vision_llm, vision_error, vision_provider, vision_model,
    graph_store, vector_store, keyword_store, embed_fn, retriever.
    """
    from medmdt.config.settings import get_settings
    from medmdt.llm.errors import VisionProviderNotSupported
    from medmdt.llm.network_policy import validate_base_url
    from medmdt.llm.provider import (
        PROVIDER_REGISTRY,
        create_chat_model,
        create_vision_model,
        get_embedding_base_url,
        resolve_embedding_api_key,
    )
    from medmdt.knowledge.graph_store import GraphStore
    from medmdt.knowledge.vector_store import VectorStore
    from medmdt.knowledge.keyword_store import KeywordStore
    from medmdt.knowledge.retriever import FusionRetriever

    from medmdt.config.runtime import load_llm_settings, get_llm_kwargs

    settings = get_settings()
    runtime = load_llm_settings()

    knowledge_ep = runtime.knowledge
    vision_ep = runtime.vision
    embed_ep = runtime.embedding

    validate_base_url(knowledge_ep.provider, knowledge_ep.base_url)
    vision_spec = PROVIDER_REGISTRY.get(vision_ep.provider)
    if vision_spec is None or vision_spec.vision_factory is not None:
        validate_base_url(vision_ep.provider, vision_ep.base_url)
    embedding_provider = embed_ep.provider or "openai"
    embedding_base_url = get_embedding_base_url(
        embedding_provider,
        embed_ep.base_url,
    )
    validate_base_url(embedding_provider, embedding_base_url)
    embedding_api_key = resolve_embedding_api_key(
        embedding_provider,
        embed_ep.api_key,
    )

    llm = create_chat_model(
        knowledge_ep.provider,
        knowledge_ep.model,
        **get_llm_kwargs(knowledge_ep),
    )

    vision_llm = None
    vision_error = None
    try:
        vision_llm = create_vision_model(
            vision_ep.provider,
            vision_ep.model,
            **get_llm_kwargs(vision_ep),
        )
    except VisionProviderNotSupported as exc:
        vision_error = str(exc)

    graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = VectorStore(
        settings.milvus_host, settings.milvus_port, "medmdt_chunks", embed_ep.dim,
    )
    keyword_store = KeywordStore(settings.elasticsearch_url)

    embed_kwargs = {"api_key": embedding_api_key}
    if embedding_base_url:
        embed_kwargs["openai_api_base"] = embedding_base_url

    def embed_fn(texts):
        from langchain_openai import OpenAIEmbeddings
        from medmdt.llm.http_clients import (
            get_shared_async_http_client,
            get_shared_http_client,
        )

        current_embedding_base_url = get_embedding_base_url(
            embedding_provider,
            embed_ep.base_url,
        )
        validate_base_url(embedding_provider, current_embedding_base_url)
        embed_kwargs["api_key"] = resolve_embedding_api_key(
            embedding_provider,
            embed_ep.api_key,
        )
        embed_kwargs["openai_api_base"] = current_embedding_base_url
        embeddings = OpenAIEmbeddings(
            model=embed_ep.model,
            http_client=get_shared_http_client(),
            http_async_client=get_shared_async_http_client(),
            **embed_kwargs,
        )
        return embeddings.embed_documents(texts)

    retriever = FusionRetriever(graph_store, vector_store, keyword_store, llm, embed_fn)

    return {
        "settings": settings,
        "llm": llm,
        "vision_llm": vision_llm,
        "vision_error": vision_error,
        "vision_provider": vision_ep.provider,
        "vision_model": vision_ep.model,
        "graph_store": graph_store,
        "vector_store": vector_store,
        "keyword_store": keyword_store,
        "embed_fn": embed_fn,
        "retriever": retriever,
    }
