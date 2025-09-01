# scripts/ingest.py
"""CLI entry point for ingesting medical documents into the knowledge base.

Usage:
    python scripts/ingest.py <file_or_directory>

Processes PDF, image, and DICOM files through the extraction pipeline
and writes entities, relations, and text chunks to the backing stores.
"""

import sys
from pathlib import Path

from medmdt.config.runtime import get_llm_kwargs, load_llm_settings
from medmdt.config.settings import get_settings
from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.http_clients import (
    close_shared_http_clients_sync,
    get_shared_async_http_client,
    get_shared_http_client,
)
from medmdt.llm.network_policy import validate_base_url
from medmdt.llm.provider import (
    PROVIDER_REGISTRY,
    create_chat_model,
    create_vision_model,
)
from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.vector_store import VectorStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.extractor.agent import ExtractionAgent
from medmdt.extractor.ingestor import IngestReport
from medmdt.extractor.file_types import MEDICAL_FILE_EXTENSIONS

SUPPORTED_EXTENSIONS = MEDICAL_FILE_EXTENSIONS


def build_agent() -> ExtractionAgent:
    """Wire up all dependencies and return a ready-to-use ExtractionAgent."""
    settings = get_settings()
    runtime = load_llm_settings()
    validate_base_url(settings.default_llm_provider, None)
    vision_spec = PROVIDER_REGISTRY.get(runtime.vision.provider)
    if vision_spec is None or vision_spec.vision_factory is not None:
        validate_base_url(runtime.vision.provider, runtime.vision.base_url)
    validate_base_url("openai", None)

    llm = create_chat_model(settings.default_llm_provider, settings.default_llm_model)
    vision_llm = None
    vision_error = None
    try:
        vision_llm = create_vision_model(
            runtime.vision.provider,
            runtime.vision.model,
            **get_llm_kwargs(runtime.vision),
        )
    except (VisionProviderNotSupported, ValueError) as exc:
        vision_error = str(exc)

    graph_store = GraphStore(settings.neo4j_uri, settings.neo4j_user, settings.neo4j_password)
    vector_store = VectorStore(
        settings.milvus_host, settings.milvus_port, "medmdt_chunks", settings.embedding_dim,
    )
    vector_store.ensure_collection()
    keyword_store = KeywordStore(settings.elasticsearch_url)
    keyword_store.ensure_index()

    def embed_fn(texts: list[str]) -> list[list[float]]:
        from langchain_openai import OpenAIEmbeddings

        embeddings = OpenAIEmbeddings(
            model=settings.embedding_model,
            http_client=get_shared_http_client(),
            http_async_client=get_shared_async_http_client(),
        )
        return embeddings.embed_documents(texts)

    return ExtractionAgent(
        settings=settings,
        graph_store=graph_store,
        vector_store=vector_store,
        keyword_store=keyword_store,
        embed_fn=embed_fn,
        llm=llm,
        vision_llm=vision_llm,
        vision_error=vision_error,
    )


def run_ingest(path: str) -> list[IngestReport]:
    """Process a file or directory and return ingest reports.

    For a single file, processes it directly.
    For a directory, processes all files with supported extensions.
    """
    try:
        agent = build_agent()
        target = Path(path)
        all_reports: list[IngestReport] = []

        if target.is_file():
            reports = agent.process_file(str(target))
            all_reports.extend(reports)
            _print_reports(str(target), reports)
        elif target.is_dir():
            files = [
                f for f in target.iterdir() if f.suffix.lower() in SUPPORTED_EXTENSIONS
            ]
            for f in sorted(files):
                try:
                    reports = agent.process_file(str(f))
                    all_reports.extend(reports)
                    _print_reports(str(f), reports)
                except Exception as e:
                    print(f"[ERROR] {f}: {e}")
        else:
            print(f"Path not found: {path}")
            sys.exit(1)

        return all_reports
    finally:
        close_shared_http_clients_sync()


def _print_reports(file_path: str, reports: list[IngestReport]) -> None:
    """Print a summary line for a processed file."""
    total_e = sum(r.entities_count for r in reports)
    total_r = sum(r.relations_count for r in reports)
    total_c = sum(r.chunks_count for r in reports)
    print(f"[OK] {file_path}: {total_e} entities, {total_r} relations, {total_c} chunks")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python scripts/ingest.py <file_or_directory>")
        sys.exit(1)
    run_ingest(sys.argv[1])
