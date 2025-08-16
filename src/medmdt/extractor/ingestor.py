# src/medmdt/extractor/ingestor.py
"""Ingestor — writes ExtractionResult data to all three backing stores.

Orchestrates writes to:
- Neo4j GraphStore (entities + relations)
- Milvus VectorStore (text embeddings)
- Elasticsearch KeywordStore (full-text + keywords)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from medmdt.extractor.schemas import ExtractionResult
from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.knowledge.vector_store import VectorStore


@dataclass
class IngestReport:
    """Summary of what was written during an ingest operation."""

    entities_count: int
    relations_count: int
    chunks_count: int


class Ingestor:
    """Fans out an ExtractionResult to the graph, vector, and keyword stores."""

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._keyword = keyword_store
        self._embed_fn = embed_fn

    def ingest(self, result: ExtractionResult) -> IngestReport:
        """Write *result* to all three stores and return a summary report.

        Entities and relations are always sent to the graph store (even if
        the lists are empty — the store handles that gracefully).  Vector
        and keyword writes are skipped when ``result.chunks`` is empty.
        """
        entities_count = self._graph.upsert_entities(result.entities)
        relations_count = self._graph.upsert_relations(result.relations)

        chunks_count = 0
        if result.chunks:
            texts = [c.text for c in result.chunks]
            embeddings = self._embed_fn(texts)
            metadatas = [
                {
                    "summary": c.summary,
                    "keywords": c.keywords,
                    "domain": result.source.type,
                    **c.metadata,
                }
                for c in result.chunks
            ]
            self._vector.insert(texts, embeddings, metadatas)
            self._keyword.index_chunks(result.chunks, result.source)
            chunks_count = len(result.chunks)

        return IngestReport(
            entities_count=entities_count,
            relations_count=relations_count,
            chunks_count=chunks_count,
        )
