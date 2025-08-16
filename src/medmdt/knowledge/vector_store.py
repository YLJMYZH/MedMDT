# src/medmdt/knowledge/vector_store.py
"""Milvus-backed vector store for semantic similarity search of medical text chunks."""

import json
from dataclasses import dataclass

from pymilvus import (
    Collection,
    CollectionSchema,
    DataType,
    FieldSchema,
    connections,
    utility,
)


@dataclass
class VectorSearchResult:
    """A single search result from the vector store."""

    text: str
    score: float
    metadata: dict


class VectorStore:
    """Manages a Milvus collection for storing and searching text embeddings.

    Uses IVF_FLAT index with COSINE metric for approximate nearest-neighbor search.
    Supports optional domain-based filtering via metadata.
    """

    def __init__(
        self, host: str, port: int, collection_name: str, embedding_dim: int
    ) -> None:
        self._host = host
        self._port = port
        self._collection_name = collection_name
        self._embedding_dim = embedding_dim
        self._collection: Collection | None = None
        connections.connect(alias="default", host=host, port=port)

    def ensure_collection(self) -> None:
        """Create the collection if it doesn't exist, or load an existing one."""
        if utility.has_collection(self._collection_name):
            self._collection = Collection(self._collection_name)
            self._collection.load()
            return

        fields = [
            FieldSchema(
                name="id", dtype=DataType.INT64, is_primary=True, auto_id=True
            ),
            FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=65535),
            FieldSchema(
                name="embedding",
                dtype=DataType.FLOAT_VECTOR,
                dim=self._embedding_dim,
            ),
            FieldSchema(name="metadata", dtype=DataType.VARCHAR, max_length=65535),
        ]
        schema = CollectionSchema(fields=fields)
        self._collection = Collection(self._collection_name, schema)

        index_params = {
            "metric_type": "COSINE",
            "index_type": "IVF_FLAT",
            "params": {"nlist": 128},
        }
        self._collection.create_index("embedding", index_params)
        self._collection.load()

    def insert(
        self,
        texts: list[str],
        embeddings: list[list[float]],
        metadatas: list[dict],
    ) -> list[int]:
        """Insert text chunks with their embeddings and metadata.

        Returns the list of auto-generated primary key IDs.
        """
        data = [
            texts,
            embeddings,
            [json.dumps(m, ensure_ascii=False) for m in metadatas],
        ]
        result = self._collection.insert(data)
        self._collection.flush()
        return result.primary_keys

    def search(
        self,
        query_embedding: list[float],
        top_k: int = 10,
        domain: str | None = None,
    ) -> list[VectorSearchResult]:
        """Search for the top_k most similar embeddings.

        Args:
            query_embedding: The query vector.
            top_k: Number of results to return.
            domain: Optional domain filter (e.g. "guideline", "case").

        Returns:
            List of VectorSearchResult ordered by similarity score (descending).
        """
        search_params = {"metric_type": "COSINE", "params": {"nprobe": 16}}
        expr = None
        if domain:
            expr = f'metadata like "%\\"domain\\": \\"{domain}\\"%"'

        results = self._collection.search(
            data=[query_embedding],
            anns_field="embedding",
            param=search_params,
            limit=top_k,
            expr=expr,
            output_fields=["text", "metadata"],
        )

        search_results = []
        for hit in results[0]:
            meta_str = hit.entity.get("metadata")
            meta = json.loads(meta_str) if meta_str else {}
            search_results.append(
                VectorSearchResult(
                    text=hit.entity.get("text"),
                    score=hit.distance,
                    metadata=meta,
                )
            )
        return search_results

    def close(self) -> None:
        """Release the collection and disconnect from Milvus."""
        if self._collection:
            self._collection.release()
        connections.disconnect("default")
