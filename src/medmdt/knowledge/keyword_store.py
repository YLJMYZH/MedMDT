# src/medmdt/knowledge/keyword_store.py
"""Elasticsearch BM25 keyword store for precise keyword matching of medical terms.

Uses the IK Chinese analyzer when available on the ES cluster; falls back
to the standard analyzer otherwise.
"""

from dataclasses import dataclass

from elasticsearch import Elasticsearch

from medmdt.extractor.schemas import SourceInfo, TextChunk


@dataclass
class KeywordSearchResult:
    """A single search result from the keyword store."""

    text: str
    score: float
    keywords: list[str]
    metadata: dict
    source: dict


# Index settings with IK smart analyzer for Chinese text segmentation.
INDEX_SETTINGS = {
    "settings": {
        "analysis": {
            "analyzer": {
                "ik_smart_analyzer": {
                    "type": "custom",
                    "tokenizer": "ik_smart",
                },
            },
        },
    },
    "mappings": {
        "properties": {
            "text": {"type": "text", "analyzer": "ik_smart_analyzer"},
            "summary": {"type": "text", "analyzer": "ik_smart_analyzer"},
            "keywords": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "source": {
                "properties": {
                    "file": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "page": {"type": "integer"},
                }
            },
        }
    },
}

# Fallback settings when IK analyzer is not installed on the ES cluster.
INDEX_SETTINGS_FALLBACK = {
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "summary": {"type": "text"},
            "keywords": {"type": "keyword"},
            "metadata": {"type": "object", "enabled": True},
            "source": {
                "properties": {
                    "file": {"type": "keyword"},
                    "type": {"type": "keyword"},
                    "page": {"type": "integer"},
                }
            },
        }
    },
}


class KeywordStore:
    """Manages an Elasticsearch index for BM25 keyword search of medical text chunks.

    Indexes TextChunk documents with source metadata and supports multi_match
    search with optional term filters.
    """

    def __init__(self, url: str, index_name: str = "medmdt_chunks") -> None:
        self._es = Elasticsearch(url)
        self._index_name = index_name

    def ensure_index(self) -> None:
        """Create the index if it doesn't exist.

        Attempts to create with IK smart analyzer first; falls back to
        standard analyzer if IK plugin is not available.
        """
        if self._es.indices.exists(index=self._index_name):
            return
        try:
            self._es.indices.create(index=self._index_name, body=INDEX_SETTINGS)
        except Exception:
            self._es.indices.create(
                index=self._index_name, body=INDEX_SETTINGS_FALLBACK
            )

    def index_chunks(self, chunks: list[TextChunk], source: SourceInfo) -> int:
        """Index a batch of text chunks with their source metadata.

        Returns the count of successfully indexed documents.
        """
        count = 0
        source_dict = source.model_dump()
        for chunk in chunks:
            doc = {
                "text": chunk.text,
                "summary": chunk.summary,
                "keywords": chunk.keywords,
                "metadata": chunk.metadata,
                "source": source_dict,
            }
            self._es.index(index=self._index_name, body=doc)
            count += 1
        return count

    def search(
        self,
        query: str,
        top_k: int = 10,
        filters: dict | None = None,
    ) -> list[KeywordSearchResult]:
        """Search for chunks matching the query using BM25.

        Args:
            query: The search query string.
            top_k: Maximum number of results to return.
            filters: Optional dict of field-value pairs for term filtering
                     (e.g. ``{"source.type": "clinical_guideline"}``).

        Returns:
            List of KeywordSearchResult ordered by BM25 score (descending).
        """
        if filters:
            filter_clauses = [{"term": {k: v}} for k, v in filters.items()]
            body = {
                "query": {
                    "bool": {
                        "must": {
                            "multi_match": {
                                "query": query,
                                "fields": ["text", "summary", "keywords"],
                            }
                        },
                        "filter": filter_clauses,
                    }
                },
                "size": top_k,
            }
        else:
            body = {
                "query": {
                    "multi_match": {
                        "query": query,
                        "fields": ["text", "summary", "keywords"],
                    }
                },
                "size": top_k,
            }

        resp = self._es.search(index=self._index_name, body=body)
        results = []
        for hit in resp["hits"]["hits"]:
            src = hit["_source"]
            results.append(
                KeywordSearchResult(
                    text=src["text"],
                    score=hit["_score"],
                    keywords=src.get("keywords", []),
                    metadata=src.get("metadata", {}),
                    source=src.get("source", {}),
                )
            )
        return results

    def close(self) -> None:
        """Close the Elasticsearch client connection."""
        self._es.close()
