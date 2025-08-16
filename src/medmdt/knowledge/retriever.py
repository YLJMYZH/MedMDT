# src/medmdt/knowledge/retriever.py
"""Fusion retriever combining graph, vector, and keyword search via Reciprocal Rank Fusion.

Uses an LLM to analyze the incoming query and dynamically adjust per-channel
weights before fusing the three ranked result lists.
"""

from __future__ import annotations

import json
import logging
from collections import defaultdict
from typing import Any, Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.knowledge.schemas import RetrievalResult
from medmdt.knowledge.vector_store import VectorStore
from medmdt.llm.prompts.retrieval.query_analysis import QUERY_ANALYSIS_PROMPT

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------
# Standalone RRF function
# ------------------------------------------------------------------


def rrf_fuse(
    ranked_lists: list[list],
    k: int = 60,
    weights: list[float] | None = None,
) -> list[tuple[Any, float]]:
    """Reciprocal Rank Fusion over multiple ranked lists.

    Args:
        ranked_lists: Each inner list contains items ordered by relevance
                      (best first).
        k: Smoothing constant (default 60 per the original RRF paper).
        weights: Optional per-list weight multipliers.  When *None* every
                 list is weighted equally at ``1.0``.

    Returns:
        A list of ``(item, fused_score)`` tuples sorted by descending score.
    """
    if not ranked_lists:
        return []

    if weights is None:
        weights = [1.0] * len(ranked_lists)

    scores: dict[str, float] = defaultdict(float)
    item_map: dict[str, Any] = {}

    for ranked_list, weight in zip(ranked_lists, weights):
        for rank, item in enumerate(ranked_list):
            key = str(item)
            item_map[key] = item
            scores[key] += weight / (k + rank + 1)

    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(item_map[key], score) for key, score in sorted_items]


# ------------------------------------------------------------------
# FusionRetriever
# ------------------------------------------------------------------

_DEFAULT_WEIGHTS = {"graph": 0.33, "vector": 0.34, "keyword": 0.33}


class FusionRetriever:
    """Multi-backend retriever that fuses graph, vector, and keyword results.

    The query is first analysed by an LLM to extract medical entities,
    keywords, and per-channel weight recommendations.  Each backend is
    queried independently and the three ranked lists are combined using
    :func:`rrf_fuse`.
    """

    def __init__(
        self,
        graph_store: GraphStore,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        llm: BaseChatModel,
        embed_fn: Callable[[list[str]], list[list[float]]],
    ) -> None:
        self._graph = graph_store
        self._vector = vector_store
        self._keyword = keyword_store
        self._llm = llm
        self._embed_fn = embed_fn

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def retrieve(self, query: str, top_k: int = 10) -> list[RetrievalResult]:
        """Retrieve the *top_k* most relevant results across all stores.

        1. Analyse the query via LLM to extract entities, keywords, and
           per-channel weights.
        2. Search each backend.
        3. Fuse the three ranked lists with weighted RRF.
        4. Return the top *top_k* results as :class:`RetrievalResult`.
        """
        analysis = self._analyze_query(query)
        keywords = analysis.get("keywords", [])
        entities = analysis.get("entities", [])
        weights_cfg = analysis.get("weights", _DEFAULT_WEIGHTS)

        graph_results = self._search_graph(entities)
        vector_results = self._search_vector(query)
        keyword_results = self._search_keyword(query, keywords)

        weights = [
            weights_cfg.get("graph", _DEFAULT_WEIGHTS["graph"]),
            weights_cfg.get("vector", _DEFAULT_WEIGHTS["vector"]),
            weights_cfg.get("keyword", _DEFAULT_WEIGHTS["keyword"]),
        ]
        fused = rrf_fuse(
            [graph_results, vector_results, keyword_results], weights=weights
        )

        results: list[RetrievalResult] = []
        for item, score in fused[:top_k]:
            if isinstance(item, RetrievalResult):
                item.score = score
                results.append(item)
            else:
                results.append(
                    RetrievalResult(text=str(item), score=score, source="fused")
                )
        return results

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _analyze_query(self, query: str) -> dict:
        """Use the LLM to decompose the query into entities / keywords / weights."""
        prompt = QUERY_ANALYSIS_PROMPT.format(query=query)
        response = self._llm.invoke([HumanMessage(content=prompt)])
        try:
            return json.loads(response.content)
        except json.JSONDecodeError:
            logger.warning("LLM returned invalid JSON for query analysis; using defaults")
            return {"keywords": [], "entities": [], "weights": _DEFAULT_WEIGHTS}

    def _search_graph(self, entities: list[str]) -> list[RetrievalResult]:
        """Query the graph store for each extracted entity."""
        results: list[RetrievalResult] = []
        for entity in entities:
            relations = self._graph.query_by_entity(entity)
            for rel in relations:
                text = f"{entity} --[{rel.get('rel_type', '')}]--> {rel.get('related', '')}"
                results.append(
                    RetrievalResult(
                        text=text,
                        score=0.0,
                        source="graph",
                        metadata={"entity": entity, "relation": rel},
                    )
                )
        return results

    def _search_vector(self, query: str) -> list[RetrievalResult]:
        """Embed the query and search the vector store."""
        embeddings = self._embed_fn([query])
        hits = self._vector.search(embeddings[0])
        return [
            RetrievalResult(
                text=h.text, score=h.score, source="vector", metadata=h.metadata
            )
            for h in hits
        ]

    def _search_keyword(
        self, query: str, keywords: list[str]
    ) -> list[RetrievalResult]:
        """Search the keyword store, enriching the query with extracted keywords."""
        search_query = f"{query} {' '.join(keywords)}" if keywords else query
        hits = self._keyword.search(search_query)
        return [
            RetrievalResult(
                text=h.text,
                score=h.score,
                source="keyword",
                metadata={**h.metadata, "keywords": h.keywords},
            )
            for h in hits
        ]
