# tests/test_retriever.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.retriever import FusionRetriever, rrf_fuse
from medmdt.knowledge.schemas import RetrievalResult


def test_retrieval_result():
    r = RetrievalResult(text="test", score=0.9, source="neo4j", metadata={})
    assert r.source == "neo4j"


def test_retrieval_result_default_metadata():
    r = RetrievalResult(text="test", score=0.5, source="vector")
    assert r.metadata == {}


def test_rrf_fuse_single_list():
    items = ["a", "b", "c"]
    fused = rrf_fuse([items])
    assert fused[0][0] == "a"
    assert fused[0][1] > fused[1][1]


def test_rrf_fuse_multiple_lists():
    list1 = ["a", "b", "c"]
    list2 = ["c", "a", "b"]
    fused = rrf_fuse([list1, list2])
    scores = {item: score for item, score in fused}
    assert scores["a"] > scores["c"]  # a is rank 1 in list1, rank 2 in list2


def test_rrf_fuse_with_weights():
    list1 = ["a", "b"]
    list2 = ["b", "a"]
    fused_equal = rrf_fuse([list1, list2], weights=[1.0, 1.0])
    fused_weighted = rrf_fuse([list1, list2], weights=[3.0, 1.0])
    scores_equal = {item: score for item, score in fused_equal}
    scores_weighted = {item: score for item, score in fused_weighted}
    assert scores_weighted["a"] > scores_weighted["b"]


def test_rrf_fuse_empty():
    result = rrf_fuse([])
    assert result == []


def test_rrf_fuse_k_parameter():
    """Changing k should affect the score magnitude but not ranking."""
    items = ["a", "b", "c"]
    fused_k60 = rrf_fuse([items], k=60)
    fused_k10 = rrf_fuse([items], k=10)
    # k=10 scores should be higher since denominator is smaller
    assert fused_k10[0][1] > fused_k60[0][1]
    # Ranking should be the same
    assert [x[0] for x in fused_k10] == [x[0] for x in fused_k60]


def test_rrf_fuse_score_formula():
    """Verify the exact RRF score for a known case."""
    # Single list with one item, k=60, weight=1.0
    # score = 1.0 / (60 + 0 + 1) = 1/61
    fused = rrf_fuse([["x"]], k=60, weights=[1.0])
    assert len(fused) == 1
    assert fused[0][0] == "x"
    assert abs(fused[0][1] - 1.0 / 61) < 1e-10


def test_fusion_retriever_retrieve():
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    llm = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 1024])

    llm_response = MagicMock()
    llm_response.content = '{"keywords": ["糖尿病"], "entities": ["糖尿病"], "weights": {"graph": 0.3, "vector": 0.4, "keyword": 0.3}}'
    llm.invoke.return_value = llm_response

    graph.query_by_entity.return_value = [
        {"related": "二甲双胍", "rel_type": "treats", "direction": "outgoing"}
    ]

    from medmdt.knowledge.vector_store import VectorSearchResult
    vector.search.return_value = [
        VectorSearchResult(text="向量结果1", score=0.9, metadata={}),
    ]

    from medmdt.knowledge.keyword_store import KeywordSearchResult
    keyword.search.return_value = [
        KeywordSearchResult(text="关键词结果1", score=5.0, keywords=["糖尿病"], metadata={}, source={}),
    ]

    retriever = FusionRetriever(graph, vector, keyword, llm, embed_fn)
    results = retriever.retrieve("糖尿病的一线治疗方案")

    assert len(results) > 0
    assert all(isinstance(r, RetrievalResult) for r in results)
    llm.invoke.assert_called_once()


def test_fusion_retriever_calls_all_stores():
    """Verify that all three stores are invoked during retrieval."""
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    llm = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 4])

    llm_response = MagicMock()
    llm_response.content = '{"keywords": ["test"], "entities": ["entity1"], "weights": {"graph": 0.33, "vector": 0.34, "keyword": 0.33}}'
    llm.invoke.return_value = llm_response

    graph.query_by_entity.return_value = []
    vector.search.return_value = []
    keyword.search.return_value = []

    retriever = FusionRetriever(graph, vector, keyword, llm, embed_fn)
    retriever.retrieve("test query")

    graph.query_by_entity.assert_called_once_with("entity1")
    vector.search.assert_called_once()
    keyword.search.assert_called_once()
    embed_fn.assert_called_once_with(["test query"])


def test_fusion_retriever_top_k():
    """Verify that top_k limits the number of results."""
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    llm = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 4])

    llm_response = MagicMock()
    llm_response.content = '{"keywords": [], "entities": ["e1", "e2"], "weights": {"graph": 0.33, "vector": 0.34, "keyword": 0.33}}'
    llm.invoke.return_value = llm_response

    # Return many graph results
    graph.query_by_entity.return_value = [
        {"related": f"r{i}", "rel_type": "t", "direction": "outgoing"}
        for i in range(10)
    ]

    from medmdt.knowledge.vector_store import VectorSearchResult
    vector.search.return_value = [
        VectorSearchResult(text=f"v{i}", score=0.9 - i * 0.1, metadata={})
        for i in range(5)
    ]

    from medmdt.knowledge.keyword_store import KeywordSearchResult
    keyword.search.return_value = [
        KeywordSearchResult(text=f"k{i}", score=5.0 - i, keywords=[], metadata={}, source={})
        for i in range(5)
    ]

    retriever = FusionRetriever(graph, vector, keyword, llm, embed_fn)
    results = retriever.retrieve("test", top_k=3)
    assert len(results) <= 3


def test_fusion_retriever_results_sorted_by_score():
    """Results should be sorted by RRF score descending."""
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    llm = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 4])

    llm_response = MagicMock()
    llm_response.content = '{"keywords": [], "entities": ["e1"], "weights": {"graph": 0.33, "vector": 0.34, "keyword": 0.33}}'
    llm.invoke.return_value = llm_response

    graph.query_by_entity.return_value = [
        {"related": "r1", "rel_type": "t", "direction": "outgoing"},
        {"related": "r2", "rel_type": "t", "direction": "outgoing"},
    ]
    vector.search.return_value = []
    keyword.search.return_value = []

    retriever = FusionRetriever(graph, vector, keyword, llm, embed_fn)
    results = retriever.retrieve("test", top_k=10)
    if len(results) >= 2:
        for i in range(len(results) - 1):
            assert results[i].score >= results[i + 1].score
