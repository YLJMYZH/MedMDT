# tests/test_keyword_store.py
"""Tests for Elasticsearch keyword store with BM25 search."""

from unittest.mock import MagicMock, patch

import pytest

from medmdt.knowledge.keyword_store import KeywordStore, KeywordSearchResult
from medmdt.extractor.schemas import TextChunk, SourceInfo


def test_keyword_search_result():
    r = KeywordSearchResult(
        text="test",
        score=5.0,
        keywords=["k1"],
        metadata={"section": "治疗"},
        source={"file": "a.pdf"},
    )
    assert r.score == 5.0
    assert r.text == "test"
    assert r.keywords == ["k1"]
    assert r.metadata == {"section": "治疗"}
    assert r.source == {"file": "a.pdf"}


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_keyword_store_init(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    assert store._es is mock_es
    mock_es_cls.assert_called_once_with("http://localhost:9200")


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_ensure_index_creates_if_missing(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.indices.exists.return_value = False

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.ensure_index()

    mock_es.indices.exists.assert_called_once_with(index="medmdt_chunks")
    mock_es.indices.create.assert_called_once()
    call_kwargs = mock_es.indices.create.call_args
    assert call_kwargs[1]["index"] == "medmdt_chunks"


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_ensure_index_skips_if_exists(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.indices.exists.return_value = True

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.ensure_index()

    mock_es.indices.exists.assert_called_once_with(index="medmdt_chunks")
    mock_es.indices.create.assert_not_called()


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_ensure_index_falls_back_on_ik_failure(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.indices.exists.return_value = False
    # First create call fails (IK analyzer not available), second succeeds
    mock_es.indices.create.side_effect = [Exception("IK not found"), None]

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.ensure_index()

    assert mock_es.indices.create.call_count == 2


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_index_chunks(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    chunks = [
        TextChunk(
            text="糖尿病治疗",
            summary="治疗方案",
            keywords=["糖尿病"],
            metadata={"section": "治疗"},
        ),
        TextChunk(
            text="高血压治疗",
            summary="治疗方案",
            keywords=["高血压"],
            metadata={"section": "治疗"},
        ),
    ]
    source = SourceInfo(file="guide.pdf", type="clinical_guideline", page=1)
    count = store.index_chunks(chunks, source)

    assert count == 2
    assert mock_es.index.call_count == 2

    # Verify the document structure of the first call
    first_call = mock_es.index.call_args_list[0]
    assert first_call[1]["index"] == "medmdt_chunks"
    doc = first_call[1]["body"]
    assert doc["text"] == "糖尿病治疗"
    assert doc["summary"] == "治疗方案"
    assert doc["keywords"] == ["糖尿病"]
    assert doc["source"]["file"] == "guide.pdf"
    assert doc["source"]["type"] == "clinical_guideline"


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_search(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.search.return_value = {
        "hits": {
            "hits": [
                {
                    "_score": 5.5,
                    "_source": {
                        "text": "糖尿病诊断标准",
                        "keywords": ["糖尿病", "诊断"],
                        "metadata": {"section": "诊断"},
                        "source": {
                            "file": "guide.pdf",
                            "type": "clinical_guideline",
                        },
                    },
                }
            ]
        }
    }

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    results = store.search("糖尿病诊断", top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], KeywordSearchResult)
    assert results[0].score == 5.5
    assert "糖尿病" in results[0].keywords
    assert results[0].source["file"] == "guide.pdf"

    # Verify multi_match query was used
    search_body = mock_es.search.call_args[1]["body"]
    assert "multi_match" in str(search_body)


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_search_with_filters(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.search.return_value = {"hits": {"hits": []}}

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.search("test", filters={"source.type": "clinical_guideline"})

    search_body = mock_es.search.call_args[1]["body"]
    assert "bool" in search_body["query"]
    assert "filter" in search_body["query"]["bool"]
    assert "must" in search_body["query"]["bool"]


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_search_empty_results(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es
    mock_es.search.return_value = {"hits": {"hits": []}}

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    results = store.search("nonexistent query")

    assert results == []


@patch("medmdt.knowledge.keyword_store.Elasticsearch")
def test_close(mock_es_cls):
    mock_es = MagicMock()
    mock_es_cls.return_value = mock_es

    store = KeywordStore("http://localhost:9200", "medmdt_chunks")
    store.close()

    mock_es.close.assert_called_once()
