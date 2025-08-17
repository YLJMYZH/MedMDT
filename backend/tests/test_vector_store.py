# tests/test_vector_store.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.vector_store import VectorStore, VectorSearchResult


def test_vector_search_result():
    r = VectorSearchResult(text="test", score=0.95, metadata={"domain": "guideline"})
    assert r.score == 0.95
    assert r.text == "test"
    assert r.metadata == {"domain": "guideline"}


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_vector_store_ensure_collection_creates_if_missing(
    mock_util, mock_coll_cls, mock_conn
):
    mock_util.has_collection.return_value = False
    store = VectorStore("localhost", 19530, "test_collection", 1024)
    store.ensure_collection()
    mock_util.has_collection.assert_called_with("test_collection")
    # Should have created a new Collection with schema
    mock_coll_cls.assert_called_once()


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_vector_store_ensure_collection_loads_existing(
    mock_util, mock_coll_cls, mock_conn
):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll

    store = VectorStore("localhost", 19530, "test_collection", 1024)
    store.ensure_collection()

    mock_util.has_collection.assert_called_with("test_collection")
    mock_coll_cls.assert_called_once_with("test_collection")
    mock_coll.load.assert_called_once()


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_insert(mock_util, mock_coll_cls, mock_conn):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll
    mock_coll.insert.return_value = MagicMock(primary_keys=[1, 2])

    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.ensure_collection()
    ids = store.insert(
        texts=["text1", "text2"],
        embeddings=[[0.1, 0.2, 0.3, 0.4], [0.5, 0.6, 0.7, 0.8]],
        metadatas=[{"domain": "guideline"}, {"domain": "case"}],
    )

    assert ids == [1, 2]
    mock_coll.insert.assert_called_once()


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_search(mock_util, mock_coll_cls, mock_conn):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll

    hit = MagicMock()
    hit.distance = 0.95
    hit.entity.get.side_effect = lambda k: {
        "text": "结果文本",
        "metadata": '{"domain": "guideline"}',
    }.get(k)
    mock_coll.search.return_value = [[hit]]

    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.ensure_collection()
    results = store.search([0.1, 0.2, 0.3, 0.4], top_k=5)

    assert len(results) == 1
    assert isinstance(results[0], VectorSearchResult)
    assert results[0].score == 0.95
    assert results[0].text == "结果文本"
    assert results[0].metadata == {"domain": "guideline"}


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_search_with_domain_filter(mock_util, mock_coll_cls, mock_conn):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll

    hit = MagicMock()
    hit.distance = 0.88
    hit.entity.get.side_effect = lambda k: {
        "text": "domain filtered result",
        "metadata": '{"domain": "guideline"}',
    }.get(k)
    mock_coll.search.return_value = [[hit]]

    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.ensure_collection()
    results = store.search([0.1, 0.2, 0.3, 0.4], top_k=5, domain="guideline")

    assert len(results) == 1
    # Verify that an expr was passed to search for domain filtering
    call_kwargs = mock_coll.search.call_args
    assert call_kwargs.kwargs.get("expr") is not None or (
        len(call_kwargs.args) > 4 and call_kwargs.args[4] is not None
    )


@patch("medmdt.knowledge.vector_store.connections")
@patch("medmdt.knowledge.vector_store.Collection")
@patch("medmdt.knowledge.vector_store.utility")
def test_close(mock_util, mock_coll_cls, mock_conn):
    mock_util.has_collection.return_value = True
    mock_coll = MagicMock()
    mock_coll_cls.return_value = mock_coll

    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.ensure_collection()
    store.close()

    mock_coll.release.assert_called_once()
    mock_conn.disconnect.assert_called_once_with("default")


@patch("medmdt.knowledge.vector_store.connections")
def test_close_without_collection(mock_conn):
    store = VectorStore("localhost", 19530, "test_collection", 4)
    store.close()  # Should not raise even without ensure_collection
    mock_conn.disconnect.assert_called_once_with("default")
