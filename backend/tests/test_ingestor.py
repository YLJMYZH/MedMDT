# tests/test_ingestor.py
from unittest.mock import MagicMock
import pytest
from medmdt.extractor.ingestor import Ingestor, IngestReport
from medmdt.extractor.schemas import (
    ExtractionResult, SourceInfo, Entity, Relation, TextChunk,
)


@pytest.fixture
def mock_stores():
    graph = MagicMock()
    vector = MagicMock()
    keyword = MagicMock()
    embed_fn = MagicMock(return_value=[[0.1] * 1024])
    return graph, vector, keyword, embed_fn


@pytest.fixture
def sample_result():
    return ExtractionResult(
        source=SourceInfo(file="guide.pdf", type="clinical_guideline", page=1),
        entities=[
            Entity(name="糖尿病", type="disease", aliases=["DM"]),
            Entity(name="二甲双胍", type="drug", aliases=[]),
        ],
        relations=[
            Relation(head="二甲双胍", relation="treats", tail="糖尿病", evidence="原文", confidence=0.9),
        ],
        chunks=[
            TextChunk(text="糖尿病治疗方案", summary="治疗", keywords=["糖尿病"], metadata={}),
        ],
    )


def test_ingest_report():
    r = IngestReport(entities_count=2, relations_count=1, chunks_count=3)
    assert r.entities_count == 2


def test_ingest_writes_to_all_stores(mock_stores, sample_result):
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    report = ingestor.ingest(sample_result)

    assert isinstance(report, IngestReport)
    graph.upsert_entities.assert_called_once_with(sample_result.entities)
    graph.upsert_relations.assert_called_once_with(sample_result.relations)
    vector.insert.assert_called_once()
    keyword.index_chunks.assert_called_once()


def test_ingest_embeds_chunk_texts(mock_stores, sample_result):
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    ingestor.ingest(sample_result)

    embed_fn.assert_called_once_with(["糖尿病治疗方案"])


def test_ingest_empty_result(mock_stores):
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 0
    graph.upsert_relations.return_value = 0

    result = ExtractionResult(
        source=SourceInfo(file="empty.pdf", type="other"),
        entities=[], relations=[], chunks=[],
    )
    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    report = ingestor.ingest(result)

    assert report.entities_count == 0
    assert report.chunks_count == 0
    vector.insert.assert_not_called()
    keyword.index_chunks.assert_not_called()


def test_ingest_report_counts_match(mock_stores, sample_result):
    """Verify the report counts reflect actual store return values."""
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    report = ingestor.ingest(sample_result)

    assert report.entities_count == 2
    assert report.relations_count == 1
    assert report.chunks_count == 1


def test_ingest_vector_metadata_includes_domain(mock_stores, sample_result):
    """Verify vector metadata includes domain from source type."""
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    ingestor.ingest(sample_result)

    # Check the metadatas argument passed to vector.insert
    call_args = vector.insert.call_args
    metadatas = call_args[0][2]  # third positional arg
    assert metadatas[0]["domain"] == "clinical_guideline"
    assert metadatas[0]["summary"] == "治疗"
    assert metadatas[0]["keywords"] == ["糖尿病"]


def test_ingest_keyword_receives_source(mock_stores, sample_result):
    """Verify keyword store receives chunks and source info."""
    graph, vector, keyword, embed_fn = mock_stores
    graph.upsert_entities.return_value = 2
    graph.upsert_relations.return_value = 1
    vector.insert.return_value = [1]
    keyword.index_chunks.return_value = 1

    ingestor = Ingestor(graph, vector, keyword, embed_fn)
    ingestor.ingest(sample_result)

    keyword.index_chunks.assert_called_once_with(sample_result.chunks, sample_result.source)
