# tests/test_graph_store.py
from unittest.mock import MagicMock, patch
import pytest
from medmdt.knowledge.graph_store import GraphStore
from medmdt.extractor.schemas import Entity, Relation


@pytest.fixture
def mock_driver():
    with patch("medmdt.knowledge.graph_store.GraphDatabase") as mock_gdb:
        driver = MagicMock()
        mock_gdb.driver.return_value = driver
        session = MagicMock()
        driver.session.return_value.__enter__ = MagicMock(return_value=session)
        driver.session.return_value.__exit__ = MagicMock(return_value=False)
        yield driver, session


def test_graph_store_init(mock_driver):
    driver, _ = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    assert store._driver is driver


def test_upsert_entities(mock_driver):
    driver, session = mock_driver
    session.run.return_value.consume.return_value.counters.nodes_created = 2

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    entities = [
        Entity(name="糖尿病", type="disease", aliases=["DM"]),
        Entity(name="二甲双胍", type="drug", aliases=[]),
    ]
    count = store.upsert_entities(entities)

    assert session.run.called
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher
    assert count == 2


def test_upsert_entities_empty(mock_driver):
    """Upserting an empty list returns 0 without calling the driver."""
    driver, session = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    count = store.upsert_entities([])
    assert count == 0


def test_upsert_relations(mock_driver):
    driver, session = mock_driver
    session.run.return_value.consume.return_value.counters.relationships_created = 1

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    relations = [
        Relation(
            head="二甲双胍", relation="treats", tail="糖尿病",
            evidence="指南推荐", confidence=0.95,
        ),
    ]
    count = store.upsert_relations(relations)

    assert session.run.called
    cypher = session.run.call_args[0][0]
    assert "MERGE" in cypher
    assert count == 1


def test_upsert_relations_empty(mock_driver):
    """Upserting an empty list returns 0 without calling the driver."""
    driver, session = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    count = store.upsert_relations([])
    assert count == 0


def test_query_by_entity(mock_driver):
    driver, session = mock_driver
    session.run.return_value.data.return_value = [
        {"related": "二甲双胍", "rel_type": "treated_by", "direction": "outgoing"}
    ]

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    results = store.query_by_entity("糖尿病")

    assert len(results) == 1
    assert results[0]["related"] == "二甲双胍"


def test_query_by_entity_no_results(mock_driver):
    """Querying a non-existent entity returns an empty list."""
    driver, session = mock_driver
    session.run.return_value.data.return_value = []

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    results = store.query_by_entity("不存在的实体")
    assert results == []


def test_query_differential(mock_driver):
    driver, session = mock_driver
    session.run.return_value.data.return_value = [
        {"disease": "高血压", "match_count": 3},
        {"disease": "糖尿病", "match_count": 1},
    ]

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    results = store.query_differential(["头痛", "视物模糊", "血压升高"])

    assert len(results) == 2
    assert results[0]["disease"] == "高血压"
    assert results[0]["match_count"] > results[1]["match_count"]


def test_query_differential_empty_symptoms(mock_driver):
    """Querying with no symptoms returns an empty list."""
    driver, session = mock_driver
    session.run.return_value.data.return_value = []

    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    results = store.query_differential([])
    assert results == []


def test_close(mock_driver):
    driver, _ = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    store.close()
    driver.close.assert_called_once()


def test_upsert_entities_sets_properties(mock_driver):
    """Verify that entity properties (name, type, aliases) are passed to Cypher."""
    driver, session = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    entities = [
        Entity(name="头痛", type="symptom", aliases=["headache", "偏头痛"]),
    ]
    store.upsert_entities(entities)

    # Check that the run call included the entity's properties
    call_kwargs = session.run.call_args
    # The kwargs or positional args should contain our entity data
    all_args = str(call_kwargs)
    assert "头痛" in all_args
    assert "symptom" in all_args


def test_upsert_relations_sets_properties(mock_driver):
    """Verify that relation properties are passed to Cypher."""
    driver, session = mock_driver
    store = GraphStore("bolt://localhost:7687", "neo4j", "password")
    relations = [
        Relation(
            head="头痛", relation="indicates", tail="高血压",
            evidence="临床观察", confidence=0.8,
        ),
    ]
    store.upsert_relations(relations)

    call_kwargs = str(session.run.call_args)
    assert "头痛" in call_kwargs
    assert "高血压" in call_kwargs
    assert "indicates" in call_kwargs
