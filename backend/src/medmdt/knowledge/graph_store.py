# src/medmdt/knowledge/graph_store.py
"""Neo4j-backed knowledge graph store for medical entities and relations.

Uses MERGE patterns for idempotent upserts so the same data can be
ingested multiple times without creating duplicates.
"""

from __future__ import annotations

from neo4j import GraphDatabase

from medmdt.extractor.schemas import Entity, Relation


class GraphStore:
    """Thin wrapper around the Neo4j Python driver for medical KG operations."""

    def __init__(self, uri: str, user: str, password: str) -> None:
        self._driver = GraphDatabase.driver(uri, auth=(user, password))

    # ------------------------------------------------------------------
    # Writes
    # ------------------------------------------------------------------

    def upsert_entities(self, entities: list[Entity]) -> int:
        """Merge medical-entity nodes into the graph.

        Each entity is matched on *name*; its ``type`` and ``aliases``
        properties are always (re-)set, and a dynamic label equal to the
        capitalised type is added via ``apoc.create.addLabels``.

        Returns the number of entities processed.
        """
        if not entities:
            return 0

        total = 0
        with self._driver.session() as session:
            for entity in entities:
                session.run(
                    "MERGE (n:MedicalEntity {name: $name}) "
                    "SET n.type = $type, n.aliases = $aliases "
                    "WITH n "
                    "CALL apoc.create.addLabels(n, [$label]) YIELD node "
                    "RETURN node",
                    name=entity.name,
                    type=entity.type,
                    aliases=entity.aliases,
                    label=entity.type.capitalize(),
                )
                total += 1
        return total

    def upsert_relations(self, relations: list[Relation]) -> int:
        """Merge relationships between medical entities.

        Both head and tail nodes are merged first (ensuring they exist),
        then the ``RELATED`` edge keyed on ``type`` is merged and its
        metadata properties updated.

        Returns the number of relations processed.
        """
        if not relations:
            return 0

        total = 0
        with self._driver.session() as session:
            for rel in relations:
                session.run(
                    "MERGE (h:MedicalEntity {name: $head}) "
                    "MERGE (t:MedicalEntity {name: $tail}) "
                    "MERGE (h)-[r:RELATED {type: $rel_type}]->(t) "
                    "SET r.evidence = $evidence, r.confidence = $confidence",
                    head=rel.head,
                    tail=rel.tail,
                    rel_type=rel.relation,
                    evidence=rel.evidence,
                    confidence=rel.confidence,
                )
                total += 1
        return total

    # ------------------------------------------------------------------
    # Reads
    # ------------------------------------------------------------------

    def query_by_entity(self, name: str) -> list[dict]:
        """Return all entities related to *name* with their relationship type
        and direction (``"outgoing"`` or ``"incoming"``).
        """
        with self._driver.session() as session:
            result = session.run(
                "MATCH (n:MedicalEntity {name: $name})-[r]-(m:MedicalEntity) "
                "RETURN m.name AS related, r.type AS rel_type, "
                "CASE WHEN startNode(r) = n THEN 'outgoing' ELSE 'incoming' END AS direction",
                name=name,
            )
            return result.data()

    def query_differential(self, symptoms: list[str]) -> list[dict]:
        """Given a list of symptom names, find diseases connected via an
        ``indicates`` relation and rank them by the number of matching
        symptoms (descending).
        """
        with self._driver.session() as session:
            result = session.run(
                "MATCH (s:MedicalEntity)-[r:RELATED {type: 'indicates'}]->"
                "(d:MedicalEntity {type: 'disease'}) "
                "WHERE s.name IN $symptoms "
                "RETURN d.name AS disease, count(s) AS match_count "
                "ORDER BY match_count DESC",
                symptoms=symptoms,
            )
            return result.data()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Shut down the underlying Neo4j driver."""
        self._driver.close()
