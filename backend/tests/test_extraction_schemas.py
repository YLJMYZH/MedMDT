# tests/test_extraction_schemas.py
import pytest
from medmdt.extractor.schemas import (
    SourceInfo, Entity, Relation, TextChunk, ExtractionResult, ParsedPage,
)


def test_source_info():
    s = SourceInfo(file="test.pdf", type="clinical_guideline", page=1)
    assert s.file == "test.pdf"
    assert s.page == 1


def test_source_info_page_optional():
    s = SourceInfo(file="test.jpg", type="image")
    assert s.page is None


def test_entity():
    e = Entity(name="2型糖尿病", type="disease", aliases=["T2DM"])
    assert e.name == "2型糖尿病"
    assert "T2DM" in e.aliases


def test_entity_type_validated():
    valid_types = {"disease", "symptom", "drug", "procedure", "anatomy", "lab_test", "other"}
    for t in valid_types:
        Entity(name="test", type=t, aliases=[])
    with pytest.raises(Exception):
        Entity(name="test", type="invalid_type", aliases=[])


def test_relation():
    r = Relation(
        head="二甲双胍", relation="first_line_treatment_for",
        tail="2型糖尿病", evidence="指南原文...", confidence=0.95,
    )
    assert r.confidence == 0.95


def test_relation_confidence_range():
    with pytest.raises(Exception):
        Relation(head="a", relation="r", tail="b", evidence="e", confidence=1.5)


def test_text_chunk():
    c = TextChunk(
        text="原始文本", summary="摘要",
        keywords=["关键词"], metadata={"section": "治疗"},
    )
    assert c.keywords == ["关键词"]


def test_extraction_result():
    result = ExtractionResult(
        source=SourceInfo(file="test.pdf", type="clinical_guideline"),
        entities=[Entity(name="糖尿病", type="disease", aliases=[])],
        relations=[],
        chunks=[TextChunk(text="text", summary="sum", keywords=[], metadata={})],
    )
    assert len(result.entities) == 1
    assert len(result.chunks) == 1


def test_extraction_result_to_dict_roundtrip():
    result = ExtractionResult(
        source=SourceInfo(file="test.pdf", type="clinical_guideline"),
        entities=[Entity(name="糖尿病", type="disease", aliases=["DM"])],
        relations=[Relation(head="a", relation="r", tail="b", evidence="e", confidence=0.9)],
        chunks=[TextChunk(text="t", summary="s", keywords=["k"], metadata={"x": 1})],
    )
    d = result.model_dump()
    restored = ExtractionResult.model_validate(d)
    assert restored == result


def test_parsed_page():
    p = ParsedPage(page_num=0, markdown="# Title", images=[b"fake-png"])
    assert p.page_num == 0
    assert len(p.images) == 1
