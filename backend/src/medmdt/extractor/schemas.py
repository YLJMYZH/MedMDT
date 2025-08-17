# src/medmdt/extractor/schemas.py
from pydantic import BaseModel, Field, field_validator


class SourceInfo(BaseModel):
    file: str
    type: str
    page: int | None = None


ENTITY_TYPES = {"disease", "symptom", "drug", "procedure", "anatomy", "lab_test", "other"}


class Entity(BaseModel):
    name: str
    type: str
    aliases: list[str] = Field(default_factory=list)

    @field_validator("type")
    @classmethod
    def validate_type(cls, v: str) -> str:
        if v not in ENTITY_TYPES:
            raise ValueError(f"Invalid entity type: {v}. Must be one of {ENTITY_TYPES}")
        return v


class Relation(BaseModel):
    head: str
    relation: str
    tail: str
    evidence: str
    confidence: float = Field(ge=0.0, le=1.0)


class TextChunk(BaseModel):
    text: str
    summary: str
    keywords: list[str] = Field(default_factory=list)
    metadata: dict = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    source: SourceInfo
    entities: list[Entity] = Field(default_factory=list)
    relations: list[Relation] = Field(default_factory=list)
    chunks: list[TextChunk] = Field(default_factory=list)


class ParsedPage(BaseModel):
    page_num: int
    markdown: str
    images: list[bytes] = Field(default_factory=list)
