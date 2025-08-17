# src/medmdt/knowledge/schemas.py
"""Shared retrieval schemas used across knowledge stores and the fusion retriever."""

from pydantic import BaseModel, Field


class RetrievalResult(BaseModel):
    """A unified result returned by the FusionRetriever after RRF fusion.

    Attributes:
        text: The retrieved text content.
        score: The fused RRF score.
        source: Origin store identifier (``"graph"``, ``"vector"``, ``"keyword"``,
                or ``"fused"`` when the source cannot be determined).
        metadata: Arbitrary key-value metadata from the originating store.
    """

    text: str
    score: float
    source: str
    metadata: dict = Field(default_factory=dict)
