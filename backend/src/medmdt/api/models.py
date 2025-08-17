import uuid
from datetime import datetime, timezone
from enum import StrEnum

from pydantic import BaseModel, Field


class ConsultationStatus(StrEnum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"


class ConsultationRequest(BaseModel):
    patient_info: dict
    medical_records: list[dict] = Field(default_factory=list)
    max_rounds: int = Field(default=3, ge=1, le=10)


class ConsultationResponse(BaseModel):
    id: str
    status: str
    created_at: str


class ConsultationResult(BaseModel):
    id: str
    status: str
    final_report: str | None = None
    discussion_rounds: list[dict] = Field(default_factory=list)
    consensus: dict | None = None
    divergences: list[str] = Field(default_factory=list)


class KnowledgeSearchRequest(BaseModel):
    query: str
    top_k: int = 10


class SearchResponse(BaseModel):
    results: list[dict]
    total: int


class IngestResponse(BaseModel):
    job_id: str
    status: str
    message: str


class FolderResultModel(BaseModel):
    folder_name: str
    status: str
    files_processed: int = 0
    files_failed: int = 0
    message: str = ""


class BatchIngestResponse(BaseModel):
    job_id: str
    status: str
    message: str
    progress: dict = Field(default_factory=lambda: {"current": 0, "total": 0})
    folders: list[FolderResultModel] = Field(default_factory=list)


class ConsultationStore:
    def __init__(self):
        self._store: dict[str, dict] = {}

    def create(self, request: ConsultationRequest) -> str:
        cid = f"mdt-{uuid.uuid4().hex[:12]}"
        self._store[cid] = {
            "id": cid,
            "status": ConsultationStatus.PENDING,
            "patient_info": request.patient_info,
            "medical_records": request.medical_records,
            "max_rounds": request.max_rounds,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "final_report": None,
            "discussion_rounds": [],
            "consensus": None,
            "divergences": [],
        }
        return cid

    def get(self, consultation_id: str) -> dict | None:
        return self._store.get(consultation_id)

    def update(self, consultation_id: str, **fields) -> None:
        if consultation_id in self._store:
            self._store[consultation_id].update(fields)

    def list_all(self) -> list[dict]:
        return list(self._store.values())
