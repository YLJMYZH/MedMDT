# src/medmdt/api/routes/knowledge.py
import logging
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from medmdt.api.models import KnowledgeSearchRequest, SearchResponse, IngestResponse

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB

_ingest_jobs: dict[str, dict] = {}


def _get_retriever():
    from medmdt.api.deps import build_infrastructure

    infra = build_infrastructure()
    return infra["retriever"]


def _run_ingest(job_id: str, file_path: str) -> None:
    from medmdt.api.deps import build_infrastructure
    from medmdt.extractor.agent import ExtractionAgent

    try:
        _ingest_jobs[job_id]["status"] = "processing"
        infra = build_infrastructure()

        infra["vector_store"].ensure_collection()
        infra["keyword_store"].ensure_index()

        agent = ExtractionAgent(
            settings=infra["settings"],
            graph_store=infra["graph_store"],
            vector_store=infra["vector_store"],
            keyword_store=infra["keyword_store"],
            embed_fn=infra["embed_fn"],
            llm=infra["llm"],
        )
        reports = agent.process_file(file_path)
        total_entities = sum(r.entities_count for r in reports)
        total_chunks = sum(r.chunks_count for r in reports)
        _ingest_jobs[job_id].update(
            status="completed",
            message=f"提取完成: {total_entities} 实体, {total_chunks} 文本块",
        )
    except Exception as e:
        logger.exception("Ingest job %s failed", job_id)
        _ingest_jobs[job_id].update(status="failed", message=str(e))
    finally:
        Path(file_path).unlink(missing_ok=True)


@router.post("/search", response_model=SearchResponse)
def search_knowledge(request: KnowledgeSearchRequest):
    retriever = _get_retriever()
    results = retriever.retrieve(request.query, top_k=request.top_k)
    return SearchResponse(
        results=[
            {
                "text": r.text,
                "score": r.score,
                "source": r.source,
                "metadata": r.metadata if hasattr(r, "metadata") else {},
            }
            for r in results
        ],
        total=len(results),
    )


@router.post("/ingest", response_model=IngestResponse, status_code=202)
def ingest_file(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    content = file.file.read()
    if len(content) > MAX_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large: {len(content)} bytes exceeds {MAX_UPLOAD_SIZE} byte limit",
        )

    job_id = f"ingest-{uuid.uuid4().hex[:12]}"
    suffix = Path(file.filename).suffix if file.filename else ".tmp"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()

    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": f"文件 {file.filename} 已接收",
        "filename": file.filename,
    }

    background_tasks.add_task(_run_ingest, job_id, tmp.name)

    return IngestResponse(
        job_id=job_id,
        status="queued",
        message=f"文件 {file.filename} 已接收，开始处理",
    )


@router.get("/ingest/{job_id}", response_model=IngestResponse)
def get_ingest_status(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Ingest job not found")
    return IngestResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job.get("message", ""),
    )
