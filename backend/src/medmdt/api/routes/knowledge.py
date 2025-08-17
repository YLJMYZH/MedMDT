# src/medmdt/api/routes/knowledge.py
import logging
import uuid
import tempfile
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, HTTPException, UploadFile, File

from medmdt.api.models import (
    KnowledgeSearchRequest,
    SearchResponse,
    IngestResponse,
    BatchIngestResponse,
    FolderResultModel,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/knowledge", tags=["knowledge"])

MAX_UPLOAD_SIZE = 100 * 1024 * 1024  # 100 MB
MAX_BATCH_UPLOAD_SIZE = 500 * 1024 * 1024  # 500 MB

ARCHIVE_EXTENSIONS = {".zip", ".tar", ".gz", ".tgz", ".bz2", ".rar", ".7z"}

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


@router.post("/ingest/batch", response_model=BatchIngestResponse, status_code=202)
def ingest_batch(
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
):
    suffix = Path(file.filename).suffix.lower() if file.filename else ""
    name_lower = (file.filename or "").lower()
    is_archive = suffix in ARCHIVE_EXTENSIONS or name_lower.endswith((".tar.gz", ".tar.bz2"))
    if not is_archive:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的文件格式: {suffix}。请上传 zip/tar/tar.gz/rar/7z 压缩包",
        )

    content = file.file.read()
    if len(content) > MAX_BATCH_UPLOAD_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"文件过大: {len(content)} 字节，超过 {MAX_BATCH_UPLOAD_SIZE // (1024*1024)}MB 限制",
        )

    job_id = f"batch-{uuid.uuid4().hex[:12]}"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    tmp.write(content)
    tmp.close()

    _ingest_jobs[job_id] = {
        "job_id": job_id,
        "status": "queued",
        "message": f"压缩包 {file.filename} 已接收",
        "progress": {"current": 0, "total": 0},
        "folders": [],
    }

    background_tasks.add_task(_run_batch_ingest, job_id, tmp.name)

    return BatchIngestResponse(
        job_id=job_id,
        status="queued",
        message=f"压缩包 {file.filename} 已接收，开始处理",
        progress={"current": 0, "total": 0},
        folders=[],
    )


def _run_batch_ingest(job_id: str, file_path: str) -> None:
    from medmdt.api.deps import build_infrastructure
    from medmdt.extractor.agent import ExtractionAgent
    from medmdt.extractor.batch_ingest import process_archive

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

        def on_folder_done(idx, result):
            folders = _ingest_jobs[job_id]["folders"]
            folders.append({
                "folder_name": result.folder_name,
                "status": result.status,
                "files_processed": result.files_processed,
                "files_failed": result.files_failed,
                "message": result.message,
            })
            _ingest_jobs[job_id]["progress"]["current"] = idx + 1
            _ingest_jobs[job_id]["message"] = f"已处理 {idx + 1}/{_ingest_jobs[job_id]['progress']['total']} 个病人文件夹"

        # Pre-count folders by doing a quick extract peek
        results = process_archive(file_path, agent, on_folder_done=on_folder_done)

        # Update total before processing starts (set via on_folder_done progress)
        total = len(results)
        _ingest_jobs[job_id]["progress"]["total"] = total
        _ingest_jobs[job_id]["progress"]["current"] = total

        completed = sum(1 for r in results if r.status == "completed")
        skipped = sum(1 for r in results if r.status == "skipped")
        failed = sum(1 for r in results if r.status == "failed")

        _ingest_jobs[job_id].update(
            status="completed",
            message=f"批量处理完成: {completed} 个文件夹成功, {skipped} 个跳过, {failed} 个失败",
        )
    except Exception as e:
        logger.exception("Batch ingest job %s failed", job_id)
        _ingest_jobs[job_id].update(status="failed", message=str(e))
    finally:
        Path(file_path).unlink(missing_ok=True)


@router.get("/ingest/batch/{job_id}", response_model=BatchIngestResponse)
def get_batch_ingest_status(job_id: str):
    job = _ingest_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="Batch ingest job not found")
    return BatchIngestResponse(
        job_id=job["job_id"],
        status=job["status"],
        message=job.get("message", ""),
        progress=job.get("progress", {"current": 0, "total": 0}),
        folders=[FolderResultModel(**f) for f in job.get("folders", [])],
    )
