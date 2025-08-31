import json
import logging
import tempfile
import uuid
from pathlib import Path

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException, UploadFile, File

from medmdt.api.models import (
    ConsultationRequest,
    ConsultationResponse,
    ConsultationResult,
    ConsultationStatus,
    ConsultationStore,
)
from medmdt.api.deps import get_consultation_store
from medmdt.llm.errors import VisionError, VisionProviderNotSupported

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/consultation", tags=["consultation"])


def run_consultation_task(consultation_id: str, store: ConsultationStore) -> None:
    from medmdt.mdt.experts.factory import create_all_experts
    from medmdt.mdt.moderator import Moderator
    from medmdt.mdt.streaming import StreamingOrchestrator
    from medmdt.api.routes.ws import get_event_bus
    from medmdt.api.deps import build_infrastructure

    entry = store.get(consultation_id)
    if not entry:
        return

    bus = get_event_bus()
    store.update(consultation_id, status=ConsultationStatus.RUNNING)
    bus.publish(consultation_id, {"type": "status", "status": "running"})

    try:
        infra = build_infrastructure()
        settings = infra["settings"]
        llm = infra["llm"]
        retriever = infra["retriever"]

        experts = create_all_experts("config/experts.yaml")
        moderator = Moderator(llm=llm, consensus_threshold=settings.mdt_consensus_threshold)

        orchestrator = StreamingOrchestrator(
            experts=experts,
            moderator=moderator,
            retriever=retriever,
            selector_llm=llm,
            event_bus=bus,
            consultation_id=consultation_id,
        )
        result = orchestrator.run(
            patient_info=entry["patient_info"],
            medical_records=entry["medical_records"],
            max_rounds=entry["max_rounds"],
        )

        store.update(
            consultation_id,
            status=ConsultationStatus.COMPLETED,
            final_report=result["final_report"],
            discussion_rounds=result["discussion_rounds"],
            consensus=result["consensus"],
            divergences=result["divergences"],
        )
        bus.publish(consultation_id, {"type": "done", "status": "completed"})
    except Exception as e:
        logger.exception("Consultation %s failed", consultation_id)
        store.update(consultation_id, status=ConsultationStatus.FAILED)
        bus.publish(consultation_id, {"type": "error", "message": str(e)})
        bus.publish(consultation_id, {"type": "done", "status": "failed"})


@router.post("", status_code=201, response_model=ConsultationResponse)
def create_consultation(
    request: ConsultationRequest,
    background_tasks: BackgroundTasks,
    store: ConsultationStore = Depends(get_consultation_store),
):
    cid = store.create(request)
    entry = store.get(cid)
    background_tasks.add_task(run_consultation_task, cid, store)
    return ConsultationResponse(
        id=cid, status=entry["status"], created_at=entry["created_at"],
    )


@router.get("", response_model=list[ConsultationResult])
def list_consultations(
    store: ConsultationStore = Depends(get_consultation_store),
):
    items = store.list_all()
    return [
        ConsultationResult(
            id=item["id"],
            status=item["status"],
            final_report=item.get("final_report"),
            discussion_rounds=item.get("discussion_rounds", []),
            consensus=item.get("consensus"),
            divergences=item.get("divergences", []),
        )
        for item in items
    ]


@router.get("/{consultation_id}", response_model=ConsultationResult)
def get_consultation(
    consultation_id: str,
    store: ConsultationStore = Depends(get_consultation_store),
):
    entry = store.get(consultation_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Consultation not found")
    return ConsultationResult(
        id=entry["id"],
        status=entry["status"],
        final_report=entry.get("final_report"),
        discussion_rounds=entry.get("discussion_rounds", []),
        consensus=entry.get("consensus"),
        divergences=entry.get("divergences", []),
    )


SUPPORTED_EXTENSIONS = {".pdf", ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".dcm", ".dicom"}


@router.post("/upload")
async def upload_consultation_files(files: list[UploadFile] = File(...)):
    """Upload medical files and extract text content for use in consultation."""
    from medmdt.api.deps import build_infrastructure

    infra = build_infrastructure()
    settings = infra["settings"]
    llm = infra["vision_llm"]
    request_id = uuid.uuid4().hex[:12]

    records = []
    for file in files:
        suffix = Path(file.filename).suffix.lower() if file.filename else ""
        if suffix not in SUPPORTED_EXTENSIONS:
            records.append({
                "record_type": "unsupported",
                "content": "",
                "filename": file.filename or "unknown",
                "error": f"不支持的文件格式: {suffix}",
            })
            continue

        content = await file.read()
        tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
        tmp.write(content)
        tmp.close()

        try:
            text = _extract_text_from_file(tmp.name, suffix, settings, llm)
            record_type = _classify_file_type(suffix)
            records.append({
                "record_type": record_type,
                "content": text,
                "filename": file.filename or "unknown",
            })
        except Exception as exc:
            logger.warning(
                "Vision upload failed request_id=%s provider=%s model=%s error_type=%s",
                request_id,
                infra["vision_provider"],
                infra["vision_model"],
                type(exc).__name__,
            )
            records.append({
                "record_type": "error",
                "content": "",
                "filename": file.filename or "unknown",
                "error": str(exc),
            })
        finally:
            Path(tmp.name).unlink(missing_ok=True)

    return {"records": records}


def _extract_text_from_file(file_path: str, suffix: str, settings, llm) -> str:
    """Extract text content from a medical file."""
    from medmdt.extractor.parsers.pdf_parser import PaddleOCRClient
    from medmdt.extractor.parsers.image_parser import ImageParser
    from medmdt.extractor.parsers.dicom_parser import DicomParser

    if suffix == ".pdf":
        parser = PaddleOCRClient(settings)
        pages = parser.parse(file_path)
        return "\n\n".join(p.markdown for p in pages if p.markdown)

    elif suffix in (".jpg", ".jpeg", ".png", ".bmp", ".tiff"):
        if llm is None:
            raise VisionProviderNotSupported("图片分析未配置可用的视觉模型")
        try:
            parser = ImageParser(llm=llm)
            with open(file_path, "rb") as f:
                image_bytes = f.read()
            result = parser.analyze(image_bytes)
            return f"[影像分析]\n模态: {result.modality or '未知'}\n描述: {result.description}\n发现: {', '.join(result.findings)}"
        except VisionError:
            raise
        except Exception as exc:
            raise ValueError("图片分析失败") from exc

    elif suffix in (".dcm", ".dicom"):
        if llm is None:
            raise VisionProviderNotSupported("图片分析未配置可用的视觉模型")
        try:
            image_parser = ImageParser(llm=llm)
            parser = DicomParser(image_parser=image_parser)
            result = parser.parse(file_path)
            return result.raw_text
        except VisionError:
            raise
        except Exception as exc:
            raise ValueError("图片分析失败") from exc

    return ""


def _classify_file_type(suffix: str) -> str:
    if suffix == ".pdf":
        return "report"
    elif suffix in (".dcm", ".dicom"):
        return "dicom"
    else:
        return "image"


@router.post("/{consultation_id}/save-to-knowledge")
def save_consultation_to_knowledge(
    consultation_id: str,
    store: ConsultationStore = Depends(get_consultation_store),
):
    """Save a completed consultation case to the knowledge base."""
    from medmdt.api.deps import build_infrastructure
    from medmdt.extractor.agent import ExtractionAgent

    entry = store.get(consultation_id)
    if not entry:
        raise HTTPException(status_code=404, detail="Consultation not found")
    if entry["status"] != ConsultationStatus.COMPLETED:
        raise HTTPException(status_code=400, detail="只能将已完成的会诊记入知识库")

    document = _format_consultation_document(entry)

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
        vision_llm=infra["vision_llm"],
        vision_error=infra["vision_error"],
    )

    result = agent._extract_from_text(document, source=f"consultation:{consultation_id}")
    from medmdt.extractor.ingestor import Ingestor
    ingestor = Ingestor(
        graph_store=infra["graph_store"],
        vector_store=infra["vector_store"],
        keyword_store=infra["keyword_store"],
        embed_fn=infra["embed_fn"],
    )
    report = ingestor.ingest(result)

    return {
        "status": "ok",
        "message": f"已记入知识库: {report.entities_count} 实体, {report.chunks_count} 文本块",
        "entities_count": report.entities_count,
        "chunks_count": report.chunks_count,
    }


def _format_consultation_document(entry: dict) -> str:
    """Format a consultation entry as a structured text document for knowledge extraction."""
    parts = []
    parts.append("# 多专家会诊病例记录\n")

    patient = entry.get("patient_info", {})
    parts.append("## 患者信息")
    for k, v in patient.items():
        parts.append(f"- {k}: {v}")

    if entry.get("medical_records"):
        parts.append("\n## 病历资料")
        for rec in entry["medical_records"]:
            parts.append(f"### {rec.get('record_type', '其他')}")
            parts.append(rec.get("content", ""))

    if entry.get("final_report"):
        parts.append("\n## 会诊结论")
        parts.append(entry["final_report"])

    if entry.get("consensus"):
        parts.append("\n## 专家共识")
        parts.append(json.dumps(entry["consensus"], ensure_ascii=False, indent=2))

    if entry.get("divergences"):
        parts.append("\n## 分歧点")
        for d in entry["divergences"]:
            parts.append(f"- {d}")

    return "\n".join(parts)
