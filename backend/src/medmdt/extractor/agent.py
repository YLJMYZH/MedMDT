# src/medmdt/extractor/agent.py
import json
from pathlib import Path
from typing import Callable

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage

from medmdt.config.settings import Settings
from medmdt.extractor.schemas import (
    ExtractionResult,
    SourceInfo,
    Entity,
    Relation,
    TextChunk,
    ParsedPage,
)
from medmdt.extractor.parsers.pdf_parser import PaddleOCRClient
from medmdt.extractor.parsers.image_parser import ImageParser
from medmdt.extractor.parsers.dicom_parser import DicomParser
from medmdt.extractor.file_types import DICOM_EXTENSIONS, IMAGE_EXTENSIONS
from medmdt.extractor.ingestor import Ingestor, IngestReport
from medmdt.knowledge.graph_store import GraphStore
from medmdt.knowledge.vector_store import VectorStore
from medmdt.knowledge.keyword_store import KeywordStore
from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.prompts.extraction.entity_extraction import ENTITY_RELATION_PROMPT
from medmdt.llm.prompts.extraction.relation_extraction import CHUNK_SUMMARY_PROMPT


class ExtractionAgent:
    """End-to-end extraction pipeline: file -> parse -> LLM extraction -> ingest.

    Orchestrates PaddleOCR (PDF), ImageParser (images), LLM-based entity/relation
    extraction, and triple-store ingestion via the Ingestor.
    """

    def __init__(
        self,
        settings: Settings,
        graph_store: GraphStore,
        vector_store: VectorStore,
        keyword_store: KeywordStore,
        embed_fn: Callable[[list[str]], list[list[float]]],
        llm: BaseChatModel,
        vision_llm: BaseChatModel | None,
        vision_error: str | None = None,
    ) -> None:
        self._settings = settings
        self._llm = llm
        self._vision_error = vision_error
        self._image_parser = ImageParser(llm=vision_llm) if vision_llm is not None else None
        self._dicom_parser = (
            DicomParser(image_parser=self._image_parser)
            if self._image_parser is not None
            else None
        )
        self._ingestor = Ingestor(graph_store, vector_store, keyword_store, embed_fn)
        self._ocr_client = PaddleOCRClient(settings)

    def _require_image_parser(self) -> ImageParser:
        if self._image_parser is None:
            raise VisionProviderNotSupported(
                self._vision_error or "图片分析未配置可用的视觉模型"
            )
        return self._image_parser

    def process_file(self, file_path: str) -> list[IngestReport]:
        """Process a file end-to-end and return ingest reports.

        Detects file type, parses content, extracts entities/relations/chunks
        via LLM, and writes everything to the backing stores.
        """
        file_type = self._detect_file_type(file_path)

        if file_type == "pdf":
            return self._process_pdf(file_path)
        elif file_type == "image":
            return [self._process_image(file_path)]
        elif file_type == "dicom":
            return [self._process_dicom(file_path)]
        else:
            raise ValueError(f"Unsupported file type: {file_path}")

    def _process_pdf(self, file_path: str) -> list[IngestReport]:
        """Parse PDF via PaddleOCR, analyze embedded images, extract from text."""
        pages = self._ocr_client.parse(file_path)
        reports: list[IngestReport] = []

        for page in pages:
            image_descriptions: list[str] = []
            if page.images:
                for img_bytes in page.images:
                    analysis = self._require_image_parser().analyze(img_bytes)
                    image_descriptions.append(analysis.description)

            full_text = page.markdown
            if image_descriptions:
                full_text += "\n\n[嵌入图像分析]\n" + "\n".join(image_descriptions)

            result = self._extract_from_text(
                text=full_text,
                source=SourceInfo(
                    file=file_path,
                    type=self._classify_document(full_text),
                    page=page.page_num,
                ),
            )
            report = self._ingestor.ingest(result)
            reports.append(report)

        return reports

    def _process_image(self, file_path: str) -> IngestReport:
        """Analyze a standalone image file and ingest the result."""
        image_parser = self._require_image_parser()
        with open(file_path, "rb") as f:
            image_data = f.read()

        analysis = image_parser.analyze(image_data)
        result = ExtractionResult(
            source=SourceInfo(file=file_path, type="image"),
            entities=[],
            relations=[],
            chunks=[
                TextChunk(
                    text=analysis.description,
                    summary=analysis.description,
                    keywords=analysis.findings,
                    metadata={"modality": analysis.modality or "unknown"},
                )
            ],
        )
        return self._ingestor.ingest(result)

    def _process_dicom(self, file_path: str) -> IngestReport:
        """Parse DICOM file and ingest metadata + image analysis."""
        dicom_parser = self._dicom_parser or DicomParser(
            image_parser=self._require_image_parser()
        )
        dicom_result = dicom_parser.parse(file_path)
        result = self._extract_from_text(
            text=dicom_result.raw_text,
            source=SourceInfo(
                file=file_path,
                type="dicom",
                page=None,
            ),
        )
        return self._ingestor.ingest(result)

    def _extract_from_text(
        self, text: str, source: SourceInfo
    ) -> ExtractionResult:
        """Call LLM twice (entity extraction + chunk summarization) and parse JSON."""
        # Step 1: Entity & relation extraction
        entity_prompt = ENTITY_RELATION_PROMPT.format(text=text)
        entity_response = self._llm.invoke([HumanMessage(content=entity_prompt)])
        entity_data = json.loads(entity_response.content)

        # Step 2: Chunk summarization
        chunk_prompt = CHUNK_SUMMARY_PROMPT.format(text=text)
        chunk_response = self._llm.invoke([HumanMessage(content=chunk_prompt)])
        chunk_data = json.loads(chunk_response.content)

        entities = [Entity.model_validate(e) for e in entity_data.get("entities", [])]
        relations = [
            Relation.model_validate(r) for r in entity_data.get("relations", [])
        ]
        chunks = [TextChunk.model_validate(c) for c in chunk_data.get("chunks", [])]

        return ExtractionResult(
            source=source,
            entities=entities,
            relations=relations,
            chunks=chunks,
        )

    def _classify_document(self, text: str) -> str:
        """Simple keyword-based document type classification."""
        text_lower = text[:500].lower()
        if any(kw in text_lower for kw in ["指南", "guideline", "共识", "consensus"]):
            return "clinical_guideline"
        if any(kw in text_lower for kw in ["病例", "case", "入院", "出院"]):
            return "case_report"
        if any(kw in text_lower for kw in ["教材", "textbook", "章", "chapter"]):
            return "textbook"
        return "other"

    @staticmethod
    def _detect_file_type(file_path: str) -> str:
        """Detect file type from extension."""
        suffix = Path(file_path).suffix.lower()
        if suffix == ".pdf":
            return "pdf"
        elif suffix in DICOM_EXTENSIONS:
            return "dicom"
        elif suffix in IMAGE_EXTENSIONS:
            return "image"
        else:
            raise ValueError(f"Unknown file extension: {suffix}")
