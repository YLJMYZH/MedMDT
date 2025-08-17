# src/medmdt/extractor/parsers/dicom_parser.py
"""DICOM parser — pydicom metadata extraction with multimodal LLM image analysis."""

import io
import logging

import numpy as np
from PIL import Image
from pydicom import dcmread
from pydantic import BaseModel, Field

from medmdt.extractor.parsers.image_parser import ImageParser, ImageAnalysisResult

logger = logging.getLogger(__name__)


class DicomMetadata(BaseModel):
    patient_id: str = ""
    patient_name: str = ""
    patient_age: str | None = None
    patient_sex: str | None = None
    study_date: str | None = None
    modality: str = ""
    study_description: str | None = None
    series_description: str | None = None
    institution: str | None = None
    manufacturer: str | None = None
    pixel_spacing: list[float] | None = None
    slice_thickness: float | None = None


class DicomParseResult(BaseModel):
    metadata: DicomMetadata
    image_analysis: ImageAnalysisResult | None = None
    raw_text: str = ""


class DicomParser:
    def __init__(self, image_parser: ImageParser):
        self._image_parser = image_parser

    def parse(self, file_path: str) -> DicomParseResult:
        ds = dcmread(file_path)
        metadata = self._extract_metadata(ds)
        image_analysis = self._analyze_pixels(ds, metadata)
        raw_text = self._build_raw_text(metadata, image_analysis)

        return DicomParseResult(
            metadata=metadata,
            image_analysis=image_analysis,
            raw_text=raw_text,
        )

    def _extract_metadata(self, ds) -> DicomMetadata:
        def safe_get(attr, default=""):
            try:
                val = getattr(ds, attr, default)
                return str(val) if val is not None else default
            except Exception:
                return default

        pixel_spacing = None
        try:
            ps = getattr(ds, "PixelSpacing", None)
            if ps is not None:
                pixel_spacing = [float(ps[0]), float(ps[1])]
        except (TypeError, IndexError):
            pass

        slice_thickness = None
        try:
            st = getattr(ds, "SliceThickness", None)
            if st is not None:
                slice_thickness = float(st)
        except (TypeError, ValueError):
            pass

        return DicomMetadata(
            patient_id=safe_get("PatientID"),
            patient_name=safe_get("PatientName"),
            patient_age=safe_get("PatientAge") or None,
            patient_sex=safe_get("PatientSex") or None,
            study_date=safe_get("StudyDate") or None,
            modality=safe_get("Modality"),
            study_description=safe_get("StudyDescription") or None,
            series_description=safe_get("SeriesDescription") or None,
            institution=safe_get("InstitutionName") or None,
            manufacturer=safe_get("Manufacturer") or None,
            pixel_spacing=pixel_spacing,
            slice_thickness=slice_thickness,
        )

    def _analyze_pixels(self, ds, metadata: DicomMetadata) -> ImageAnalysisResult | None:
        try:
            pixel_array = ds.pixel_array
        except (AttributeError, TypeError):
            logger.info("No pixel data in DICOM file")
            return None

        image_bytes = self._pixels_to_png(pixel_array)
        context = f"Modality: {metadata.modality}"
        if metadata.study_description:
            context += f", Study: {metadata.study_description}"
        if metadata.series_description:
            context += f", Series: {metadata.series_description}"

        return self._image_parser.analyze(image_bytes, context=context)

    @staticmethod
    def _pixels_to_png(pixel_array: np.ndarray) -> bytes:
        arr = pixel_array.astype(np.float64)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr = (arr - arr_min) / (arr_max - arr_min) * 255.0
        arr = arr.astype(np.uint8)
        image = Image.fromarray(arr)
        buf = io.BytesIO()
        image.save(buf, format="PNG")
        return buf.getvalue()

    @staticmethod
    def _build_raw_text(metadata: DicomMetadata, analysis: ImageAnalysisResult | None) -> str:
        lines = [
            f"患者ID: {metadata.patient_id}",
            f"患者姓名: {metadata.patient_name}",
        ]
        if metadata.patient_age:
            lines.append(f"年龄: {metadata.patient_age}")
        if metadata.patient_sex:
            lines.append(f"性别: {metadata.patient_sex}")
        if metadata.study_date:
            lines.append(f"检查日期: {metadata.study_date}")
        lines.append(f"检查类型: {metadata.modality}")
        if metadata.study_description:
            lines.append(f"检查描述: {metadata.study_description}")
        if metadata.institution:
            lines.append(f"机构: {metadata.institution}")
        if analysis:
            lines.append(f"\n影像分析: {analysis.description}")
            if analysis.findings:
                lines.append("发现: " + ", ".join(analysis.findings))
        return "\n".join(lines)
