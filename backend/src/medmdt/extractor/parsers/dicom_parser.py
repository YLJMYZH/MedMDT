# src/medmdt/extractor/parsers/dicom_parser.py
"""DICOM parser — pydicom metadata extraction with multimodal LLM image analysis."""

import io
import logging

import numpy as np
from PIL import Image
from pydicom import dcmread
from pydicom.dataset import Dataset
from pydantic import BaseModel, Field

from medmdt.extractor.parsers.image_parser import ImageParser, ImageAnalysisResult
from medmdt.llm.errors import InvalidImageError

logger = logging.getLogger(__name__)

DEFAULT_MAX_DECODED_SAMPLES = 40_000_000
DEFAULT_MAX_DECODED_BYTES = 160 * 1024 * 1024


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
    def __init__(
        self,
        image_parser: ImageParser,
        max_decoded_samples: int = DEFAULT_MAX_DECODED_SAMPLES,
        max_decoded_bytes: int = DEFAULT_MAX_DECODED_BYTES,
    ):
        self._image_parser = image_parser
        self._max_decoded_samples = max_decoded_samples
        self._max_decoded_bytes = max_decoded_bytes

    def parse(self, file_path: str) -> DicomParseResult:
        read_error = False
        try:
            ds = dcmread(file_path)
        except Exception:
            read_error = True
        if read_error:
            raise InvalidImageError("DICOM 文件无法安全解析")
        metadata_error = False
        try:
            metadata = self._extract_metadata(ds)
        except Exception:
            metadata_error = True
        if metadata_error:
            raise InvalidImageError("DICOM 元数据无法安全解析")
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
        if isinstance(ds, Dataset) and not any(
            keyword in ds
            for keyword in ("PixelData", "FloatPixelData", "DoubleFloatPixelData")
        ):
            logger.info("No pixel data in DICOM file")
            return None

        invalid_budget = False
        try:
            rows = int(ds.Rows)
            columns = int(ds.Columns)
            samples_per_pixel = int(getattr(ds, "SamplesPerPixel", 1) or 1)
            number_of_frames = int(getattr(ds, "NumberOfFrames", 1) or 1)
            bits_allocated = int(ds.BitsAllocated)
            expected_samples = rows * columns * samples_per_pixel * number_of_frames
            expected_bytes = expected_samples * ((bits_allocated + 7) // 8)
            invalid_budget = (
                rows <= 0
                or columns <= 0
                or samples_per_pixel <= 0
                or bits_allocated <= 0
                or number_of_frames != 1
                or expected_samples > self._max_decoded_samples
                or expected_bytes > self._max_decoded_bytes
            )
        except Exception:
            invalid_budget = True
        if invalid_budget:
            raise InvalidImageError("DICOM 图像超过静态解码安全限制")

        decode_error = False
        try:
            pixel_array = ds.pixel_array
        except AttributeError:
            logger.info("No pixel data in DICOM file")
            return None
        except Exception:
            decode_error = True
        if decode_error:
            raise InvalidImageError("DICOM 像素数据无法安全解码")

        actual_budget_invalid = False
        try:
            actual_budget_invalid = (
                pixel_array.size > expected_samples
                or pixel_array.nbytes > expected_bytes
                or pixel_array.size > self._max_decoded_samples
                or pixel_array.nbytes > self._max_decoded_bytes
            )
        except Exception:
            actual_budget_invalid = True
        if actual_budget_invalid:
            raise InvalidImageError("DICOM 解码结果超过安全限制")

        conversion_error = False
        try:
            image_bytes = self._pixels_to_png(pixel_array)
        except Exception:
            conversion_error = True
        if conversion_error:
            raise InvalidImageError("DICOM 像素数据无法安全转换")
        context = f"Modality: {metadata.modality}"
        if metadata.study_description:
            context += f", Study: {metadata.study_description}"
        if metadata.series_description:
            context += f", Series: {metadata.series_description}"

        return self._image_parser.analyze(image_bytes, context=context)

    @staticmethod
    def _pixels_to_png(pixel_array: np.ndarray) -> bytes:
        arr = pixel_array.astype(np.float32)
        arr_min, arr_max = arr.min(), arr.max()
        if arr_max > arr_min:
            arr -= arr_min
            arr *= 255.0 / (arr_max - arr_min)
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
