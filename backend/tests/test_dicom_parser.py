# tests/test_dicom_parser.py
import io
import struct
import pytest
from unittest.mock import MagicMock, patch
import numpy as np
from pydicom.dataset import Dataset

from medmdt.extractor.parsers.dicom_parser import DicomParser, DicomMetadata, DicomParseResult
from medmdt.extractor.parsers.image_parser import ImageAnalysisResult
from medmdt.llm.errors import InvalidImageError


@pytest.fixture
def mock_image_parser():
    parser = MagicMock()
    parser.analyze.return_value = ImageAnalysisResult(
        description="胸部X线显示心影增大",
        findings=["心影增大", "肺纹理增粗"],
        modality="X-ray",
    )
    return parser


@pytest.fixture
def dicom_parser(mock_image_parser):
    return DicomParser(image_parser=mock_image_parser)


@pytest.fixture
def mock_dicom_dataset():
    ds = MagicMock()
    ds.PatientID = "P001"
    ds.PatientName = "张三"
    ds.PatientAge = "055Y"
    ds.PatientSex = "M"
    ds.StudyDate = "20260101"
    ds.Modality = "CR"
    ds.StudyDescription = "胸部正侧位"
    ds.SeriesDescription = "PA"
    ds.InstitutionName = "某医院"
    ds.Manufacturer = "Siemens"
    ds.PixelSpacing = [0.15, 0.15]
    ds.SliceThickness = None
    ds.Rows = 512
    ds.Columns = 512
    ds.SamplesPerPixel = 1
    ds.NumberOfFrames = 1
    ds.BitsAllocated = 16
    ds.pixel_array = np.zeros((512, 512), dtype=np.uint16)
    return ds


def test_dicom_metadata_model():
    meta = DicomMetadata(
        patient_id="P001",
        patient_name="张三",
        patient_age="055Y",
        patient_sex="M",
        study_date="20260101",
        modality="CR",
    )
    assert meta.patient_id == "P001"
    assert meta.modality == "CR"


def test_dicom_metadata_optional_fields():
    meta = DicomMetadata(
        patient_id="P001",
        patient_name="",
        modality="CT",
    )
    assert meta.study_description is None
    assert meta.slice_thickness is None


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_parser_parse(mock_dcmread, dicom_parser, mock_dicom_dataset):
    mock_dcmread.return_value = mock_dicom_dataset

    result = dicom_parser.parse("/path/to/scan.dcm")

    assert isinstance(result, DicomParseResult)
    assert result.metadata.patient_id == "P001"
    assert result.metadata.modality == "CR"
    assert result.image_analysis is not None
    assert "心影增大" in result.image_analysis.description
    assert len(result.raw_text) > 0


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_parser_no_pixel_data(mock_dcmread, dicom_parser, mock_dicom_dataset):
    del mock_dicom_dataset.pixel_array
    mock_dicom_dataset.pixel_array = property(lambda self: (_ for _ in ()).throw(AttributeError()))
    type(mock_dicom_dataset).pixel_array = property(lambda self: (_ for _ in ()).throw(AttributeError("No pixel data")))
    mock_dcmread.return_value = mock_dicom_dataset

    result = dicom_parser.parse("/path/to/sr.dcm")
    assert result.image_analysis is None


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_real_dicom_dataset_without_pixel_data_remains_metadata_only(
    mock_dcmread, mock_image_parser
):
    dataset = Dataset()
    dataset.PatientID = "P001"
    dataset.Modality = "SR"
    mock_dcmread.return_value = dataset

    result = DicomParser(mock_image_parser).parse("/path/to/report.dcm")

    assert result.image_analysis is None
    mock_image_parser.analyze.assert_not_called()


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_parser_context_includes_metadata(mock_dcmread, dicom_parser, mock_dicom_dataset, mock_image_parser):
    mock_dcmread.return_value = mock_dicom_dataset

    dicom_parser.parse("/path/to/scan.dcm")

    call_args = mock_image_parser.analyze.call_args
    context = call_args[1].get("context", call_args[0][1] if len(call_args[0]) > 1 else "")
    assert "CR" in context or "胸部" in context


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_parser_raw_text(mock_dcmread, dicom_parser, mock_dicom_dataset):
    mock_dcmread.return_value = mock_dicom_dataset
    result = dicom_parser.parse("/path/to/scan.dcm")
    assert "P001" in result.raw_text
    assert "张三" in result.raw_text


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("NumberOfFrames", 2),
        ("Rows", 100_000),
        ("Columns", 100_000),
        ("SamplesPerPixel", 100),
        ("BitsAllocated", 65_536),
    ],
)
@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_metadata_budget_rejects_before_pixel_decode(
    mock_dcmread, mock_image_parser, mock_dicom_dataset, field, value
):
    type(mock_dicom_dataset).pixel_array = property(
        lambda self: (_ for _ in ()).throw(AssertionError("pixel_array accessed"))
    )
    setattr(mock_dicom_dataset, field, value)
    mock_dcmread.return_value = mock_dicom_dataset

    with pytest.raises(InvalidImageError, match="DICOM") as exc_info:
        DicomParser(mock_image_parser).parse("/path/to/oversized.dcm")

    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    mock_image_parser.analyze.assert_not_called()


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_actual_array_mismatch_is_rejected(
    mock_dcmread, mock_image_parser, mock_dicom_dataset
):
    mock_dicom_dataset.Rows = 8
    mock_dicom_dataset.Columns = 8
    mock_dicom_dataset.pixel_array = np.zeros((16, 16), dtype=np.uint16)
    mock_dcmread.return_value = mock_dicom_dataset

    with pytest.raises(InvalidImageError, match="DICOM") as exc_info:
        DicomParser(mock_image_parser, max_decoded_samples=512).parse(
            "/path/to/mismatch.dcm"
        )

    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    mock_image_parser.analyze.assert_not_called()


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_actual_nbytes_budget_is_rejected(
    mock_dcmread, mock_image_parser, mock_dicom_dataset
):
    mock_dicom_dataset.Rows = 8
    mock_dicom_dataset.Columns = 8
    mock_dicom_dataset.BitsAllocated = 16
    mock_dicom_dataset.pixel_array = np.zeros((8, 8), dtype=np.uint32)
    mock_dcmread.return_value = mock_dicom_dataset

    with pytest.raises(InvalidImageError, match="DICOM") as exc_info:
        DicomParser(mock_image_parser, max_decoded_bytes=256).parse(
            "/path/to/nbytes.dcm"
        )

    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    mock_image_parser.analyze.assert_not_called()


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
def test_dicom_reader_exception_is_sanitized(mock_dcmread, mock_image_parser):
    mock_dcmread.side_effect = RuntimeError("sensitive DICOM parser internals")

    with pytest.raises(InvalidImageError, match="DICOM") as exc_info:
        DicomParser(mock_image_parser).parse("/path/to/broken.dcm")

    assert "sensitive" not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    mock_image_parser.analyze.assert_not_called()


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
@patch.object(DicomParser, "_extract_metadata")
def test_dicom_metadata_parser_exception_is_sanitized(
    extract_metadata, mock_dcmread, mock_image_parser, mock_dicom_dataset
):
    mock_dcmread.return_value = mock_dicom_dataset
    extract_metadata.side_effect = RuntimeError("sensitive metadata parser internals")

    with pytest.raises(InvalidImageError, match="DICOM") as exc_info:
        DicomParser(mock_image_parser).parse("/path/to/broken-metadata.dcm")

    assert "sensitive" not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    mock_image_parser.analyze.assert_not_called()


@patch("medmdt.extractor.parsers.dicom_parser.dcmread")
@patch.object(DicomParser, "_pixels_to_png")
def test_dicom_png_conversion_exception_is_sanitized(
    pixels_to_png, mock_dcmread, mock_image_parser, mock_dicom_dataset
):
    mock_dcmread.return_value = mock_dicom_dataset
    pixels_to_png.side_effect = RuntimeError("sensitive numpy/Pillow internals")

    with pytest.raises(InvalidImageError, match="DICOM") as exc_info:
        DicomParser(mock_image_parser).parse("/path/to/broken-pixels.dcm")

    assert "sensitive" not in str(exc_info.value)
    assert exc_info.value.__context__ is None
    assert exc_info.value.__cause__ is None
    mock_image_parser.analyze.assert_not_called()
