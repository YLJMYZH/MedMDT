# tests/test_dicom_parser.py
import io
import struct
import pytest
from unittest.mock import MagicMock, patch
import numpy as np

from medmdt.extractor.parsers.dicom_parser import DicomParser, DicomMetadata, DicomParseResult
from medmdt.extractor.parsers.image_parser import ImageAnalysisResult


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
