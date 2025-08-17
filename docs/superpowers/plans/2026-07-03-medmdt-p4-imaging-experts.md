# MedMDT P4: Image Processing Enhancement + More Experts — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add DICOM file parsing via pydicom + multimodal LLM, enhance the extraction agent to handle DICOM files end-to-end, and expand the expert panel with 4 additional specialists (cardiologist, neurologist, oncologist, pathologist) for richer MDT consultations.

**Architecture:** DICOM parser uses pydicom to extract metadata (patient info, study info, series/equipment details) and Pillow to convert pixel data into PNG images for multimodal LLM analysis. The extraction agent's `_process_dicom` method orchestrates the DICOM pipeline. New experts are purely configuration-driven additions to `config/experts.yaml`.

**Tech Stack:** Python 3.12, pydicom (already in deps), Pillow (already in deps), pytest

## Global Constraints

- Python >=3.12, managed by uv
- Source layout: `src/medmdt/` (src-layout)
- pydicom and Pillow already in pyproject.toml dependencies
- Tests: pytest with mocks; no real DICOM files or LLM calls in unit tests
- Existing `ImageParser.analyze(image_data: bytes, context: str) -> ImageAnalysisResult` used for the pixel-data analysis step
- Existing `ExtractionAgent._detect_file_type` already returns "dicom" for `.dcm`/`.dicom` extensions
- Expert configs driven by `config/experts.yaml`; factory at `medmdt.mdt.experts.factory`

---

### Task 1: DICOM Parser

**Files:**
- Create: `src/medmdt/extractor/parsers/dicom_parser.py`
- Test: `tests/test_dicom_parser.py`

**Interfaces:**
- Consumes: `ImageParser.analyze(image_data, context) -> ImageAnalysisResult` from P1; `BaseChatModel` from langchain
- Produces:
  - `DicomMetadata(BaseModel)` — patient_id, patient_name, patient_age, patient_sex, study_date, modality, study_description, series_description, institution, manufacturer, pixel_spacing, slice_thickness
  - `DicomParseResult(BaseModel)` — metadata: DicomMetadata, image_analysis: ImageAnalysisResult | None, raw_text: str (combined textual representation)
  - `DicomParser.__init__(image_parser: ImageParser)`
  - `DicomParser.parse(file_path: str) -> DicomParseResult`

- [ ] **Step 1: Write the failing test**

```python
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
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_dicom_parser.py -v`
Expected: FAIL — module not found

- [ ] **Step 2: Implement dicom_parser.py**

```python
# src/medmdt/extractor/parsers/dicom_parser.py
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
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_dicom_parser.py -v`
Expected: 6 passed

- [ ] **Step 4: Commit**

```bash
git add src/medmdt/extractor/parsers/dicom_parser.py tests/test_dicom_parser.py
git commit -m "feat: DICOM parser — pydicom metadata extraction with multimodal LLM image analysis"
```

---

### Task 2: Wire DICOM Parser into Extraction Agent

**Files:**
- Modify: `src/medmdt/extractor/agent.py`
- Test: `tests/test_extraction_agent_dicom.py`

**Interfaces:**
- Consumes: `DicomParser.parse(file_path) -> DicomParseResult` from Task 1; existing `ExtractionAgent` class, `Ingestor.ingest()`, `_extract_from_text()`
- Produces:
  - `ExtractionAgent._process_dicom(file_path: str) -> IngestReport` — replaces the NotImplementedError stub

- [ ] **Step 1: Write the failing test**

```python
# tests/test_extraction_agent_dicom.py
import pytest
from unittest.mock import MagicMock, patch
from medmdt.extractor.agent import ExtractionAgent
from medmdt.extractor.parsers.dicom_parser import DicomParseResult, DicomMetadata
from medmdt.extractor.parsers.image_parser import ImageAnalysisResult
from medmdt.extractor.ingestor import IngestReport


@pytest.fixture
def mock_settings():
    settings = MagicMock()
    settings.paddleocr_api_url = "http://test"
    settings.paddleocr_token = "token"
    settings.paddleocr_use_doc_orientation_classify = False
    settings.paddleocr_use_doc_unwarping = False
    settings.paddleocr_use_chart_recognition = False
    return settings


@pytest.fixture
def agent(mock_settings):
    return ExtractionAgent(
        settings=mock_settings,
        graph_store=MagicMock(),
        vector_store=MagicMock(),
        keyword_store=MagicMock(),
        embed_fn=MagicMock(return_value=[[0.1] * 1024]),
        llm=MagicMock(),
    )


@patch("medmdt.extractor.agent.DicomParser")
def test_process_dicom(MockDicomParser, agent):
    mock_parser_instance = MockDicomParser.return_value
    mock_parser_instance.parse.return_value = DicomParseResult(
        metadata=DicomMetadata(
            patient_id="P001", patient_name="张三", modality="CT",
            study_description="胸部CT",
        ),
        image_analysis=ImageAnalysisResult(
            description="肺部见结节影",
            findings=["右肺结节", "直径约1cm"],
            modality="CT",
        ),
        raw_text="患者ID: P001\n患者姓名: 张三\n检查类型: CT\n影像分析: 肺部见结节影",
    )

    # Mock the LLM response for entity extraction
    import json
    entity_response = MagicMock()
    entity_response.content = json.dumps({
        "entities": [{"name": "肺结节", "type": "disease", "aliases": []}],
        "relations": [],
    })
    chunk_response = MagicMock()
    chunk_response.content = json.dumps({
        "chunks": [{"text": "CT显示右肺结节", "summary": "右肺结节影", "keywords": ["结节", "CT"]}],
    })
    agent._llm.invoke.side_effect = [entity_response, chunk_response]

    reports = agent.process_file("/path/scan.dcm")
    assert len(reports) == 1
    assert isinstance(reports[0], IngestReport)
    mock_parser_instance.parse.assert_called_once_with("/path/scan.dcm")


@patch("medmdt.extractor.agent.DicomParser")
def test_process_dicom_no_image(MockDicomParser, agent):
    mock_parser_instance = MockDicomParser.return_value
    mock_parser_instance.parse.return_value = DicomParseResult(
        metadata=DicomMetadata(patient_id="P002", patient_name="李四", modality="SR"),
        image_analysis=None,
        raw_text="患者ID: P002\n检查类型: SR",
    )

    import json
    entity_response = MagicMock()
    entity_response.content = json.dumps({"entities": [], "relations": []})
    chunk_response = MagicMock()
    chunk_response.content = json.dumps({"chunks": [{"text": "SR报告", "summary": "结构化报告", "keywords": ["SR"]}]})
    agent._llm.invoke.side_effect = [entity_response, chunk_response]

    reports = agent.process_file("/path/report.dcm")
    assert len(reports) == 1
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_extraction_agent_dicom.py -v`
Expected: FAIL — `_process_dicom` raises NotImplementedError

- [ ] **Step 2: Implement _process_dicom in agent.py**

Modify `src/medmdt/extractor/agent.py`:
1. Add import: `from medmdt.extractor.parsers.dicom_parser import DicomParser`
2. In `__init__`, add: `self._dicom_parser = DicomParser(image_parser=self._image_parser)`
3. Replace the `elif file_type == "dicom"` branch:
```python
elif file_type == "dicom":
    return [self._process_dicom(file_path)]
```
4. Add the `_process_dicom` method:
```python
def _process_dicom(self, file_path: str) -> IngestReport:
    """Parse DICOM file and ingest metadata + image analysis."""
    dicom_result = self._dicom_parser.parse(file_path)
    result = self._extract_from_text(
        text=dicom_result.raw_text,
        source=SourceInfo(
            file=file_path,
            type="dicom",
            page=None,
        ),
    )
    return self._ingestor.ingest(result)
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_extraction_agent_dicom.py -v`
Expected: 2 passed

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add src/medmdt/extractor/agent.py tests/test_extraction_agent_dicom.py
git commit -m "feat: wire DICOM parser into extraction agent — end-to-end DICOM processing"
```

---

### Task 3: Expand Expert Panel (4 New Specialists)

**Files:**
- Modify: `config/experts.yaml`
- Test: `tests/test_expert_configs.py`

**Interfaces:**
- Consumes: `load_expert_configs(yaml_path) -> list[ExpertConfig]` from P2 Task 3
- Produces: 4 additional expert entries — cardiologist, neurologist, oncologist, pathologist

- [ ] **Step 1: Write the test**

```python
# tests/test_expert_configs.py
import pytest
from pathlib import Path
from medmdt.mdt.experts.factory import load_expert_configs


EXPERTS_YAML = str(Path(__file__).parent.parent / "config" / "experts.yaml")


def test_load_all_experts():
    configs = load_expert_configs(EXPERTS_YAML)
    assert len(configs) == 7  # internist, radiologist, surgeon + 4 new


def test_expert_ids():
    configs = load_expert_configs(EXPERTS_YAML)
    ids = {c.expert_id for c in configs}
    expected = {"internist", "radiologist", "surgeon", "cardiologist", "neurologist", "oncologist", "pathologist"}
    assert ids == expected


def test_all_experts_have_system_prompt():
    configs = load_expert_configs(EXPERTS_YAML)
    for cfg in configs:
        assert len(cfg.system_prompt) > 20, f"{cfg.expert_id} has insufficient system_prompt"


def test_all_experts_have_knowledge_domains():
    configs = load_expert_configs(EXPERTS_YAML)
    for cfg in configs:
        assert len(cfg.knowledge_domains) >= 1, f"{cfg.expert_id} has no knowledge_domains"


def test_cardiologist_config():
    configs = load_expert_configs(EXPERTS_YAML)
    cardio = next(c for c in configs if c.expert_id == "cardiologist")
    assert "cardiology" in cardio.knowledge_domains
    assert "心" in cardio.name or "心" in cardio.system_prompt


def test_oncologist_config():
    configs = load_expert_configs(EXPERTS_YAML)
    onco = next(c for c in configs if c.expert_id == "oncologist")
    assert "oncology" in onco.knowledge_domains
```

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_expert_configs.py -v`
Expected: FAIL — only 3 experts, expected 7

- [ ] **Step 2: Add 4 experts to experts.yaml**

Append these entries after the `surgeon` entry in `config/experts.yaml`:

```yaml
  cardiologist:
    name: "心内科专家"
    knowledge_domains: ["cardiology", "internal_medicine"]
    system_prompt: |
      你是一位资深心内科主任医师，擅长冠心病、心律失常、心力衰竭、高血压、瓣膜病等心血管疾病的诊断与治疗。
      你熟悉心电图、超声心动图、冠脉造影等检查结果的解读。
      请基于提供的患者资料和医学知识，从心血管专科角度给出专业分析和建议。
    llm:
      provider: "openai"
      model: "gpt-4o"

  neurologist:
    name: "神经内科专家"
    knowledge_domains: ["neurology", "internal_medicine"]
    system_prompt: |
      你是一位资深神经内科主任医师，擅长脑血管病、癫痫、帕金森病、多发性硬化、周围神经病等神经系统疾病的诊断与治疗。
      你熟悉CT、MRI脑影像、脑电图、肌电图等检查结果的解读。
      请基于提供的患者资料和医学知识，从神经内科专科角度给出专业分析和建议。
    llm:
      provider: "openai"
      model: "gpt-4o"

  oncologist:
    name: "肿瘤科专家"
    knowledge_domains: ["oncology", "internal_medicine"]
    system_prompt: |
      你是一位资深肿瘤科主任医师，擅长各类实体肿瘤和血液肿瘤的诊断、分期和综合治疗方案制定。
      你熟悉肿瘤标志物、病理报告、影像学分期、基因检测等检查结果的解读。
      请基于提供的患者资料和医学知识，从肿瘤专科角度给出专业分析、分期评估和治疗建议。
    llm:
      provider: "deepseek"
      model: "deepseek-chat"

  pathologist:
    name: "病理科专家"
    knowledge_domains: ["pathology"]
    system_prompt: |
      你是一位资深病理科主任医师，擅长组织病理、细胞病理、免疫组化和分子病理的诊断。
      你熟悉各类活检标本、手术切除标本的病理形态学特征和鉴别诊断。
      请基于提供的病理描述和临床资料，给出专业的病理诊断意见和鉴别诊断分析。
    llm:
      provider: "anthropic"
      model: "claude-sonnet-4-6"
```

- [ ] **Step 3: Run tests, verify pass**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/test_expert_configs.py -v`
Expected: 6 passed

- [ ] **Step 4: Run full test suite**

Run: `cd /Users/yljm/MedMDT && uv run pytest tests/ -v`
Expected: All tests pass

- [ ] **Step 5: Commit**

```bash
git add config/experts.yaml tests/test_expert_configs.py
git commit -m "feat: expand expert panel — cardiologist, neurologist, oncologist, pathologist"
```
