# Multimodal Vision Provider Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a dedicated, capability-aware image-understanding path that keeps native text integrations while routing OpenAI-compatible vision providers through a uniform LangChain message format.

**Architecture:** A single `ProviderSpec` registry owns provider metadata plus separate chat and vision factories. `ImageParser` remains provider-agnostic and normalizes image bytes, MIME types, LangChain responses, validation failures, and one JSON-correction retry. API wiring tolerates legacy unsupported vision settings at startup, while settings exposes provider capability and a real image test.

**Tech Stack:** Python 3.12, LangChain Core 1.x, `langchain-openai`, `langchain-anthropic`, `langchain-deepseek`, `langchain-community`, Pillow, Pydantic v2, FastAPI, pytest, React 18, TypeScript, Vite.

## Global Constraints

- Scope is image understanding only: image input and structured text output.
- Keep existing native text routes for Anthropic, Qwen, Zhipu, and DeepSeek.
- Route Qwen, Zhipu, Moonshot, and Custom vision calls through OpenAI Chat Completions-compatible `ChatOpenAI` clients.
- Do not add `langchain-glm`.
- Keep DeepSeek registered and visible, but mark official vision support unavailable until its API publishes an image-input contract.
- Do not infer model-level vision capability from model names; use a real image request.
- Do not silently resize medical images.
- Never log API keys, image bytes, Base64 Data URLs, or clinical context.
- Default maximum transmitted image size is 5 MiB and remains overrideable through `ImageParser(max_image_bytes=...)`.
- Current baseline note: `tests/test_llm_provider.py` has two stale failures because its Qwen and DeepSeek mocks still expect `ChatOpenAI`; Task 1 replaces those expectations.

---

## File Structure

- Create `backend/src/medmdt/llm/errors.py`: shared vision exception taxonomy.
- Modify `backend/src/medmdt/llm/provider.py`: `ProviderSpec`, unified metadata, separate chat/vision factories, provider URL lookup.
- Modify `backend/src/medmdt/extractor/parsers/image_parser.py`: image normalization, MIME handling, response normalization, retry, request error classification.
- Modify `backend/src/medmdt/extractor/agent.py`: inject a separate optional vision model and fail only when an image actually needs analysis.
- Modify `backend/src/medmdt/api/deps.py`: build chat and vision clients independently without breaking startup for a legacy unsupported vision setting.
- Modify `backend/src/medmdt/api/routes/settings.py`: registry-backed provider metadata, save validation, and real vision test endpoint.
- Modify `backend/src/medmdt/api/routes/consultation.py`: propagate precise vision errors and pass the dedicated vision client to extraction.
- Modify `backend/src/medmdt/api/routes/knowledge.py`: pass the dedicated vision client to ingestion agents.
- Modify `backend/scripts/ingest.py`: construct the configured runtime vision client for CLI ingestion.
- Modify `backend/tests/test_llm_provider.py`, `backend/tests/test_image_parser.py`, and extraction/API tests: backend TDD coverage.
- Create `backend/tests/test_api_settings.py` and `backend/tests/test_api_deps.py`: settings and legacy-startup behavior.
- Create `backend/tests/contract/test_vision_providers.py`: opt-in real-provider contract coverage.
- Modify `backend/pyproject.toml`: register the `contract` pytest marker.
- Modify `frontend/src/lib/types.ts`, `frontend/src/lib/api.ts`, and `frontend/src/pages/SettingsPage.tsx`: visual capability state and visual test UI.

---

### Task 1: Provider Registry and Dedicated Vision Factory

**Files:**
- Create: `backend/src/medmdt/llm/errors.py`
- Modify: `backend/src/medmdt/llm/provider.py`
- Test: `backend/tests/test_llm_provider.py`

**Interfaces:**
- Produces: `ProviderSpec`, `PROVIDER_REGISTRY: dict[str, ProviderSpec]`, `create_chat_model(...) -> BaseChatModel`, `create_vision_model(...) -> BaseChatModel`, `list_provider_metadata() -> list[dict[str, object]]`, and `get_provider_base_url(provider: str) -> str | None`.
- Produces: `VisionError`, `VisionProviderNotSupported`, `VisionModelNotSupported`, `InvalidImageError`, `VisionRequestError`, and `InvalidVisionResponse`.
- Consumes: existing LangChain model classes and existing provider credentials.

- [ ] **Step 1: Replace stale provider expectations and add failing vision-factory tests**

Update `backend/tests/test_llm_provider.py` so native text routes are mocked correctly, then add these cases:

```python
from unittest.mock import MagicMock, patch

import pytest

from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.provider import (
    PROVIDER_REGISTRY,
    create_chat_model,
    create_vision_model,
    get_provider_base_url,
    list_provider_metadata,
)


@patch("medmdt.llm.provider.ChatTongyi")
def test_create_qwen_chat_uses_native_tongyi(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("qwen", "qwen-plus", api_key="qwen-key")
    mock_cls.assert_called_once()
    assert mock_cls.call_args.kwargs["dashscope_api_key"] == "qwen-key"


@patch("medmdt.llm.provider.ChatDeepSeek")
def test_create_deepseek_chat_uses_partner_integration(mock_cls):
    mock_cls.return_value = MagicMock()
    create_chat_model("deepseek", "deepseek-chat", api_key="deepseek-key")
    mock_cls.assert_called_once()


@pytest.mark.parametrize(
    ("provider", "expected_base_url"),
    [
        ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("zhipu", "https://open.bigmodel.cn/api/paas/v4"),
        ("moonshot", "https://api.moonshot.cn/v1"),
    ],
)
@patch("medmdt.llm.provider.ChatOpenAI")
def test_compatible_vision_routes_use_chat_completions(
    mock_cls, provider, expected_base_url
):
    mock_cls.return_value = MagicMock()
    create_vision_model(provider, "vision-model", api_key="vision-key")
    kwargs = mock_cls.call_args.kwargs
    assert kwargs["base_url"] == expected_base_url
    assert kwargs["use_responses_api"] is False


@patch("medmdt.llm.provider.ChatAnthropic")
def test_anthropic_vision_uses_native_integration(mock_cls):
    mock_cls.return_value = MagicMock()
    create_vision_model("anthropic", "claude-vision", api_key="anthropic-key")
    mock_cls.assert_called_once()


def test_deepseek_vision_is_registered_but_unavailable():
    assert PROVIDER_REGISTRY["deepseek"].vision_status == "unavailable"
    with pytest.raises(VisionProviderNotSupported, match="DeepSeek"):
        create_vision_model("deepseek", "deepseek-chat", api_key="key")


def test_custom_vision_requires_base_url():
    with pytest.raises(ValueError, match="base_url"):
        create_vision_model("custom", "local-vlm", api_key="key")


def test_provider_metadata_and_base_urls_come_from_registry():
    metadata = {item["key"]: item for item in list_provider_metadata()}
    assert metadata["deepseek"]["vision_status"] == "unavailable"
    assert metadata["custom"]["vision_status"] == "unknown"
    assert get_provider_base_url("qwen") == "https://dashscope.aliyuncs.com/compatible-mode/v1"
```

- [ ] **Step 2: Run the provider tests and confirm the new API is missing**

Run: `cd backend && uv run pytest tests/test_llm_provider.py -q`

Expected: collection fails because `medmdt.llm.errors`, `create_vision_model`, and provider metadata helpers do not exist yet.

- [ ] **Step 3: Add the shared vision exception taxonomy**

Create `backend/src/medmdt/llm/errors.py`:

```python
"""Stable error categories for image-understanding calls."""


class VisionError(RuntimeError):
    """Base class for expected image-understanding failures."""


class VisionProviderNotSupported(VisionError):
    """The provider has no supported image-input transport."""


class VisionModelNotSupported(VisionError):
    """The selected model rejected image input."""


class InvalidImageError(VisionError):
    """The supplied image cannot be safely sent to a model."""


class VisionRequestError(VisionError):
    """Authentication, network, rate-limit, or provider request failure."""


class InvalidVisionResponse(VisionError):
    """The provider response contains no valid structured image analysis."""
```

- [ ] **Step 4: Introduce `ProviderSpec` and separate text and vision factories**

In `backend/src/medmdt/llm/provider.py`, add these types and helpers, retain the existing native `_create_*` text factories, and replace the callable-only registry:

```python
from dataclasses import dataclass
from typing import Callable, Literal

from medmdt.llm.errors import VisionProviderNotSupported

ModelFactory = Callable[..., BaseChatModel]
VisionStatus = Literal["supported", "unavailable", "unknown"]


@dataclass(frozen=True)
class ProviderSpec:
    key: str
    label: str
    chat_factory: ModelFactory
    vision_factory: ModelFactory | None
    vision_status: VisionStatus
    api_base_url: str | None = None
    needs_key: bool = True
    needs_base_url: bool = False


_PROVIDER_BASE_URLS = {
    "openai": "https://api.openai.com/v1",
    "deepseek": "https://api.deepseek.com/v1",
    "qwen": "https://dashscope.aliyuncs.com/compatible-mode/v1",
    "zhipu": "https://open.bigmodel.cn/api/paas/v4",
    "moonshot": "https://api.moonshot.cn/v1",
}


def _create_compatible_vision(
    provider: str, model: str, temperature: float, **kwargs
) -> BaseChatModel:
    base_url = kwargs.pop("base_url", None) or _PROVIDER_BASE_URLS.get(provider)
    if not base_url:
        raise ValueError(f"{provider} vision provider requires base_url")
    api_key = kwargs.pop("api_key", None)
    params = {
        "model": model,
        "base_url": base_url,
        "temperature": temperature,
        "use_responses_api": False,
        **kwargs,
    }
    if api_key:
        params["api_key"] = api_key
    return ChatOpenAI(**params)


PROVIDER_REGISTRY: dict[str, ProviderSpec] = {
    "openai": ProviderSpec(
        "openai", "OpenAI", _create_openai, _create_openai, "supported",
        api_base_url=_PROVIDER_BASE_URLS["openai"],
    ),
    "anthropic": ProviderSpec(
        "anthropic", "Anthropic", _create_anthropic, _create_anthropic, "supported",
    ),
    "deepseek": ProviderSpec(
        "deepseek", "DeepSeek", _create_deepseek, None, "unavailable",
        api_base_url=_PROVIDER_BASE_URLS["deepseek"],
    ),
    "qwen": ProviderSpec(
        "qwen", "Qwen (通义千问)", _create_tongyi,
        lambda m, t, **kw: _create_compatible_vision("qwen", m, t, **kw),
        "supported", api_base_url=_PROVIDER_BASE_URLS["qwen"],
    ),
    "zhipu": ProviderSpec(
        "zhipu", "ZhiPu (智谱)", _create_zhipu,
        lambda m, t, **kw: _create_compatible_vision("zhipu", m, t, **kw),
        "supported", api_base_url=_PROVIDER_BASE_URLS["zhipu"],
    ),
    "moonshot": ProviderSpec(
        "moonshot", "Moonshot (月之暗面)",
        lambda m, t, **kw: _create_openai_compat("moonshot", m, t, **kw),
        lambda m, t, **kw: _create_compatible_vision("moonshot", m, t, **kw),
        "supported", api_base_url=_PROVIDER_BASE_URLS["moonshot"],
    ),
    "custom": ProviderSpec(
        "custom", "自定义 (OpenAI 兼容)", _create_custom,
        lambda m, t, **kw: _create_compatible_vision("custom", m, t, **kw),
        "unknown", needs_base_url=True,
    ),
}


def _get_provider(provider: str) -> ProviderSpec:
    spec = PROVIDER_REGISTRY.get(provider)
    if spec is None:
        raise ValueError(
            f"Unknown provider: {provider}. Available: {list(PROVIDER_REGISTRY)}"
        )
    return spec


def create_chat_model(
    provider: str, model: str, temperature: float = 0.0, **kwargs
) -> BaseChatModel:
    return _get_provider(provider).chat_factory(model, temperature, **kwargs)


def create_vision_model(
    provider: str, model: str, temperature: float = 0.0, **kwargs
) -> BaseChatModel:
    spec = _get_provider(provider)
    if spec.vision_factory is None:
        raise VisionProviderNotSupported(
            f"{spec.label} 官方 API 暂不支持图像理解，请选择其他视觉模型"
        )
    return spec.vision_factory(model, temperature, **kwargs)


def get_provider_base_url(provider: str) -> str | None:
    return _get_provider(provider).api_base_url


def list_provider_metadata() -> list[dict[str, object]]:
    return [
        {
            "key": spec.key,
            "label": spec.label,
            "needs_key": spec.needs_key,
            "needs_base_url": spec.needs_base_url,
            "vision_status": spec.vision_status,
        }
        for spec in PROVIDER_REGISTRY.values()
    ]
```

- [ ] **Step 5: Run the provider tests**

Run: `cd backend && uv run pytest tests/test_llm_provider.py -q`

Expected: all provider tests pass with no API key lookup from unmocked Qwen or DeepSeek classes.

- [ ] **Step 6: Commit the registry and vision factory**

```bash
git add backend/src/medmdt/llm/errors.py backend/src/medmdt/llm/provider.py backend/tests/test_llm_provider.py
git commit -m "feat(llm): add capability-aware vision provider factory"
```

---

### Task 2: Provider-Agnostic Image Normalization and Response Handling

**Files:**
- Modify: `backend/src/medmdt/extractor/parsers/image_parser.py`
- Modify: `backend/tests/test_image_parser.py`

**Interfaces:**
- Consumes: `BaseChatModel` and the vision errors created in Task 1.
- Produces: `ImageParser(llm: BaseChatModel, max_image_bytes: int = 5 * 1024 * 1024)` with unchanged `analyze(...)` and `analyze_batch(...)` public methods.
- Produces: normalized OpenAI-style `HumanMessage` image blocks usable by all Task 1 vision routes.

- [ ] **Step 1: Add real-image fixtures and failing normalization tests**

Add the following helpers and cases to `backend/tests/test_image_parser.py`:

```python
from io import BytesIO
from unittest.mock import MagicMock

import pytest
from langchain_core.messages import AIMessage
from PIL import Image

from medmdt.llm.errors import InvalidImageError, InvalidVisionResponse


def image_bytes(fmt: str, color: str = "red") -> bytes:
    output = BytesIO()
    Image.new("RGB", (8, 8), color=color).save(output, format=fmt)
    return output.getvalue()


@pytest.mark.parametrize(
    ("fmt", "mime"),
    [
        ("PNG", "image/png"),
        ("JPEG", "image/jpeg"),
        ("WEBP", "image/webp"),
        ("GIF", "image/gif"),
    ],
)
def test_analyze_uses_detected_mime(fmt, mime):
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"description":"ok","findings":[],"modality":null}'
    )
    ImageParser(llm).analyze(image_bytes(fmt))
    block = llm.invoke.call_args.args[0][0].content[1]
    assert block["image_url"]["url"].startswith(f"data:{mime};base64,")


def test_bmp_is_converted_to_png_before_sending():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(
        content='{"description":"ok","findings":[],"modality":null}'
    )
    ImageParser(llm).analyze(image_bytes("BMP"))
    block = llm.invoke.call_args.args[0][0].content[1]
    assert block["image_url"]["url"].startswith("data:image/png;base64,")


@pytest.mark.parametrize("payload", [b"", b"not-an-image"])
def test_invalid_images_fail_before_model_call(payload):
    llm = MagicMock()
    with pytest.raises(InvalidImageError):
        ImageParser(llm).analyze(payload)
    llm.invoke.assert_not_called()


def test_transmission_size_limit_does_not_resize():
    llm = MagicMock()
    with pytest.raises(InvalidImageError, match="大小限制"):
        ImageParser(llm, max_image_bytes=10).analyze(image_bytes("PNG"))
    llm.invoke.assert_not_called()


def test_content_block_response_is_normalized_to_text():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content=[
        {"type": "thinking", "thinking": "private"},
        {
            "type": "text",
            "text": '{"description":"ok","findings":[],"modality":null}',
        },
    ])
    result = ImageParser(llm).analyze(image_bytes("PNG"))
    assert result.description == "ok"


def test_invalid_json_is_retried_once_with_same_image():
    llm = MagicMock()
    llm.invoke.side_effect = [
        AIMessage(content="not json"),
        AIMessage(content='{"description":"fixed","findings":[],"modality":null}'),
    ]
    result = ImageParser(llm).analyze(image_bytes("PNG"))
    assert result.description == "fixed"
    assert llm.invoke.call_count == 2
    assert "上一次响应不是有效 JSON" in llm.invoke.call_args.args[0][0].content[0]["text"]


def test_second_invalid_json_raises_stable_error():
    llm = MagicMock()
    llm.invoke.return_value = AIMessage(content="not json")
    with pytest.raises(InvalidVisionResponse):
        ImageParser(llm).analyze(image_bytes("PNG"))
    assert llm.invoke.call_count == 2
```

Make these exact substitutions in the existing tests so success-path tests continue to use valid images:

```python
# test_image_parser_analyze
result = parser.analyze(image_bytes("PNG"))

# test_image_parser_analyze_with_context
result = parser.analyze(image_bytes("PNG"), context="患者咳嗽2周")

# test_image_parser_analyze_batch
results = parser.analyze_batch([image_bytes("PNG"), image_bytes("JPEG")])

# test_image_parser_sends_base64_image
image_data = image_bytes("PNG")
parser.analyze(image_data)

# test_image_parser_analyze_json_with_markdown_fence
result = parser.analyze(image_bytes("PNG"))
```

Delete the earlier `b"fake-image-bytes"`, `b"fake-image"`, `b"img1"`, `b"img2"`, `b"PNG-fake-data"`, and `b"img"` success inputs; invalid-byte behavior is covered only by `test_invalid_images_fail_before_model_call`.

- [ ] **Step 2: Run image-parser tests and confirm normalization cases fail**

Run: `cd backend && uv run pytest tests/test_image_parser.py -q`

Expected: failures for fake images, JPEG/WebP MIME, content-block parsing, size validation, and retry behavior.

- [ ] **Step 3: Implement image validation and normalized Data URLs**

Add these definitions to `image_parser.py` and call `_prepare_image()` once at the start of `analyze()`:

```python
from io import BytesIO
from typing import Any

from PIL import Image, UnidentifiedImageError
from pydantic import ValidationError

from medmdt.llm.errors import (
    InvalidImageError,
    InvalidVisionResponse,
    VisionModelNotSupported,
    VisionRequestError,
)

DEFAULT_MAX_IMAGE_BYTES = 5 * 1024 * 1024
DIRECT_IMAGE_MIME = {
    "PNG": "image/png",
    "JPEG": "image/jpeg",
    "WEBP": "image/webp",
    "GIF": "image/gif",
}


def _prepare_image(image_data: bytes, max_image_bytes: int) -> tuple[bytes, str]:
    if not image_data:
        raise InvalidImageError("图片内容为空")
    try:
        with Image.open(BytesIO(image_data)) as image:
            image_format = image.format
            image.verify()
    except (UnidentifiedImageError, OSError, ValueError) as exc:
        raise InvalidImageError("图片损坏或格式无法识别") from exc

    if image_format in DIRECT_IMAGE_MIME:
        normalized = image_data
        mime_type = DIRECT_IMAGE_MIME[image_format]
    else:
        output = BytesIO()
        with Image.open(BytesIO(image_data)) as image:
            image.save(output, format="PNG")
        normalized = output.getvalue()
        mime_type = "image/png"

    if len(normalized) > max_image_bytes:
        raise InvalidImageError(
            f"图片超过 {max_image_bytes} 字节传输大小限制；系统不会静默缩放医学影像"
        )
    return normalized, mime_type
```

Change the constructor and message builder:

```python
def __init__(
    self,
    llm: BaseChatModel,
    max_image_bytes: int = DEFAULT_MAX_IMAGE_BYTES,
):
    self._llm = llm
    self._max_image_bytes = max_image_bytes


def _build_message(self, image_data: bytes, mime_type: str, prompt: str) -> HumanMessage:
    encoded = base64.b64encode(image_data).decode("ascii")
    return HumanMessage(content=[
        {"type": "text", "text": prompt},
        {
            "type": "image_url",
            "image_url": {"url": f"data:{mime_type};base64,{encoded}"},
        },
    ])
```

- [ ] **Step 4: Implement response normalization, request classification, and one correction retry**

Add these helpers and replace `analyze()` with the loop below:

```python
def _extract_response_text(content: Any) -> str:
    if isinstance(content, str) and content.strip():
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for block in content:
            if isinstance(block, str) and block.strip():
                text_parts.append(block)
            elif isinstance(block, dict) and block.get("type") in {"text", "output_text"}:
                text = block.get("text")
                if isinstance(text, str) and text.strip():
                    text_parts.append(text)
        if text_parts:
            return "\n".join(text_parts)
    raise InvalidVisionResponse("视觉模型响应中没有可用文本")


def _invoke_model(llm: BaseChatModel, message: HumanMessage):
    try:
        return llm.invoke([message])
    except Exception as exc:
        response = getattr(exc, "response", None)
        status_code = getattr(exc, "status_code", None) or getattr(
            response, "status_code", None
        )
        error_text = str(exc).lower()
        image_markers = ("image", "vision", "multimodal", "image_url", "图片", "视觉")
        if status_code in {400, 404, 422} and any(
            marker in error_text for marker in image_markers
        ):
            raise VisionModelNotSupported("所选模型不接受图片输入") from None
        # Do not retain the SDK exception chain: some clients attach the full
        # request body, which contains the Base64 medical image.
        raise VisionRequestError("视觉模型请求失败，请检查认证、限流和服务状态") from None


def analyze(self, image_data: bytes, context: str = "") -> ImageAnalysisResult:
    normalized, mime_type = _prepare_image(image_data, self._max_image_bytes)
    validation_error: ValidationError | None = None
    for attempt in range(2):
        prompt = self._build_prompt(context)
        if attempt == 1:
            prompt += "\n\n上一次响应不是有效 JSON。请严格只返回符合上述字段定义的 JSON。"
        message = self._build_message(normalized, mime_type, prompt)
        response = _invoke_model(self._llm, message)
        raw = _extract_response_text(response.content)
        try:
            return self._parse_response(raw)
        except ValidationError as exc:
            validation_error = exc
    raise InvalidVisionResponse("视觉模型未返回有效的结构化 JSON") from validation_error
```

Remove the unused `json` import if it remains unused.

- [ ] **Step 5: Run focused and parser-adjacent tests**

Run: `cd backend && uv run pytest tests/test_image_parser.py tests/test_dicom_parser.py -q`

Expected: all tests pass; DICOM rendering still sends valid PNG bytes through `ImageParser`.

- [ ] **Step 6: Commit image normalization**

```bash
git add backend/src/medmdt/extractor/parsers/image_parser.py backend/tests/test_image_parser.py
git commit -m "feat(extractor): normalize multimodal image requests"
```

---

### Task 3: Wire the Dedicated Vision Client Through Runtime and Extraction

**Files:**
- Modify: `backend/src/medmdt/api/deps.py`
- Modify: `backend/src/medmdt/extractor/agent.py`
- Modify: `backend/src/medmdt/api/routes/knowledge.py`
- Modify: `backend/src/medmdt/api/routes/consultation.py`
- Modify: `backend/scripts/ingest.py`
- Create: `backend/tests/test_api_deps.py`
- Modify: `backend/tests/test_extraction_agent.py`
- Modify: `backend/tests/test_extraction_agent_dicom.py`
- Modify: `backend/tests/test_ingest_script.py`

**Interfaces:**
- Consumes: `create_vision_model()` and vision error classes from Tasks 1–2.
- Produces: infrastructure keys `vision_llm: BaseChatModel | None`, `vision_error: str | None`, `vision_provider: str`, and `vision_model: str`.
- Produces: `ExtractionAgent(..., llm: BaseChatModel, vision_llm: BaseChatModel | None, vision_error: str | None = None)`.

- [ ] **Step 1: Add failing tests for independent clients and legacy unsupported settings**

Create `backend/tests/test_api_deps.py` with model/store constructors patched so no external services are contacted:

```python
from types import SimpleNamespace
from unittest.mock import MagicMock, patch

from medmdt.llm.errors import VisionProviderNotSupported


@patch("medmdt.config.settings.get_settings")
@patch("medmdt.config.runtime.load_llm_settings")
@patch("medmdt.llm.provider.create_vision_model")
@patch("medmdt.llm.provider.create_chat_model")
@patch("medmdt.knowledge.retriever.FusionRetriever")
@patch("medmdt.knowledge.keyword_store.KeywordStore")
@patch("medmdt.knowledge.vector_store.VectorStore")
@patch("medmdt.knowledge.graph_store.GraphStore")
def test_build_infrastructure_keeps_running_when_legacy_vision_is_unsupported(
    graph_cls,
    vector_cls,
    keyword_cls,
    retriever_cls,
    chat_factory,
    vision_factory,
    load_runtime,
    get_settings,
):
    from medmdt.api.deps import build_infrastructure

    get_settings.return_value = MagicMock(
        neo4j_uri="bolt://test",
        neo4j_user="neo4j",
        neo4j_password="password",
        milvus_host="localhost",
        milvus_port=19530,
        elasticsearch_url="http://localhost:9200",
    )
    load_runtime.return_value = SimpleNamespace(
        knowledge=SimpleNamespace(provider="openai", model="text", api_key="k", base_url=None),
        vision=SimpleNamespace(provider="deepseek", model="deepseek-chat", api_key="k", base_url=None),
        embedding=SimpleNamespace(provider="openai", model="embed", api_key="k", base_url=None, dim=3),
    )
    chat_factory.return_value = MagicMock()
    vision_factory.side_effect = VisionProviderNotSupported("DeepSeek 暂不支持图像理解")

    infra = build_infrastructure()

    assert infra["vision_llm"] is None
    assert infra["vision_error"] == "DeepSeek 暂不支持图像理解"
```

Add this focused test to `backend/tests/test_extraction_agent.py`:

```python
def test_extraction_agent_uses_separate_vision_model(settings, mock_deps):
    graph, vector, keyword, embed_fn, text_llm = mock_deps
    vision_llm = MagicMock()
    agent = ExtractionAgent(
        settings=settings,
        graph_store=graph,
        vector_store=vector,
        keyword_store=keyword,
        embed_fn=embed_fn,
        llm=text_llm,
        vision_llm=vision_llm,
    )
    assert agent._llm is text_llm
    assert agent._image_parser is not None
    assert agent._image_parser._llm is vision_llm
```

- [ ] **Step 2: Run the new focused tests and confirm signatures/wiring fail**

Run: `cd backend && uv run pytest tests/test_api_deps.py tests/test_extraction_agent.py::test_extraction_agent_uses_separate_vision_model -q`

Expected: failures because infrastructure still calls `create_chat_model` for vision and `ExtractionAgent` has no `vision_llm` parameter.

- [ ] **Step 3: Build vision separately while tolerating an unavailable legacy provider**

In `backend/src/medmdt/api/deps.py`, replace the vision construction with:

```python
from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.provider import create_chat_model, create_vision_model

vision_ep = runtime.vision
vision_llm = None
vision_error = None
try:
    vision_llm = create_vision_model(
        vision_ep.provider,
        vision_ep.model,
        **get_llm_kwargs(vision_ep),
    )
except VisionProviderNotSupported as exc:
    vision_error = str(exc)
```

Add the following fields next to `"vision_llm": vision_llm` in the returned dictionary. Keep text model construction unchanged.

```python
"vision_error": vision_error,
"vision_provider": vision_ep.provider,
"vision_model": vision_ep.model,
```

- [ ] **Step 4: Inject the separate vision model into `ExtractionAgent`**

Change the constructor and add a guard in `backend/src/medmdt/extractor/agent.py`:

```python
from medmdt.llm.errors import VisionProviderNotSupported


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
    self._image_parser = ImageParser(vision_llm) if vision_llm is not None else None
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
```

Use `_require_image_parser()` in embedded PDF image, standalone image, and DICOM paths. Construct `DicomParser(image_parser=self._require_image_parser())` at the point of DICOM use instead of storing a parser when no vision model exists.

- [ ] **Step 5: Pass vision dependencies at every production call site**

In both `backend/src/medmdt/api/routes/knowledge.py` agent constructions and the construction in `backend/src/medmdt/api/routes/consultation.py`, add:

```python
vision_llm=infra["vision_llm"],
vision_error=infra["vision_error"],
```

For `upload_consultation_files`, retain `vision_llm=None` in infrastructure but reject only image/DICOM extraction. At the start of the image and DICOM branches in `_extract_text_from_file`, add:

```python
if llm is None:
    raise VisionProviderNotSupported("图片分析未配置可用的视觉模型")
```

Remove the broad exception rewrites that always recommend `qwen-vl-max`; allow stable `VisionError` messages to reach the per-file error response, and convert only unexpected exceptions to `ValueError("图片分析失败")`.

Import `uuid` in `consultation.py`. Generate `request_id = uuid.uuid4().hex[:12]` once in `upload_consultation_files`. In its per-file exception handler, log only safe metadata:

```python
logger.warning(
    "Vision upload failed request_id=%s provider=%s model=%s error_type=%s",
    request_id,
    infra["vision_provider"],
    infra["vision_model"],
    type(exc).__name__,
)
```

In `_run_ingest` and `_run_batch_ingest`, add an `except VisionError as exc` branch before the existing general branch. Use `job_id` as the request trace identifier, log `infra["vision_provider"]`, `infra["vision_model"]`, and `type(exc).__name__`, and update the job message with the stable `str(exc)`. Do not use `logger.exception` for `VisionError`.

In `backend/scripts/ingest.py`, load the runtime vision endpoint and construct it independently:

```python
from medmdt.config.runtime import get_llm_kwargs, load_llm_settings
from medmdt.llm.errors import VisionProviderNotSupported
from medmdt.llm.provider import create_chat_model, create_vision_model

runtime = load_llm_settings()
vision_llm = None
vision_error = None
try:
    vision_llm = create_vision_model(
        runtime.vision.provider,
        runtime.vision.model,
        **get_llm_kwargs(runtime.vision),
    )
except (VisionProviderNotSupported, ValueError) as exc:
    vision_error = str(exc)
```

Pass `vision_llm=vision_llm` and `vision_error=vision_error` to `ExtractionAgent`.

- [ ] **Step 6: Update test constructors explicitly**

For each `ExtractionAgent(...)` in `backend/tests/test_extraction_agent.py` and `backend/tests/test_extraction_agent_dicom.py`, pass `vision_llm=llm` or the fixture's existing mocked vision model. Update `backend/tests/test_ingest_script.py` to patch `create_vision_model` and assert the result is passed into the agent.

- [ ] **Step 7: Run runtime, extraction, and upload tests**

Run: `cd backend && uv run pytest tests/test_api_deps.py tests/test_extraction_agent.py tests/test_extraction_agent_dicom.py tests/test_api_consultation.py tests/test_api_knowledge.py tests/test_ingest_script.py -q`

Expected: all selected tests pass; an unsupported legacy DeepSeek vision setting does not prevent text-only infrastructure from building.

- [ ] **Step 8: Commit runtime wiring**

```bash
git add backend/src/medmdt/api/deps.py backend/src/medmdt/extractor/agent.py backend/src/medmdt/api/routes/knowledge.py backend/src/medmdt/api/routes/consultation.py backend/scripts/ingest.py backend/tests/test_api_deps.py backend/tests/test_extraction_agent.py backend/tests/test_extraction_agent_dicom.py backend/tests/test_ingest_script.py
git commit -m "feat(api): wire dedicated vision model through extraction"
```

---

### Task 4: Registry-Backed Settings API and Real Vision Test

**Files:**
- Modify: `backend/src/medmdt/api/routes/settings.py`
- Create: `backend/tests/test_api_settings.py`

**Interfaces:**
- Consumes: `list_provider_metadata()`, `get_provider_base_url()`, `create_vision_model()`, and `ImageParser`.
- Produces: `POST /api/v1/settings/test-vision` using the existing `TestRequest` schema.
- Changes: `GET /api/v1/settings` provider items include `vision_status`.

- [ ] **Step 1: Add failing settings API tests**

Create `backend/tests/test_api_settings.py`:

```python
from unittest.mock import MagicMock, patch

from fastapi.testclient import TestClient

from medmdt.api.app import create_app


def test_settings_provider_metadata_includes_vision_status():
    with patch("medmdt.api.routes.settings.load_llm_settings") as load:
        load.return_value = MagicMock(
            consultation=MagicMock(provider="", model="", api_key=None, base_url=None),
            knowledge=MagicMock(provider="", model="", api_key=None, base_url=None),
            vision=MagicMock(provider="", model="", api_key=None, base_url=None),
            embedding=MagicMock(provider="", model="", api_key=None, base_url=None, dim=1024),
            experts={},
            paddleocr_token=None,
        )
        response = TestClient(create_app()).get("/api/v1/settings")
    assert response.status_code == 200
    providers = {item["key"]: item for item in response.json()["providers"]}
    assert providers["deepseek"]["vision_status"] == "unavailable"
    assert providers["custom"]["vision_status"] == "unknown"


def test_cannot_save_new_deepseek_vision_configuration():
    response = TestClient(create_app()).put(
        "/api/v1/settings",
        json={"vision": {"provider": "deepseek", "model": "deepseek-chat"}},
    )
    assert response.status_code == 400
    assert "暂不支持图像理解" in response.json()["detail"]


@patch("medmdt.api.routes.settings.ImageParser")
@patch("medmdt.api.routes.settings.create_vision_model")
def test_vision_connection_uses_real_image_parser(vision_factory, parser_cls):
    vision_factory.return_value = MagicMock()
    parser_cls.return_value.analyze.return_value = MagicMock(description="红色测试图")
    response = TestClient(create_app()).post(
        "/api/v1/settings/test-vision",
        json={
            "provider": "qwen",
            "model": "qwen-vl-max",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 200
    parser_cls.return_value.analyze.assert_called_once()
    test_bytes = parser_cls.return_value.analyze.call_args.args[0]
    assert test_bytes.startswith(b"\x89PNG")


def test_deepseek_vision_test_returns_capability_error():
    response = TestClient(create_app()).post(
        "/api/v1/settings/test-vision",
        json={
            "provider": "deepseek",
            "model": "deepseek-chat",
            "api_key": "test-key",
        },
    )
    assert response.status_code == 400
    assert "暂不支持图像理解" in response.json()["detail"]
```

- [ ] **Step 2: Run settings tests and confirm missing metadata/route behavior**

Run: `cd backend && uv run pytest tests/test_api_settings.py -q`

Expected: failures because provider metadata is duplicated locally and `/test-vision` does not exist.

- [ ] **Step 3: Replace duplicated provider metadata and Base URLs**

In `settings.py`, import:

```python
from medmdt.llm.provider import (
    PROVIDER_REGISTRY,
    create_chat_model,
    create_vision_model,
    get_provider_base_url,
    list_provider_metadata,
)
from medmdt.llm.errors import VisionError
from medmdt.extractor.parsers.image_parser import ImageParser
```

Delete the local `PROVIDERS` and `_PROVIDER_BASE_URLS`. Return `providers=list_provider_metadata()` in `get_settings()`. Replace `_PROVIDER_BASE_URLS.get(body.provider)` lookups with `get_provider_base_url(body.provider)` after validating registry membership.

- [ ] **Step 4: Reject only newly submitted unavailable vision settings**

At the beginning of `update_settings()` after loading the current settings, add:

```python
if body.vision is not None:
    spec = PROVIDER_REGISTRY.get(body.vision.provider)
    if spec is None:
        raise HTTPException(status_code=400, detail="不支持的视觉 provider")
    if spec.vision_status == "unavailable":
        raise HTTPException(
            status_code=400,
            detail=f"{spec.label} 官方 API 暂不支持图像理解",
        )
```

This check does not reject unrelated settings updates when a legacy saved vision value is still DeepSeek.

- [ ] **Step 5: Add the real image capability endpoint**

Add a helper and route to `settings.py` (including `import uuid`):

```python
from io import BytesIO
from PIL import Image


def _vision_test_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(output, format="PNG")
    return output.getvalue()


@router.post("/test-vision")
def test_vision_connection(body: TestRequest):
    request_id = uuid.uuid4().hex[:12]
    api_key = body.api_key
    if api_key and "****" in api_key:
        api_key = load_llm_settings().vision.api_key

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if body.base_url:
        kwargs["base_url"] = body.base_url

    try:
        llm = create_vision_model(body.provider, body.model, **kwargs)
        result = ImageParser(llm).analyze(
            _vision_test_png(),
            context="视觉连接测试图片；请按要求返回结构化 JSON。",
        )
        return {"status": "ok", "message": f"图片能力测试成功: {result.description[:80]}"}
    except VisionError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.warning(
            "Vision connection test failed request_id=%s provider=%s model=%s error_type=%s",
            request_id,
            body.provider,
            body.model,
            type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail="图片能力测试失败") from None
```

- [ ] **Step 6: Run settings and OpenAPI tests**

Run: `cd backend && uv run pytest tests/test_api_settings.py tests/test_api_app.py -q`

Expected: all tests pass and the OpenAPI document includes `/api/v1/settings/test-vision`.

- [ ] **Step 7: Commit settings API changes**

```bash
git add backend/src/medmdt/api/routes/settings.py backend/tests/test_api_settings.py
git commit -m "feat(settings): expose and test vision capabilities"
```

---

### Task 5: Vision-Aware Settings UI

**Files:**
- Modify: `frontend/src/lib/types.ts`
- Modify: `frontend/src/lib/api.ts`
- Modify: `frontend/src/pages/SettingsPage.tsx`

**Interfaces:**
- Consumes: provider `vision_status` and `POST /api/v1/settings/test-vision` from Task 4.
- Produces: vision-specific provider labels, disabled unavailable choices, a real image-test button, and an untested/unknown warning.

- [ ] **Step 1: Extend frontend types and API client**

Change `ProviderInfo` in `frontend/src/lib/types.ts` to:

```typescript
export type VisionStatus = 'supported' | 'unavailable' | 'unknown'

export interface ProviderInfo {
  key: string
  label: string
  needs_key: boolean
  needs_base_url?: boolean
  vision_status: VisionStatus
}
```

Add to `frontend/src/lib/api.ts` next to `testConnection`:

```typescript
testVisionConnection: (data: { provider: string; model: string; api_key?: string | null; base_url?: string | null }) =>
  request<{ status: string; message: string }>('/api/v1/settings/test-vision', {
    method: 'POST',
    body: JSON.stringify(data),
  }),
```

- [ ] **Step 2: Make `LLMSection` explicitly text or vision aware**

Replace the `showTest` prop with `testMode?: 'text' | 'vision'`. Add:

```typescript
const isVision = testMode === 'vision'
const visionUnavailable = isVision && currentProvider?.vision_status === 'unavailable'
const visionUnknown = isVision && currentProvider?.vision_status === 'unknown'
const [visionVerified, setVisionVerified] = useState(false)

const resetVisionVerification = () => {
  if (isVision) setVisionVerified(false)
}
```

Call `resetVisionVerification()` when provider, model, API key, or Base URL changes. Update the test handler:

```typescript
const handleTest = async () => {
  setTesting(true)
  setMessage(null)
  try {
    const payload = {
      provider,
      model,
      api_key: apiKey || null,
      base_url: baseUrl || null,
    }
    const res = isVision
      ? await api.testVisionConnection(payload)
      : await api.testConnection(payload)
    if (isVision) setVisionVerified(true)
    setMessage({ type: 'success', text: res.message })
  } catch (err) {
    if (isVision) setVisionVerified(false)
    setMessage({ type: 'error', text: err instanceof Error ? err.message : '连接失败' })
  } finally {
    setTesting(false)
  }
}
```

- [ ] **Step 3: Render capability state without hiding DeepSeek**

Render provider options as:

```tsx
{providers.map(p => (
  <option
    key={p.key}
    value={p.key}
    disabled={isVision && p.vision_status === 'unavailable'}
  >
    {p.label}
    {isVision && p.vision_status === 'unavailable' ? '（暂不支持图片）' : ''}
    {isVision && p.vision_status === 'unknown' ? '（需实测）' : ''}
  </option>
))}
```

Keep a legacy saved DeepSeek value visible in the select, show this warning below it, and disable Save/Test while it remains selected:

```tsx
{visionUnavailable && (
  <p className="text-sm text-destructive">
    该 Provider 的官方 API 暂不支持图像理解，请选择其他视觉模型。
  </p>
)}
{isVision && !visionUnavailable && !visionVerified && (
  <p className="text-xs text-muted-foreground">
    当前模型尚未通过真实图片能力测试，保存前建议先测试。
  </p>
)}
{visionUnknown && (
  <p className="text-xs text-amber-600">
    自定义端点的多模态兼容性未知，必须以图片能力测试结果为准。
  </p>
)}
```

Use button state and copy:

```tsx
<Button onClick={handleTest} disabled={testing || visionUnavailable} variant="outline">
  {testing ? '测试中...' : isVision ? '测试图片能力' : '测试连接'}
</Button>
<Button onClick={handleSave} disabled={saving || visionUnavailable}>
  {saving ? '保存中...' : '保存'}
</Button>
```

- [ ] **Step 4: Use the new mode at all settings call sites**

Pass `testMode="text"` to consultation and knowledge `LLMSection` instances. Pass `testMode="vision"` to the image-analysis instance. Change the initial image-analysis state from DeepSeek to empty values:

```typescript
const [vision, setVision] = useState<EndpointData>({
  provider: '',
  model: '',
  api_key: null,
  base_url: null,
})
```

Server-loaded legacy values still replace this initial state after settings load.

- [ ] **Step 5: Build the frontend**

Run: `cd frontend && npm run build`

Expected: TypeScript and Vite build succeed; no provider metadata type errors remain.

- [ ] **Step 6: Commit frontend behavior**

```bash
git add frontend/src/lib/types.ts frontend/src/lib/api.ts frontend/src/pages/SettingsPage.tsx
git commit -m "feat(frontend): add vision-aware model settings"
```

---

### Task 6: Optional Real-Provider Contracts and Full Verification

**Files:**
- Create: `backend/tests/contract/test_vision_providers.py`
- Modify: `backend/pyproject.toml`

**Interfaces:**
- Consumes: the public `create_vision_model()` and `ImageParser` APIs.
- Produces: opt-in contract checks controlled by provider-specific environment variables.

- [ ] **Step 1: Register and add opt-in contract coverage**

Add to `backend/pyproject.toml`:

```toml
[tool.pytest.ini_options]
markers = [
    "contract: calls a real external provider and may incur cost",
]
```

Create `backend/tests/contract/test_vision_providers.py`:

```python
import os
from io import BytesIO

import pytest
from PIL import Image

from medmdt.extractor.parsers.image_parser import ImageParser
from medmdt.llm.provider import create_vision_model


def _test_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(output, format="PNG")
    return output.getvalue()


PROVIDERS = [
    ("openai", "OPENAI_API_KEY", "OPENAI_VISION_MODEL", None),
    ("anthropic", "ANTHROPIC_API_KEY", "ANTHROPIC_VISION_MODEL", None),
    ("qwen", "DASHSCOPE_API_KEY", "QWEN_VISION_MODEL", None),
    ("zhipu", "ZHIPUAI_API_KEY", "ZHIPU_VISION_MODEL", None),
    ("moonshot", "MOONSHOT_API_KEY", "MOONSHOT_VISION_MODEL", None),
    ("custom", "CUSTOM_VISION_API_KEY", "CUSTOM_VISION_MODEL", "CUSTOM_VISION_BASE_URL"),
]


@pytest.mark.contract
@pytest.mark.parametrize("provider,key_env,model_env,base_url_env", PROVIDERS)
def test_real_provider_accepts_image(provider, key_env, model_env, base_url_env):
    api_key = os.getenv(key_env)
    model = os.getenv(model_env)
    base_url = os.getenv(base_url_env) if base_url_env else None
    if not api_key or not model or (provider == "custom" and not base_url):
        pytest.skip(f"{provider} contract credentials are not configured")

    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    llm = create_vision_model(provider, model, **kwargs)
    result = ImageParser(llm).analyze(
        _test_png(),
        context="技术连通性测试，不用于医学诊断。",
    )
    assert result.description
```

- [ ] **Step 2: Run contract tests in default skip mode**

Run: `cd backend && uv run pytest tests/contract/test_vision_providers.py -q`

Expected: six skipped tests when no contract environment variables are configured, with no unknown-marker warning.

- [ ] **Step 3: Run the complete backend suite**

Run: `cd backend && uv run pytest -q`

Expected: all non-contract tests pass and all unconfigured contract cases skip. If a failure predates this work, record its exact test name and confirm it also fails on the plan's starting commit before treating it as unrelated.

- [ ] **Step 4: Run final frontend and diff verification**

Run: `cd frontend && npm run build`

Expected: TypeScript and Vite build pass.

Run: `git diff --check`

Expected: no whitespace errors.

Run: `git status --short`

Expected: only Task 6 contract-test and pytest-marker files are uncommitted at this point.

- [ ] **Step 5: Commit contract coverage**

```bash
git add backend/pyproject.toml backend/tests/contract/test_vision_providers.py
git commit -m "test(vision): add opt-in provider contracts"
```

- [ ] **Step 6: Record final evidence**

Run: `git log -n 6 --oneline`

Expected: the vision implementation is split into registry, parser, runtime wiring, settings API, frontend, and contract-test commits. Include the exact final backend pytest summary and frontend build result in the handoff; do not claim clinical validation.
