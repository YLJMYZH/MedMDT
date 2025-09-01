import logging
import uuid
from io import BytesIO
from typing import Literal

import requests as http_requests
from fastapi import APIRouter, HTTPException
from PIL import Image
from pydantic import BaseModel

from medmdt.config.runtime import (
    LLMSettings,
    LLMEndpoint,
    EmbeddingEndpoint,
    ExpertLLM,
    load_llm_settings,
    save_llm_settings,
    mask_api_key,
)
from medmdt.extractor.parsers.image_parser import ImageParser
from medmdt.llm.network_policy import (
    canonical_effective_base_url,
    validate_user_base_url,
)
from medmdt.llm.errors import VisionError
from medmdt.llm.provider import (
    PROVIDER_REGISTRY,
    create_chat_model,
    create_vision_model,
    get_embedding_base_url,
    get_provider_base_url,
    list_provider_metadata,
    resolve_embedding_api_key,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])


def _reject_redirect_response(response) -> None:
    if 300 <= response.status_code < 400:
        raise http_requests.HTTPError("redirect response rejected")


class EndpointData(BaseModel):
    provider: str = ""
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None


class EmbeddingData(BaseModel):
    provider: str = ""
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    dim: int = 1024


class ExpertLLMData(BaseModel):
    provider: str = ""
    model: str = ""


class SettingsResponse(BaseModel):
    consultation: EndpointData
    knowledge: EndpointData
    vision: EndpointData
    embedding: EmbeddingData
    experts: dict[str, ExpertLLMData]
    expert_names: dict[str, str]
    paddleocr_token: str | None
    providers: list[dict]


class SettingsUpdate(BaseModel):
    consultation: EndpointData | None = None
    knowledge: EndpointData | None = None
    vision: EndpointData | None = None
    embedding: EmbeddingData | None = None
    experts: dict[str, ExpertLLMData] | None = None
    paddleocr_token: str | None = None


class TestRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    credential_scope: Literal["consultation", "knowledge", "vision"] = "consultation"


def _load_expert_names() -> dict[str, str]:
    from medmdt.mdt.experts.factory import load_expert_configs
    try:
        configs = load_expert_configs("config/experts.yaml")
        return {cfg.expert_id: cfg.name for cfg in configs}
    except Exception:
        return {}


def _mask_endpoint(ep: LLMEndpoint) -> EndpointData:
    return EndpointData(
        provider=ep.provider,
        model=ep.model,
        api_key=mask_api_key(ep.api_key),
        base_url=ep.base_url,
    )


def _mask_embedding(ep: EmbeddingEndpoint) -> EmbeddingData:
    return EmbeddingData(
        provider=ep.provider,
        model=ep.model,
        api_key=mask_api_key(ep.api_key),
        base_url=ep.base_url,
        dim=ep.dim,
    )


@router.get("", response_model=SettingsResponse)
def get_settings():
    settings = load_llm_settings()
    expert_names = _load_expert_names()
    experts_data = {}
    for eid in expert_names:
        ecfg = settings.experts.get(eid, ExpertLLM())
        experts_data[eid] = ExpertLLMData(provider=ecfg.provider, model=ecfg.model)

    return SettingsResponse(
        consultation=_mask_endpoint(settings.consultation),
        knowledge=_mask_endpoint(settings.knowledge),
        vision=_mask_endpoint(settings.vision),
        embedding=_mask_embedding(settings.embedding),
        experts=experts_data,
        expert_names=expert_names,
        paddleocr_token=mask_api_key(settings.paddleocr_token),
        providers=list_provider_metadata(),
    )


CredentialScope = Literal["consultation", "knowledge", "vision", "embedding"]


def _require_embedding_base_url(provider: str, base_url: str | None) -> str:
    try:
        return get_embedding_base_url(provider, base_url)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="该 Provider 暂不支持向量模型或缺少有效 Base URL",
        ) from None


def _require_embedding_api_key(provider: str, api_key: str | None) -> str:
    try:
        return resolve_embedding_api_key(provider, api_key)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail="该 Provider 缺少有效的向量 API Key",
        ) from None


def _prepare_outbound_api_key(
    api_key: str | None,
    provider: str,
    base_url: str | None,
    credential_scope: CredentialScope,
) -> str | None:
    if api_key and "****" in api_key:
        current = load_llm_settings()
        saved = (
            current.embedding
            if credential_scope == "embedding"
            else getattr(current, credential_scope)
        )
        reusable = False
        try:
            reusable = (
                bool(saved.api_key)
                and provider == saved.provider
                and canonical_effective_base_url(provider, base_url)
                == canonical_effective_base_url(saved.provider, saved.base_url)
            )
        except (TypeError, ValueError):
            reusable = False
        if not reusable:
            raise HTTPException(
                status_code=400,
                detail="无法复用已保存的 API Key",
            )
        api_key = saved.api_key

    try:
        validate_user_base_url(provider, base_url)
    except (TypeError, ValueError):
        raise HTTPException(status_code=400, detail="Base URL 不安全或无效") from None
    return api_key


@router.put("")
def update_settings(body: SettingsUpdate):
    if body.embedding is not None:
        _require_embedding_base_url(
            body.embedding.provider,
            body.embedding.base_url,
        )

    if body.vision is not None:
        spec = PROVIDER_REGISTRY.get(body.vision.provider)
        if spec is None:
            raise HTTPException(status_code=400, detail="不支持的视觉 provider")
        if not body.vision.model.strip():
            raise HTTPException(status_code=400, detail="视觉 model 不能为空")
        if spec.vision_status == "unavailable":
            raise HTTPException(
                status_code=400,
                detail=f"{spec.label} 官方 API 暂不支持图像理解",
            )
        if spec.needs_base_url and not (body.vision.base_url or "").strip():
            raise HTTPException(status_code=400, detail="视觉 Provider 需要提供 Base URL")

    current = load_llm_settings()

    def _update_endpoint(
        new: EndpointData | None,
        cur: LLMEndpoint,
        scope: Literal["consultation", "knowledge", "vision"],
    ) -> LLMEndpoint:
        if not new:
            return cur
        return LLMEndpoint(
            provider=new.provider,
            model=new.model,
            api_key=_prepare_outbound_api_key(
                new.api_key, new.provider, new.base_url, scope
            ),
            base_url=new.base_url,
        )

    def _update_embedding(new: EmbeddingData | None, cur: EmbeddingEndpoint) -> EmbeddingEndpoint:
        if not new:
            return cur
        return EmbeddingEndpoint(
            provider=new.provider,
            model=new.model,
            api_key=_prepare_outbound_api_key(
                new.api_key, new.provider, new.base_url, "embedding"
            ),
            base_url=new.base_url,
            dim=new.dim,
        )

    consultation = _update_endpoint(
        body.consultation, current.consultation, "consultation"
    )
    knowledge = _update_endpoint(body.knowledge, current.knowledge, "knowledge")
    vision = _update_endpoint(body.vision, current.vision, "vision")
    embedding = _update_embedding(body.embedding, current.embedding)

    experts = current.experts.copy()
    if body.experts is not None:
        for eid, ecfg in body.experts.items():
            experts[eid] = ExpertLLM(provider=ecfg.provider, model=ecfg.model)

    paddleocr_token = body.paddleocr_token
    if paddleocr_token and "****" in paddleocr_token:
        paddleocr_token = current.paddleocr_token
    elif paddleocr_token is None:
        paddleocr_token = current.paddleocr_token

    settings = LLMSettings(
        consultation=consultation,
        knowledge=knowledge,
        vision=vision,
        embedding=embedding,
        experts=experts,
        paddleocr_token=paddleocr_token,
    )
    save_llm_settings(settings)
    return {"status": "ok", "message": "设置已保存"}


@router.post("/test")
def test_connection(body: TestRequest):
    if body.provider not in PROVIDER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"不支持的 provider: {body.provider}")

    if body.credential_scope not in {"consultation", "knowledge"}:
        raise HTTPException(status_code=400, detail="不支持的凭据范围")
    api_key = _prepare_outbound_api_key(
        body.api_key,
        body.provider,
        body.base_url,
        body.credential_scope,
    )

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if body.base_url:
        kwargs["base_url"] = body.base_url

    try:
        llm = create_chat_model(body.provider, body.model, **kwargs)
        from langchain_core.messages import HumanMessage
        response = llm.invoke([HumanMessage(content="请回复'连接成功'")])
        return {"status": "ok", "message": f"连接成功: {response.content[:100]}"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"连接失败: {str(e)}")


def _vision_test_png() -> bytes:
    output = BytesIO()
    Image.new("RGB", (32, 32), color="red").save(output, format="PNG")
    return output.getvalue()


@router.post("/test-vision")
def test_vision_connection(body: TestRequest):
    request_id = uuid.uuid4().hex[:12]
    spec = PROVIDER_REGISTRY.get(body.provider)
    if spec is None:
        raise HTTPException(status_code=400, detail="不支持的视觉 provider")
    if spec.vision_status == "unavailable":
        raise HTTPException(
            status_code=400,
            detail=f"{spec.label} 官方 API 暂不支持图像理解",
        )

    api_key = _prepare_outbound_api_key(
        body.api_key,
        body.provider,
        body.base_url,
        "vision",
    )

    kwargs = {}
    if api_key:
        kwargs["api_key"] = api_key
    if body.base_url:
        kwargs["base_url"] = body.base_url

    try:
        llm = create_vision_model(body.provider, body.model, **kwargs)
        ImageParser(llm).analyze(
            _vision_test_png(),
            context="视觉连接测试图片；请按要求返回结构化 JSON。",
        )
        return {"status": "ok", "message": "图片能力测试成功"}
    except VisionError as exc:
        logger.warning(
            "Vision connection test failed request_id=%s provider=%s model=%s error_type=%s",
            request_id,
            body.provider,
            "<redacted>",
            type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail="图片能力测试失败") from None
    except Exception as exc:
        logger.warning(
            "Vision connection test failed request_id=%s provider=%s model=%s error_type=%s",
            request_id,
            body.provider,
            "<redacted>",
            type(exc).__name__,
        )
        raise HTTPException(status_code=400, detail="图片能力测试失败") from None


@router.post("/test-embedding")
def test_embedding_connection(body: TestRequest):
    if body.provider not in PROVIDER_REGISTRY:
        raise HTTPException(status_code=400, detail="不支持的 provider")
    base_url = _require_embedding_base_url(body.provider, body.base_url)
    api_key = _prepare_outbound_api_key(
        body.api_key,
        body.provider,
        body.base_url,
        "embedding",
    )
    api_key = _require_embedding_api_key(body.provider, api_key)
    url = f"{base_url.rstrip('/')}/embeddings"
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    try:
        resp = http_requests.post(
            url, headers=headers, timeout=15,
            json={"model": body.model, "input": "连接测试"},
            allow_redirects=False,
        )
        _reject_redirect_response(resp)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        if data and "embedding" in data[0]:
            dim = len(data[0]["embedding"])
            return {"status": "ok", "message": f"连接成功，向量维度: {dim}"}
        return {"status": "ok", "message": "连接成功"}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"连接失败: {str(e)}")


class ModelsRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None
    credential_scope: Literal["consultation", "knowledge", "vision"] = "consultation"


@router.post("/models")
def list_models(body: ModelsRequest):
    if body.provider not in PROVIDER_REGISTRY:
        raise HTTPException(status_code=400, detail="不支持的 provider")
    api_key = _prepare_outbound_api_key(
        body.api_key,
        body.provider,
        body.base_url,
        body.credential_scope,
    )

    if body.provider == "anthropic":
        return _fetch_anthropic_models(api_key)

    base_url = body.base_url or get_provider_base_url(body.provider)
    if not base_url:
        raise HTTPException(status_code=400, detail="需要提供 Base URL")

    return _fetch_openai_compat_models(base_url, api_key)


@router.post("/embedding-models")
def list_embedding_models(body: ModelsRequest):
    if body.provider not in PROVIDER_REGISTRY:
        raise HTTPException(status_code=400, detail="不支持的 provider")
    base_url = _require_embedding_base_url(body.provider, body.base_url)
    api_key = _prepare_outbound_api_key(
        body.api_key,
        body.provider,
        body.base_url,
        "embedding",
    )
    api_key = _require_embedding_api_key(body.provider, api_key)

    if body.provider == "qwen":
        return _fetch_dashscope_embedding_models(api_key)

    return _fetch_openai_compat_models(base_url, api_key)


def _fetch_dashscope_embedding_models(api_key: str | None) -> dict:
    """Fetch models from DashScope native API (paginated)."""
    if not api_key:
        raise HTTPException(status_code=400, detail="需要 API Key")
    headers = {"Authorization": f"Bearer {api_key}"}
    models = []
    page = 1
    try:
        while True:
            resp = http_requests.get(
                f"https://dashscope.aliyuncs.com/api/v1/models?page_size=100&page_no={page}",
                headers=headers, timeout=15, allow_redirects=False,
            )
            _reject_redirect_response(resp)
            resp.raise_for_status()
            output = resp.json().get("output", {})
            items = output.get("models", [])
            if not items:
                break
            for item in items:
                mid = item.get("model", "")
                if mid:
                    models.append(mid)
            total = output.get("total", 0)
            if page * 100 >= total:
                break
            page += 1
        return {"models": sorted(models)}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取模型列表失败: {str(e)}")


def _fetch_openai_compat_models(base_url: str, api_key: str | None) -> dict:
    url = f"{base_url.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = http_requests.get(
            url, headers=headers, timeout=10, allow_redirects=False
        )
        _reject_redirect_response(resp)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = sorted(item["id"] for item in data if "id" in item)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取模型列表失败: {str(e)}")


def _fetch_anthropic_models(api_key: str | None) -> dict:
    if not api_key:
        raise HTTPException(status_code=400, detail="Anthropic 需要 API Key")
    headers = {
        "x-api-key": api_key,
        "anthropic-version": "2023-06-01",
    }
    try:
        resp = http_requests.get(
            "https://api.anthropic.com/v1/models",
            headers=headers,
            timeout=10,
            allow_redirects=False,
        )
        _reject_redirect_response(resp)
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = sorted(item["id"] for item in data if "id" in item)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取模型列表失败: {str(e)}")
