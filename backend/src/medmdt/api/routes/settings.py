import logging
from dataclasses import asdict

import requests as http_requests
from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from medmdt.config.runtime import (
    LLMSettings,
    load_llm_settings,
    save_llm_settings,
    mask_api_key,
)
from medmdt.llm.provider import create_chat_model, PROVIDER_REGISTRY, _OPENAI_COMPAT_URLS

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1/settings", tags=["settings"])

PROVIDERS = [
    {"key": "openai", "label": "OpenAI", "needs_key": True},
    {"key": "anthropic", "label": "Anthropic", "needs_key": True},
    {"key": "deepseek", "label": "DeepSeek", "needs_key": True},
    {"key": "qwen", "label": "Qwen (通义千问)", "needs_key": True},
    {"key": "zhipu", "label": "ZhiPu (智谱)", "needs_key": True},
    {"key": "moonshot", "label": "Moonshot (月之暗面)", "needs_key": True},
    {"key": "ollama", "label": "Ollama (本地)", "needs_key": False},
    {"key": "custom", "label": "自定义 (OpenAI 兼容)", "needs_key": False, "needs_base_url": True},
]


class SettingsResponse(BaseModel):
    provider: str
    model: str
    api_key: str | None
    base_url: str | None
    paddleocr_token: str | None
    providers: list[dict]


class SettingsUpdate(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None
    paddleocr_token: str | None = None


class TestRequest(BaseModel):
    provider: str
    model: str
    api_key: str | None = None
    base_url: str | None = None


@router.get("", response_model=SettingsResponse)
def get_settings():
    settings = load_llm_settings()
    return SettingsResponse(
        provider=settings.provider,
        model=settings.model,
        api_key=mask_api_key(settings.api_key),
        base_url=settings.base_url,
        paddleocr_token=mask_api_key(settings.paddleocr_token),
        providers=PROVIDERS,
    )


@router.put("")
def update_settings(body: SettingsUpdate):
    if body.provider not in PROVIDER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"不支持的 provider: {body.provider}")

    current = load_llm_settings()
    api_key = body.api_key
    if api_key and "****" in api_key:
        api_key = current.api_key
    paddleocr_token = body.paddleocr_token
    if paddleocr_token and "****" in paddleocr_token:
        paddleocr_token = current.paddleocr_token

    settings = LLMSettings(
        provider=body.provider,
        model=body.model,
        api_key=api_key,
        base_url=body.base_url,
        paddleocr_token=paddleocr_token,
    )
    save_llm_settings(settings)
    return {"status": "ok", "message": "设置已保存"}


@router.post("/test")
def test_connection(body: TestRequest):
    if body.provider not in PROVIDER_REGISTRY:
        raise HTTPException(status_code=400, detail=f"不支持的 provider: {body.provider}")

    api_key = body.api_key
    if api_key and "****" in api_key:
        current = load_llm_settings()
        api_key = current.api_key

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


_PROVIDER_BASE_URLS: dict[str, str] = {
    "openai": "https://api.openai.com/v1",
    **_OPENAI_COMPAT_URLS,
    "ollama": "http://127.0.0.1:11434/v1",
}


class ModelsRequest(BaseModel):
    provider: str
    api_key: str | None = None
    base_url: str | None = None


@router.post("/models")
def list_models(body: ModelsRequest):
    api_key = body.api_key
    if api_key and "****" in api_key:
        current = load_llm_settings()
        api_key = current.api_key

    if body.provider == "anthropic":
        return _fetch_anthropic_models(api_key)

    base_url = body.base_url or _PROVIDER_BASE_URLS.get(body.provider)
    if not base_url:
        raise HTTPException(status_code=400, detail="需要提供 Base URL")

    return _fetch_openai_compat_models(base_url, api_key)


def _fetch_openai_compat_models(base_url: str, api_key: str | None) -> dict:
    url = f"{base_url.rstrip('/')}/models"
    headers = {}
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    try:
        resp = http_requests.get(url, headers=headers, timeout=10)
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
            "https://api.anthropic.com/v1/models", headers=headers, timeout=10,
        )
        resp.raise_for_status()
        data = resp.json().get("data", [])
        models = sorted(item["id"] for item in data if "id" in item)
        return {"models": models}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"获取模型列表失败: {str(e)}")
