import json
import logging
from dataclasses import dataclass, asdict, field
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "llm_settings.json"


@dataclass
class LLMEndpoint:
    provider: str = ""
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None


@dataclass
class EmbeddingEndpoint:
    provider: str = ""
    model: str = ""
    api_key: str | None = None
    base_url: str | None = None
    dim: int = 1024


@dataclass
class ExpertLLM:
    provider: str = ""
    model: str = ""


@dataclass
class LLMSettings:
    consultation: LLMEndpoint = field(default_factory=LLMEndpoint)
    knowledge: LLMEndpoint = field(default_factory=LLMEndpoint)
    vision: LLMEndpoint = field(default_factory=LLMEndpoint)
    embedding: EmbeddingEndpoint = field(default_factory=EmbeddingEndpoint)
    experts: dict[str, ExpertLLM] = field(default_factory=dict)
    paddleocr_token: str | None = None


def load_llm_settings() -> LLMSettings:
    if not CONFIG_PATH.exists():
        return LLMSettings()
    try:
        data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    except Exception:
        logger.warning("Failed to load %s, using defaults", CONFIG_PATH)
        return LLMSettings()

    # Migrate old flat format
    if "provider" in data and "consultation" not in data:
        endpoint = LLMEndpoint(
            provider=data.get("provider", ""),
            model=data.get("model", ""),
            api_key=data.get("api_key"),
            base_url=data.get("base_url"),
        )
        return LLMSettings(
            consultation=endpoint,
            knowledge=LLMEndpoint(
                provider=endpoint.provider,
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
            ),
            vision=LLMEndpoint(
                provider=endpoint.provider,
                model=endpoint.model,
                api_key=endpoint.api_key,
                base_url=endpoint.base_url,
            ),
            paddleocr_token=data.get("paddleocr_token"),
        )

    def _parse_endpoint(d: dict) -> LLMEndpoint:
        return LLMEndpoint(
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            api_key=d.get("api_key"),
            base_url=d.get("base_url"),
        )

    def _parse_embedding(d: dict) -> EmbeddingEndpoint:
        return EmbeddingEndpoint(
            provider=d.get("provider", ""),
            model=d.get("model", ""),
            api_key=d.get("api_key"),
            base_url=d.get("base_url"),
            dim=d.get("dim", 1024),
        )

    experts = {}
    for eid, ecfg in data.get("experts", {}).items():
        experts[eid] = ExpertLLM(
            provider=ecfg.get("provider", ""),
            model=ecfg.get("model", ""),
        )

    return LLMSettings(
        consultation=_parse_endpoint(data.get("consultation", {})),
        knowledge=_parse_endpoint(data.get("knowledge", {})),
        vision=_parse_endpoint(data.get("vision", {})),
        embedding=_parse_embedding(data.get("embedding", {})),
        experts=experts,
        paddleocr_token=data.get("paddleocr_token"),
    )


def save_llm_settings(settings: LLMSettings) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    data = {
        "consultation": asdict(settings.consultation),
        "knowledge": asdict(settings.knowledge),
        "vision": asdict(settings.vision),
        "embedding": asdict(settings.embedding),
        "experts": {eid: asdict(e) for eid, e in settings.experts.items()},
        "paddleocr_token": settings.paddleocr_token,
    }
    CONFIG_PATH.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def get_llm_kwargs(endpoint: LLMEndpoint) -> dict:
    kwargs = {}
    if endpoint.api_key:
        kwargs["api_key"] = endpoint.api_key
    if endpoint.base_url:
        kwargs["base_url"] = endpoint.base_url
    return kwargs


def get_expert_endpoint(settings: LLMSettings, expert_id: str) -> LLMEndpoint:
    ecfg = settings.experts.get(expert_id)
    if ecfg and ecfg.provider and ecfg.model:
        return LLMEndpoint(
            provider=ecfg.provider,
            model=ecfg.model,
            api_key=settings.consultation.api_key,
            base_url=settings.consultation.base_url,
        )
    return settings.consultation


def mask_api_key(key: str | None) -> str | None:
    if not key or len(key) < 8:
        return key
    return key[:4] + "****" + key[-4:]
