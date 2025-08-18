import json
import logging
from dataclasses import dataclass, asdict
from pathlib import Path

logger = logging.getLogger(__name__)

CONFIG_PATH = Path(__file__).resolve().parent.parent.parent.parent / "config" / "llm_settings.json"


@dataclass
class LLMSettings:
    provider: str = "openai"
    model: str = "gpt-4o"
    api_key: str | None = None
    base_url: str | None = None
    paddleocr_token: str | None = None


def load_llm_settings() -> LLMSettings:
    if CONFIG_PATH.exists():
        try:
            data = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
            return LLMSettings(
                provider=data.get("provider", "openai"),
                model=data.get("model", "gpt-4o"),
                api_key=data.get("api_key"),
                base_url=data.get("base_url"),
                paddleocr_token=data.get("paddleocr_token"),
            )
        except Exception:
            logger.warning("Failed to load %s, using defaults", CONFIG_PATH)
    return LLMSettings()


def save_llm_settings(settings: LLMSettings) -> None:
    CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
    CONFIG_PATH.write_text(
        json.dumps(asdict(settings), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def mask_api_key(key: str | None) -> str | None:
    if not key or len(key) < 8:
        return key
    return key[:4] + "****" + key[-4:]
