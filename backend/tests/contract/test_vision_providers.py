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
    (
        "custom",
        "CUSTOM_VISION_API_KEY",
        "CUSTOM_VISION_MODEL",
        "CUSTOM_VISION_BASE_URL",
    ),
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
