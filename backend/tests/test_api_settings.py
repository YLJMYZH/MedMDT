import io
import logging
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from medmdt.api.app import create_app
from medmdt.config.runtime import EmbeddingEndpoint, LLMEndpoint, LLMSettings
from medmdt.llm.errors import VisionRequestError


def _settings(
    *,
    consultation: LLMEndpoint | None = None,
    knowledge: LLMEndpoint | None = None,
    vision: LLMEndpoint | None = None,
) -> LLMSettings:
    return LLMSettings(
        consultation=consultation or LLMEndpoint(),
        knowledge=knowledge or LLMEndpoint(),
        vision=vision or LLMEndpoint(),
        embedding=EmbeddingEndpoint(),
    )


def test_settings_provider_metadata_includes_vision_status():
    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=_settings()),
        patch("medmdt.api.routes.settings._load_expert_names", return_value={}),
    ):
        response = TestClient(create_app()).get("/api/v1/settings")

    assert response.status_code == 200
    providers = {item["key"]: item for item in response.json()["providers"]}
    assert providers["deepseek"]["vision_status"] == "unavailable"
    assert providers["custom"]["vision_status"] == "unknown"


def test_cannot_save_new_deepseek_vision_configuration():
    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=_settings()),
        patch("medmdt.api.routes.settings.save_llm_settings") as save,
    ):
        response = TestClient(create_app()).put(
            "/api/v1/settings",
            json={"vision": {"provider": "deepseek", "model": "deepseek-chat"}},
        )

    assert response.status_code == 400
    assert "暂不支持图像理解" in response.json()["detail"]
    save.assert_not_called()


@pytest.mark.parametrize(
    ("vision", "detail"),
    [
        ({"provider": "", "model": "qwen-vl-max"}, "provider"),
        ({"provider": "unknown", "model": "model"}, "provider"),
        ({"provider": "qwen", "model": ""}, "model"),
        ({"provider": "qwen", "model": "   "}, "model"),
        ({"provider": "custom", "model": "vision", "base_url": ""}, "Base URL"),
        ({"provider": "custom", "model": "vision", "base_url": "   "}, "Base URL"),
    ],
)
def test_invalid_vision_configuration_is_rejected_without_persisting(vision, detail):
    current = _settings(
        vision=LLMEndpoint(provider="qwen", model="qwen-vl-max", api_key="saved")
    )
    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=current),
        patch("medmdt.api.routes.settings.save_llm_settings") as save,
    ):
        response = TestClient(create_app()).put(
            "/api/v1/settings", json={"vision": vision}
        )

    assert response.status_code == 400
    assert detail in response.json()["detail"]
    save.assert_not_called()


def test_legacy_deepseek_vision_does_not_block_unrelated_update():
    current = _settings(
        vision=LLMEndpoint(
            provider="deepseek",
            model="deepseek-chat",
            api_key="legacy-vision-key",
        )
    )
    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=current),
        patch("medmdt.api.routes.settings.save_llm_settings") as save,
    ):
        response = TestClient(create_app()).put(
            "/api/v1/settings",
            json={"consultation": {"provider": "openai", "model": "gpt-4o"}},
        )

    assert response.status_code == 200
    assert save.call_args.args[0].vision == current.vision


def test_masked_vision_key_falls_back_to_saved_vision_key():
    current = _settings(
        vision=LLMEndpoint(
            provider="qwen",
            model="qwen-vl-max",
            api_key="saved-vision-key",
        )
    )
    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=current),
        patch("medmdt.api.routes.settings.save_llm_settings") as save,
    ):
        response = TestClient(create_app()).put(
            "/api/v1/settings",
            json={
                "vision": {
                    "provider": "qwen",
                    "model": "qwen-vl-plus",
                    "api_key": "save****-key",
                }
            },
        )

    assert response.status_code == 200
    assert save.call_args.args[0].vision.api_key == "saved-vision-key"


@patch("medmdt.api.routes.settings.ImageParser", create=True)
@patch("medmdt.api.routes.settings.create_vision_model", create=True)
def test_vision_connection_uses_real_red_png(vision_factory, parser_cls):
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
    assert response.json()["message"] == "图片能力测试成功"
    vision_factory.assert_called_once_with("qwen", "qwen-vl-max", api_key="test-key")
    parser_cls.assert_called_once_with(vision_factory.return_value)
    parser_cls.return_value.analyze.assert_called_once()
    test_bytes = parser_cls.return_value.analyze.call_args.args[0]
    assert test_bytes.startswith(b"\x89PNG")
    with Image.open(io.BytesIO(test_bytes)) as image:
        assert image.size == (32, 32)
        assert image.convert("RGB").getpixel((0, 0)) == (255, 0, 0)


@patch("medmdt.api.routes.settings.ImageParser")
@patch("medmdt.api.routes.settings.create_vision_model")
def test_vision_success_response_does_not_echo_provider_controlled_description(
    vision_factory, parser_cls
):
    sensitive_markers = [
        "secret-key",
        "data:image/png;base64,AAAA",
        "临床背景：患者隐私",
    ]
    parser_cls.return_value.analyze.return_value = MagicMock(
        description=" | ".join(sensitive_markers)
    )

    response = TestClient(create_app()).post(
        "/api/v1/settings/test-vision",
        json={"provider": "qwen", "model": "qwen-vl-max"},
    )

    assert response.status_code == 200
    assert response.json()["message"] == "图片能力测试成功"
    for marker in sensitive_markers:
        assert marker not in response.text


@patch("medmdt.api.routes.settings.ImageParser", create=True)
@patch("medmdt.api.routes.settings.create_vision_model", create=True)
def test_masked_vision_test_key_falls_back_to_saved_key(vision_factory, parser_cls):
    parser_cls.return_value.analyze.return_value = MagicMock(description="测试图")
    current = _settings(vision=LLMEndpoint(api_key="saved-vision-key"))

    with patch("medmdt.api.routes.settings.load_llm_settings", return_value=current):
        response = TestClient(create_app()).post(
            "/api/v1/settings/test-vision",
            json={
                "provider": "qwen",
                "model": "qwen-vl-max",
                "api_key": "save****-key",
            },
        )

    assert response.status_code == 200
    vision_factory.assert_called_once_with(
        "qwen", "qwen-vl-max", api_key="saved-vision-key"
    )


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


@patch("medmdt.api.routes.settings.create_vision_model", create=True)
def test_vision_error_response_and_log_do_not_leak_sensitive_text(
    vision_factory, caplog
):
    sensitive_markers = [
        "raw-sdk-message",
        "secret-key",
        "data:image/png;base64,AAAA",
        "临床背景：患者隐私",
    ]
    vision_factory.side_effect = VisionRequestError(" | ".join(sensitive_markers))

    with caplog.at_level(logging.WARNING, logger="medmdt.api.routes.settings"):
        response = TestClient(create_app()).post(
            "/api/v1/settings/test-vision",
            json={
                "provider": "qwen",
                "model": "qwen-vl-max",
                "api_key": "secret-key",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "图片能力测试失败"
    for marker in sensitive_markers:
        assert marker not in response.text
        assert marker not in caplog.text
    assert "provider=qwen" in caplog.text
    assert "qwen-vl-max" not in caplog.text
    assert "model=<redacted>" in caplog.text
    assert "error_type=VisionRequestError" in caplog.text


@pytest.mark.parametrize(
    ("model", "sensitive_markers"),
    [
        pytest.param(
            "sk-proj-abc123",
            ["sk-proj-abc123"],
            id="ascii-secret-only",
        ),
        pytest.param(
            "data:image/png;base64,AAAA",
            ["data:image/png;base64,AAAA"],
            id="data-url-only",
        ),
        pytest.param(
            "临床背景：患者隐私",
            ["临床背景", "患者隐私"],
            id="clinical-context-only",
        ),
        pytest.param(
            "line-one\nforged-log-entry\x1b",
            ["line-one", "forged-log-entry"],
            id="control-characters-only",
        ),
        pytest.param("m" * 128, ["m" * 128], id="length-boundary"),
        pytest.param("m" * 129, ["m" * 129], id="overlong"),
        pytest.param("qwen-vl-max", ["qwen-vl-max"], id="normal-model"),
    ],
)
@patch("medmdt.api.routes.settings.create_vision_model")
def test_vision_failure_always_redacts_request_model(
    vision_factory, model, sensitive_markers, caplog
):
    vision_factory.side_effect = VisionRequestError("safe failure category")

    with caplog.at_level(logging.WARNING, logger="medmdt.api.routes.settings"):
        response = TestClient(create_app()).post(
            "/api/v1/settings/test-vision",
            json={
                "provider": "qwen",
                "model": model,
                "api_key": "test-key",
            },
        )

    assert response.status_code == 400
    assert model not in response.text
    assert model not in caplog.text
    for marker in sensitive_markers:
        assert marker not in response.text
        assert marker not in caplog.text
    assert "provider=qwen" in caplog.text
    assert "model=<redacted>" in caplog.text
    assert "error_type=VisionRequestError" in caplog.text


def test_openapi_includes_vision_test_route():
    response = TestClient(create_app()).get("/openapi.json")

    assert response.status_code == 200
    assert "/api/v1/settings/test-vision" in response.json()["paths"]


@patch("medmdt.api.routes.settings._fetch_openai_compat_models")
@patch("medmdt.api.routes.settings.get_provider_base_url", create=True)
def test_model_listing_uses_registry_base_url(base_url, fetch_models):
    base_url.return_value = "https://registry.example/v1"
    fetch_models.return_value = {"models": []}

    response = TestClient(create_app()).post(
        "/api/v1/settings/models",
        json={"provider": "openai"},
    )

    assert response.status_code == 200
    base_url.assert_called_once_with("openai")
    fetch_models.assert_called_once_with("https://registry.example/v1", None)


@pytest.mark.parametrize(
    ("scope", "expected_key"),
    [
        ("consultation", "consultation-secret"),
        ("knowledge", "knowledge-secret"),
        ("vision", "vision-secret"),
    ],
)
@patch("medmdt.api.routes.settings._fetch_openai_compat_models")
def test_masked_model_listing_key_uses_endpoint_scope(fetch_models, scope, expected_key):
    fetch_models.return_value = {"models": []}
    current = _settings(
        consultation=LLMEndpoint(api_key="consultation-secret"),
        knowledge=LLMEndpoint(api_key="knowledge-secret"),
        vision=LLMEndpoint(api_key="vision-secret"),
    )

    with patch("medmdt.api.routes.settings.load_llm_settings", return_value=current):
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "openai",
                "api_key": "mask****-key",
                "base_url": "https://models.example/v1",
                "credential_scope": scope,
            },
        )

    assert response.status_code == 200
    fetch_models.assert_called_once_with("https://models.example/v1", expected_key)


def test_model_listing_rejects_unknown_credential_scope():
    response = TestClient(create_app()).post(
        "/api/v1/settings/models",
        json={"provider": "openai", "credential_scope": "embedding"},
    )

    assert response.status_code == 422
