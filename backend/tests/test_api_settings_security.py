import socket
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from medmdt.api.app import create_app
from medmdt.config.runtime import EmbeddingEndpoint, LLMEndpoint, LLMSettings


PUBLIC_DNS = [
    (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
]


@pytest.fixture(autouse=True)
def private_base_url_opt_in_is_disabled(monkeypatch):
    monkeypatch.delenv("MEDMDT_ALLOW_PRIVATE_BASE_URLS", raising=False)


def _saved_settings() -> LLMSettings:
    return LLMSettings(
        consultation=LLMEndpoint(
            provider="openai", model="gpt", api_key="consultation-saved-key"
        ),
        knowledge=LLMEndpoint(
            provider="qwen", model="qwen", api_key="knowledge-saved-key"
        ),
        vision=LLMEndpoint(
            provider="qwen", model="qwen-vl", api_key="vision-saved-key"
        ),
        embedding=EmbeddingEndpoint(
            provider="openai", model="embed", api_key="embedding-saved-key"
        ),
    )


@pytest.mark.parametrize(
    ("path", "payload"),
    [
        (
            "/api/v1/settings/test",
            {
                "provider": "custom",
                "model": "chat",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
                "credential_scope": "consultation",
            },
        ),
        (
            "/api/v1/settings/test",
            {
                "provider": "custom",
                "model": "chat",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
                "credential_scope": "knowledge",
            },
        ),
        (
            "/api/v1/settings/test-vision",
            {
                "provider": "custom",
                "model": "vision",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            "/api/v1/settings/test-embedding",
            {
                "provider": "custom",
                "model": "embed",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            "/api/v1/settings/models",
            {
                "provider": "custom",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
                "credential_scope": "consultation",
            },
        ),
        (
            "/api/v1/settings/models",
            {
                "provider": "custom",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
                "credential_scope": "knowledge",
            },
        ),
        (
            "/api/v1/settings/models",
            {
                "provider": "custom",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
                "credential_scope": "vision",
            },
        ),
        (
            "/api/v1/settings/embedding-models",
            {
                "provider": "custom",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
            },
        ),
    ],
)
def test_masked_saved_key_cannot_cross_endpoint_target(path, payload):
    downstream = [
        patch("medmdt.api.routes.settings.create_chat_model"),
        patch("medmdt.api.routes.settings.create_vision_model"),
        patch("medmdt.api.routes.settings.http_requests.post"),
        patch("medmdt.api.routes.settings._fetch_openai_compat_models"),
        patch("medmdt.api.routes.settings._fetch_anthropic_models"),
        patch("medmdt.api.routes.settings._fetch_dashscope_embedding_models"),
    ]
    mocks = [item.start() for item in downstream]
    try:
        with (
            patch(
                "medmdt.api.routes.settings.load_llm_settings",
                return_value=_saved_settings(),
            ),
            patch("socket.getaddrinfo", return_value=PUBLIC_DNS),
        ):
            response = TestClient(create_app()).post(path, json=payload)
    finally:
        for item in reversed(downstream):
            item.stop()

    assert response.status_code == 400
    assert response.json()["detail"] == "无法复用已保存的 API Key"
    for mock in mocks:
        mock.assert_not_called()


def test_masked_key_rejects_same_provider_with_different_effective_base_url():
    current = _saved_settings()
    current.consultation = LLMEndpoint(
        provider="custom",
        model="chat",
        api_key="saved-key",
        base_url="https://saved.example/v1/",
    )

    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=current),
        patch("medmdt.api.routes.settings.create_chat_model") as factory,
        patch("socket.getaddrinfo", return_value=PUBLIC_DNS),
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/test",
            json={
                "provider": "custom",
                "model": "chat",
                "api_key": "save****-key",
                "base_url": "https://attacker.example/v1",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "无法复用已保存的 API Key"
    factory.assert_not_called()


def test_masked_key_rejects_when_saved_endpoint_has_no_key():
    current = _saved_settings()
    current.consultation.api_key = None

    with (
        patch("medmdt.api.routes.settings.load_llm_settings", return_value=current),
        patch("medmdt.api.routes.settings.create_chat_model") as factory,
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/test",
            json={
                "provider": "openai",
                "model": "gpt",
                "api_key": "mask****-key",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "无法复用已保存的 API Key"
    factory.assert_not_called()


@pytest.mark.parametrize(
    ("scope", "payload"),
    [
        (
            "consultation",
            {
                "provider": "custom",
                "model": "chat",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            "knowledge",
            {
                "provider": "custom",
                "model": "chat",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            "vision",
            {
                "provider": "custom",
                "model": "vision",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
            },
        ),
        (
            "embedding",
            {
                "provider": "custom",
                "model": "embed",
                "api_key": "mask****-key",
                "base_url": "https://attacker.example/v1",
                "dim": 1024,
            },
        ),
    ],
)
def test_settings_save_cannot_rebind_masked_key(scope, payload):
    with (
        patch(
            "medmdt.api.routes.settings.load_llm_settings",
            return_value=_saved_settings(),
        ),
        patch("medmdt.api.routes.settings.save_llm_settings") as save,
        patch("socket.getaddrinfo", return_value=PUBLIC_DNS),
    ):
        response = TestClient(create_app()).put(
            "/api/v1/settings", json={scope: payload}
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "无法复用已保存的 API Key"
    save.assert_not_called()


@pytest.mark.parametrize(
    "base_url",
    [
        "ftp://example.com/v1",
        "http://user:password@example.com/v1",
        "http://127.0.0.1/v1",
        "http://10.0.0.1/v1",
        "http://169.254.169.254/latest",
        "http://0.0.0.0/v1",
        "http://224.0.0.1/v1",
        "http://192.0.2.1/v1",
        "http://[::1]/v1",
    ],
)
def test_unsafe_user_base_url_is_rejected_without_outbound_call(base_url):
    with (
        patch("medmdt.api.routes.settings._fetch_openai_compat_models") as fetch,
        patch("socket.getaddrinfo", return_value=PUBLIC_DNS),
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={"provider": "custom", "api_key": "new-key", "base_url": base_url},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Base URL 不安全或无效"
    fetch.assert_not_called()


def test_dns_resolution_to_private_address_is_rejected():
    private_dns = [
        (socket.AF_INET, socket.SOCK_STREAM, 6, "", ("10.1.2.3", 443)),
    ]
    with (
        patch("socket.getaddrinfo", return_value=private_dns) as resolver,
        patch("medmdt.api.routes.settings._fetch_openai_compat_models") as fetch,
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "https://public-name.example/v1",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Base URL 不安全或无效"
    resolver.assert_called_once()
    fetch.assert_not_called()


def test_public_custom_url_is_allowed_with_mocked_dns():
    with (
        patch("socket.getaddrinfo", return_value=PUBLIC_DNS) as resolver,
        patch(
            "medmdt.api.routes.settings._fetch_openai_compat_models",
            return_value={"models": []},
        ) as fetch,
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "https://public.example/v1",
            },
        )

    assert response.status_code == 200
    resolver.assert_called_once()
    fetch.assert_called_once_with("https://public.example/v1", "new-key")


def test_private_custom_url_is_allowed_only_with_explicit_opt_in(monkeypatch):
    monkeypatch.setenv("MEDMDT_ALLOW_PRIVATE_BASE_URLS", "true")
    with (
        patch("socket.getaddrinfo") as resolver,
        patch(
            "medmdt.api.routes.settings._fetch_openai_compat_models",
            return_value={"models": []},
        ) as fetch,
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "http://127.0.0.1:8080/v1",
            },
        )

    assert response.status_code == 200
    resolver.assert_not_called()
    fetch.assert_called_once_with("http://127.0.0.1:8080/v1", "new-key")


def test_registry_default_url_bypasses_dns_validation():
    with (
        patch("socket.getaddrinfo") as resolver,
        patch(
            "medmdt.api.routes.settings._fetch_openai_compat_models",
            return_value={"models": []},
        ) as fetch,
    ):
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={"provider": "openai", "api_key": "new-key"},
        )

    assert response.status_code == 200
    resolver.assert_not_called()
    fetch.assert_called_once_with("https://api.openai.com/v1", "new-key")
