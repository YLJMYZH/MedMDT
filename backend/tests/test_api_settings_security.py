from unittest.mock import MagicMock, call, patch

import pytest
from fastapi.testclient import TestClient

from medmdt.api.app import create_app
from medmdt.config.runtime import EmbeddingEndpoint, LLMEndpoint, LLMSettings


@pytest.fixture(autouse=True)
def outbound_base_url_allowlists_are_disabled(monkeypatch):
    monkeypatch.delenv("MEDMDT_ALLOW_PRIVATE_BASE_URLS", raising=False)
    monkeypatch.delenv("MEDMDT_ALLOWED_BASE_URL_HOSTS", raising=False)
    monkeypatch.delenv("MEDMDT_ALLOWED_BASE_URL_CIDRS", raising=False)


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
    ):
        response = TestClient(create_app()).put(
            "/api/v1/settings", json={scope: payload}
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "无法复用已保存的 API Key"
    save.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("custom", "ftp://example.com/v1"),
        ("custom", "https://user:password@example.com/v1"),
        ("custom", "https://example.com/v1?target=private"),
        ("custom", "https://example.com/v1#fragment"),
        ("custom", "https://example.com:99999/v1"),
        ("custom", "http://93.184.216.34/v1"),
        ("custom", "https://169.254.169.254/latest"),
        ("custom", "https://224.0.0.1/v1"),
        ("openai", "https://93.184.216.34/v1"),
    ],
)
def test_unsafe_user_base_url_is_rejected_without_outbound_call(provider, base_url):
    with patch("medmdt.api.routes.settings._fetch_openai_compat_models") as fetch:
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={"provider": provider, "api_key": "new-key", "base_url": base_url},
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Base URL 不安全或无效"
    fetch.assert_not_called()


def test_untrusted_custom_hostname_is_denied_without_dns_or_outbound_call():
    with patch("medmdt.api.routes.settings._fetch_openai_compat_models") as fetch:
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "https://untrusted.example/v1",
            },
        )

    assert response.status_code == 400
    assert response.json()["detail"] == "Base URL 不安全或无效"
    fetch.assert_not_called()


def test_exact_deployment_allowlisted_custom_hostname_is_allowed(monkeypatch):
    monkeypatch.setenv(
        "MEDMDT_ALLOWED_BASE_URL_HOSTS", "other.example, PUBLIC.EXAMPLE."
    )
    with patch(
        "medmdt.api.routes.settings._fetch_openai_compat_models",
        return_value={"models": []},
    ) as fetch:
        response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "https://public.example/v1",
            },
        )
        insecure_response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "http://public.example/v1",
            },
        )

    assert response.status_code == 200
    assert insecure_response.status_code == 400
    fetch.assert_called_once_with("https://public.example/v1", "new-key")


def test_private_cidr_is_precise_and_special_ranges_remain_denied(monkeypatch):
    monkeypatch.setenv("MEDMDT_ALLOW_PRIVATE_BASE_URLS", "true")
    monkeypatch.setenv("MEDMDT_ALLOWED_BASE_URL_CIDRS", "10.20.30.0/24")
    with patch(
        "medmdt.api.routes.settings._fetch_openai_compat_models",
        return_value={"models": []},
    ) as fetch:
        allowed = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "http://10.20.30.40:8080/v1",
            },
        )
        outside_cidr = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "http://10.20.31.40:8080/v1",
            },
        )

        monkeypatch.setenv("MEDMDT_ALLOWED_BASE_URL_CIDRS", "127.0.0.1/32")
        loopback = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "http://127.0.0.1:8080/v1",
            },
        )

        monkeypatch.setenv("MEDMDT_ALLOWED_BASE_URL_CIDRS", "0.0.0.0/0,::/0")
        special_responses = [
            TestClient(create_app()).post(
                "/api/v1/settings/models",
                json={
                    "provider": "custom",
                    "api_key": "new-key",
                    "base_url": f"https://{host}/v1",
                },
            )
            for host in ("0.0.0.0", "169.254.169.254", "192.0.2.1")
        ]

    assert allowed.status_code == 200
    assert outside_cidr.status_code == 400
    assert loopback.status_code == 200
    assert all(response.status_code == 400 for response in special_responses)
    assert fetch.call_args_list == [
        call("http://10.20.30.40:8080/v1", "new-key"),
        call("http://127.0.0.1:8080/v1", "new-key"),
    ]


def test_global_literal_https_and_registry_default_are_trusted():
    with patch(
        "medmdt.api.routes.settings._fetch_openai_compat_models",
        return_value={"models": []},
    ) as fetch:
        literal_response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "custom",
                "api_key": "new-key",
                "base_url": "https://93.184.216.34/v1",
            },
        )
        registry_response = TestClient(create_app()).post(
            "/api/v1/settings/models",
            json={
                "provider": "openai",
                "api_key": "new-key",
                "base_url": "https://api.openai.com/v1/",
            },
        )

    assert literal_response.status_code == 200
    assert registry_response.status_code == 200
    assert fetch.call_args_list == [
        call("https://93.184.216.34/v1", "new-key"),
        call("https://api.openai.com/v1/", "new-key"),
    ]
