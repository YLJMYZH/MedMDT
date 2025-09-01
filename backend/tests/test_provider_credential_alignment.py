from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from medmdt.api.app import create_app


@pytest.mark.parametrize(
    "provider", ["anthropic", "deepseek", "moonshot", "unknown"]
)
def test_unsupported_embedding_provider_fails_before_key_or_outbound_use(provider):
    downstream_patches = [
        patch("medmdt.api.routes.settings.save_llm_settings"),
        patch("medmdt.api.routes.settings.http_requests.post"),
        patch("medmdt.api.routes.settings.http_requests.get"),
        patch("medmdt.api.routes.settings._fetch_openai_compat_models"),
        patch("medmdt.api.routes.settings._fetch_dashscope_embedding_models"),
    ]
    downstream = [item.start() for item in downstream_patches]
    try:
        client = TestClient(create_app())
        responses = [
            client.put(
                "/api/v1/settings",
                json={
                    "embedding": {
                        "provider": provider,
                        "model": "embed",
                        "api_key": "unsupported-secret",
                        "dim": 3,
                    }
                },
            ),
            client.post(
                "/api/v1/settings/test-embedding",
                json={
                    "provider": provider,
                    "model": "embed",
                    "api_key": "unsupported-secret",
                },
            ),
            client.post(
                "/api/v1/settings/embedding-models",
                json={
                    "provider": provider,
                    "api_key": "unsupported-secret",
                },
            ),
        ]
    finally:
        for item in reversed(downstream_patches):
            item.stop()

    assert all(response.status_code == 400 for response in responses)
    for mock in downstream:
        mock.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("openai", "https://api.openai.com/v1"),
        ("qwen", "https://dashscope.aliyuncs.com/compatible-mode/v1"),
        ("zhipu", "https://open.bigmodel.cn/api/paas/v4"),
        ("custom", "https://embedding.example/v1"),
    ],
)
def test_supported_embedding_connection_uses_matching_effective_target(
    provider, base_url, monkeypatch
):
    if provider == "custom":
        monkeypatch.setenv("MEDMDT_ALLOWED_BASE_URL_HOSTS", "embedding.example")
    response_mock = MagicMock(status_code=200)
    response_mock.json.return_value = {"data": [{"embedding": [0.1, 0.2]}]}
    payload = {"provider": provider, "model": "embed", "api_key": "key"}
    if provider == "custom":
        payload["base_url"] = base_url

    with patch(
        "medmdt.api.routes.settings.http_requests.post",
        return_value=response_mock,
    ) as request:
        response = TestClient(create_app()).post(
            "/api/v1/settings/test-embedding", json=payload
        )

    assert response.status_code == 200
    assert request.call_args.args[0] == f"{base_url}/embeddings"


@pytest.mark.parametrize("provider", ["anthropic", "deepseek", "moonshot"])
def test_runtime_rejects_unsupported_embedding_before_any_construction(provider):
    from medmdt.api.deps import build_infrastructure

    runtime = SimpleNamespace(
        knowledge=SimpleNamespace(
            provider="openai", model="text", api_key="knowledge-key", base_url=None
        ),
        vision=SimpleNamespace(
            provider="qwen", model="vision", api_key="vision-key", base_url=None
        ),
        embedding=SimpleNamespace(
            provider=provider,
            model="embed",
            api_key="unsupported-secret",
            base_url=None,
            dim=3,
        ),
    )
    with (
        patch("medmdt.config.settings.get_settings", return_value=MagicMock()),
        patch("medmdt.config.runtime.load_llm_settings", return_value=runtime),
        patch("medmdt.llm.provider.create_chat_model") as chat_factory,
        patch("medmdt.llm.provider.create_vision_model") as vision_factory,
        patch("medmdt.knowledge.graph_store.GraphStore") as graph_cls,
        patch("medmdt.knowledge.vector_store.VectorStore") as vector_cls,
        patch("medmdt.knowledge.keyword_store.KeywordStore") as keyword_cls,
        patch("medmdt.knowledge.retriever.FusionRetriever"),
        patch("langchain_openai.OpenAIEmbeddings") as embedding_cls,
    ):
        with pytest.raises(ValueError, match="embedding"):
            build_infrastructure()

    chat_factory.assert_not_called()
    vision_factory.assert_not_called()
    graph_cls.assert_not_called()
    vector_cls.assert_not_called()
    keyword_cls.assert_not_called()
    embedding_cls.assert_not_called()
