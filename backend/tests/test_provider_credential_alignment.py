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
        patch("medmdt.api.routes.settings.load_llm_settings"),
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
                        "api_key": "fake****masked",
                        "dim": 3,
                    }
                },
            ),
            client.post(
                "/api/v1/settings/test-embedding",
                json={
                    "provider": provider,
                    "model": "embed",
                    "api_key": "fake****masked",
                },
            ),
            client.post(
                "/api/v1/settings/embedding-models",
                json={
                    "provider": provider,
                    "api_key": "fake****masked",
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


def _runtime_with_embedding(provider, api_key=None, base_url=None):
    return SimpleNamespace(
        knowledge=SimpleNamespace(
            provider="openai", model="text", api_key="knowledge-key", base_url=None
        ),
        vision=SimpleNamespace(
            provider="qwen", model="vision", api_key="vision-key", base_url=None
        ),
        embedding=SimpleNamespace(
            provider=provider,
            model="embed",
            api_key=api_key,
            base_url=base_url,
            dim=3,
        ),
    )


@pytest.mark.parametrize(
    ("provider", "key_env", "base_url"),
    [
        ("openai", "OPENAI_API_KEY", "https://api.openai.com/v1"),
        (
            "qwen",
            "DASHSCOPE_API_KEY",
            "https://dashscope.aliyuncs.com/compatible-mode/v1",
        ),
        ("zhipu", "ZHIPUAI_API_KEY", "https://open.bigmodel.cn/api/paas/v4"),
    ],
)
def test_runtime_embedding_binds_only_provider_environment_key(
    provider, key_env, base_url, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv(key_env, f"{provider}-embedding-key")
    runtime = _runtime_with_embedding(provider)
    sync_client = MagicMock(follow_redirects=False)
    async_client = MagicMock(follow_redirects=False)

    with (
        patch("medmdt.config.settings.get_settings", return_value=MagicMock()),
        patch("medmdt.config.runtime.load_llm_settings", return_value=runtime),
        patch("medmdt.llm.provider.create_chat_model", return_value=MagicMock()),
        patch("medmdt.llm.provider.create_vision_model", return_value=MagicMock()),
        patch("medmdt.knowledge.graph_store.GraphStore"),
        patch("medmdt.knowledge.vector_store.VectorStore"),
        patch("medmdt.knowledge.keyword_store.KeywordStore"),
        patch("medmdt.knowledge.retriever.FusionRetriever"),
        patch("medmdt.llm.http_clients.get_shared_http_client", return_value=sync_client),
        patch(
            "medmdt.llm.http_clients.get_shared_async_http_client",
            return_value=async_client,
        ),
        patch("langchain_openai.OpenAIEmbeddings") as embedding_cls,
    ):
        from medmdt.api.deps import build_infrastructure

        build_infrastructure()["embed_fn"](["document"])

    kwargs = embedding_cls.call_args.kwargs
    assert kwargs["api_key"] == f"{provider}-embedding-key"
    assert kwargs["openai_api_base"] == base_url


@pytest.mark.parametrize(
    ("provider", "key_env", "base_url"),
    [
        ("qwen", "DASHSCOPE_API_KEY", None),
        ("zhipu", "ZHIPUAI_API_KEY", None),
        ("custom", None, "https://custom-embedding.example/v1"),
    ],
)
def test_runtime_embedding_rejects_ambient_openai_key_before_construction(
    provider, key_env, base_url, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    if key_env:
        monkeypatch.delenv(key_env, raising=False)
    if provider == "custom":
        monkeypatch.setenv(
            "MEDMDT_ALLOWED_BASE_URL_HOSTS", "custom-embedding.example"
        )
    runtime = _runtime_with_embedding(provider, base_url=base_url)

    with (
        patch("medmdt.config.settings.get_settings", return_value=MagicMock()),
        patch("medmdt.config.runtime.load_llm_settings", return_value=runtime),
        patch("medmdt.llm.provider.create_chat_model") as chat_factory,
        patch("medmdt.llm.provider.create_vision_model") as vision_factory,
        patch("medmdt.knowledge.graph_store.GraphStore") as graph_cls,
        patch("medmdt.knowledge.vector_store.VectorStore") as vector_cls,
        patch("medmdt.knowledge.keyword_store.KeywordStore") as keyword_cls,
        patch("medmdt.knowledge.retriever.FusionRetriever"),
        patch("medmdt.llm.http_clients.get_shared_http_client") as sync_client,
        patch("medmdt.llm.http_clients.get_shared_async_http_client") as async_client,
        patch("langchain_openai.OpenAIEmbeddings") as embedding_cls,
    ):
        from medmdt.api.deps import build_infrastructure

        with pytest.raises(ValueError, match="embedding API key"):
            build_infrastructure()

    chat_factory.assert_not_called()
    vision_factory.assert_not_called()
    graph_cls.assert_not_called()
    vector_cls.assert_not_called()
    keyword_cls.assert_not_called()
    sync_client.assert_not_called()
    async_client.assert_not_called()
    embedding_cls.assert_not_called()


@pytest.mark.parametrize(
    ("provider", "base_url"),
    [
        ("qwen", None),
        ("custom", "https://custom-embedding.example/v1"),
    ],
)
def test_runtime_embedding_saved_explicit_key_wins_over_environment(
    provider, base_url, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "ambient-qwen-key")
    if provider == "custom":
        monkeypatch.setenv(
            "MEDMDT_ALLOWED_BASE_URL_HOSTS", "custom-embedding.example"
        )
    runtime = _runtime_with_embedding(
        provider, api_key="saved-explicit-key", base_url=base_url
    )
    sync_client = MagicMock(follow_redirects=False)
    async_client = MagicMock(follow_redirects=False)

    with (
        patch("medmdt.config.settings.get_settings", return_value=MagicMock()),
        patch("medmdt.config.runtime.load_llm_settings", return_value=runtime),
        patch("medmdt.llm.provider.create_chat_model", return_value=MagicMock()),
        patch("medmdt.llm.provider.create_vision_model", return_value=MagicMock()),
        patch("medmdt.knowledge.graph_store.GraphStore"),
        patch("medmdt.knowledge.vector_store.VectorStore"),
        patch("medmdt.knowledge.keyword_store.KeywordStore"),
        patch("medmdt.knowledge.retriever.FusionRetriever"),
        patch("medmdt.llm.http_clients.get_shared_http_client", return_value=sync_client),
        patch(
            "medmdt.llm.http_clients.get_shared_async_http_client",
            return_value=async_client,
        ),
        patch("langchain_openai.OpenAIEmbeddings") as embedding_cls,
    ):
        from medmdt.api.deps import build_infrastructure

        build_infrastructure()["embed_fn"](["document"])

    assert embedding_cls.call_args.kwargs["api_key"] == "saved-explicit-key"


@pytest.mark.parametrize(
    ("provider", "key_env"),
    [
        ("openai", "OPENAI_API_KEY"),
        ("qwen", "DASHSCOPE_API_KEY"),
        ("zhipu", "ZHIPUAI_API_KEY"),
    ],
)
def test_embedding_connection_binds_provider_environment_key(
    provider, key_env, monkeypatch
):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv(key_env, f"{provider}-embedding-key")
    response_mock = MagicMock(status_code=200)
    response_mock.json.return_value = {"data": [{"embedding": [0.1]}]}

    with patch(
        "medmdt.api.routes.settings.http_requests.post",
        return_value=response_mock,
    ) as request:
        response = TestClient(create_app()).post(
            "/api/v1/settings/test-embedding",
            json={"provider": provider, "model": "embed"},
        )

    assert response.status_code == 200
    assert request.call_args.kwargs["headers"]["Authorization"] == (
        f"Bearer {provider}-embedding-key"
    )


def test_embedding_model_list_binds_qwen_environment_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("DASHSCOPE_API_KEY", "qwen-embedding-key")

    with patch(
        "medmdt.api.routes.settings._fetch_dashscope_embedding_models",
        return_value={"models": []},
    ) as fetch_models:
        response = TestClient(create_app()).post(
            "/api/v1/settings/embedding-models",
            json={"provider": "qwen"},
        )

    assert response.status_code == 200
    fetch_models.assert_called_once_with("qwen-embedding-key")


def test_custom_embedding_http_paths_reject_ambient_openai_key(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "ambient-openai-key")
    monkeypatch.setenv("MEDMDT_ALLOWED_BASE_URL_HOSTS", "custom-embedding.example")
    base_url = "https://custom-embedding.example/v1"
    response_mock = MagicMock(status_code=200)
    response_mock.json.return_value = {"data": []}

    with (
        patch(
            "medmdt.api.routes.settings.http_requests.post",
            return_value=response_mock,
        ) as post,
        patch(
            "medmdt.api.routes.settings._fetch_openai_compat_models",
            return_value={"models": []},
        ) as fetch_models,
    ):
        client = TestClient(create_app())
        responses = [
            client.post(
                "/api/v1/settings/test-embedding",
                json={"provider": "custom", "model": "embed", "base_url": base_url},
            ),
            client.post(
                "/api/v1/settings/embedding-models",
                json={"provider": "custom", "base_url": base_url},
            ),
        ]

    assert all(response.status_code == 400 for response in responses)
    post.assert_not_called()
    fetch_models.assert_not_called()
