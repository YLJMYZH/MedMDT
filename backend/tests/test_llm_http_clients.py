import asyncio
import importlib

import httpx
import pytest


def test_shared_http_clients_are_reused_closed_idempotently_and_recreated():
    try:
        clients = importlib.import_module("medmdt.llm.http_clients")
    except ModuleNotFoundError:
        pytest.fail("shared LLM HTTP client ownership module is missing")

    async def exercise_clients():
        sync_one = clients.get_shared_http_client()
        sync_two = clients.get_shared_http_client()
        async_one = clients.get_shared_async_http_client()
        async_two = clients.get_shared_async_http_client()

        assert isinstance(sync_one, httpx.Client)
        assert isinstance(async_one, httpx.AsyncClient)
        assert sync_one.follow_redirects is False
        assert async_one.follow_redirects is False
        assert sync_one is sync_two
        assert async_one is async_two

        clients.reset_shared_http_client()
        clients.reset_shared_http_client()
        sync_recreated = clients.get_shared_http_client()
        assert sync_recreated is not sync_one
        assert sync_recreated.follow_redirects is False

        await clients.reset_shared_async_http_client()
        await clients.reset_shared_async_http_client()
        async_recreated = clients.get_shared_async_http_client()
        assert async_recreated is not async_one
        assert async_recreated.follow_redirects is False

        await clients.reset_shared_http_clients()
        await clients.reset_shared_http_clients()

    asyncio.run(exercise_clients())

    sync_client = clients.get_shared_http_client()
    async_client = clients.get_shared_async_http_client()
    clients.close_shared_http_clients_sync()
    clients.close_shared_http_clients_sync()
    assert sync_client.is_closed
    assert async_client.is_closed
