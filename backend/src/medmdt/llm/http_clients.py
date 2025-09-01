"""Process-owned redirect-safe HTTP clients for LLM integrations."""

import asyncio
import threading

import httpx


_client_lock = threading.RLock()
_sync_client: httpx.Client | None = None
_async_client: httpx.AsyncClient | None = None


def get_shared_http_client() -> httpx.Client:
    global _sync_client
    with _client_lock:
        if _sync_client is None or _sync_client.is_closed:
            _sync_client = httpx.Client(follow_redirects=False)
        return _sync_client


def get_shared_async_http_client() -> httpx.AsyncClient:
    global _async_client
    with _client_lock:
        if _async_client is None or _async_client.is_closed:
            _async_client = httpx.AsyncClient(follow_redirects=False)
        return _async_client


def close_shared_http_client() -> None:
    global _sync_client
    with _client_lock:
        client = _sync_client
        _sync_client = None
    if client is not None and not client.is_closed:
        client.close()


async def close_shared_async_http_client() -> None:
    global _async_client
    with _client_lock:
        client = _async_client
        _async_client = None
    if client is not None and not client.is_closed:
        await client.aclose()


async def close_shared_http_clients() -> None:
    close_shared_http_client()
    await close_shared_async_http_client()


def reset_shared_http_client() -> None:
    close_shared_http_client()


async def reset_shared_async_http_client() -> None:
    await close_shared_async_http_client()


async def reset_shared_http_clients() -> None:
    await close_shared_http_clients()


def close_shared_http_clients_sync() -> None:
    """Close both pools from a synchronous process or CLI boundary."""
    close_shared_http_client()
    asyncio.run(close_shared_async_http_client())
