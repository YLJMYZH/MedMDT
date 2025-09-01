"""Outbound Base URL validation for settings connectivity endpoints.

Private and local targets are denied by default. Deployments that intentionally
connect to an on-premises model server may explicitly set
``MEDMDT_ALLOW_PRIVATE_BASE_URLS=true``. URL syntax and embedded credentials
remain forbidden even when that opt-in is enabled.
"""

import ipaddress
import os
import socket
from urllib.parse import urlsplit

from medmdt.llm.provider import get_provider_base_url


PRIVATE_URL_OPT_IN = "MEDMDT_ALLOW_PRIVATE_BASE_URLS"


def canonical_effective_base_url(provider: str, base_url: str | None) -> str | None:
    effective = base_url or get_provider_base_url(provider)
    if effective is None:
        return None
    parts = urlsplit(effective.strip())
    if parts.scheme.lower() not in {"http", "https"}:
        raise ValueError("unsupported URL scheme")
    if not parts.hostname or parts.username is not None or parts.password is not None:
        raise ValueError("invalid URL authority")
    if parts.query or parts.fragment:
        raise ValueError("query and fragment are not valid in a Base URL")
    try:
        port = parts.port
    except ValueError as exc:
        raise ValueError("invalid URL port") from exc

    scheme = parts.scheme.lower()
    host = parts.hostname.rstrip(".").lower()
    if not host:
        raise ValueError("invalid URL host")
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    authority = rendered_host if port in {None, default_port} else f"{rendered_host}:{port}"
    path = parts.path.rstrip("/")
    return f"{scheme}://{authority}{path}"


def validate_user_base_url(provider: str, base_url: str | None) -> None:
    if not base_url:
        return
    registry_default = get_provider_base_url(provider)
    if registry_default is not None and base_url.strip() == registry_default:
        return

    canonical = canonical_effective_base_url(provider, base_url)
    assert canonical is not None
    parts = urlsplit(canonical)
    host = parts.hostname
    assert host is not None

    if os.getenv(PRIVATE_URL_OPT_IN, "").strip().lower() in {"1", "true", "yes"}:
        return

    try:
        literal = ipaddress.ip_address(host)
    except ValueError:
        try:
            resolved = socket.getaddrinfo(
                host,
                parts.port or (443 if parts.scheme == "https" else 80),
                type=socket.SOCK_STREAM,
            )
        except OSError as exc:
            raise ValueError("Base URL host could not be resolved") from exc
        if not resolved:
            raise ValueError("Base URL host did not resolve")
        addresses = []
        try:
            for result in resolved:
                addresses.append(ipaddress.ip_address(result[4][0]))
        except (IndexError, ValueError) as exc:
            raise ValueError("invalid resolved address") from exc
    else:
        addresses = [literal]

    if any(
        not address.is_global
        or address.is_loopback
        or address.is_private
        or address.is_link_local
        or address.is_reserved
        or address.is_multicast
        or address.is_unspecified
        for address in addresses
    ):
        raise ValueError("Base URL resolves to a non-public address")
