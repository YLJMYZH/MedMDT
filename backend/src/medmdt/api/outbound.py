"""Deterministic outbound Base URL policy for settings endpoints.

Provider registry defaults are trusted application configuration. Custom
targets must instead be literal addresses governed by the deployment CIDR
allowlist or exact hostnames governed by the deployment hostname allowlist.
The policy performs no DNS lookup: the allowlists, not resolver results, are
the trust boundary. ``MEDMDT_ALLOWED_BASE_URL_HOSTS`` is a comma-separated
exact hostname list; ``MEDMDT_ALLOWED_BASE_URL_CIDRS`` is a comma-separated
strict CIDR list for private and loopback literal addresses.
"""

import ipaddress
import os
from urllib.parse import urlsplit

from medmdt.llm.provider import get_provider_base_url


ALLOWED_BASE_URL_HOSTS_ENV = "MEDMDT_ALLOWED_BASE_URL_HOSTS"
ALLOWED_BASE_URL_CIDRS_ENV = "MEDMDT_ALLOWED_BASE_URL_CIDRS"

_RFC1918_NETWORKS = tuple(
    ipaddress.ip_network(cidr)
    for cidr in ("10.0.0.0/8", "172.16.0.0/12", "192.168.0.0/16")
)
_ULA_NETWORK = ipaddress.ip_network("fc00::/7")


def _canonicalize_base_url(value: str) -> str:
    raw = value.strip()
    if not raw or any(character.isspace() for character in raw) or "\\" in raw:
        raise ValueError("invalid URL")

    parts = urlsplit(raw)
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
    if parts.netloc.endswith(":") or port == 0:
        raise ValueError("invalid URL port")

    scheme = parts.scheme.lower()
    host = parts.hostname.rstrip(".").lower()
    if not host:
        raise ValueError("invalid URL host")
    rendered_host = f"[{host}]" if ":" in host else host
    default_port = 443 if scheme == "https" else 80
    authority = rendered_host if port in {None, default_port} else f"{rendered_host}:{port}"
    path = parts.path.rstrip("/")
    return f"{scheme}://{authority}{path}"


def canonical_effective_base_url(provider: str, base_url: str | None) -> str | None:
    effective = base_url or get_provider_base_url(provider)
    if effective is None:
        return None
    return _canonicalize_base_url(effective)


def _allowed_hostnames() -> set[str]:
    allowed: set[str] = set()
    for raw_entry in os.getenv(ALLOWED_BASE_URL_HOSTS_ENV, "").split(","):
        entry = raw_entry.strip().rstrip(".").lower()
        if not entry or "*" in entry or ":" in entry or "/" in entry:
            continue
        try:
            ipaddress.ip_address(entry)
        except ValueError:
            allowed.add(entry)
    return allowed


def _allowed_networks() -> tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...]:
    networks: list[ipaddress.IPv4Network | ipaddress.IPv6Network] = []
    for raw_entry in os.getenv(ALLOWED_BASE_URL_CIDRS_ENV, "").split(","):
        entry = raw_entry.strip()
        if not entry:
            continue
        try:
            networks.append(ipaddress.ip_network(entry, strict=True))
        except ValueError:
            continue
    return tuple(networks)


def _is_private_or_loopback(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    if address.is_loopback:
        return True
    if isinstance(address, ipaddress.IPv4Address):
        return any(address in network for network in _RFC1918_NETWORKS)
    return address in _ULA_NETWORK


def _is_in_allowed_network(
    address: ipaddress.IPv4Address | ipaddress.IPv6Address,
) -> bool:
    return any(
        address.version == network.version and address in network
        for network in _allowed_networks()
    )


def _validate_custom_target(canonical: str) -> None:
    parts = urlsplit(canonical)
    host = parts.hostname
    assert host is not None

    try:
        address = ipaddress.ip_address(host)
    except ValueError:
        if parts.scheme != "https" or host not in _allowed_hostnames():
            raise ValueError("Custom Base URL hostname is not allowed")
        return

    if _is_private_or_loopback(address):
        if not _is_in_allowed_network(address):
            raise ValueError("private Base URL address is not allowed")
        return

    if (
        not address.is_global
        or address.is_link_local
        or address.is_multicast
        or address.is_reserved
        or address.is_unspecified
    ):
        raise ValueError("special-purpose Base URL address is forbidden")
    if parts.scheme != "https":
        raise ValueError("public Base URL addresses require HTTPS")


def validate_user_base_url(provider: str, base_url: str | None) -> None:
    if not base_url:
        return

    canonical = _canonicalize_base_url(base_url)
    registry_default = get_provider_base_url(provider)
    if registry_default is not None:
        if canonical == _canonicalize_base_url(registry_default):
            return

    if provider != "custom":
        raise ValueError("Base URL overrides are forbidden for this provider")

    _validate_custom_target(canonical)
