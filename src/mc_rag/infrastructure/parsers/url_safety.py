"""SSRF-safe URL validation and fetching.

Any code that downloads a remote resource (web pages, Google Sheets) must go
through :func:`safe_fetch`, which enforces:

* only ``http`` / ``https`` schemes (no ``file://``, ``ftp://``, ...);
* no embedded credentials;
* the resolved IP of **every** hop must be public: loopback, private, link-local
  (including the cloud metadata address ``169.254.169.254``), CGNAT, multicast
  and reserved ranges are refused;
* redirects are followed manually so each hop is re-validated.

Residual risk: the IP check and the actual connection are two steps, so a
hostile DNS server could rebind between them. The window is small and the
scheme/host checks still hold; it is documented in README.md.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from urllib.parse import urljoin, urlsplit

import httpx

from mc_rag.config import get_settings
from mc_rag.domain.exceptions import ParsingError, UnsafeURLError

logger = logging.getLogger(__name__)

_ALLOWED_SCHEMES = frozenset({"http", "https"})
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address
Resolver = Callable[[str], Sequence[IPAddress]]

# Ranges that are not flagged by ``ipaddress`` on every Python version.
_EXTRA_BLOCKED: tuple[ipaddress.IPv4Network | ipaddress.IPv6Network, ...] = (
    ipaddress.IPv4Network("100.64.0.0/10"),  # carrier-grade NAT
    ipaddress.IPv4Network("192.0.0.0/24"),
    ipaddress.IPv4Network("198.18.0.0/15"),
)


@dataclass(slots=True, frozen=True)
class FetchResult:
    """A validated HTTP response."""

    url: str
    status_code: int
    content_type: str
    body: bytes

    @property
    def is_html(self) -> bool:
        """Whether the response body looks like HTML (login pages, errors)."""
        return "html" in self.content_type.lower()


def validate_url(url: str) -> tuple[str, int]:
    """Check the *shape* of *url* and return ``(host, port)``.

    Raises:
        UnsafeURLError: The scheme, the credentials or the port are unusable.
    """
    candidate = (url or "").strip()
    try:
        parts = urlsplit(candidate)
    except ValueError as exc:
        raise UnsafeURLError(f"Malformed URL: {candidate!r}") from exc

    scheme = parts.scheme.lower()
    if scheme not in _ALLOWED_SCHEMES:
        raise UnsafeURLError(f"Only http(s) URLs are allowed, got {scheme or '<none>'!r}.")
    if parts.username or parts.password:
        raise UnsafeURLError("URLs embedding credentials are not allowed.")
    if not parts.hostname:
        raise UnsafeURLError(f"URL has no host: {candidate!r}")
    try:
        port = parts.port or (443 if scheme == "https" else 80)
    except ValueError as exc:
        raise UnsafeURLError(f"URL has an invalid port: {candidate!r}") from exc
    return parts.hostname, port


def is_public_ip(ip: IPAddress) -> bool:
    """Whether *ip* may be reached from the service."""
    if isinstance(ip, ipaddress.IPv6Address) and ip.ipv4_mapped is not None:
        ip = ip.ipv4_mapped
    if (
        ip.is_unspecified
        or ip.is_loopback
        or ip.is_link_local
        or ip.is_private
        or ip.is_reserved
        or ip.is_multicast
        or (ip.version == 6 and ip.is_site_local)
    ):
        return False
    if ip.version == 4:
        return not any(ip in network for network in _EXTRA_BLOCKED if network.version == 4)
    return not any(ip in network for network in _EXTRA_BLOCKED if network.version == 6)


def default_resolver(host: str) -> list[IPAddress]:
    """Resolve *host* to its IP addresses (system resolver)."""
    try:
        answers = socket.getaddrinfo(host, None)
    except socket.gaierror as exc:
        raise UnsafeURLError(f"Cannot resolve host: {host}") from exc
    addresses: list[IPAddress] = []
    for answer in answers:
        sockaddr = answer[4]
        if not sockaddr:
            continue
        try:
            addresses.append(ipaddress.ip_address(sockaddr[0]))
        except ValueError:
            continue
    if not addresses:
        raise UnsafeURLError(f"Cannot resolve host: {host}")
    return addresses


def assert_public_host(host: str, *, resolver: Resolver | None = None) -> list[IPAddress]:
    """Resolve *host* and refuse it when any address is not public."""
    resolve = resolver or default_resolver
    addresses = resolve(host)
    if not addresses:
        raise UnsafeURLError(f"Cannot resolve host: {host}")
    blocked = [str(ip) for ip in addresses if not is_public_ip(ip)]
    if blocked:
        raise UnsafeURLError(f"Blocked non-public address for host {host}: {', '.join(blocked)}")
    return list(addresses)


def safe_fetch(
    url: str,
    *,
    timeout: float | None = None,
    max_redirects: int | None = None,
    headers: Mapping[str, str] | None = None,
    transport: httpx.BaseTransport | None = None,
    resolver: Resolver | None = None,
    raise_for_status: bool = True,
) -> FetchResult:
    """GET *url* after validating the scheme, the host and every redirect.

    Args:
        url: The URL to fetch (``http``/``https`` only).
        timeout: Per-request timeout in seconds (defaults to ``URL_FETCH_TIMEOUT``).
        max_redirects: Redirect budget (defaults to ``URL_FETCH_MAX_REDIRECTS``).
        headers: Extra headers merged over a default user agent.
        transport: Injected :class:`httpx.BaseTransport` (tests only).
        resolver: Injected DNS resolver (tests only).
        raise_for_status: Raise :class:`ParsingError` on HTTP 4xx/5xx.

    Returns:
        The final :class:`FetchResult`.

    Raises:
        UnsafeURLError: A scheme, host, address or redirect budget is refused.
        ParsingError: The HTTP request itself failed.
    """
    settings = get_settings()
    budget = settings.url_fetch_max_redirects if max_redirects is None else max_redirects
    wait = settings.url_fetch_timeout if timeout is None else timeout
    request_headers = dict(headers or {})
    request_headers.setdefault("User-Agent", "mc-club-rag/0.1")

    current = (url or "").strip()
    with httpx.Client(
        timeout=wait,
        follow_redirects=False,
        transport=transport,
        headers=request_headers,
    ) as client:
        for _hop in range(budget + 1):
            assert_public_host(validate_url(current)[0], resolver=resolver)
            try:
                response = client.get(current)
            except httpx.HTTPError as exc:
                logger.warning("Network error fetching %s: %s", current, exc)
                raise ParsingError(f"Unable to fetch URL: {current}") from exc

            location = response.headers.get("location")
            if 300 <= response.status_code < 400 and location:
                current = urljoin(current, location)
                continue
            if raise_for_status and response.status_code >= 400:
                logger.warning("HTTP error fetching %s: %s", current, response.status_code)
                raise ParsingError(
                    f"HTTP {response.status_code} while fetching: {current}"
                )
            return FetchResult(
                url=str(response.url),
                status_code=response.status_code,
                content_type=response.headers.get("content-type", ""),
                body=response.content,
            )
    raise UnsafeURLError(f"Too many redirects (max {budget}) while fetching: {url}")
