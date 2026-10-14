"""Security tests for the SSRF guard in :mod:`mc_rag.infrastructure.parsers.url_safety`.

The guard is what allows the ``mode=url`` document endpoint to stay open:
submitted URLs must never reach loopback, private, link-local or metadata
addresses, including after a redirect.
"""

from __future__ import annotations

import ipaddress
from collections.abc import Sequence
from pathlib import Path

import httpx
import pytest

from mc_rag.domain.exceptions import ParsingError, UnsafeURLError
from mc_rag.infrastructure.parsers.url_safety import (
    FetchResult,
    assert_public_host,
    is_public_ip,
    safe_fetch,
    validate_url,
)
from mc_rag.infrastructure.parsers.web import parse_web

PUBLIC_IP = "93.184.216.34"
IPAddress = ipaddress.IPv4Address | ipaddress.IPv6Address


def resolver(host: str) -> Sequence[IPAddress]:
    """Resolve IP literals to themselves and any name to a public address."""
    try:
        return [ipaddress.ip_address(host)]
    except ValueError:
        return [ipaddress.ip_address(PUBLIC_IP)]


def blocking_handler(status: int, headers: dict[str, str] | None = None) -> httpx.MockTransport:
    """Transport returning *status* (and *headers*) for every request."""

    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(status, headers=headers or {}, content=b"payload")

    return httpx.MockTransport(handler)


# --------------------------------------------------------------------------- #
# URL shape
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "url",
    [
        "ftp://example.com/file.csv",
        "file:///etc/passwd",
        "javascript:alert(1)",
        "gopher://example.com",
        "http://",
        "https:///nohost",
        "http://user:secret@example.com/",
        "http://example.com:notaport/",
    ],
)
def test_validate_url_refuses_unsafe_shapes(url: str) -> None:
    with pytest.raises(UnsafeURLError):
        validate_url(url)


@pytest.mark.parametrize(
    ("url", "host", "port"),
    [
        ("https://example.com/a", "example.com", 443),
        ("http://example.com:8080/a", "example.com", 8080),
        ("HTTP://Example.COM/x", "example.com", 80),
    ],
)
def test_validate_url_returns_host_and_port(url: str, host: str, port: int) -> None:
    assert validate_url(url) == (host, port)


# --------------------------------------------------------------------------- #
# Address policy
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "address",
    [
        "127.0.0.1",
        "127.1.2.3",
        "10.0.0.1",
        "172.16.0.5",
        "192.168.1.1",
        "169.254.169.254",  # cloud metadata
        "100.64.0.1",  # carrier-grade NAT
        "0.0.0.0",
        "224.0.0.1",
        "255.255.255.255",
        "::1",
        "fc00::1",
        "fe80::1",
        "::ffff:127.0.0.1",  # IPv4-mapped loopback
        "::ffff:192.168.0.1",
    ],
)
def test_is_public_ip_blocks_internal_addresses(address: str) -> None:
    assert is_public_ip(ipaddress.ip_address(address)) is False


@pytest.mark.parametrize("address", ["93.184.216.34", "8.8.8.8", "2606:4700:4700::1111"])
def test_is_public_ip_allows_public_addresses(address: str) -> None:
    assert is_public_ip(ipaddress.ip_address(address)) is True


def test_assert_public_host_blocks_a_private_resolution() -> None:
    def private(_host: str) -> Sequence[IPAddress]:
        return [ipaddress.ip_address("10.0.0.5")]

    with pytest.raises(UnsafeURLError, match="non-public"):
        assert_public_host("intranet.test", resolver=private)


def test_assert_public_host_allows_a_public_resolution() -> None:
    assert assert_public_host("example.com", resolver=resolver)


# --------------------------------------------------------------------------- #
# Fetching
# --------------------------------------------------------------------------- #


def test_safe_fetch_returns_the_body() -> None:
    result = safe_fetch(
        "http://example.com/page",
        transport=blocking_handler(200, {"content-type": "text/html; charset=utf-8"}),
        resolver=resolver,
    )
    assert isinstance(result, FetchResult)
    assert result.status_code == 200
    assert result.is_html
    assert result.body == b"payload"
    assert result.url == "http://example.com/page"


def test_safe_fetch_follows_public_redirects() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.host == "example.com":
            return httpx.Response(
                302, headers={"location": "http://other.example.org/final"}
            )
        return httpx.Response(200, content=b"done")

    result = safe_fetch(
        "http://example.com/start",
        transport=httpx.MockTransport(handler),
        resolver=resolver,
    )
    assert result.status_code == 200
    assert result.body == b"done"
    assert "other.example.org" in result.url


def test_safe_fetch_refuses_a_redirect_to_a_private_host() -> None:
    transport = blocking_handler(302, {"location": "http://127.0.0.1/secret"})
    with pytest.raises(UnsafeURLError):
        safe_fetch("http://example.com/start", transport=transport, resolver=resolver)


def test_safe_fetch_refuses_a_redirect_to_the_metadata_address() -> None:
    transport = blocking_handler(302, {"location": "http://169.254.169.254/latest/meta-data"})
    with pytest.raises(UnsafeURLError):
        safe_fetch("http://example.com/start", transport=transport, resolver=resolver)


def test_safe_fetch_enforces_the_redirect_budget() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        return httpx.Response(302, headers={"location": "http://example.com/again"})

    with pytest.raises(UnsafeURLError, match="Too many redirects"):
        safe_fetch(
            "http://example.com/start",
            transport=httpx.MockTransport(handler),
            resolver=resolver,
            max_redirects=2,
        )


def test_safe_fetch_raises_on_http_error_by_default() -> None:
    with pytest.raises(ParsingError, match="HTTP 404"):
        safe_fetch(
            "http://example.com/missing",
            transport=blocking_handler(404),
            resolver=resolver,
        )


def test_safe_fetch_can_return_http_errors() -> None:
    result = safe_fetch(
        "http://example.com/missing",
        transport=blocking_handler(404),
        resolver=resolver,
        raise_for_status=False,
    )
    assert result.status_code == 404


def test_safe_fetch_maps_network_errors_to_parsing_error() -> None:
    def handler(_request: httpx.Request) -> httpx.Response:
        raise httpx.ConnectError("boom")

    with pytest.raises(ParsingError, match="Unable to fetch"):
        safe_fetch(
            "http://example.com/x",
            transport=httpx.MockTransport(handler),
            resolver=resolver,
        )


# --------------------------------------------------------------------------- #
# parse_web integration
# --------------------------------------------------------------------------- #


def test_parse_web_reads_a_local_file_by_default(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<html><title>Page</title><p>Hello</p></html>", encoding="utf-8")
    document = parse_web(str(path))
    assert "Hello" in document.text


def test_parse_web_refuses_a_local_file_when_disabled(tmp_path: Path) -> None:
    path = tmp_path / "page.html"
    path.write_text("<html><p>Hello</p></html>", encoding="utf-8")
    with pytest.raises(ParsingError, match="neither an HTTP"):
        parse_web(str(path), allow_local_files=False)


@pytest.mark.parametrize("value", ["file:///etc/passwd", "ftp://example.com/f"])
def test_parse_web_refuses_non_http_schemes(value: str) -> None:
    with pytest.raises(ParsingError):
        parse_web(value, allow_local_files=False)
