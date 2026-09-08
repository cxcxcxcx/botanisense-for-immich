"""Regression tests for probe_existing_async's status-code contract.

The purge flow (POST /api/assets/purge-missing) deletes every asset the
probe reports missing. Only 200 (exists) and 404 (missing) are
definitive answers; any other status — 401/403 (bad or expired key),
429 (rate limit), 5xx (Immich restarting) — must ABORT the probe with
ImmichProbeError. Treating those as "missing" would make every asset
look deleted and the confirmed purge would wipe the whole local DB.
"""

import asyncio

import httpx
import pytest

import immich_client
from immich_client import ImmichClient, ImmichProbeError


def _client_with_status(status_code: int, monkeypatch) -> ImmichClient:
    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(status_code)

    real = httpx.AsyncClient

    def factory(*args, **kwargs):
        return real(transport=httpx.MockTransport(handler), *args, **kwargs)

    # Patch for the whole test (probe_existing_async builds its own
    # AsyncClient internally, so the patch must cover the probe call,
    # not just client construction).
    monkeypatch.setattr(immich_client.httpx, "AsyncClient", factory)
    return ImmichClient(base_url="http://immich.test", api_key="k")


@pytest.mark.parametrize("status", [200])
def test_200_means_exists(status, monkeypatch):
    c = _client_with_status(status, monkeypatch)
    got = asyncio.run(c.probe_existing_async(["a", "b"]))
    assert got == {"a", "b"}


@pytest.mark.parametrize("status", [404])
def test_404_means_missing(status, monkeypatch):
    c = _client_with_status(status, monkeypatch)
    got = asyncio.run(c.probe_existing_async(["a", "b"]))
    assert got == set()


@pytest.mark.parametrize("status", [401, 403, 429, 500, 502, 503])
def test_other_statuses_abort(status, monkeypatch):
    c = _client_with_status(status, monkeypatch)
    with pytest.raises(ImmichProbeError):
        asyncio.run(c.probe_existing_async(["a", "b"]))
