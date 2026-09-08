"""Tiny Immich REST client used by the demo script.

In Immich 2.7, the CLIP-powered "text search" lives at POST /api/search/smart
(see server/src/controllers/search.controller.ts → searchSmart / SmartSearchDto).
The request body is `{ "query": "...", "page": 1, "size": N }` and the response
contains an `assets.items` array of full AssetResponseDTOs.

Endpoints we hit:
  - POST /api/search/smart         -- CLIP text→image smart search
  - GET  /api/assets/{id}/thumbnail -- preview bytes (size=preview is small)
  - GET  /api/assets/{id}/original  -- full-resolution bytes

All endpoints accept either `x-api-key: <key>` (preferred for API keys) or
`Authorization: Bearer <jwt>` (after /api/auth/login). We use x-api-key.
"""

from __future__ import annotations

import asyncio
import logging
import os
from dataclasses import dataclass
from typing import Any

import httpx

_log = logging.getLogger("immich_client")


class ImmichProbeError(RuntimeError):
    """Raised when an asset existence probe gets a response that is
    neither 200 (exists) nor 404 (missing) — 401/403 (bad/expired key),
    429 (rate limit), 5xx (Immich restarting). The purge flow must ABORT
    on this: treating those as "missing" would delete every local row.
    """


@dataclass
class ImmichAsset:
    id: str
    original_file_name: str
    type: str  # "IMAGE" | "VIDEO"

    @property
    def original_url(self) -> str:
        return f"/api/assets/{self.id}/original"

    @property
    def thumbnail_url(self) -> str:
        return f"/api/assets/{self.id}/thumbnail"


class ImmichClient:
    def __init__(
        self,
        base_url: str,
        api_key: str | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        api_key = api_key or os.environ.get("IMMICH_API_KEY")
        if not api_key:
            raise RuntimeError(
                "Missing Immich API key. Set IMMICH_API_KEY in the environment "
                "or pass api_key=... explicitly."
            )
        self._client = httpx.Client(
            base_url=self.base_url,
            headers={"x-api-key": api_key, "Accept": "application/json"},
            timeout=timeout,
        )

    # ---- search ----------------------------------------------------------

    # Immich's smart-search enforces a hard cap of 1000 results per page.
    # To request more than that we paginate the calls underneath.
    _MAX_PAGE_SIZE = 1000

    def search_text(
        self,
        query: str,
        *,
        size: int = 5,
        page: int = 1,
        with_extras: bool = False,
    ) -> dict[str, Any]:
        """POST /api/search/smart — Immich's CLIP text→image "smart" search.

        The DTO is { query, page, size, withExtras } (SmartSearchDto). The
        response is the standard SearchResponseDto:
            { "albums": {...}, "assets": { "items": [...], "total": N } }.

        Caps the per-page size at 1000 (Immich's hard limit) and
        transparently paginates when the caller asks for more, returning a
        merged payload whose `assets.items` contains every result up to
        `size`. Capped at the actual `total` Immich reports so we don't
        fetch pages that would be empty.
        """
        if size <= self._MAX_PAGE_SIZE:
            return self._search_text_one_page(
                query, size=size, page=page, with_extras=with_extras
            )

        # Paginate. We don't know the true total up front, so we keep
        # fetching pages until we get an empty one or hit `size`.
        merged_items: list[dict] = []
        cur_page = 1
        total = None
        while len(merged_items) < size:
            payload = self._search_text_one_page(
                query,
                size=self._MAX_PAGE_SIZE,
                page=cur_page,
                with_extras=with_extras,
            )
            assets_container = payload.get("assets") or {}
            items = assets_container.get("items") or []
            total = assets_container.get("total", total)
            if not items:
                break
            merged_items.extend(items)
            cur_page += 1
            # If the server reported a total, stop once we'd be past it.
            if total is not None and len(merged_items) >= total:
                break
        # Trim to the requested size and stitch the items back into the
        # first page's payload shape so callers (list_search_assets) keep
        # working unchanged.
        merged_items = merged_items[:size]
        return {
            "albums": {"items": [], "total": 0},
            "assets": {
                "items": merged_items,
                "total": total if total is not None else len(merged_items),
            },
        }

    def _search_text_one_page(
        self,
        query: str,
        *,
        size: int,
        page: int,
        with_extras: bool = False,
    ) -> dict[str, Any]:
        body = {
            "query": query,
            "page": page,
            "size": size,
            "withExtras": with_extras,
        }
        r = self._client.post("/api/search/smart", json=body)
        r.raise_for_status()
        return r.json()

    # ---- asset helpers ---------------------------------------------------

    def list_search_assets(self, payload: dict[str, Any]) -> list[ImmichAsset]:
        """Immich's search/text response shape:
        { "albums": {...}, "assets": {"items": [...], "total": N, ...} }
        Each item is an AssetResponseDTO.
        """
        assets_container = payload.get("assets") or {}
        items = assets_container.get("items") or []
        return [
            ImmichAsset(
                id=item["id"],
                original_file_name=item.get("originalFileName", ""),
                type=item.get("type", "IMAGE"),
            )
            for item in items
            if item.get("type") == "IMAGE"
        ]

    # ---- asset enumeration ----------------------------------------------

    # How many concurrent GETs to fire at Immich when probing per-asset
    # existence. Each call is ~30-50ms round-trip on a LAN; 100 in
    # flight processes ~2000 assets/sec when using HEAD (no response
    # body). GET (with full asset JSON) is ~10× slower.
    _PROBE_CONCURRENCY = 100

    # ---- per-asset EXIF --------------------------------------------------

    # Cache GET /api/assets/{id} for the duration of one ingest batch.
    # The pipeline hits this once per asset; without a cache the same
    # asset would be re-fetched if it shows up twice (e.g. across two
    # search pages).
    _exif_cache: dict[str, dict] | None = None

    def _exif_cache_init(self) -> None:
        if self._exif_cache is None:
            self._exif_cache = {}

    def get_asset_metadata(self, asset_id: str) -> dict | None:
        """GET /api/assets/{id} — return the AssetResponseDTO, or None
        if Immich 404s. Pulled by the pipeline on ingest to extract
        EXIF capture time + GPS for the local DB.

        The full payload is large (~1.7 KB) and only a few fields are
        needed downstream; we return the whole dict so callers can
        extract anything else without another round-trip.

        Results are cached for the lifetime of this client instance
        (one pipeline run) so repeated lookups for the same asset
        are free.
        """
        self._exif_cache_init()
        if asset_id in self._exif_cache:
            return self._exif_cache[asset_id]
        try:
            r = self._client.get(f"/api/assets/{asset_id}")
        except httpx.RequestError as e:
            # Transport-level error (DNS, connect, timeout). Treat as
            # "no EXIF this run" so a long ingest isn't aborted, but
            # log so the user can see why their EXIF never landed.
            _log.warning("get_asset_metadata %s: %s", asset_id, e)
            return None
        if r.status_code == 200:
            data = r.json()
            self._exif_cache[asset_id] = data
            return data
        if r.status_code == 404:
            # Asset genuinely gone — not worth warning for, since the
            # caller will treat as "no EXIF" and the next ingest run
            # will skip it via FK behaviour.
            self._exif_cache[asset_id] = None
            return None
        _log.warning(
            "get_asset_metadata %s: unexpected status %s",
            asset_id, r.status_code,
        )
        # NOT cached: a 429/5xx is Immich having a moment, not the
        # asset having no metadata. Caching it here would pin "no EXIF"
        # for the rest of the run even after Immich recovers. (Only the
        # definitive 404 above is cached as None.)
        return None

    @staticmethod
    def extract_exif(asset_payload: dict | None) -> dict:
        """Pull (captured_at, latitude, longitude) out of an
        AssetResponseDTO. All three fields are nullable — photos
        without EXIF or GPS return None.

        `captured_at` prefers `exifInfo.dateTimeOriginal` (EXIF
        capture time) and falls back to `fileCreatedAt` (filesystem
        mtime) when EXIF is absent. Both arrive as ISO 8601 strings.
        """
        if not asset_payload:
            return {"captured_at": None, "latitude": None, "longitude": None}
        exif = asset_payload.get("exifInfo") or {}
        captured_at = exif.get("dateTimeOriginal") or asset_payload.get("fileCreatedAt")
        latitude = exif.get("latitude")
        longitude = exif.get("longitude")
        # Immich serializes missing GPS as 0/0 in some versions; treat
        # (0.0, 0.0) as "no location" so the UI doesn't render a fake
        # pin in the Atlantic.
        if latitude == 0.0 and longitude == 0.0:
            latitude = None
            longitude = None
        return {
            "captured_at": captured_at,
            "latitude": latitude if isinstance(latitude, (int, float)) else None,
            "longitude": longitude if isinstance(longitude, (int, float)) else None,
        }

    def probe_existing(self, asset_ids: list[str]) -> set[str]:
        """Sync wrapper around probe_existing_async(). Use from CLI
        scripts / non-async contexts."""
        return asyncio.run(self.probe_existing_async(asset_ids))

    async def probe_existing_async(self, asset_ids: list[str]) -> set[str]:
        """Probe the given asset_ids against Immich's
        HEAD /api/assets/{id}; return the subset that exists (200).
        Anything else (404 / network error) is treated as missing.

        Used by the local DB's "purge missing" flow: assets in the
        local DB but missing from this set are candidates for purging.
        We don't try to enumerate the full Immich library — that would
        require GET /api/assets (not exposed in Immich 2.7) or
        brute-force random sampling (which doesn't converge with the
        user's 100k+ photo library). Instead we probe only what the
        DB knows about and report anything 404ing as missing.

        Note: Immich's trash retention means trashed photos still
        return 200 here until the trash TTL expires, so callers should
        let users confirm before deleting anything.

        Async because FastAPI runs the endpoint in an event loop;
        calling asyncio.run() inside it would fail with
        "asyncio.run() cannot be called from a running event loop".
        """
        sem = asyncio.Semaphore(self._PROBE_CONCURRENCY)
        api_key = self._client.headers["x-api-key"]

        # Reuse ONE AsyncClient across all probes + use HEAD to skip the
        # ~1.7KB JSON body per asset. With 100 concurrency, ~17k probes
        # complete in 10-15s on a LAN.
        async with httpx.AsyncClient(
            base_url=self.base_url,
            headers={"x-api-key": api_key, "Accept": "application/json"},
            timeout=10.0,
            limits=httpx.Limits(max_connections=self._PROBE_CONCURRENCY * 2),
        ) as ac:
            async def probe(aid: str) -> str | None:
                async with sem:
                    # Only 200 (exists) and 404 (missing) are
                    # definitive. Transport errors MUST bubble too —
                    # if Immich is unreachable, rate-limiting (429),
                    # rejecting the key (401/403), or 5xx'ing mid-restart
                    # and we treated any of those as "missing", the
                    # purge endpoint would mark EVERY asset as missing
                    # and DELETE them. Abort the whole purge instead.
                    r = await ac.head(f"/api/assets/{aid}")
                    if r.status_code == 200:
                        return aid
                    if r.status_code == 404:
                        return None
                    raise ImmichProbeError(
                        f"Immich returned HTTP {r.status_code} while "
                        f"probing assets (first id: {aid}) — aborting "
                        f"purge; check the API key and that Immich is up."
                    )

            results = await asyncio.gather(*(probe(a) for a in asset_ids))
        return {x for x in results if x is not None}

    def download_bytes(self, path: str) -> bytes:
        r = self._client.get(path)
        r.raise_for_status()
        return r.content

    # ---- album management -----------------------------------------------

    def create_album(self, *, name: str, description: str = "") -> dict:
        """POST /api/albums — create a new album.

        Body shape (CreateAlbumDto): { albumName, description? }.
        Returns the created album dict (includes 'id').
        """
        r = self._client.post(
            "/api/albums",
            json={"albumName": name, "description": description},
        )
        r.raise_for_status()
        return r.json()

    def list_albums(self) -> list[dict]:
        """GET /api/albums — list every album the API key has access to.

        Returns a list of {id, albumName, ...} dicts. We don't paginate
        (a personal Immich install typically has <100 albums) but if a
        future user has more, callers can chain `paged_albums`. Used by
        the Plants-album flow to find a pre-existing "Plantae
        (BioCLIP)" album before creating a new one, so a deleted
        album_meta row doesn't lead to a duplicate album.
        """
        r = self._client.get("/api/albums")
        r.raise_for_status()
        data = r.json()
        # Immich returns either a bare list or a {albums: [...], total: N}
        # paged envelope depending on version; handle both.
        if isinstance(data, dict):
            return list(data.get("albums") or [])
        return list(data)

    def find_album_by_name(self, name: str) -> dict | None:
        """Convenience: return the first album whose `albumName` exactly
        matches `name`, or None. Match is case-sensitive and exact — we
        own the album name, so we don't want to silently merge into an
        unrelated album the user renamed.
        """
        for a in self.list_albums():
            if a.get("albumName") == name:
                return a
        return None

    def add_assets_to_album(
        self, *, album_id: str, asset_ids: list[str]
    ) -> dict:
        """PUT /api/albums/{id}/assets — bulk-add assets to an album.

        Immich's body schema is { ids: [asset_uuid, ...] } (not
        `assetIds`). Returns the updated album summary. We chunk in the
        caller to mirror the project's _MAX_PAGE_SIZE boundary, although
        Immich's bulk endpoint accepts very large lists.
        """
        r = self._client.put(
            f"/api/albums/{album_id}/assets",
            json={"ids": list(asset_ids)},
        )
        r.raise_for_status()
        return r.json()

    def get_album(self, album_id: str) -> dict:
        """GET /api/albums/{id}. Raises (via raise_for_status) if the album
        doesn't exist (404) — used by /api/album/plants to detect a stale
        saved id after a user manually deletes the album in Immich.
        """
        r = self._client.get(f"/api/albums/{album_id}")
        r.raise_for_status()
        return r.json()

    def delete_album(self, *, album_id: str) -> None:
        """DELETE /api/albums/{id}. Raises httpx.HTTPStatusError on
        non-2xx; callers should handle 404 as a no-op (idempotent delete
        — the album was already gone before this call)."""
        r = self._client.delete(f"/api/albums/{album_id}")
        r.raise_for_status()

    def close(self) -> None:
        self._client.close()