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

import os
from dataclasses import dataclass
from typing import Any

import httpx


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

    def download_bytes(self, path: str) -> bytes:
        r = self._client.get(path)
        r.raise_for_status()
        return r.content

    def close(self) -> None:
        self._client.close()