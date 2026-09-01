"""Regression test for the grid's default sort = capture date DESC.

Pins the project decision to make the browse-list default "Newest first"
(capture date DESC, NULLs last) instead of Immich's search rank. A
regression here would silently re-rank the grid by whatever order
BioCLIP last saw the photos — confusing for a photo-library browse.

These tests hit /api/results via FastAPI's TestClient so the full
sort-application code path is exercised (storage → app.py time-sort
post-processing → response shape), not just one piece.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(p))
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    storage._GLOBAL_NEAR_DUP_CACHE = None


def _seed_asset(conn, asset_id: str, captured_at: str | None) -> None:
    conn.execute(
        "INSERT INTO assets "
        "(asset_id, original_file_name, original_url, thumbnail_url, "
        " captured_at) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            asset_id,
            f"{asset_id}.jpg",
            f"http://x/orig/{asset_id}.jpg",
            f"http://x/thumb/{asset_id}.jpg",
            captured_at,
        ),
    )
    conn.execute(
        "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
        "VALUES (?, 'kingdom', 'Plantae', ?)",
        (asset_id, 0.90),
    )


def _seed(db_path: Path) -> None:
    """Four assets: one with no EXIF (nulls last), three with distinct
    capture dates that span a year. Plantae scores are all 0.90 so the
    plant-only default doesn't filter anything out."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        for aid, ts in [
            ("a-oldest",    "2024-03-01T10:00:00Z"),
            ("a-middle",    "2024-08-15T12:00:00Z"),
            ("a-newest",    "2025-02-10T09:00:00Z"),
            ("a-no-exif",   None),
        ]:
            _seed_asset(conn, aid, ts)
        conn.commit()


def _hit(path: str = "/api/results", query: str = ""):
    """GET the FastAPI endpoint and return the parsed response."""
    import storage  # noqa: E402
    from fastapi.testclient import TestClient
    import app  # noqa: E402
    storage._GLOBAL_NEAR_DUP_CACHE = {}
    with TestClient(app.app) as client:
        r = client.get(f"{path}?{query}" if query else path)
    r.raise_for_status()
    return r.json()


def test_default_sort_is_capture_date_descending(db_path):
    """Calling /api/results with no explicit `sort=` param returns
    rows ordered by captured_at DESC, NULLs last — the new 'Newest
    first' default. A bare URL is the user-visible test case (typing
    the IP into the browser shouldn't see Immich's search rank)."""
    _seed(db_path)
    resp = _hit("/api/results", "page_size=10")
    ids_in_order = [r["asset_id"] for r in resp["results"]]
    assert ids_in_order == ["a-newest", "a-middle", "a-oldest", "a-no-exif"], (
        f"default sort should put newest first with NULLs last; got {ids_in_order}"
    )


def test_explicit_sort_param_overrides_default(db_path):
    """Passing ?sort=rank falls back to the SQL ORDER BY (Immich search
    rank + cluster_id tiebreak). Pins that the URL param still works
    after the default change so a bookmarked `?sort=rank` URL keeps
    rendering the legacy order."""
    _seed(db_path)
    resp = _hit("/api/results", "sort=rank&page_size=10")
    # The exact ordering under sort=rank depends on the SQL ORDER BY
    # (latest_rank ASC, cluster_id ASC, filename ASC) — none of our
    # seeded rows have search_results rows, so latest_rank is NULL
    # for all; tiebreak goes to original_file_name ASC. Assert at
    # least that all 4 are present (the sort doesn't drop anything).
    assert len(resp["results"]) == 4
    ids_in_order = [r["asset_id"] for r in resp["results"]]
    # Filename ordering:
    expected = sorted([
        "a-oldest", "a-middle", "a-newest", "a-no-exif",
    ])
    assert ids_in_order == expected, (
        f"sort=rank with no search_results should fall back to "
        f"filename ASC tiebreak; got {ids_in_order}"
    )

