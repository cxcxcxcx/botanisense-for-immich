"""Tests for the plants→kingdom album-sync merge.

The old /api/album/plants* endpoints (a dedicated worker, registry,
and four routes that duplicated the kingdom ones for Plantae) were
removed; the UI now calls /api/album/kingdom* with kingdom="Plantae".
These tests pin the two user-visible contracts of that merge:

1. The kingdom endpoints serve the Plantae case end to end (preview
   counts, saved album id).
2. Upgrade safety: a library that synced via the old plants code has
   its album stored under album_meta key "plants_album_id". The merged
   read path must adopt that legacy row so the existing Immich album
   is reused instead of duplicated.
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
    # app.py hard-exits at import without these; tests never talk to a
    # real Immich instance.
    monkeypatch.setenv("IMMICH_API_KEY", "test-fixture-do-not-use")
    monkeypatch.setenv("IMMICH_BASE_URL", "http://immich.example:2283")
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    storage._GLOBAL_NEAR_DUP_CACHE = None


def _seed_plantae(db_path: Path, n: int = 3) -> None:
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        for i in range(n):
            conn.execute(
                "INSERT INTO assets "
                "(asset_id, original_file_name, original_url, thumbnail_url) "
                "VALUES (?, ?, ?, ?)",
                (f"a-{i}", f"a-{i}.jpg", f"http://x/orig/{i}", f"http://x/thumb/{i}"),
            )
            conn.execute(
                "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
                "VALUES (?, 'kingdom', 'Plantae', 0.90)",
                (f"a-{i}",),
            )
        conn.commit()


def _seed_legacy_plants_album(db_path: Path, album_id: str = "legacy-album") -> None:
    """Write the album_meta row the OLD /api/album/plants code created."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        storage.set_album_meta(
            conn,
            "plants_album_id",
            {"album_id": album_id, "n_assets": 3, "min_score": 0.6},
        )
        conn.commit()


def _client():
    import storage  # noqa: E402
    from fastapi.testclient import TestClient
    import app  # noqa: E402
    storage._GLOBAL_NEAR_DUP_CACHE = {}
    return TestClient(app.app)


def test_kingdom_preview_counts_plantae(db_path):
    _seed_plantae(db_path)
    with _client() as client:
        r = client.get("/api/album/kingdom/preview", params={"kingdom": "Plantae"})
    r.raise_for_status()
    body = r.json()
    assert body["n_assets"] == 3
    assert body["kingdom"] == "Plantae"
    assert body["saved_album_id"] is None


def test_legacy_plants_album_row_is_adopted(db_path):
    """The upgrade case: album saved by the old plants code (key
    "plants_album_id"), no row under the new "Plantae" key. Preview and
    status must surface the legacy album id so the sync reuses it."""
    _seed_plantae(db_path)
    _seed_legacy_plants_album(db_path)
    with _client() as client:
        preview = client.get(
            "/api/album/kingdom/preview", params={"kingdom": "Plantae"}
        ).json()
        status = client.get(
            "/api/kingdom-album-status", params={"kingdom": "Plantae"}
        ).json()
    assert preview["saved_album_id"] == "legacy-album"
    assert status["exists"] is True
    assert status["album_id"] == "legacy-album"


def test_new_kingdom_key_wins_over_legacy(db_path):
    """Once a sync has run under the merged code (row under "Plantae"),
    that row takes precedence over any stale legacy row."""
    _seed_plantae(db_path)
    _seed_legacy_plants_album(db_path)
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        storage.set_kingdom_album_meta(
            conn, "Plantae", {"album_id": "new-album", "n_assets": 3}
        )
        conn.commit()
    with _client() as client:
        status = client.get(
            "/api/kingdom-album-status", params={"kingdom": "Plantae"}
        ).json()
    assert status["album_id"] == "new-album"


def test_other_kingdoms_ignore_legacy_row(db_path):
    _seed_legacy_plants_album(db_path)
    with _client() as client:
        status = client.get(
            "/api/kingdom-album-status", params={"kingdom": "Fungi"}
        ).json()
    assert status["exists"] is False


def test_old_plants_routes_are_gone(db_path):
    """/api/album/plants was removed in the merge — it must not linger
    as a half-working alias. Checked against the OpenAPI schema rather
    than status codes: a request to a removed path falls through to the
    SPA StaticFiles mount when frontend/dist exists, which answers 405
    for POST instead of 404."""
    _seed_plantae(db_path)
    with _client() as client:
        paths = set(client.get("/openapi.json").json()["paths"])
    assert not any("album/plants" in p or p == "/api/plants-album-status"
                   for p in paths), (
        f"plants routes still registered: "
        f"{[p for p in paths if 'plants' in p]}"
    )
    assert "/api/album/kingdom" in paths
    assert "/api/album/kingdom/preview" in paths
