"""Regression tests for the sync-all junk-cluster filter.

Background: cluster 499 in the user's DB held 10,827 of 19,784 photos
(54.7%) — and every photo in it had only kingdom-rank taxon_scores,
no species-rank. HDBSCAN's "cluster of last resort" had captured the
scattered kingdom-only points. The sync-all worker happily tried to
slurp all 10,827 of them into one giant Immich album called
"Cluster 499 (BioCLIP)", which is exactly the wrong user-facing
behavior.

The fix: filter out clusters with no species-rank scores from the
sync-all path. These tests pin:

- `storage.cluster_has_species_predictions` answers correctly.
- `_cluster_album_sync_all_worker` excludes junk clusters from both
  the cluster count and the asset total in the job snapshot.
- `clusters_done` never exceeds the post-filter cluster count.
- Junk clusters don't appear in `cluster_albums` (no album was
  created for them).
"""

from __future__ import annotations

import os
import sys
import tempfile
import uuid
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Fresh DB per test. Patches DB_PATH before app is imported so the
    module picks up the test path on first import; subsequent imports
    inside the test get the patched app.DB_PATH via mock.patch."""
    p = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(p))
    # Force a re-import of app + storage so DB_PATH-bound state (none
    # in storage, but app holds globals) is reset cleanly. Drop both
    # from sys.modules if present.
    for mod in ("app", "storage"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402,F401
    import app  # noqa: E402,F401
    storage.init_db(p)
    yield p
    # Clean up the jobs dict so a test that crashed mid-run doesn't
    # leak entries to the next test.
    app._cluster_album_jobs.clear()


def _seed_assets(conn, cluster_id: int, n: int) -> list[str]:
    """Insert `n` assets in `cluster_id`. Returns asset_ids."""
    ids = []
    for i in range(n):
        aid = f"asset-{cluster_id}-{i:04d}"
        ids.append(aid)
        conn.execute(
            "INSERT INTO assets "
            "(asset_id, cluster_id, original_file_name, original_url, thumbnail_url) "
            "VALUES (?, ?, ?, ?, ?)",
            (
                aid,
                cluster_id,
                f"img_{cluster_id}_{i:04d}.jpg",
                f"http://x/orig/{aid}.jpg",
                f"http://x/thumb/{aid}.jpg",
            ),
        )
    conn.commit()
    return ids


def _seed_species_score(conn, asset_id: str, taxon: str = "Quercus robur") -> None:
    """Insert one species-rank prediction row for the asset. The species
    stage writes top-k species to the `predictions` table (not
    taxon_scores); we mirror that here so the storage helper's check
    against `predictions.rank='species'` finds the row."""
    conn.execute(
        "INSERT INTO predictions "
        "(asset_id, rank, position, species, common_name, score) "
        "VALUES (?, 'species', 0, ?, 'oak', 0.85)",
        (asset_id, taxon),
    )
    conn.commit()


def _setup_clusters(
    db_path: Path, species_clusters: dict[int, int], junk_clusters: list[int]
) -> None:
    """Seed assets + species predictions. `species_clusters` maps
    cluster_id to asset count; each asset gets a species prediction.
    `junk_clusters` lists cluster_ids whose assets get only kingdom
    scores in taxon_scores (no species predictions)."""
    import storage  # local re-import in case test fixture dropped the module
    with storage.connect(db_path) as conn:
        for cid, n in species_clusters.items():
            ids = _seed_assets(conn, cid, n)
            for aid in ids:
                _seed_species_score(conn, aid)
        for cid in junk_clusters:
            ids = _seed_assets(conn, cid, 5)
            for aid in ids:
                conn.execute(
                    "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
                    "VALUES (?, 'kingdom', 'Plantae', 0.5)",
                    (aid,),
                )
        conn.commit()


class _FakeImmichClient:
    """Records calls; never hits the network. Albums are created with
    deterministic ids so we can assert against them."""

    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.created_albums: list[str] = []
        self.added: dict[str, list[str]] = {}
        self._next = 0

    def _new_id(self) -> str:
        self._next += 1
        return f"album-{self._next:03d}"

    def get_album(self, album_id: str):
        import httpx

        raise httpx.HTTPStatusError(
            "not found", request=mock.MagicMock(), response=mock.MagicMock(status_code=404)
        )

    def find_album_by_name(self, name: str):
        return None

    def create_album(self, *, name: str, description: str = "") -> dict:
        album_id = self._new_id()
        self.created_albums.append(album_id)
        self.added[album_id] = []
        return {"id": album_id, "albumName": name}

    def add_assets_to_album(self, *, album_id: str, asset_ids: list[str]) -> None:
        self.added.setdefault(album_id, []).extend(asset_ids)

    def close(self) -> None:
        pass


def _new_job_dict() -> dict:
    """Mirror the shape _cluster_album_sync_all_worker expects."""
    return {
        "status": "running",
        "scope": "all",
        "force_recreate": False,
        "people_filter": None,
        "n_clusters": 0,
        "n_clusters_skipped": 0,
        "n_assets_total": 0,
        "clusters_done": 0,
        "photos_added": 0,
        "current_cluster_id": None,
        "started_at": 0.0,
        "finished_at": None,
        "error": None,
    }


# ---------- storage helper ------------------------------------------------


def test_cluster_has_species_predictions_true_when_any_species_row(db_path):
    import storage  # noqa
    with storage.connect(db_path) as conn:
        _seed_assets(conn, 1, 3)
    with storage.connect(db_path) as conn:
        _seed_species_score(conn, "asset-1-0001")
    with storage.connect(db_path) as conn:
        assert storage.cluster_has_species_predictions(conn, 1) is True


def test_cluster_has_species_predictions_false_when_only_kingdom_scores(db_path):
    import storage  # noqa
    with storage.connect(db_path) as conn:
        _seed_assets(conn, 7, 3)
        conn.execute(
            "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
            "VALUES ('asset-7-0000', 'kingdom', 'Plantae', 0.7)"
        )
        conn.commit()
    with storage.connect(db_path) as conn:
        assert storage.cluster_has_species_predictions(conn, 7) is False


def test_cluster_has_species_predictions_false_for_empty_cluster(db_path):
    import storage  # noqa
    with storage.connect(db_path) as conn:
        assert storage.cluster_has_species_predictions(conn, 999) is False


# ---------- worker -------------------------------------------------------


def test_worker_excludes_junk_clusters_from_totals_and_progress(db_path):
    """Two real clusters (10 + 20 photos, both with species scores) and
    one junk cluster (5 photos, kingdom-only) should produce n_clusters=2
    and n_assets_total=30 — the junk cluster is excluded from both the
    cluster count and the asset total."""
    import app
    import storage  # noqa
    _setup_clusters(db_path, {10: 10, 20: 20}, junk_clusters=[30])

    fake = _FakeImmichClient("http://x", "k")
    job_id = uuid.uuid4().hex
    with app._cluster_album_jobs_lock:
        app._cluster_album_jobs[job_id] = _new_job_dict()

    with mock.patch.object(app, "_immich_client", return_value=fake), \
         mock.patch.dict(os.environ, {"IMMICH_BASE_URL": "http://x"}), \
         mock.patch("time.sleep"):
        app._cluster_album_sync_all_worker(job_id, force_recreate=False)

    snap = app._cluster_album_jobs[job_id]
    assert snap["status"] == "done"
    assert snap["n_clusters"] == 2, (
        f"junk cluster should be excluded from n_clusters, got {snap['n_clusters']}"
    )
    assert snap["n_assets_total"] == 30, (
        f"junk cluster's 5 photos should be excluded from n_assets_total, "
        f"got {snap['n_assets_total']}"
    )
    assert snap["clusters_done"] == 2, snap["clusters_done"]
    assert snap["n_clusters_skipped"] == 1, snap["n_clusters_skipped"]
    # Real clusters got albums; junk cluster did not.
    assert len(fake.created_albums) == 2, fake.created_albums
    with storage.connect(db_path) as conn:
        albums = storage.list_cluster_albums(conn)
        synced_ids = {a["cluster_id"] for a in albums}
    assert synced_ids == {10, 20}, synced_ids


def test_worker_progress_never_exceeds_filtered_count(db_path):
    """Even when junk clusters exist, clusters_done must stop at the
    filtered count — not iterate through them."""
    import app
    import storage  # noqa
    # 1 real + 5 junk. Progress should plateau at clusters_done=1.
    _setup_clusters(db_path, {1: 4}, junk_clusters=[100, 101, 102, 103, 104])

    fake = _FakeImmichClient("http://x", "k")
    job_id = uuid.uuid4().hex
    with app._cluster_album_jobs_lock:
        app._cluster_album_jobs[job_id] = _new_job_dict()

    observed_max = {"v": 0}

    def _track_set_cluster_album(conn, **kw):
        observed_max["v"] = max(
            observed_max["v"],
            app._cluster_album_jobs[job_id].get("clusters_done", 0),
        )
        return storage.set_cluster_album(conn, **kw)

    with mock.patch.object(app, "_immich_client", return_value=fake), \
         mock.patch.dict(os.environ, {"IMMICH_BASE_URL": "http://x"}), \
         mock.patch.object(storage, "set_cluster_album", _track_set_cluster_album), \
         mock.patch("time.sleep"):
        app._cluster_album_sync_all_worker(job_id, force_recreate=False)

    assert observed_max["v"] <= 1, (
        f"progress jumped past filtered count: observed max {observed_max['v']}, "
        f"expected at most 1"
    )


def test_worker_all_clusters_junk_yields_zero_cluster_run(db_path):
    """Edge case: every cluster is junk. The worker should still
    terminate cleanly with n_clusters=0 and no albums created — though
    in practice the preview endpoint would 400 before this runs."""
    import app
    import storage  # noqa
    _setup_clusters(db_path, {}, junk_clusters=[1, 2, 3])

    fake = _FakeImmichClient("http://x", "k")
    job_id = uuid.uuid4().hex
    with app._cluster_album_jobs_lock:
        app._cluster_album_jobs[job_id] = _new_job_dict()

    with mock.patch.object(app, "_immich_client", return_value=fake), \
         mock.patch.dict(os.environ, {"IMMICH_BASE_URL": "http://x"}), \
         mock.patch("time.sleep"):
        app._cluster_album_sync_all_worker(job_id, force_recreate=False)

    snap = app._cluster_album_jobs[job_id]
    assert snap["status"] == "done"
    assert snap["n_clusters"] == 0
    assert snap["clusters_done"] == 0
    assert fake.created_albums == []
