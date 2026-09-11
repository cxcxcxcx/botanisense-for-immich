"""Tests for cluster-to-cluster similarity (related-clusters browse).

The UI's "Related" strip under the Level-2 back-bar is driven by
/api/clusters/related, which ranks OTHER clusters by cosine similarity
between cluster centroids (mean of member embeddings, L2-normalized).
These tests pin the math and the endpoint contract:

  - centroids are unit-length and one-per-cluster (embedded assets only)
  - the queried cluster never appears in its own related list
  - ranking matches the geometry (nearest group wins)
  - the endpoint joins display metadata (name/n/sample) from the same
    cached payload /api/clusters serves

Embeddings here are hand-built 4-D vectors — no BioCLIP involved. The
storage layer keeps vectors as little-endian float32 blobs, which is all
these tests rely on.
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _vec_blob(v: list[float]) -> bytes:
    """Pack a vector the same way pipeline.py stores embeddings."""
    return struct.pack(f"<{len(v)}f", *v)


@pytest.fixture()
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(p))
    for mod in ("storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    storage._GLOBAL_NEAR_DUP_CACHE = None


def _seed_cluster(db_path, cluster_id: int, vectors: list[list[float]]) -> None:
    """One cluster with `len(vectors)` embedded plant assets."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        for i, v in enumerate(vectors):
            aid = f"c{cluster_id}-{i}"
            conn.execute(
                "INSERT INTO assets (asset_id, cluster_id, original_file_name,"
                " original_url, thumbnail_url) VALUES (?, ?, ?, ?, ?)",
                (aid, cluster_id, f"{aid}.jpg", f"http://x/orig/{aid}.jpg",
                 f"http://x/thumb/{aid}.jpg"),
            )
            conn.execute(
                "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score)"
                " VALUES (?, 'kingdom', 'Plantae', 0.9)",
                (aid,),
            )
            conn.execute(
                "INSERT INTO embeddings (asset_id, dim, vector) VALUES (?, ?, ?)",
                (aid, len(v), _vec_blob(v)),
            )
        conn.commit()


# Three well-separated directions + one cluster that straddles A and B.
A_DIR = [1.0, 0.0, 0.0, 0.0]
B_DIR = [0.0, 1.0, 0.0, 0.0]
C_DIR = [0.0, 0.0, 1.0, 0.0]


@pytest.fixture()
def seeded(db_path):
    # Clusters 1..3 sit on orthogonal axes; cluster 4 mixes A and B so
    # its centroid is closest to whichever of 1/2 contributed more.
    _seed_cluster(db_path, 1, [A_DIR, [0.9, 0.1, 0.0, 0.0]])
    _seed_cluster(db_path, 2, [B_DIR, [0.0, 0.9, 0.1, 0.0]])
    _seed_cluster(db_path, 3, [C_DIR, [0.0, 0.0, 0.9, 0.1]])
    _seed_cluster(db_path, 4, [[0.9, 0.5, 0.0, 0.0]])  # mostly A, some B
    return db_path


def test_centroids_are_unit_length_and_aligned(seeded):
    import storage  # noqa: E402
    import clusters  # noqa: E402
    with storage.connect(seeded) as conn:
        cluster_ids, centroids = clusters.cluster_centroids(conn)
    assert cluster_ids == [1, 2, 3, 4]
    assert centroids.shape == (4, 4)
    # Every centroid must be normalized so cosine == dot product.
    assert np.allclose(np.linalg.norm(centroids, axis=1), 1.0, atol=1e-5)
    # Cluster 1 is pure A-direction.
    assert centroids[0].argmax() == 0


def test_assets_without_cluster_id_are_ignored(seeded):
    import storage  # noqa: E402
    import clusters  # noqa: E402
    with storage.connect(seeded) as conn:
        # An embedded asset with no cluster (non-plant / HDBSCAN noise).
        conn.execute(
            "INSERT INTO assets (asset_id, original_file_name, original_url,"
            " thumbnail_url) VALUES ('orphan', 'o.jpg', 'http://x/orig/o.jpg',"
            " 'http://x/thumb/o.jpg')"
        )
        conn.execute(
            "INSERT INTO embeddings (asset_id, dim, vector) VALUES (?, ?, ?)",
            ("orphan", len(A_DIR), _vec_blob(A_DIR)),
        )
        conn.commit()
        cluster_ids, centroids = clusters.cluster_centroids(conn)
    assert cluster_ids == [1, 2, 3, 4]  # no phantom cluster for the orphan


def test_nearest_excludes_self_and_ranks_by_geometry(seeded):
    import storage  # noqa: E402
    import clusters  # noqa: E402
    with storage.connect(seeded) as conn:
        cluster_ids, centroids = clusters.cluster_centroids(conn)
        neighbors = clusters.nearest_clusters(
            cluster_ids, centroids, cluster_id=4, k=3
        )
    ids = [n["cluster_id"] for n in neighbors]
    # Cluster 4 = 0.9A + 0.5B → closest to 1 (pure A), then 2 (pure B),
    # then 3 (orthogonal to both). Self (4) must never appear.
    assert ids == [1, 2, 3]
    sims = [n["similarity"] for n in neighbors]
    # Cluster 3 is exactly orthogonal to 4 → its similarity rounds to 0.
    assert sims[0] > sims[1] > sims[2] >= 0.0


def test_nearest_respects_k_and_missing_cluster(seeded):
    import storage  # noqa: E402
    import clusters  # noqa: E402
    with storage.connect(seeded) as conn:
        cluster_ids, centroids = clusters.cluster_centroids(conn)
        one = clusters.nearest_clusters(cluster_ids, centroids, cluster_id=1, k=1)
        none = clusters.nearest_clusters(cluster_ids, centroids, cluster_id=999)
    assert len(one) == 1
    # Unknown cluster (no embedded assets) → nothing to compare.
    assert none == []


def test_sim_cache_invalidated_with_clusters_cache(seeded, monkeypatch):
    """app._invalidate_clusters_cache must also drop the centroid index —
    a refit renumbers clusters, so a stale centroid matrix would relate
    the wrong IDs."""
    import storage  # noqa: E402
    import clusters  # noqa: E402
    db = seeded
    monkeypatch.setenv("IMMICH_API_KEY", "test-key")
    monkeypatch.setenv("IMMICH_BASE_URL", "http://immich.test")
    import dotenv  # noqa: E402
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app",):
        sys.modules.pop(mod, None)
    import app  # noqa: E402

    with storage.connect(db) as conn:
        ids, centroids = clusters.cluster_centroids(conn)
    app._cluster_sim_cache["index"] = (ids, centroids)
    app._invalidate_clusters_cache()
    assert app._cluster_sim_cache == {}


def test_related_endpoint_contract(seeded, monkeypatch):
    import storage  # noqa: E402
    import clusters  # noqa: E402
    db = seeded
    monkeypatch.setenv("IMMICH_API_KEY", "test-key")
    monkeypatch.setenv("IMMICH_BASE_URL", "http://immich.test")
    import dotenv  # noqa: E402
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app",):
        sys.modules.pop(mod, None)
    import app  # noqa: E402
    app.DB_PATH = db
    app._invalidate_clusters_cache()

    # Name every cluster the way compute_cluster_names would have, so
    # the test proves the endpoint joins labels from cluster_names. All
    # four rows are needed: ensure_cluster_names recomputes (wiping the
    # table) whenever assets carry MORE distinct cluster_ids than the
    # table has rows.
    with storage.connect(db) as conn:
        for cid, name in ((1, "Quercus"), (2, "Acer"), (3, "Betula"), (4, "Mixed")):
            conn.execute(
                "INSERT INTO cluster_names (cluster_id, name, n_total,"
                " top_species_json, top_genus_json) VALUES (?, ?, ?, '[]', '[]')",
                (cid, name, 2),
            )
        conn.commit()

    from fastapi.testclient import TestClient
    client = TestClient(app.app)
    r = client.get("/api/clusters/related", params={"cluster_id": 4, "limit": 2})
    assert r.status_code == 200
    body = r.json()
    assert body["cluster_id"] == 4
    related = body["related"]
    assert [x["cluster_id"] for x in related] == [1, 2]  # top-2, self excluded
    assert related[0]["name"] == "Quercus"
    assert related[0]["n"] == 2
    assert related[0]["sample_asset_id"] == "c1-0"
    assert 0.0 < related[0]["similarity"] < 1.0
