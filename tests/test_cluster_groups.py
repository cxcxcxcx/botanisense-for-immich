"""Tests for cluster super-groups (the hierarchy level above clusters).

/api/cluster-groups cuts a complete-linkage dendrogram over cluster
centroids at a cosine-distance threshold (clusters.CLUSTER_GROUP_DISTANCE):
only genuinely-close clusters merge into a family, the rest land in the
Ungrouped bucket (group_id 0). /api/results?cluster_group=N filters the
grid to a family's member clusters, resolving N through the SAME cached
sorted member list the cards were numbered with.

Pins, on hand-built 4-D centroids (no BioCLIP):
  - tight pairs merge, a distant cluster stays solo
  - every input cluster appears exactly once (families + solo)
  - group-id resolution covers 0, 1..N, and rejects out-of-range
  - /api/cluster-groups cards and /api/results filtering agree
"""

from __future__ import annotations

import struct
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _vec_blob(v: list[float]) -> bytes:
    return struct.pack(f"<{len(v)}f", *v)


def _ids_and_centroids(spec: dict[int, list[list[float]]]):
    """Centroid matrix for hand-built clusters: mean of member vectors,
    L2-normalized — mirrors clusters.cluster_centroids."""
    import numpy as np  # noqa: E402

    ids = sorted(spec)
    rows = []
    for cid in ids:
        m = np.mean(np.array(spec[cid], dtype=np.float32), axis=0)
        rows.append(m / max(float(np.linalg.norm(m)), 1e-9))
    return ids, np.stack(rows)


# Two tight pairs on orthogonal axes + one far-away singleton.
A1, A2 = [1.0, 0.0, 0.0, 0.0], [0.95, 0.1, 0.0, 0.0]
B1, B2 = [0.0, 1.0, 0.0, 0.0], [0.1, 0.95, 0.0, 0.0]
SOLO = [0.0, 0.0, 1.0, 0.0]
SPEC = {1: [A1, A1], 2: [A2, A2], 3: [B1, B1], 4: [B2, B2], 5: [SOLO, SOLO]}


def test_pairs_merge_and_stray_stays_solo():
    import clusters  # noqa: E402
    ids, cent = _ids_and_centroids(SPEC)
    families, solo = clusters.build_cluster_groups(ids, cent)
    assert families == [[1, 2], [3, 4]]  # size desc, ids asc within
    assert solo == [5]
    # Every cluster appears exactly once.
    assert sorted(sum(families, []) + solo) == ids


def test_single_cluster_and_empty_inputs():
    import clusters  # noqa: E402
    import numpy as np  # noqa: E402
    assert clusters.build_cluster_groups([], np.empty((0, 4), dtype=np.float32)) == ([], [])
    ids, cent = _ids_and_centroids({7: [A1, A1]})
    assert clusters.build_cluster_groups(ids, cent) == ([], [7])


def test_loose_threshold_merges_everything_into_one_family():
    """A distance cut ≥ the largest pairwise distance collapses the tree
    into one family with no solo members — pins that max_distance is
    actually honored (0.15 must NOT merge the orthogonal axes)."""
    import clusters  # noqa: E402
    ids, cent = _ids_and_centroids(SPEC)
    families, solo = clusters.build_cluster_groups(ids, cent, max_distance=2.0)
    assert families == [[1, 2, 3, 4, 5]]
    assert solo == []
    families, solo = clusters.build_cluster_groups(ids, cent, max_distance=0.15)
    assert [1, 2] in families and [3, 4] in families and solo == [5]


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    monkeypatch.setenv("IMMICH_API_KEY", "test-key")
    monkeypatch.setenv("IMMICH_BASE_URL", "http://immich.test")
    monkeypatch.delenv("CORS_ORIGINS", raising=False)
    import dotenv  # noqa: E402
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(db_path)
    storage._GLOBAL_NEAR_DUP_CACHE = {}
    import app  # noqa: E402
    app.DB_PATH = db_path
    return app


def _seed(app, cluster_id: int, vectors: list[list[float]], names_all: dict[int, str]) -> None:
    import storage  # noqa: E402
    with storage.connect(app.DB_PATH) as conn:
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
        conn.execute(
            "INSERT INTO cluster_names (cluster_id, name, n_total,"
            " top_species_json, top_genus_json) VALUES (?, ?, ?, '[]', '[]')",
            (cluster_id, names_all[cluster_id], len(vectors)),
        )
        conn.commit()


def test_group_card_votes_genus_and_noise_falls_back_to_mixed():
    import app as app_mod  # noqa: E402
    meta = {
        1: {"n": 10, "top_genus": [{"genus": "Rosa", "count": 3}]},
        2: {"n": 4, "top_genus": [{"genus": "Rosa", "count": 3}]},
        3: {"n": 2, "top_genus": [{"genus": "Cirsium", "count": 3},
                                   {"genus": "Malus", "count": 3}]},
    }
    card = app_mod._group_card(1, [1, 2, 3], meta)
    assert card["name"] == "Rosa"       # 6/12 ≥ 40%
    assert card["n"] == 16 and card["n_clusters"] == 3
    mixed = app_mod._group_card(2, [1, 3], {1: meta[1], 3: meta[3]})
    assert mixed["name"] == "Mixed"     # top genus 3/9 < 40%
    assert app_mod._group_card(3, [1], {}) is None  # no metadata at all


def test_cluster_groups_endpoint_and_results_filter_agree(app_module):
    app = app_module
    _seed(app, 1, [A1, A1], {1: "Rosa"})
    _seed(app, 2, [A2, A2], {2: "China rose"})
    _seed(app, 3, [B1, B1], {3: "Acer"})
    _seed(app, 4, [B2, B2], {4: "Japanese Maple"})
    _seed(app, 5, [SOLO, SOLO], {5: "Lilium"})
    app._invalidate_clusters_cache()

    from fastapi.testclient import TestClient  # noqa: E402
    client = TestClient(app.app)

    body = client.get("/api/cluster-groups").json()
    assert body["ungrouped"]["name"] == "Ungrouped"
    assert body["ungrouped"]["cluster_ids"] == [5]
    assert body["ungrouped"]["n"] == 2
    families = body["groups"]
    assert len(families) == 2
    # Sorted by photo count desc; both families here have n=4 → stable
    # family order ([[1,2] before [3,4]]) decides the numbering.
    assert families[0]["cluster_ids"] == [1, 2]
    assert families[0]["group_id"] == 1
    assert families[1]["cluster_ids"] == [3, 4]

    # /api/results resolves through the SAME member lists.
    r = client.get("/api/results", params={"cluster_group": 1, "page_size": 50})
    rows = r.json()["results"]
    assert {row["cluster_id"] for row in rows} == {1, 2}
    assert r.json()["pagination"]["total"] == 4  # 2 assets × 2 clusters

    r0 = client.get("/api/results", params={"cluster_group": 0, "page_size": 50})
    assert {row["cluster_id"] for row in r0.json()["results"]} == {5}

    # Out-of-range id: empty grid, never the whole library.
    r_bad = client.get("/api/results", params={"cluster_group": 99})
    assert r_bad.json()["pagination"]["total"] == 0

    # Ungrouped bucket card name is forced, family names come from the
    # genus vote (no predictions seeded → "Mixed").
    assert families[0]["name"] == "Mixed"


def test_invalidate_clears_group_index(app_module):
    app = app_module
    app._cluster_sim_cache["groups"] = ([[1, 2]], [3])
    app._invalidate_clusters_cache()
    assert app._cluster_sim_cache == {}
