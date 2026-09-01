"""Regression tests for plant-only clustering.

Background: the cluster algorithm (HDBSCAN) used to fit on every
embedding regardless of BioCLIP kingdom score, so a single high-quality
photo of a person or car (BioCLIP Plantae score < 0.35) would land in
a random plant cluster via the BioCLIP embedding similarity. The fix
scopes every step of cluster computation to assets with Plantae score
>= CLUSTER_PLANTAE_MIN (0.35, the yellow-floor boundary in the UI):

  - `clusters.refit_with_remap` only fits HDBSCAN on plant embeddings
    and clears `cluster_id` on non-plant assets after each refit.
  - `clusters.assign_new_points` filters the input asset_ids to plants
    only before calling `approximate_predict`.
  - `clusters.should_refit` compares the current plant-count to
    `n_assets_at_fit`, not the total embedding count (so non-plant
    photos flowing in don't trigger spurious refits).

These tests pin those three behaviors end-to-end against a fresh DB.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Fresh DB per test. Uses a temporary DB_PATH so we can import
    `clusters` and `storage` without dragging in the rest of `app`
    (BioCLIP subprocess init, etc.)."""
    p = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(p))
    # Drop any cached imports so storage picks up the new DB_PATH.
    for mod in ("clusters", "storage"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    # Clean up the clusterer_state singleton so a parallel-pickled
    # clusterer doesn't leak across tests. (clear_clusterer_state is
    # only safe inside a connection; we just rely on per-test module
    # re-import for isolation.)


def _seed_asset(
    conn,
    asset_id: str,
    *,
    cluster_id: int | None = None,
    vector: np.ndarray | None = None,
) -> None:
    """Asset row + embedding in one helper. `vector` defaults to a
    16-dim zero placeholder — tests that need distinct clustering
    structure can pass in their own vector and `_seed_with_*` helpers
    will UPSERT it on top."""
    conn.execute(
        "INSERT INTO assets "
        "(asset_id, cluster_id, original_file_name, original_url, "
        "thumbnail_url) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            asset_id,
            cluster_id,
            f"{asset_id}.jpg",
            f"http://x/orig/{asset_id}.jpg",
            f"http://x/thumb/{asset_id}.jpg",
        ),
    )
    vec = vector if vector is not None else np.zeros(16, dtype="<f4")
    conn.execute(
        "INSERT INTO embeddings (asset_id, dim, vector) "
        "VALUES (?, ?, ?)",
        (asset_id, int(vec.size), vec.astype("<f4").tobytes()),
    )


def _seed_plantae(conn, asset_id: str, score: float) -> None:
    """Plantae kingdom score for an asset. Mirrors what the pipeline's
    kingdom stage writes."""
    conn.execute(
        "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
        "VALUES (?, 'kingdom', 'Plantae', ?)",
        (asset_id, float(score)),
    )


def _make_distinct_vectors(n: int, dim: int = 16) -> list[np.ndarray]:
    """Return n embeddings that are pairwise orthogonal-ish so HDBSCAN
    can find at least 2 dense clusters. Without this the test would be
    dominated by HDBSCAN's noise emission and the assertions on
    cluster counts would be flaky."""
    rng = np.random.default_rng(seed=42)
    out = []
    for i in range(n):
        v = rng.standard_normal(dim).astype("<f4")
        v /= np.linalg.norm(v) + 1e-9
        # Force a per-vector signature: each pole owns the mass at one
        # coordinate so HDBSCAN sees tight groups rather than one
        # diffuse cloud.
        pole = i % 3  # 3 poles
        sig = np.zeros(dim, dtype="<f4")
        sig[pole * (dim // 3)] = 10.0
        v = v + sig
        v /= np.linalg.norm(v) + 1e-9
        out.append(v)
    return out


def _seed_with_distinct_vecs(
    conn, asset_id: str, vec: np.ndarray
) -> None:
    """UPSERT the embedding row for `asset_id` with the given vector.
    Use this AFTER `_seed_asset` to overwrite the zero-placeholder
    with a structured vector."""
    conn.execute(
        "INSERT INTO embeddings (asset_id, dim, vector) "
        "VALUES (?, ?, ?) "
        "ON CONFLICT(asset_id) DO UPDATE SET "
        "  dim = excluded.dim, vector = excluded.vector",
        (asset_id, int(vec.size), vec.astype("<f4").tobytes()),
    )


def _cluster_id_for(conn, asset_id: str) -> int | None:
    row = conn.execute(
        "SELECT cluster_id FROM assets WHERE asset_id = ?",
        (asset_id,),
    ).fetchone()
    return int(row["cluster_id"]) if row and row["cluster_id"] is not None else None


# ---------- refit_with_remap ------------------------------------------


def test_refit_only_clusters_plants(db_path):
    """Three plant assets + three non-plant assets. After refit: every
    plant asset has a cluster_id; every non-plant asset has cluster_id
    = NULL."""
    import storage  # noqa: E402
    import clusters  # noqa: E402

    vec = _make_distinct_vectors(1)[0]  # same vector for everyone
    with storage.connect(db_path) as conn:
        # 3 plants.
        for aid in ("plant-A", "plant-B", "plant-C"):
            _seed_asset(conn, aid)
            _seed_plantae(conn, aid, 0.85)
            _seed_with_distinct_vecs(conn, aid, vec)
        # 3 non-plants (Plantae score below the 0.35 floor).
        for aid in ("non-A", "non-B", "non-C"):
            _seed_asset(conn, aid)
            _seed_plantae(conn, aid, 0.10)
            _seed_with_distinct_vecs(conn, aid, vec)
        conn.commit()

    with storage.connect(db_path) as conn:
        asset_to_cluster, n_clusters, _remap = clusters.refit_with_remap(conn)

    # Plant assets appear in the mapping.
    for aid in ("plant-A", "plant-B", "plant-C"):
        assert aid in asset_to_cluster, (
            f"plant {aid} missing from final mapping: {sorted(asset_to_cluster)}"
        )
    # Non-plants do NOT.
    for aid in ("non-A", "non-B", "non-C"):
        assert aid not in asset_to_cluster, (
            f"non-plant {aid} leaked into final mapping: "
            f"{sorted(asset_to_cluster)}"
        )

    with storage.connect(db_path) as conn:
        # Non-plant assets have their cluster_id cleared.
        for aid in ("non-A", "non-B", "non-C"):
            assert _cluster_id_for(conn, aid) is None, (
                f"non-plant {aid} retained cluster_id "
                f"{_cluster_id_for(conn, aid)} after refit"
            )
        # Plants still have one.
        for aid in ("plant-A", "plant-B", "plant-C"):
            assert _cluster_id_for(conn, aid) is not None, (
                f"plant {aid} lost cluster_id after refit"
            )


def test_refit_clears_stale_cluster_id_on_non_plants(db_path):
    """A non-plant asset that was previously clustered (cluster_id set
    to e.g. 17 by an earlier run) must have its cluster_id cleared on
    the next refit. Pins the plant-only invariant for legacy data."""
    import storage  # noqa: E402
    import clusters  # noqa: E402

    vec = _make_distinct_vectors(1)[0]
    with storage.connect(db_path) as conn:
        # Non-plant asset with a stale cluster_id from an earlier botani
        # run. Plantae score 0.20 puts it well below the 0.35 floor.
        _seed_asset(conn, "legacy-non", cluster_id=17)
        _seed_plantae(conn, "legacy-non", 0.20)
        _seed_with_distinct_vecs(conn, "legacy-non", vec)
        # Three plant assets — enough for HDBSCAN (min_cluster_size=2)
        # to fit at least one cluster. Distinct vecs so they're not
        # collapsed into noise.
        for aid, pole in (("plant-a", 0), ("plant-b", 1), ("plant-c", 2)):
            _seed_asset(conn, aid)
            _seed_plantae(conn, aid, 0.80)
            _seed_with_distinct_vecs(conn, aid, vec)
        conn.commit()

    with storage.connect(db_path) as conn:
        clusters.refit_with_remap(conn)

    with storage.connect(db_path) as conn:
        assert _cluster_id_for(conn, "legacy-non") is None, (
            "non-plant asset with stale cluster_id was not cleared"
        )
        for aid in ("plant-a", "plant-b", "plant-c"):
            assert _cluster_id_for(conn, aid) is not None, (
                f"{aid} lost its cluster_id after refit"
            )


def test_refit_with_only_non_plants_clears_everything(db_path):
    """Edge case: every asset is non-plant. refit must clear the
    clusterer state and every cluster_id, returning an empty mapping."""
    import storage  # noqa: E402
    import clusters  # noqa: E402

    with storage.connect(db_path) as conn:
        for aid in ("non-A", "non-B", "non-C"):
            _seed_asset(conn, aid, cluster_id=5)
            _seed_plantae(conn, aid, 0.10)
            vec = _make_distinct_vectors(1)[0]
            _seed_with_distinct_vecs(conn, aid, vec)
        conn.commit()

    with storage.connect(db_path) as conn:
        asset_to_cluster, n_clusters, _remap = clusters.refit_with_remap(conn)

    assert asset_to_cluster == {}
    assert n_clusters == 0

    with storage.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT COUNT(*) AS n FROM assets WHERE cluster_id IS NOT NULL"
        ).fetchone()
        assert int(rows["n"]) == 0, (
            f"non-plant cluster_ids not cleared: {rows['n']} still set"
        )
        # Clusterer state row should be cleared too — no model to load.
        state = storage.get_clusterer_state(conn)
        assert state is None, (
            f"clusterer_state should be cleared when no plants fit, "
            f"got {state}"
        )


# ---------- should_refit --------------------------------------------


def test_should_refit_compares_plant_count(db_path):
    """Adding non-plant assets must NOT trigger a refit. Adding plant
    assets must. Pins the policy: `growth_threshold_pct` is measured
    against the plant population that HDBSCAN actually fits against."""
    import storage  # noqa: E402
    import clusters  # noqa: E402

    with storage.connect(db_path) as conn:
        # Pretend a previous fit had 5 plant assets.
        # We can't easily run HDBSCAN here without real embeddings, so
        # write the clusterer_state row directly with n_assets_at_fit=5.
        import pickle as _pickle
        # Use a tiny but valid HDBSCAN clusterer to satisfy unpickle.
        # Easier: just bypass the clusterer load by setting the state
        # to a sentinel that load_clusterer rejects and falls back to
        # first-fit (always True). Then exercise should_refit
        # against an "already fit" state by inserting a state row.
        # We use a dict-shaped blob that load_clusterer accepts.
        sentinel_clusterer = clusters.hdbscan.HDBSCAN(
            min_cluster_size=2, metric="euclidean"
        )
        blob = _pickle.dumps(
            {"clusterer": sentinel_clusterer, "label_map": {-1: 0}}
        )
        conn.execute(
            "INSERT INTO clusterer_state "
            "(id, clusterer_blob, min_cluster_size, n_assets_at_fit, "
            " n_clusters) VALUES (1, ?, 2, 5, 1)",
            (blob,),
        )
        # 5 plant assets (matches n_assets_at_fit).
        for i in range(5):
            aid = f"plant-{i}"
            _seed_asset(conn, aid)
            _seed_plantae(conn, aid, 0.80)
        conn.commit()

    with storage.connect(db_path) as conn:
        # Right at threshold — should NOT refit (growth 0%).
        assert clusters.should_refit(conn, growth_threshold_pct=0.10) is False

        # Add only non-plant assets — growth is irrelevant to HDBSCAN.
        with storage.connect(db_path) as conn:
            for i in range(20):
                aid = f"non-{i}"
                _seed_asset(conn, aid)
                _seed_plantae(conn, aid, 0.10)
            conn.commit()
        with storage.connect(db_path) as conn:
            assert clusters.should_refit(conn, growth_threshold_pct=0.10) is False, (
                "non-plant additions should not trigger refit"
            )

        # Add a plant asset — bring plant-count from 5 to 6 (20% growth).
        with storage.connect(db_path) as conn:
            _seed_asset(conn, "plant-extra")
            _seed_plantae(conn, "plant-extra", 0.80)
            conn.commit()
        with storage.connect(db_path) as conn:
            assert clusters.should_refit(conn, growth_threshold_pct=0.10) is True, (
                "20% growth in plant count must trigger refit"
            )


# ---------- assign_new_points ---------------------------------------


def test_assign_new_points_skips_non_plants(db_path):
    """Incremental-assign path: a non-plant asset_id handed in must
    not appear in the returned mapping."""
    import storage  # noqa: E402
    import clusters  # noqa: E402

    # Stage: a small plant population has been clustered once (need at
    # least min_cluster_size=2 points for HDBSCAN to fit).
    import pickle as _pickle
    vec = _make_distinct_vectors(1)[0]
    with storage.connect(db_path) as conn:
        for aid in ("seed-plant-1", "seed-plant-2", "seed-plant-3"):
            _seed_asset(conn, aid)
            _seed_plantae(conn, aid, 0.90)
            _seed_with_distinct_vecs(conn, aid, vec)
        conn.commit()
    with storage.connect(db_path) as conn:
        clusters.refit_with_remap(conn)

    # Now feed a mix: one plant + one non-plant.
    with storage.connect(db_path) as conn:
        for aid, score in (("new-plant", 0.90), ("new-non", 0.10)):
            _seed_asset(conn, aid)
            _seed_plantae(conn, aid, score)
            _seed_with_distinct_vecs(conn, aid, vec)
        conn.commit()

    ids = ["new-plant", "new-non"]
    vecs = np.stack([vec, vec])
    with storage.connect(db_path) as conn:
        result = clusters.assign_new_points(
            conn, new_asset_ids=ids, new_vectors=vecs
        )

    assert result is not None
    assert "new-plant" in result, (
        f"new-plant missing from assign result: {sorted(result)}"
    )
    assert "new-non" not in result, (
        f"new-non (non-plant) should not appear in assign result: "
        f"{sorted(result)}"
    )
