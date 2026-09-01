"""Regression test for `cluster_summary` ordering = size DESC.

Pins the project decision to render the cluster-cards grid with the
largest clusters at the top. A regression here (e.g. reverting to
cluster_id ASC) would push tiny 1-2 photo clusters above the user's
biggest plant groupings — confusing on a personal-library browse.

The cluster_id ASC tiebreaker is also tested so two clusters with the
same photo count render in a stable order across refits.
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
    for mod in ("storage",):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    storage._GLOBAL_NEAR_DUP_CACHE = None


def _seed_asset(conn, asset_id: str, cluster_id: int | None = None) -> None:
    """Minimal asset row + kingdom-tier taxon score. Plant-only is
    intrinsic now (clusters.py clears non-plant cluster_ids), so we
    give every asset a Plantae score above the floor — the test cares
    about ordering, not the cluster policy."""
    conn.execute(
        "INSERT INTO assets "
        "(asset_id, cluster_id, original_file_name, original_url, "
        " thumbnail_url) "
        "VALUES (?, ?, ?, ?, ?)",
        (
            asset_id,
            cluster_id,
            f"{asset_id}.jpg",
            f"http://x/orig/{asset_id}.jpg",
            f"http://x/thumb/{asset_id}.jpg",
        ),
    )
    # Plantae score so the asset keeps its cluster_id after a (hypothetical)
    # refit; the test doesn't actually refit, so this is belt-and-suspenders.
    conn.execute(
        "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
        "VALUES (?, 'kingdom', 'Plantae', ?)",
        (asset_id, 0.90),
    )


def _seed_cluster(conn, cluster_id: int, n: int) -> None:
    """Insert `n` assets assigned to cluster_id."""
    for i in range(n):
        _seed_asset(conn, f"c{cluster_id}-{i:04d}", cluster_id=cluster_id)


def _seed(db_path: Path, sizes: dict[int, int]) -> None:
    """Seed clusters with the given {cluster_id: asset_count}. Sizes
    are intentionally non-monotonic so naive cluster_id ASC would
    obviously NOT match the expected order."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        for cid, n in sizes.items():
            _seed_cluster(conn, cid, n)
        conn.commit()


def test_cluster_summary_orders_by_size_desc(db_path):
    """cluster_summary must return rows in `n DESC, cluster_id ASC`
    order — largest cluster first."""
    import storage  # noqa: E402

    # Non-monotonic sizes so cluster_id ASC would diverge from size DESC.
    _seed(db_path, {7: 5, 42: 100, 100: 12, 3: 50, 88: 1})

    with storage.connect(db_path) as conn:
        rows = storage.cluster_summary(conn)

    cluster_counts = [(r["cluster_id"], r["n"]) for r in rows]
    assert cluster_counts == [
        (42, 100),  # largest
        (3, 50),
        (100, 12),
        (7, 5),
        (88, 1),   # smallest
    ], f"size-desc order expected; got {cluster_counts}"


def test_cluster_summary_ties_break_by_cluster_id_asc(db_path):
    """When two clusters have identical photo counts, cluster_id ASC
    is the tiebreaker — so the order is stable across refits (HDBSCAN
    reshuffles membership; cluster_ids are persisted)."""
    import storage  # noqa: E402

    # Three clusters, all 3 photos — tiebreak by cluster_id ASC.
    _seed(db_path, {99: 3, 7: 3, 50: 3})

    with storage.connect(db_path) as conn:
        rows = storage.cluster_summary(conn)

    ids_in_order = [r["cluster_id"] for r in rows]
    assert ids_in_order == [7, 50, 99], (
        f"tied counts should tiebreak by cluster_id ASC; got {ids_in_order}"
    )


def test_cluster_summary_empty_when_no_clusters(db_path):
    """No assets (or no clustered assets) → empty list. The ORDER BY
    clause must not crash on an empty result set."""
    import storage  # noqa: E402
    # No seeds — every cluster_id is NULL or absent.
    with storage.connect(db_path) as conn:
        rows = storage.cluster_summary(conn)
    assert rows == []
