"""Regression tests for `include_non_plants` on the grid endpoint.

Pins the project-wide "no non-plant counts in pages" policy applied to
`storage.get_all_results` (the underlying query for /api/results):

  - Default (`include_non_plants=False`): the all-view drops non-plant
    assets from both the page slice AND `total_assets`. Plantae score
    < plantae_yellow (0.35) → excluded.
  - Opt-in (`include_non_plants=True`): every asset is counted,
    regardless of Plantae score. Legacy behavior.
  - Explicit `plantae` tier always wins regardless of the toggle —
    e.g. `plantae="non_plant"` returns non-plants even when
    `include_non_plants=False`.

Edges and per-filter cases are folded into single tests for compactness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    """Fresh DB per test, mirroring the fixture used by
    test_cluster_excludes_non_plants. Drops both `storage` and
    `clusters` from sys.modules so a seeded `_GLOBAL_NEAR_DUP_CACHE`
    can't leak across tests."""
    p = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(p))
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    # Reset the per-process near-dup cache so the next test starts with
    # a clean (None → recompute) state instead of inheriting ours.
    storage._GLOBAL_NEAR_DUP_CACHE = None


def _seed(db_path: Path, plantae_scores: dict[str, float]) -> None:
    """Bulk-seed a small gallery with the given Plantae scores.

    Spread is intentional: we want plants in all three tiers (yellow
    floor, green tier) and non-plants across the spectrum so the
    policy is exercised in both directions. Opens its own connection
    so callers don't have to thread a `with storage.connect(...)` block
    through the `_call` helper."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        for aid, score in plantae_scores.items():
            conn.execute(
                "INSERT INTO assets "
                "(asset_id, original_file_name, original_url, thumbnail_url) "
                "VALUES (?, ?, ?, ?)",
                (
                    aid,
                    f"{aid}.jpg",
                    f"http://x/orig/{aid}.jpg",
                    f"http://x/thumb/{aid}.jpg",
                ),
            )
            conn.execute(
                "INSERT INTO taxon_scores "
                "(asset_id, rank, taxon_name, score) "
                "VALUES (?, 'kingdom', 'Plantae', ?)",
                (aid, float(score)),
            )
        conn.commit()


def _call(conn, **kwargs) -> tuple[list, int, int]:
    """Run storage.get_all_results with an empty near-dup map (every
    asset is its own representative, so total_reps == total_assets).
    The plumbing (joins, ORDER BY, LIMIT) is irrelevant to the filter
    invariants this test pins."""
    import storage  # noqa: E402
    storage._GLOBAL_NEAR_DUP_CACHE = {}  # bypass compute_near_dup_groups
    return storage.get_all_results(
        conn,
        page=0,
        page_size=10_000_000,  # fetch all matching; we slice in Python
        **kwargs,
    )


# ---------- default: include_non_plants=False ------------------------------


def test_default_all_view_excludes_non_plants(db_path):
    """Default grid (no plantae filter, no opt-in): non-plant assets
    must be absent from the page slice AND from total_assets."""
    import storage  # noqa: E402
    _seed(
        db_path,
        {
            "plant-green": 0.90,      # >= plantae_yellow (0.35) → included
            "plant-yellow": 0.55,     # >= plantae_yellow
            "uncertain-edge": 0.35,   # boundary, included
            "non-plant": 0.10,        # < plantae_yellow → excluded
            "no-score": 0.0,          # No taxon_scores row → COALESCE=0 → non-plant
        },
    )
    with storage.connect(db_path) as conn:
        results, total_reps, total_assets = _call(conn)

    expected_ids = {"plant-green", "plant-yellow", "uncertain-edge"}
    actual_ids = {r["asset_id"] for r in results}

    assert actual_ids == expected_ids, (
        f"all-view default returned wrong set: {actual_ids}"
    )
    assert total_assets == 3, f"total_assets should be 3 plants, got {total_assets}"
    assert total_reps == 3, f"total_reps should be 3, got {total_reps}"


def test_opt_in_all_view_includes_every_asset(db_path):
    """With `include_non_plants=True`, every asset regardless of
    Plantae score appears in the page slice and total_assets."""
    import storage  # noqa: E402
    _seed(
        db_path,
        {
            "plant-green": 0.90,
            "plant-yellow": 0.55,
            "non-plant": 0.10,
            "no-score": 0.0,
        },
    )
    with storage.connect(db_path) as conn:
        results, total_reps, total_assets = _call(
            conn, include_non_plants=True,
        )

    assert total_assets == 4
    assert total_reps == 4
    assert {r["asset_id"] for r in results} == {
        "plant-green", "plant-yellow", "non-plant", "no-score",
    }


# ---------- explicit plantae tier always wins -----------------------------


def test_explicit_non_plant_returns_non_plants_even_with_toggle_off(db_path):
    """Asking for non_plant explicitly always wins, regardless of
    include_non_plants. A future refactor must NOT silently make the
    explicit filter a no-op when the toggle is off."""
    import storage  # noqa: E402
    _seed(
        db_path,
        {
            "plant-green": 0.90,
            "non-plant": 0.10,
        },
    )
    with storage.connect(db_path) as conn:
        # include_non_plants=False (default), plantae='non_plant'.
        # The explicit filter wins → non-plant IS returned.
        results, _r, total_assets = _call(conn, plantae="non_plant")
        assert total_assets == 1, (
            f"explicit non_plant must override include_non_plants, "
            f"got total_assets={total_assets}"
        )
        assert {r["asset_id"] for r in results} == {"non-plant"}


def test_explicit_plant_tier_unaffected_by_toggle(db_path):
    """plantae='plant' (>= plantae_green=0.60) doesn't change with
    include_non_plants — the explicit tier always wins."""
    import storage  # noqa: E402
    _seed(
        db_path,
        {
            "plant-green": 0.90,      # >= 0.60
            "plant-yellow": 0.55,     # tier: uncertain (< 0.60)
            "non-plant": 0.10,
        },
    )
    with storage.connect(db_path) as conn:
        off_results, _, off_total = _call(conn, plantae="plant")
    with storage.connect(db_path) as conn:
        on_results, _, on_total = _call(
            conn, plantae="plant", include_non_plants=True,
        )
    assert off_total == on_total == 1
    assert {r["asset_id"] for r in off_results} == {"plant-green"}
    assert {r["asset_id"] for r in on_results} == {"plant-green"}


# ---------- behavioral pinning -----------------------------------------


def test_no_assets_returns_empty(db_path):
    """Empty DB + default filters returns (no crash, no rows)."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        results, total_reps, total_assets = _call(conn)
    assert results == []
    assert total_reps == 0
    assert total_assets == 0
