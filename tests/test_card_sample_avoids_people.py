"""Card representative photos demote has_person=1 assets.

Cluster cards (`storage.cluster_summary`) and taxonomy cards
(`storage.get_global_taxonomy_summary`) both pick one representative
photo per card. That photo must be person-free whenever the card has
ANY person-free member — a people photo is only acceptable as the
fallback when every member is stamped has_person=1 (or a cluster's
members have no predictions at all). has_person IS NULL means "not
scanned yet" and is treated as person-free, matching the grid's
people filter.

The taxonomy rule uses the same Python-side comparison, so it is
pinned here too.
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
    sys.modules.pop("storage", None)
    import storage  # noqa: E402
    storage.init_db(p)
    yield p
    storage._GLOBAL_NEAR_DUP_CACHE = None


def _asset(conn, asset_id: str, *, cluster_id: int = 1,
           has_person: int | None = 0) -> None:
    conn.execute(
        "INSERT INTO assets "
        "(asset_id, cluster_id, original_file_name, original_url, "
        " thumbnail_url, has_person) "
        "VALUES (?, ?, ?, ?, ?, ?)",
        (
            asset_id,
            cluster_id,
            f"{asset_id}.jpg",
            f"http://x/orig/{asset_id}.jpg",
            f"http://x/thumb/{asset_id}.jpg",
            has_person,
        ),
    )
    conn.execute(
        "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) "
        "VALUES (?, 'kingdom', 'Plantae', 0.90)",
        (asset_id,),
    )


def _species_pred(conn, asset_id: str, score: float, *,
                  species: str = "Rosa rugosa", genus: str = "Rosa",
                  family: str = "Rosaceae") -> None:
    conn.execute(
        "INSERT INTO predictions "
        "(asset_id, rank, position, species, genus, family, score) "
        "VALUES (?, 'species', 0, ?, ?, ?, ?)",
        (asset_id, species, genus, family, score),
    )


def test_cluster_sample_prefers_lower_score_plant_over_people(db_path):
    """Even a much higher-scoring people photo loses the card slot to
    a person-free member — the demotion outranks the BioCLIP score."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        _asset(conn, "person-high", cluster_id=1, has_person=1)
        _species_pred(conn, "person-high", 0.99)
        _asset(conn, "plant-low", cluster_id=1, has_person=0)
        _species_pred(conn, "plant-low", 0.20)
        conn.commit()

        row = storage.cluster_summary(conn)[0]
    assert row["sample_asset_id"] == "plant-low"
    assert row["sample_thumbnail_url"] == "http://x/thumb/plant-low.jpg"


def test_cluster_sample_null_has_person_counts_as_person_free(db_path):
    """Unscanned members (has_person NULL) are eligible before any
    explicit people photo."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        _asset(conn, "person", cluster_id=1, has_person=1)
        _species_pred(conn, "person", 0.99)
        _asset(conn, "unscanned", cluster_id=1, has_person=None)
        _species_pred(conn, "unscanned", 0.10)
        conn.commit()

        row = storage.cluster_summary(conn)[0]
    assert row["sample_asset_id"] == "unscanned"


def test_cluster_sample_falls_back_to_people_when_all_have_people(db_path):
    """A cluster whose every member is has_person=1 still gets a
    thumbnail — the demotion is a sort, not a filter."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        _asset(conn, "person-a", cluster_id=2, has_person=1)
        _species_pred(conn, "person-a", 0.50)
        _asset(conn, "person-b", cluster_id=2, has_person=1)
        _species_pred(conn, "person-b", 0.80)
        conn.commit()

        row = storage.cluster_summary(conn)[0]
    # Inside the all-people pool the usual highest-score rule applies.
    assert row["sample_asset_id"] == "person-b"


def test_cluster_sample_people_demotion_holds_without_predictions(db_path):
    """Non-plant clusters have no predictions; the person demotion is
    ahead of the `p.score IS NULL` key so it still applies there."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        _asset(conn, "person-nopred", cluster_id=3, has_person=1)
        _asset(conn, "plant-nopred", cluster_id=3, has_person=0)
        conn.commit()

        row = storage.cluster_summary(conn)[0]
    assert row["sample_asset_id"] == "plant-nopred"


def test_taxonomy_sample_prefers_person_free_photo(db_path):
    """Species/genus/family card thumbnails follow the same rule:
    person-free first, then highest score."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        _asset(conn, "person-high", has_person=1)
        _species_pred(conn, "person-high", 0.97)
        _asset(conn, "plant-low", has_person=0)
        _species_pred(conn, "plant-low", 0.30)
        conn.commit()

        summary = storage.get_global_taxonomy_summary(conn)

    species = {s["name"]: s for s in summary["species"]}["Rosa rugosa"]
    assert species["sample_asset_id"] == "plant-low"
    genus = {g["name"]: g for g in summary["genus"]}["Rosa"]
    assert genus["sample_asset_id"] == "plant-low"
    family = {f["name"]: f for f in summary["family"]}["Rosaceae"]
    assert family["sample_asset_id"] == "plant-low"


def test_taxonomy_sample_falls_back_to_people(db_path):
    """When every photo of a taxon has people, the best-scoring people
    photo is still used as the thumbnail."""
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        _asset(conn, "person-a", has_person=1)
        _species_pred(conn, "person-a", 0.40)
        _asset(conn, "person-b", has_person=1)
        _species_pred(conn, "person-b", 0.70)
        conn.commit()

        species = storage.get_global_taxonomy_summary(conn)["species"][0]
    assert species["sample_asset_id"] == "person-b"
