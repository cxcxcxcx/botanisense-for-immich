"""Tests for the KNN reference-photo proxy allowlist.

/api/knn/ref-thumb fetches arbitrary external URLs on the server's
behalf, so the gate that decides *which* URLs it will fetch is the
security-relevant part. These tests pin that gate:

  - a URL we actually stored as a neighbor identifier is accepted
  - anything else is rejected (the proxy is closed, not open)
  - a prefix of a stored URL does not match (no attacker-controlled
    host smuggled in by being a prefix of a legitimate entry)
  - SQL LIKE metacharacters in the candidate URL don't widen the match
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

REAL_URL = "https://content.eol.org/data/media/59/d2/0f/509.fe24ffde.jpg"


@pytest.fixture
def conn(tmp_path):
    import storage
    p = tmp_path / "state.db"
    storage.init_db(p)
    with storage.connect(p) as c:
        yield c


def _seed_knn(conn, neighbors: list[dict]) -> None:
    # knn_predictions.asset_id is a FK into assets, so the parent row
    # has to exist first.
    conn.execute(
        "INSERT OR IGNORE INTO assets (asset_id, original_file_name, "
        "original_url, thumbnail_url) VALUES ('a1', 'f.jpg', 'u', 't')"
    )
    conn.execute(
        """
        INSERT INTO knn_predictions
          (asset_id, index_version, k, top_species, common_name,
           score, kingdom, family, genus, neighbors)
        VALUES ('a1', 'v1', 10, 'Acer rubrum', 'Red maple',
                0.9, 'Plantae', 'Sapindaceae', 'Acer', ?)
        """,
        (json.dumps(neighbors),),
    )
    conn.commit()


def test_stored_identifier_is_allowed(conn):
    import storage
    _seed_knn(conn, [{"uuid": "u1", "species": "rubrum", "sim": 20.0,
                      "identifier": REAL_URL, "source_dataset": "eol"}])
    assert storage.knn_reference_url_known(conn, REAL_URL) is True


def test_unknown_url_is_rejected(conn):
    import storage
    _seed_knn(conn, [{"uuid": "u1", "species": "rubrum", "sim": 20.0,
                      "identifier": REAL_URL, "source_dataset": "eol"}])
    assert storage.knn_reference_url_known(
        conn, "https://evil.test/pwn.jpg"
    ) is False


def test_prefix_of_stored_url_is_rejected(conn):
    """The pattern is anchored by the surrounding JSON quotes, so a
    shorter prefix of a stored URL must not match — otherwise
    'https://' itself would authorize the whole internet."""
    import storage
    _seed_knn(conn, [{"uuid": "u1", "species": "rubrum", "sim": 20.0,
                      "identifier": REAL_URL, "source_dataset": "eol"}])
    assert storage.knn_reference_url_known(conn, "https://content.eol.org/") is False
    assert storage.knn_reference_url_known(conn, "https://") is False


def test_like_wildcards_do_not_widen_the_match(conn):
    """A candidate URL full of LIKE metacharacters must be matched
    literally. Without escaping, '%' would match any stored row and
    turn the allowlist into a no-op."""
    import storage
    _seed_knn(conn, [{"uuid": "u1", "species": "rubrum", "sim": 20.0,
                      "identifier": REAL_URL, "source_dataset": "eol"}])
    assert storage.knn_reference_url_known(conn, "https://%") is False
    assert storage.knn_reference_url_known(conn, "%") is False
    assert storage.knn_reference_url_known(conn, "https://content.eol.org/_") is False


def test_bbox_table_is_searched_too(conn):
    """Per-crop reclassification writes to bbox_knn_predictions; its
    reference photos must be viewable as well."""
    import storage
    conn.execute(
        "INSERT OR IGNORE INTO assets (asset_id, original_file_name, "
        "original_url, thumbnail_url) VALUES ('a1', 'f.jpg', 'u', 't')"
    )
    conn.execute(
        "INSERT INTO detections (id, asset_id, label, score, x, y, w, h) "
        "VALUES (1, 'a1', 'plant', 0.9, 0, 0, 1, 1)"
    )
    conn.execute(
        """
        INSERT INTO bbox_knn_predictions
          (detection_id, index_version, k, top_species, common_name,
           score, kingdom, family, genus, neighbors)
        VALUES (1, 'v1', 10, 'Acer rubrum', 'Red maple', 0.9,
                'Plantae', 'Sapindaceae', 'Acer', ?)
        """,
        (json.dumps([{"uuid": "u2", "species": "rubrum", "sim": 19.0,
                      "identifier": REAL_URL, "source_dataset": "eol"}]),),
    )
    conn.commit()
    assert storage.knn_reference_url_known(conn, REAL_URL) is True


def test_no_rows_means_everything_rejected(conn):
    import storage
    assert storage.knn_reference_url_known(conn, REAL_URL) is False
