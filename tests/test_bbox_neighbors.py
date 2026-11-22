"""Box-neighbor aggregation: one result per source image.

The crop index is intentionally crop-level, but a target photo with many
flower boxes must not fill the whole Similar panel. These tests pin:

  - the query's own photo is excluded (all its crops and its whole photo),
  - a target with N boxes appears once, with n_boxes=N and its best crop,
  - include_photos=False returns only crop rows,
  - k counts distinct source images, not crops.
"""

from __future__ import annotations

import numpy as np
import pytest

import storage


def _vec(*values: float) -> bytes:
    v = np.zeros(768, dtype=np.float32)
    v[: len(values)] = values
    v /= max(float(np.linalg.norm(v)), 1e-12)
    return v.tobytes()


def _add_asset(conn, asset_id: str, name: str) -> None:
    storage.upsert_asset(
        conn,
        asset_id=asset_id,
        original_file_name=name,
        original_url=f"http://immich.test/{asset_id}/original",
        thumbnail_url=f"http://immich.test/{asset_id}/thumbnail",
    )


def _add_boxes(conn, asset_id: str, vectors: list[bytes]) -> list[int]:
    dets = [
        {
            "label": "flower",
            "score": 0.9 - i * 0.05,
            "x": 0.05 * i,
            "y": 0.05,
            "w": 0.1,
            "h": 0.1,
        }
        for i in range(len(vectors))
    ]
    ids = storage.replace_detections(conn, asset_id=asset_id, detections=dets)
    for detection_id, vector in zip(ids, vectors):
        conn.execute(
            """
            INSERT INTO bbox_embeddings
              (detection_id, model_version, dim, vector)
            VALUES (?, 'test-teacher', 768, ?)
            """,
            (detection_id, vector),
        )
    return ids


def _add_photo(conn, asset_id: str, vector: bytes) -> None:
    conn.execute(
        """
        INSERT INTO embeddings (asset_id, dim, vector)
        VALUES (?, 768, ?)
        """,
        (asset_id, vector),
    )


@pytest.fixture()
def neighbor_db(tmp_path):
    db = tmp_path / "state.db"
    storage.init_db(db)
    with storage.connect(db) as conn:
        for aid in ("query", "bouquet", "single", "photo-only"):
            _add_asset(conn, aid, f"{aid}.jpg")

        q_ids = _add_boxes(conn, "query", [_vec(1.0), _vec(1.0)])
        b_ids = _add_boxes(
            conn,
            "bouquet",
            [_vec(0.99, 0.1), _vec(0.8, 0.2), _vec(0.7, 0.3)],
        )
        c_ids = _add_boxes(conn, "single", [_vec(0.6, 0.4)])

        _add_photo(conn, "query", _vec(1.0))
        _add_photo(conn, "bouquet", _vec(0.5, 0.5))
        _add_photo(conn, "photo-only", _vec(0.4, 0.6))

    return db, q_ids, b_ids, c_ids


def test_neighbors_collapse_to_one_row_per_source_image(neighbor_db):
    db, q_ids, b_ids, c_ids = neighbor_db
    with storage.connect(db) as conn:
        result = storage.bbox_neighbors(
            conn, detection_id=q_ids[0], k=10, include_photos=True
        )

    assert result is not None
    rows = result["neighbors"]
    asset_ids = [row["asset_id"] for row in rows]

    # The query image never matches itself — neither its other crop nor
    # its whole-photo vector.
    assert "query" not in asset_ids
    # One card per source image, even though "bouquet" has three crops.
    assert asset_ids.count("bouquet") == 1
    assert asset_ids.count("single") == 1
    assert asset_ids.count("photo-only") == 1

    by_asset = {row["asset_id"]: row for row in rows}
    assert by_asset["bouquet"]["kind"] == "box"
    assert by_asset["bouquet"]["detection_id"] == b_ids[0]
    assert by_asset["bouquet"]["n_boxes"] == 3
    assert by_asset["single"]["detection_id"] == c_ids[0]
    assert by_asset["single"]["n_boxes"] == 1
    assert by_asset["photo-only"]["kind"] == "photo"
    assert by_asset["photo-only"]["n_boxes"] == 0

    # k means distinct source images. With k=2 the bouquet's other crops
    # must not consume the second slot.
    with storage.connect(db) as conn:
        two = storage.bbox_neighbors(
            conn, detection_id=q_ids[0], k=2, include_photos=False
        )
    assert two is not None
    assert {row["asset_id"] for row in two["neighbors"]} == {"bouquet", "single"}


def test_neighbors_can_exclude_whole_photos(neighbor_db):
    db, q_ids, _b_ids, _c_ids = neighbor_db
    with storage.connect(db) as conn:
        result = storage.bbox_neighbors(
            conn, detection_id=q_ids[0], k=10, include_photos=False
        )

    assert result is not None
    assert all(row["kind"] == "box" for row in result["neighbors"])
    assert "photo-only" not in {row["asset_id"] for row in result["neighbors"]}
