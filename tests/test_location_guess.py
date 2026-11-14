"""Location inference (local-only) + effective coordinates in storage."""

from __future__ import annotations

import pytest

import location_guess
import storage


def _row(asset_id, when, lat=None, lng=None):
    return {
        "asset_id": asset_id,
        "captured_at": when,
        "latitude": lat,
        "longitude": lng,
    }


def test_interpolates_between_two_gps_neighbours():
    rows = [
        _row("a", "2026-05-01T10:00:00+00:00", 37.0, -122.0),
        _row("target", "2026-05-01T10:05:00+00:00"),
        _row("b", "2026-05-01T10:10:00+00:00", 37.002, -122.002),
    ]
    guesses = location_guess.compute_guesses(rows)
    assert len(guesses) == 1
    g = guesses[0]
    assert g["asset_id"] == "target"
    assert g["latitude"] == pytest.approx(37.001)
    assert g["longitude"] == pytest.approx(-122.001)
    assert g["confidence"] == "medium"  # 5 min each side, span ~270 m
    assert g["prev_asset_id"] == "a"
    assert g["next_asset_id"] == "b"
    assert g["prev_gap_seconds"] == 300
    assert g["next_gap_seconds"] == 300


def test_high_confidence_when_tight():
    rows = [
        _row("a", "2026-05-01T10:00:00+00:00", 37.0, -122.0),
        _row("target", "2026-05-01T10:01:00+00:00"),
        _row("b", "2026-05-01T10:02:00+00:00", 37.0005, -122.0),
    ]
    g = location_guess.compute_guesses(rows)[0]
    assert g["confidence"] == "high"
    assert g["span_meters"] < 200


def test_window_and_span_are_enforced():
    # 20 min gap on the right → outside the 15 min window.
    rows = [
        _row("a", "2026-05-01T10:00:00+00:00", 37.0, -122.0),
        _row("target", "2026-05-01T10:01:00+00:00"),
        _row("b", "2026-05-01T10:21:00+00:00", 37.0, -122.0),
    ]
    assert location_guess.compute_guesses(rows) == []

    # Both sides inside the window, but ~2.8 km apart → not the same place.
    far = [
        _row("a", "2026-05-01T10:00:00+00:00", 37.0, -122.0),
        _row("target", "2026-05-01T10:05:00+00:00"),
        _row("b", "2026-05-01T10:10:00+00:00", 37.025, -122.0),
    ]
    assert location_guess.compute_guesses(far) == []


def test_needs_both_sides_and_a_capture_time():
    assert location_guess.compute_guesses(
        [
            _row("a", "2026-05-01T10:00:00+00:00", 37.0, -122.0),
            _row("target", "2026-05-01T10:05:00+00:00"),
        ]
    ) == []
    assert location_guess.compute_guesses(
        [
            _row("target", "2026-05-01T10:05:00+00:00"),
            _row("b", "2026-05-01T10:10:00+00:00", 37.0, -122.0),
        ]
    ) == []
    assert location_guess.compute_guesses(
        [
            _row("a", "2026-05-01T10:00:00+00:00", 37.0, -122.0),
            _row("target", None),
            _row("b", "2026-05-01T10:10:00+00:00", 37.0, -122.0),
        ]
    ) == []


def test_null_island_is_not_an_anchor():
    rows = [
        _row("a", "2026-05-01T10:00:00+00:00", 0.0, 0.0),
        _row("target", "2026-05-01T10:05:00+00:00"),
        _row("b", "2026-05-01T10:10:00+00:00", 37.0, -122.0),
    ]
    assert location_guess.compute_guesses(rows) == []


def _seed(tmp_path):
    db = tmp_path / "state.db"
    storage.init_db(db)
    with storage.connect(db) as conn:
        conn.execute(
            "INSERT INTO assets (asset_id, original_file_name, original_url,"
            " thumbnail_url, captured_at, latitude, longitude) VALUES"
            " ('a1', 'a1.jpg', 'u', 't', '2026-05-01T10:00:00+00:00', 37.0, -122.0),"
            " ('a2', 'a2.jpg', 'u', 't', '2026-05-01T10:05:00+00:00', NULL, NULL),"
            " ('a3', 'a3.jpg', 'u', 't', '2026-05-01T11:00:00+00:00', NULL, NULL)"
        )
        conn.execute(
            "INSERT INTO location_guesses (asset_id, latitude, longitude,"
            " confidence, prev_asset_id, next_asset_id) VALUES"
            " ('a2', 37.001, -122.001, 'high', 'a1', 'a1')"
        )
    return db


def test_geo_assets_guessed_modes(tmp_path):
    db = _seed(tmp_path)
    with storage.connect(db) as conn:
        include = storage.get_geo_assets(conn, include_non_plants=True)
        exclude = storage.get_geo_assets(
            conn, include_non_plants=True, guessed="exclude"
        )
        only = storage.get_geo_assets(
            conn, include_non_plants=True, guessed="only"
        )
    by_id = {r["asset_id"]: r for r in include}
    assert by_id["a1"]["latitude"] == 37.0 and by_id["a1"]["guessed"] == 0
    assert by_id["a2"]["latitude"] == pytest.approx(37.001)
    assert by_id["a2"]["guessed"] == 1
    assert by_id["a3"]["latitude"] is None
    # exclude means "no inferred coordinates". a2 is still in the payload,
    # but with its real (NULL) coordinates so the map counts it as
    # unlocated; a3 likewise.
    exclude_by_id = {r["asset_id"]: r for r in exclude}
    assert exclude_by_id["a2"]["latitude"] is None
    assert exclude_by_id["a2"]["guessed"] == 0
    assert exclude_by_id["a1"]["latitude"] == 37.0
    assert [r["asset_id"] for r in only] == ["a2"]


def test_results_expose_per_photo_location_guessed(tmp_path):
    db = _seed(tmp_path)
    with storage.connect(db) as conn:
        rows, _reps, _total = storage.get_all_results(
            conn, include_non_plants=True
        )
    by_id = {r["asset_id"]: r for r in rows}
    assert by_id["a1"]["location_guessed"] == 0
    assert by_id["a2"]["location_guessed"] == 1  # guess is effective
    assert by_id["a3"]["location_guessed"] == 0


def test_near_filter_uses_inferred_coordinates(tmp_path):
    db = _seed(tmp_path)
    with storage.connect(db) as conn:
        rows, _reps, _total = storage.get_all_results(
            conn,
            include_non_plants=True,
            near_lat=37.001,
            near_lng=-122.001,
            near_radius_m=60,
        )
    assert [r["asset_id"] for r in rows] == ["a2"]


def test_guess_stats_ignore_shadowed_guesses(tmp_path):
    db = _seed(tmp_path)
    with storage.connect(db) as conn:
        stats = storage.get_location_guess_stats(conn, plantae_yellow=0.35)
        assert stats["total"] == 1
        assert stats["high"] == 1
        # A real fix arriving later shadows the guess without deleting it.
        conn.execute(
            "UPDATE assets SET latitude = 1.0, longitude = 2.0 WHERE asset_id = 'a2'"
        )
        conn.commit()
        stats = storage.get_location_guess_stats(conn, plantae_yellow=0.35)
        assert stats["total"] == 0
