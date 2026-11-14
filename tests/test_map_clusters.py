"""Unit tests for map_clusters — the location-mode port used by /api/map/clusters."""

from __future__ import annotations

import math

import map_clusters
from map_clusters import (
    PLACE_RADIUS_FINE_M,
    PLACE_RADIUS_WIDE_M,
    cluster_by_location,
    haversine_meters,
)


def _row(asset_id: str, lat: float, lng: float, captured_at: str | None = None) -> dict:
    return {
        "asset_id": asset_id,
        "latitude": lat,
        "longitude": lng,
        "captured_at": captured_at,
    }


def _offset(lat: float, lng: float, north_m: float, east_m: float) -> tuple[float, float]:
    """Move a coordinate by metres in each axis (small distances)."""
    dlat = north_m / map_clusters.METERS_PER_DEGREE_LAT
    dlng = east_m / (map_clusters.METERS_PER_DEGREE_LAT * math.cos(math.radians(lat)))
    return lat + dlat, lng + dlng


def test_haversine_matches_known_distance():
    # ~111.32 km per degree of latitude.
    assert abs(haversine_meters(0.0, 0.0, 1.0, 0.0) - 111_195) < 500


def test_nearby_photos_group_into_one_place():
    lat, lng = 37.7749, -122.4194
    rows = [
        _row("a", lat, lng, "2024-05-01T10:00:00+00:00"),
        _row("b", *(_offset(lat, lng, 40, 30)), "2024-05-01T10:05:00+00:00"),
        _row("c", *(_offset(lat, lng, -20, 60)), "2024-05-02T09:00:00+00:00"),
    ]
    result = cluster_by_location(rows, radius_m=PLACE_RADIUS_FINE_M, min_photos=2)

    assert result["cluster_count"] == 1
    cluster = result["clusters"][0]
    assert cluster["count"] == 3
    assert cluster["start"] == "2024-05-01T10:00:00+00:00"
    assert cluster["end"] == "2024-05-02T09:00:00+00:00"
    # The click-through radius must actually reach every member.
    for row in rows:
        assert haversine_meters(
            cluster["latitude"], cluster["longitude"], row["latitude"], row["longitude"]
        ) <= cluster["radius_m"]


def test_distinct_venues_stay_separate_and_sort_by_size():
    lat, lng = 37.7749, -122.4194
    far = _offset(lat, lng, 600, 0)  # 600 m north: a different venue at 150 m
    rows = [
        _row("solo", *far, "2024-05-01T10:00:00+00:00"),
        _row("a", lat, lng, "2024-05-01T11:00:00+00:00"),
        _row("b", *(_offset(lat, lng, 20, 0)), "2024-05-01T11:05:00+00:00"),
        _row("c", *(_offset(lat, lng, 0, 20)), "2024-05-01T11:10:00+00:00"),
    ]
    result = cluster_by_location(rows, radius_m=PLACE_RADIUS_FINE_M, min_photos=2)

    assert result["cluster_count"] == 1  # the singleton is excluded by min_photos
    assert result["singleton_places"] == 1
    assert result["clusters"][0]["count"] == 3
    # "N places · M photos" must describe the visible markers only;
    # located_photos still reports everything with a fix.
    assert result["photo_count"] == 3
    assert result["located_photos"] == 4


def test_wide_granularity_merges_neighbourhoods():
    lat, lng = 37.7749, -122.4194
    rows = [
        _row("a", lat, lng, "2024-05-01T10:00:00+00:00"),
        _row("b", *(_offset(lat, lng, 600, 0)), "2024-05-01T10:05:00+00:00"),
    ]

    fine = cluster_by_location(rows, radius_m=PLACE_RADIUS_FINE_M)
    assert fine["cluster_count"] == 2

    wide = cluster_by_location(rows, radius_m=PLACE_RADIUS_WIDE_M)
    assert wide["cluster_count"] == 1
    assert wide["clusters"][0]["count"] == 2


def test_photos_without_usable_gps_are_counted_not_plotted():
    rows = [
        _row("a", 37.0, -122.0),
        _row("null-lat", None, -122.0),
        _row("null-lng", 37.0, None),
        _row("null-island", 0.0, 0.0),
        _row("out-of-range", 95.0, -122.0),
    ]
    result = cluster_by_location(rows, radius_m=PLACE_RADIUS_FINE_M, min_photos=1)

    assert result["unlocated"] == 4
    assert result["photo_count"] == 1
    assert result["total_assets"] == 5
    assert [c["count"] for c in result["clusters"]] == [1]


def test_visits_count_temporal_runs_inside_one_place():
    lat, lng = 37.7749, -122.4194
    rows = [
        _row("a", lat, lng, "2024-05-01T10:00:00+00:00"),
        _row("b", *(_offset(lat, lng, 20, 0)), "2024-05-01T10:30:00+00:00"),
        # > 7 days later, same place → second visit
        _row("c", *(_offset(lat, lng, 30, 0)), "2024-05-20T08:00:00+00:00"),
        # 3 days after the second visit → still the same visit
        _row("d", *(_offset(lat, lng, -10, 0)), "2024-05-23T08:00:00+00:00"),
    ]
    result = cluster_by_location(rows, radius_m=PLACE_RADIUS_FINE_M, min_photos=1)

    assert result["cluster_count"] == 1
    assert result["clusters"][0]["visits"] == 2


def test_cluster_ids_are_prefixed_and_dense():
    lat, lng = 37.7749, -122.4194
    rows = [
        _row("a", lat, lng),
        _row("b", *(_offset(lat, lng, 20, 0))),
        _row("c", *(_offset(lat, lng, 1000, 0))),
        _row("d", *(_offset(lat, lng, 1020, 0))),
    ]
    result = cluster_by_location(
        rows, radius_m=PLACE_RADIUS_FINE_M, min_photos=2, id_prefix="loc-wide-"
    )
    assert [c["id"] for c in result["clusters"]] == ["loc-wide-0", "loc-wide-1"]


def test_empty_library_returns_empty_payload():
    result = cluster_by_location([], radius_m=PLACE_RADIUS_FINE_M)
    assert result["clusters"] == []
    assert result["photo_count"] == 0
    assert result["unlocated"] == 0
