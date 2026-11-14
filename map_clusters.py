"""Location clustering for the map view.

Ported from the Solar app (`src/utils/clustering/locationClustering.ts`).
The idea is deliberately simple: photos are grouped by **distance alone**,
so the same park or balcony visited across months lands in one place. Time
is not a splitter — it only orders the greedy walk (making the partition
deterministic) and defines *visits*, the temporal runs inside a place.

Differences from Solar, all driven by this repo's data model:

- Input is `allocated assets` from SQLite (already filtered and near-dup
  deduped by `storage.get_all_results`), not native photo-library rows.
- A uniform lat/lng grid indexes place centroids, so a 13k-photo library
  does not cost 13k × n_places haversine calls. Solar's library was small
  enough for the O(n²) walk.
- `(0, 0)` is treated as "no GPS": EXIF writers emit it when the receiver
  never got a fix, and plotting those photos puts them in the Gulf of
  Guinea.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

EARTH_RADIUS_M = 6_371_008.8

# Same knobs as Solar's location mode. `fine` is venue level (one place per
# building / park corner); `wide` folds a whole market, campus or park into
# one place. A radius is the whole difference — no hierarchy.
PLACE_RADIUS_FINE_M = 150.0
PLACE_RADIUS_WIDE_M = 1000.0
# A third, deliberately coarse read of the same library: one place per
# ~10 km neighbourhood (a city / day-trip area). Not a hierarchy level —
# just another radius for the same greedy walk.
PLACE_RADIUS_REGION_M = 10_000.0

# Temporal runs inside a place are separated by gaps longer than this. Used
# only for the tooltip's "N visits" line; it never splits a place.
VISIT_GAP_S = 7 * 24 * 60 * 60

METERS_PER_DEGREE_LAT = 111_320.0


def haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    """Great-circle distance in metres.

    Fine for place-scale distances (hundreds of metres); the spherical
    approximation is far below GPS error at that scale.
    """
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _coords(row: dict) -> tuple[float, float] | None:
    """Return usable (lat, lng) or None.

    Nulls and the exact `(0, 0)` placeholder are treated as missing. A real
    photo at 0,0 is possible in principle (Null Island) but vanishingly
    unlikely compared to a camera writing a zeroed fix.
    """
    lat = row.get("latitude")
    lng = row.get("longitude")
    if lat is None or lng is None:
        return None
    try:
        lat_f = float(lat)
        lng_f = float(lng)
    except (TypeError, ValueError):
        return None
    if not (-90.0 <= lat_f <= 90.0 and -180.0 <= lng_f <= 180.0):
        return None
    if lat_f == 0.0 and lng_f == 0.0:
        return None
    return lat_f, lng_f


def _sort_time(row: dict) -> str:
    # ISO 8601 strings sort chronologically. Missing capture time goes last
    # but keeps the partition deterministic through the asset id tiebreak.
    return row.get("captured_at") or "9999"


def _seconds_between(a: str, b: str) -> float | None:
    """Parse two ISO timestamps and return the gap in seconds.

    `datetime.fromisoformat` handles the shapes SQLite/Immich produce
    (`2024-05-01T10:00:00+00:00`, `2024-05-01 10:00:00`). A malformed or
    missing value returns None and simply starts a new visit.
    """
    from datetime import datetime

    try:
        da = datetime.fromisoformat(a)
        db = datetime.fromisoformat(b)
    except (TypeError, ValueError):
        return None
    return abs((db - da).total_seconds())


@dataclass
class _Place:
    rows: list[dict] = field(default_factory=list)
    sum_lat: float = 0.0
    sum_lng: float = 0.0
    cell: tuple[int, int] = (0, 0)

    @property
    def centroid(self) -> tuple[float, float]:
        n = len(self.rows)
        return self.sum_lat / n, self.sum_lng / n

    def add(self, row: dict, lat: float, lng: float) -> None:
        self.rows.append(row)
        self.sum_lat += lat
        self.sum_lng += lng


def _grid_steps(points: list[tuple[float, float]], radius_m: float) -> tuple[float, float]:
    """Degree size of one grid cell, wide enough that a 3×3 search cannot
    miss a point within `radius_m`.

    Longitude degrees shrink with latitude, so the worst-case (largest)
    latitude in the set dictates the cell width. Clamped near the poles
    because 1/cos(lat) explodes and the grid would become uselessly wide.
    """
    max_abs_lat = max(abs(lat) for lat, _ in points)
    max_abs_lat = min(max_abs_lat + 0.5, 89.0)
    cos_lat = max(0.01, math.cos(math.radians(max_abs_lat)))
    lat_step = radius_m / METERS_PER_DEGREE_LAT
    lng_step = radius_m / (METERS_PER_DEGREE_LAT * cos_lat)
    return lat_step, lng_step


def _cell(lat: float, lng: float, lat_step: float, lng_step: float) -> tuple[int, int]:
    return math.floor(lat / lat_step), math.floor(lng / lng_step)


def _nearest_place(
    grid: dict[tuple[int, int], list[_Place]],
    lat: float,
    lng: float,
    cell: tuple[int, int],
    radius_m: float,
) -> _Place | None:
    """Nearest place whose centroid is within the radius, else None.

    Nearest rather than first-match so the partition does not depend on the
    order places happened to be created in.
    """
    best: _Place | None = None
    best_distance = radius_m
    cx, cy = cell
    for dx in (-1, 0, 1):
        for dy in (-1, 0, 1):
            for place in grid.get((cx + dx, cy + dy), ()):
                plat, plng = place.centroid
                distance = haversine_meters(plat, plng, lat, lng)
                if distance <= best_distance:
                    best = place
                    best_distance = distance
    return best


def _visit_count(rows: list[dict]) -> int:
    """Number of temporal runs (gaps > 7 days) among a place's photos."""
    visits = 0
    previous: str | None = None
    for row in sorted(rows, key=_sort_time):
        ts = row.get("captured_at")
        if not ts:
            continue
        if previous is None:
            visits += 1
        else:
            gap = _seconds_between(previous, ts)
            if gap is None or gap > VISIT_GAP_S:
                visits += 1
        previous = ts
    return visits


def _summarize(
    place: _Place, prefix: str, index: int, *, include_asset_ids: bool = False
) -> dict:
    lat, lng = place.centroid
    cover_row = place.rows[0]
    cover_distance = float("inf")
    max_distance = 0.0
    for row in place.rows:
        coords = _coords(row)
        if coords is None:
            continue
        distance = haversine_meters(lat, lng, coords[0], coords[1])
        if distance < cover_distance:
            cover_distance = distance
            cover_row = row
        if distance > max_distance:
            max_distance = distance

    times = sorted(row["captured_at"] for row in place.rows if row.get("captured_at"))
    return {
        "id": f"{prefix}{index}",
        "latitude": round(lat, 6),
        "longitude": round(lng, 6),
        "count": len(place.rows),
        # Largest member distance from the centroid. The UI uses this as
        # the click-through radius so the resulting grid is the place
        # itself, not an arbitrary fixed box.
        "radius_m": round(max_distance + 5.0, 1),
        "sample_asset_id": cover_row.get("asset_id"),
        "start": times[0] if times else None,
        "end": times[-1] if times else None,
        "visits": _visit_count(place.rows),
        # Photos in this place whose coordinate came from location_guesses
        # rather than EXIF. The map styles mixed/inferred markers
        # differently and the tooltip explains the difference.
        "guessed_count": sum(1 for row in place.rows if row.get("guessed")),
        # Only when the caller needs exact click-through membership (the
        # /api/results resolver). The map payload itself leaves this out:
        # 16k UUIDs would dominate the response for no visual benefit.
        **(
            {"asset_ids": [row.get("asset_id") for row in place.rows]}
            if include_asset_ids
            else {}
        ),
    }


def cluster_by_location(
    rows: list[dict],
    *,
    radius_m: float = PLACE_RADIUS_FINE_M,
    min_photos: int = 1,
    id_prefix: str = "loc-",
    include_asset_ids: bool = False,
) -> dict:
    """Group rows into places and return a map payload.

    `rows` must carry `asset_id`, `latitude`, `longitude`, `captured_at`.
    Photos without usable coordinates are counted (`unlocated`) but cannot
    be plotted.

    Places are ordered largest-first (a stable sort keeps the chronological
    walk order for ties), then `min_photos` drops one-off places from the
    marker list while still reporting `singleton_places` so the UI can offer
    a "show single-photo places" toggle without a second query.
    """
    located: list[tuple[dict, float, float]] = []
    unlocated = 0
    for row in rows:
        coords = _coords(row)
        if coords is None:
            unlocated += 1
        else:
            located.append((row, coords[0], coords[1]))

    # Ascending capture time makes the greedy walk deterministic and matches
    # the order a photo library naturally reads.
    located.sort(key=lambda item: (_sort_time(item[0]), item[0].get("asset_id") or ""))

    if located:
        lat_step, lng_step = _grid_steps([(lat, lng) for _, lat, lng in located], radius_m)
    else:
        lat_step = lng_step = radius_m / METERS_PER_DEGREE_LAT

    grid: dict[tuple[int, int], list[_Place]] = {}
    places: list[_Place] = []
    for row, lat, lng in located:
        cell = _cell(lat, lng, lat_step, lng_step)
        place = _nearest_place(grid, lat, lng, cell, radius_m)
        if place is None:
            place = _Place(rows=[], sum_lat=0.0, sum_lng=0.0, cell=cell)
            places.append(place)
            grid.setdefault(cell, []).append(place)
        place.add(row, lat, lng)
        # The running centroid can drift across a cell boundary; keep the
        # index honest so later photos still find this place.
        new_cell = _cell(*place.centroid, lat_step, lng_step)
        if new_cell != place.cell:
            grid[place.cell].remove(place)
            grid.setdefault(new_cell, []).append(place)
            place.cell = new_cell

    places.sort(
        key=lambda place: (
            -len(place.rows),
            min((r.get("captured_at") or "9999") for r in place.rows),
        )
    )

    singleton_places = sum(1 for place in places if len(place.rows) == 1)
    visible = [place for place in places if len(place.rows) >= max(1, min_photos)]
    clusters = [
        _summarize(place, id_prefix, index, include_asset_ids=include_asset_ids)
        for index, place in enumerate(visible)
    ]

    return {
        "radius_m": radius_m,
        # What the map currently plots, and what qualifies before the
        # min_photos cut. The panel uses `photo_count` so "N places ·
        # M photos" always describes the visible markers; the hidden
        # single-photo places are offered as an explicit toggle.
        "photo_count": sum(cluster["count"] for cluster in clusters),
        "located_photos": sum(len(place.rows) for place in places),
        "guessed_photos": sum(
            1 for place in places for row in place.rows if row.get("guessed")
        ),
        "cluster_count": len(clusters),
        "singleton_places": singleton_places,
        "unlocated": unlocated,
        "total_assets": len(rows),
        "clusters": clusters,
    }
