"""Infer coordinates for no-GPS photos from their temporal neighbours.

**Local only.** Nothing here writes to Immich and nothing overwrites
`assets.latitude/longitude`. Guesses live in their own table
(`location_guesses`); the map/query layer reads
`COALESCE(assets.latitude, location_guesses.latitude)`, so a real EXIF fix
always wins and deleting the guess table is a complete rollback.

Method (the conservative tier agreed from the dry-run):

- Sort every asset with a capture time on one timeline.
- For a no-GPS photo, find the nearest GPS-tagged photo before and after.
- Interpolate only when *both* sides are within `window_seconds` (15 min)
  and the two reference points are within `max_span_m` (1 km). A moving
  window of that size is the "same outing, same place" regime; wider
  windows turn into "same day", which is not a location.
- `high` confidence: both sides within 5 min and the span within 200 m.
  Everything else that passes is `medium`.

This mirrors Dawarich's timestamp interpolation, but uses the photo
library itself (a phone in the same pocket) instead of a recorded track.
"""

from __future__ import annotations

import bisect
import math
from datetime import datetime, timezone

EARTH_RADIUS_M = 6_371_008.8

DEFAULT_WINDOW_SECONDS = 15 * 60
DEFAULT_MAX_SPAN_M = 1_000.0
HIGH_WINDOW_SECONDS = 5 * 60
HIGH_MAX_SPAN_M = 200.0


def haversine_meters(lat1: float, lng1: float, lat2: float, lng2: float) -> float:
    phi1 = math.radians(lat1)
    phi2 = math.radians(lat2)
    dphi = phi2 - phi1
    dlambda = math.radians(lng2 - lng1)
    a = (
        math.sin(dphi / 2.0) ** 2
        + math.cos(phi1) * math.cos(phi2) * math.sin(dlambda / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def parse_capture_time(value: str | None) -> float | None:
    """ISO 8601 → epoch seconds. Naive timestamps are treated as UTC."""
    if not value:
        return None
    try:
        dt = datetime.fromisoformat(value)
    except (TypeError, ValueError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    try:
        return dt.timestamp()
    except (OverflowError, OSError):
        return None


def _has_gps(row: dict) -> bool:
    lat = row.get("latitude")
    lng = row.get("longitude")
    if lat is None or lng is None:
        return False
    # (0, 0) is an EXIF placeholder, not a real fix — same rule as the map.
    return not (float(lat) == 0.0 and float(lng) == 0.0)


def compute_guesses(
    rows: list[dict],
    *,
    window_seconds: int = DEFAULT_WINDOW_SECONDS,
    max_span_m: float = DEFAULT_MAX_SPAN_M,
    high_window_seconds: int = HIGH_WINDOW_SECONDS,
    high_max_span_m: float = HIGH_MAX_SPAN_M,
) -> list[dict]:
    """Return one guess per fillable no-GPS asset.

    `rows` must carry `asset_id`, `captured_at`, `latitude`, `longitude`.
    Rows without a usable capture time or without GPS on at least one side
    of the window are skipped — there is nothing to interpolate from.
    """
    located: list[tuple[float, str, float, float]] = []
    missing: list[tuple[float, dict]] = []
    for row in rows:
        ts = parse_capture_time(row.get("captured_at"))
        if ts is None:
            continue
        if _has_gps(row):
            located.append(
                (ts, row["asset_id"], float(row["latitude"]), float(row["longitude"]))
            )
        else:
            missing.append((ts, row))

    located.sort(key=lambda item: (item[0], item[1]))
    missing.sort(key=lambda item: (item[0], item[1]["asset_id"]))
    times = [item[0] for item in located]

    guesses: list[dict] = []
    for ts, row in missing:
        index = bisect.bisect_left(times, ts)
        prev = located[index - 1] if index > 0 else None
        nxt = located[index] if index < len(located) else None
        if prev is None or nxt is None:
            continue
        prev_gap = ts - prev[0]
        next_gap = nxt[0] - ts
        if prev_gap > window_seconds or next_gap > window_seconds:
            continue
        span = haversine_meters(prev[2], prev[3], nxt[2], nxt[3])
        if span > max_span_m:
            continue

        total = prev_gap + next_gap
        fraction = 0.5 if total <= 0 else prev_gap / total
        latitude = prev[2] + fraction * (nxt[2] - prev[2])
        longitude = prev[3] + fraction * (nxt[3] - prev[3])
        confidence = (
            "high"
            if prev_gap <= high_window_seconds
            and next_gap <= high_window_seconds
            and span <= high_max_span_m
            else "medium"
        )
        guesses.append(
            {
                "asset_id": row["asset_id"],
                "latitude": latitude,
                "longitude": longitude,
                "method": "interpolate",
                "confidence": confidence,
                "prev_asset_id": prev[1],
                "next_asset_id": nxt[1],
                "prev_gap_seconds": int(round(prev_gap)),
                "next_gap_seconds": int(round(next_gap)),
                "span_meters": round(span, 1),
            }
        )
    return guesses
