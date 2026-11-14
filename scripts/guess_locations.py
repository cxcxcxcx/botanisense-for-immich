#!/usr/bin/env python3
"""Infer coordinates for no-GPS photos from GPS-tagged temporal neighbours.

**Local only.** Writes `location_guesses` in the app's SQLite DB; never
`assets.latitude/longitude`, never Immich. The map reads
`COALESCE(real, guess)`, so a real EXIF fix always wins, and `--clear` is a
complete rollback.

    # preview (default)
    .venv/bin/python scripts/guess_locations.py

    # write locally
    .venv/bin/python scripts/guess_locations.py --apply

    # rollback
    .venv/bin/python scripts/guess_locations.py --clear

The same computation is available from the Map tab ("Infer from nearby
GPS"), which is the normal way to refresh after a pipeline run.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import location_guess  # noqa: E402
import storage  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=str(ROOT / "data" / "state.db"),
        help="SQLite DB path (default: data/state.db)",
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="write the guesses (default is a read-only preview)",
    )
    parser.add_argument(
        "--clear",
        action="store_true",
        help="delete every guess and exit",
    )
    parser.add_argument(
        "--window-minutes",
        type=int,
        default=15,
        help="max gap to the GPS anchor on each side (default: 15)",
    )
    parser.add_argument(
        "--max-span-m",
        type=float,
        default=1000.0,
        help="max distance between the two anchors (default: 1000)",
    )
    args = parser.parse_args()

    db = Path(args.db)
    if not db.exists():
        print(f"no such DB: {db}", file=sys.stderr)
        return 2
    # Idempotent: creates location_guesses on DBs from older versions.
    storage.init_db(db)

    if args.clear:
        with storage.connect(db) as conn:
            n = storage.clear_location_guesses(conn)
        print(f"cleared {n:,} inferred locations")
        return 0

    with storage.connect(db) as conn:
        rows = storage.get_assets_for_location_guess(conn)
        guesses = location_guess.compute_guesses(
            rows,
            window_seconds=args.window_minutes * 60,
            max_span_m=args.max_span_m,
        )
        high = sum(1 for g in guesses if g["confidence"] == "high")
        medium = len(guesses) - high
        print(
            f"{len(rows):,} assets | {len(guesses):,} inferable "
            f"(high {high:,}, medium {medium:,}) "
            f"| window ±{args.window_minutes} min, span ≤{args.max_span_m:.0f} m"
        )
        if args.apply:
            storage.replace_location_guesses(conn, guesses)
            stats = storage.get_location_guess_stats(
                conn, plantae_yellow=0.35
            )
            print(
                f"applied: {stats['total']:,} effective guesses "
                f"({stats['high']:,} high / {stats['medium']:,} medium; "
                f"{stats['plants']:,} Plantae ≥0.35)"
            )
        else:
            print("preview only — re-run with --apply to write")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
