#!/usr/bin/env python3
"""Create the shadow (S0-engine) A/B database from the production DB.

Reads the live production state.db (READ ONLY — sqlite backup API, safe
against WAL), writes a fully independent copy to data/ab/state.db, then
clears every MODEL-side table so the next run under ENGINE=mobile-*
recomputes them from scratch. Teacher-side kingdom scores
(taxon_scores rank='kingdom') are deliberately KEPT: the Plantae gate /
badge calibration then stays identical between the two instances, so
the A/B isolates exactly what changed (species names, clusters,
detections) instead of re-testing the library gate — see
docs/web_model_ab_plan.md §5 (hybrid).

Assets / search_results / Immich credentials (album_meta) come along
with the copy, so the shadow instance streams Immich in the same order
and the comparison is asset-for-asset aligned.

Usage:
    python scripts/setup_shadow_ab.py \
        [--baseline ~/open_source/botanisense-fresh/data/state.db] \
        [--shadow data/ab/state.db]

The baseline is never written. The shadow DB is likewise NEVER copied
back — it is throwaway by design (it rewrites cluster_id and friends).
"""

from __future__ import annotations

import argparse
import sqlite3
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]

DEFAULT_BASELINE = Path(
    "~/open_source/botanisense-fresh/data/state.db"
).expanduser()
DEFAULT_SHADOW = REPO / "data/ab/state.db"

# Model-side state to wipe. Everything here is a pure function of
# (model output × detections); the copy would otherwise let the
# pipeline's explicit per-stage cache checks short-circuit and the run
# would silently keep TEACHER results under ENGINE=mobile.
CLEAR_TABLES = [
    # embeddings are the dangerous one: teacher AND mobile are both
    # 768-dim, so pipeline's dim guard can NOT catch a mixed table.
    "embeddings",
    "predictions",
    "detections",
    "bbox_predictions",
    "knn_predictions",        # teacher-space FAISS results — stale here
    "bbox_knn_predictions",
    "cluster_names",
    "cluster_albums",
    "clusterer_state",
    "cluster_remap",
]

# taxon_scores is NOT in CLEAR_TABLES: this is the hybrid layout — the
# teacher's rank='kingdom' rows are kept so the Plantae gate and badges
# keep their calibration, everything else (none written by today's
# pipeline, but guard anyway) goes.
CLEAR_NON_KINGDOM_TAXON_SCORES = (
    "DELETE FROM taxon_scores WHERE rank <> 'kingdom'"
)

# Person flag + detection scan marker: the shadow instance re-derives
# both with the gemini1 detector (compose sets YOLO_BACKEND=gemini1).
RESET_ASSET_COLUMNS = """
UPDATE assets SET cluster_id = NULL, has_person = NULL,
                  yolo_scanned_at = NULL
"""


def backup(baseline: Path, shadow: Path) -> None:
    """sqlite backup API — the only safe copy of a live WAL database
    (a plain `cp` races writers and loses the -wal tail)."""
    shadow.parent.mkdir(parents=True, exist_ok=True)
    if shadow.exists():
        sys.exit(f"refusing to overwrite existing shadow DB {shadow}")
    src = sqlite3.connect(f"file:{baseline}?mode=ro", uri=True)
    dst = sqlite3.connect(shadow)
    with dst:
        src.backup(dst)
    src.close()
    dst.close()


def clear_model_side(shadow: Path, *, engine: str, version: str) -> dict:
    conn = sqlite3.connect(shadow)
    conn.row_factory = sqlite3.Row
    counts: dict[str, int] = {}
    with conn:
        for table in CLEAR_TABLES:
            cur = conn.execute(f"SELECT COUNT(*) FROM {table}")
            counts[table] = cur.fetchone()[0]
            conn.execute(f"DELETE FROM {table}")
        cur = conn.execute(CLEAR_NON_KINGDOM_TAXON_SCORES)
        counts["taxon_scores(non-kingdom)"] = cur.rowcount
        cur = conn.execute(RESET_ASSET_COLUMNS)
        counts["assets_reset"] = cur.rowcount
        # Self-identification markers (reads as: which model built the
        # model-side rows in this DB).
        conn.execute(
            "INSERT INTO album_meta (key, value_json, updated_at) "
            "VALUES ('model_version', ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
            "updated_at=excluded.updated_at",
            (version,),
        )
        conn.execute(
            "INSERT INTO album_meta (key, value_json, updated_at) "
            "VALUES ('engine', ?, datetime('now')) "
            "ON CONFLICT(key) DO UPDATE SET value_json=excluded.value_json, "
            "updated_at=excluded.updated_at",
            (engine,),
        )
    conn.close()
    return counts


def verify(baseline: Path, shadow: Path) -> None:
    """Assertions the runbook relies on: same asset corpus, same search
    order, kingdom scores intact."""
    def _scalar(db: Path, sql: str) -> int:
        conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
        try:
            return conn.execute(sql).fetchone()[0]
        finally:
            conn.close()

    checks = {
        "assets": "SELECT COUNT(*) FROM assets",
        "search_results": "SELECT COUNT(*) FROM search_results",
        "kingdom_taxon_scores": (
            "SELECT COUNT(*) FROM taxon_scores WHERE rank='kingdom'"
        ),
    }
    for name, sql in checks.items():
        b, s = _scalar(baseline, sql), _scalar(shadow, sql)
        if b != s:
            sys.exit(
                f"verify FAILED: {name} baseline={b} shadow={s} — "
                f"the copy or the clear step lost rows"
            )
        print(f"  ok {name}: {b}")
    print("shadow DB verified against baseline")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--shadow", type=Path, default=DEFAULT_SHADOW)
    ap.add_argument(
        "--engine", default="mobile-v0",
        help="value for the album_meta engine marker",
    )
    ap.add_argument(
        "--version", default="",
        help="identity key for the model_version marker; default asks "
             "mobile_engine for the configured package's version",
    )
    args = ap.parse_args()

    if not args.baseline.exists():
        sys.exit(f"baseline DB not found: {args.baseline}")

    version = args.version
    if not version:
        sys.path.insert(0, str(REPO))
        from mobile_engine import MobileEngine, default_package_dir

        version = MobileEngine(default_package_dir()).version_key()

    print(f"baseline: {args.baseline}")
    print(f"shadow:   {args.shadow}")
    print(f"engine:   {args.engine}")
    print(f"version:  {version}")
    backup(args.baseline, args.shadow)
    counts = clear_model_side(args.shadow, engine=args.engine, version=version)
    for table, n in counts.items():
        print(f"  cleared {table}: {n} rows")
    verify(args.baseline, args.shadow)
    near_dup = args.shadow.parent / "near_dup_cache.pkl"
    if near_dup.exists():
        near_dup.unlink()
        print(f"removed {near_dup}")
    print(
        "\ndone. Next:\n"
        "  docker compose -f compose.s0.yml up --build -d   # :8771\n"
        "  then trigger a run from the UI (same query as production)."
    )


if __name__ == "__main__":
    main()
