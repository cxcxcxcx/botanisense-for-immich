#!/usr/bin/env python3
"""Scan every asset in the local DB with the active YOLO backend.

Why this exists: the normal pipeline's YOLO stage only scans the assets
returned by that run's Immich smart-search query. The standalone "Detect
people" backfill scans the whole library but persists person/hand boxes only.
There was no "run the current detector over every photo and persist all
labels" path — so after switching a shadow DB to a new detector, most of the
library had no plant/pot boxes at all.

This script fills that gap: it walks the assets table, downloads previews,
runs the detector selected by $YOLO_BACKEND (world or gemini1), and writes
all labels via storage.replace_detections + mark_yolo_scanned.

Usage (run from the repo root, with the same env as the target app):

  YOLO_BACKEND=gemini1 \
  YOLO_GEMINI1_ONNX=$PWD/data/models/gemini1_seg_best.fp16.onnx \
  DB_PATH=$PWD/data/tg/state.db \
  .venv/bin/python scripts/yolo_backfill.py --batch-size 8

  # rescan everything (also replaces old detector rows):
  ... --force

The app can stay up; SQLite WAL serializes the writes. Do not run this
concurrently with a pipeline run or the person backfill.
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
# The app modules live at the repo root; when this file is run as
# `python scripts/yolo_backfill.py`, sys.path[0] is scripts/.
sys.path.insert(0, str(ROOT))

# Same .env the app loads (IMMICH_BASE_URL / IMMICH_API_KEY etc.).
load_dotenv(ROOT / ".env")

# Importing app.py would pull in Flask + Sentry; these modules are enough.
import pipeline
import storage
from immich_client import ImmichClient


def _connection(db_path: Path) -> tuple[str, str]:
    """Env wins, then the UI-saved connection row."""
    with storage.connect(db_path) as conn:
        saved = storage.get_connection_settings(conn) or {}
    base_url = os.environ.get("IMMICH_BASE_URL") or str(saved.get("base_url") or "")
    api_key = os.environ.get("IMMICH_API_KEY") or str(saved.get("api_key") or "")
    if not base_url or not api_key:
        raise SystemExit(
            "No Immich connection: set IMMICH_BASE_URL / IMMICH_API_KEY, or "
            "save a connection in the app first."
        )
    return base_url, api_key


def _asset_ids(db_path: Path, *, force: bool, limit: int) -> list[str]:
    with storage.connect(db_path) as conn:
        where = "" if force else "WHERE yolo_scanned_at IS NULL"
        sql = f"SELECT asset_id FROM assets {where} ORDER BY first_seen_at, asset_id"
        if limit > 0:
            sql += f" LIMIT {int(limit)}"
        return [row[0] for row in conn.execute(sql)]


def _split_labels(dets: list[dict]) -> list[dict]:
    """Same tier split as the pipeline's yolo stage."""
    plant = [
        d for d in dets
        if d["label"] not in storage.PERSON_LABELS and d["label"] != storage.HAND_LABEL
    ]
    person = [
        d for d in dets
        if d["label"] in storage.PERSON_LABELS
        and d["label"] != "face"
        and d["score"] >= storage.PERSON_MIN_CONF
    ]
    hand = [
        d for d in dets
        if d["label"] == storage.HAND_LABEL and d["score"] >= storage.HAND_MIN_CONF
    ]
    face = [d for d in dets if d["label"] == "face"]
    return plant + person + hand + face


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--db",
        default=os.environ.get("DB_PATH", "data/state.db"),
        help="SQLite DB path (default: $DB_PATH or data/state.db)",
    )
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--limit", type=int, default=0, help="0 = all assets")
    parser.add_argument(
        "--force",
        action="store_true",
        help="rescan assets already marked yolo_scanned_at (replaces their detections)",
    )
    parser.add_argument("--conf", type=float, default=0.05)
    args = parser.parse_args()

    db_path = Path(args.db)
    if not db_path.exists():
        raise SystemExit(f"DB not found: {db_path}")
    base_url, api_key = _connection(db_path)
    ids = _asset_ids(db_path, force=args.force, limit=args.limit)
    backend = os.environ.get("YOLO_BACKEND", "world")
    print(
        f"yolo_backfill: backend={backend} db={db_path} "
        f"assets={len(ids)} batch={args.batch_size} force={args.force}",
        flush=True,
    )
    if not ids:
        return 0

    client = ImmichClient(base_url=base_url, api_key=api_key, timeout=60.0)
    decode_size = pipeline._yolo_decode_size()
    started = time.monotonic()
    scanned = 0
    total_dets = 0
    failed = 0

    with storage.connect(db_path) as conn:
        for start in range(0, len(ids), args.batch_size):
            batch = ids[start:start + args.batch_size]
            images, dead = pipeline.fetch_thumbs(
                client,
                [(asset_id, asset_id) for asset_id in batch],
                "yolo-backfill",
                max_size=decode_size,
            )
            failed += len(dead)
            ok_ids = [asset_id for asset_id in batch if asset_id in images]
            if not ok_ids:
                continue
            det_lists = pipeline.yolo.detect(
                [images[asset_id] for asset_id in ok_ids], conf=args.conf
            )
            for asset_id, raw_dets, image in zip(ok_ids, det_lists, [images[a] for a in ok_ids]):
                dets = pipeline._merge_overlapping_boxes(raw_dets)
                dets = pipeline._augment_with_yunet_faces(dets, image)
                kept = _split_labels(dets)
                storage.replace_detections(conn, asset_id=asset_id, detections=kept)
                storage.mark_yolo_scanned(
                    conn, asset_id=asset_id, has_person=storage.has_real_person(kept)
                )
                total_dets += len(kept)
            conn.commit()
            scanned += len(ok_ids)
            elapsed = max(time.monotonic() - started, 0.001)
            rate = scanned / elapsed
            eta = (len(ids) - scanned) / rate if rate > 0 else 0
            print(
                f"[{scanned}/{len(ids)}] dets={total_dets} "
                f"rate={rate:.1f}/s eta={eta/60:.1f}m failed={failed}",
                flush=True,
            )

    print(
        f"done: scanned={scanned} detections={total_dets} failed={failed} "
        f"elapsed={(time.monotonic() - started)/60:.1f}m",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
