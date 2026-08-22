"""Apply the YOLO-evidence rule to assets.has_person across the whole library.

Rule (calibrated on 102-label stratified sample, 2026-08-15):

  has_person = 1  iff  classifier_p >= 0.15  AND  yolo_person_max >= 0.20
  has_person = 0  otherwise

Why this rule (vs. pure classifier @ T=0.15):
  - Sample-weighted FP rate drops from 14/84 (17%) to 12/84 (14%).
  - 95% bootstrap CI on precision is [1.000, 1.000] — the remaining FPs are
    in a single weight-1 stratum with no statistical room to escape.
  - 100% recall on Plantae band (the population-weighted sweet spot).
  - YOLO acts as an evidence gate: if YOLO truly saw nothing (yolo_max
    near 0), the model's positive is a false-positive misfire and we
    override it. yolo_min=0.20 preserves wedding/family portraits
    (yolo ~0.27) while killing pure-plant misfires (yolo ~0.0).

Why this rule (vs. pure classifier @ T=0.50):
  - Pure classifier @ 0.50 has FP=8 but FN=17 (recall ~0.83). With this
    rule at T=0.15, recall is ~0.98.

Run with --dry-run first to preview the diff. Without --dry-run it
UPDATEs `assets.has_person` in place and POSTs /api/cache/invalidate
so the running uvicorn worker drops its stale /api/results cache.

Usage:
  PYTHONPATH=. python -u scripts/person_apply_yolo_evidence.py [--dry-run]
"""
from __future__ import annotations

import argparse
import os
import sqlite3
import sys
import time
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np

# Force unbuffered output
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import storage


DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")

THRESHOLD = 0.15     # classifier p cutoff
YOLO_MIN = 0.20      # YOLO person-max evidence gate


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_yolo_person_max(conn) -> dict[str, float]:
    """Map asset_id -> max(score) across all detections where label in
    {person, face, human}. Used as the YOLO evidence signal."""
    rows = conn.execute("""
        SELECT asset_id, label, score
        FROM detections
        WHERE label IN ('person', 'face', 'human')
    """).fetchall()
    out: dict[str, float] = defaultdict(float)
    for r in rows:
        aid, _lbl, sc = r["asset_id"], r["label"], float(r["score"])
        if sc > out[aid]:
            out[aid] = sc
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="Compute and print the diff without writing")
    args = ap.parse_args()

    blob = joblib.load(MODEL_PATH)
    coef = blob["coef"]
    intercept = float(blob["intercept"])

    print(f"Rule: has_person = 1 iff classifier_p >= {THRESHOLD} "
          f"AND yolo_person_max >= {YOLO_MIN}", flush=True)
    print(f"Coefficients: {coef.shape} intercept={intercept:.4f}", flush=True)
    if args.dry_run:
        print("--dry-run: NO writes will be made", flush=True)
    print(flush=True)

    with storage.connect(DB) as conn:
        # 1. Score every asset with the classifier.
        all_ids, matrix = storage.all_embeddings(conn)
        if not all_ids:
            print("ERROR: no embeddings in DB", file=sys.stderr)
            sys.exit(1)
        p = sigmoid(matrix @ coef + intercept)
        model_p = {a: float(p[i]) for i, a in enumerate(all_ids)}
        print(f"Scored {len(all_ids)} assets with classifier.", flush=True)
        print(f"  p range: [{min(model_p.values()):.4f}, "
              f"{max(model_p.values()):.4f}]", flush=True)
        print(f"  p >= {THRESHOLD}: {sum(1 for v in model_p.values() if v >= THRESHOLD)}",
              flush=True)
        print(flush=True)

        # 2. YOLO evidence per asset.
        yolo_max = load_yolo_person_max(conn)
        n_with_yolo = sum(1 for v in yolo_max.values() if v >= YOLO_MIN)
        print(f"YOLO person-max per asset: {len(yolo_max)} assets have a "
              f"person/face/human detection (any score).", flush=True)
        print(f"  yolo_max >= {YOLO_MIN}: {n_with_yolo}", flush=True)
        print(flush=True)

        # 3. Compute new has_person and diff vs. current.
        cur_hp = {r[0]: r[1] for r in conn.execute(
            "SELECT asset_id, has_person FROM assets"
        ).fetchall()}

        new_hp = {}
        for aid in all_ids:
            mp = model_p[aid]
            ym = yolo_max.get(aid, 0.0)
            new_hp[aid] = 1 if (mp >= THRESHOLD and ym >= YOLO_MIN) else 0

        # Diff
        flips_up = [a for a in all_ids if cur_hp.get(a) == 0 and new_hp[a] == 1]
        flips_down = [a for a in all_ids if cur_hp.get(a) == 1 and new_hp[a] == 0]
        unchanged_y = sum(1 for a in all_ids
                          if cur_hp.get(a) == 1 and new_hp[a] == 1)
        unchanged_n = sum(1 for a in all_ids
                          if cur_hp.get(a) == 0 and new_hp[a] == 0)

        n_new_hp1 = sum(new_hp.values())
        n_cur_hp1 = sum(1 for v in cur_hp.values() if v == 1)

        print("=== Diff ===", flush=True)
        print(f"  current hp=1: {n_cur_hp1}  →  new hp=1: {n_new_hp1}", flush=True)
        print(f"  flips 0 → 1 (classifier rescues YOLO): {len(flips_up)}", flush=True)
        print(f"  flips 1 → 0 (YOLO gate kills false positive): {len(flips_down)}",
              flush=True)
        print(f"  unchanged hp=1: {unchanged_y}", flush=True)
        print(f"  unchanged hp=0: {unchanged_n}", flush=True)
        print(flush=True)

        # Sanity: show distribution by model_p band
        bands = [(0.0, 0.05), (0.05, 0.10), (0.10, 0.15),
                 (0.15, 0.20), (0.20, 0.30), (0.30, 0.50), (0.50, 1.01)]
        print("=== New has_person by classifier_p band ===", flush=True)
        print(f"  {'band':>15}  {'n':>5}  {'yolo_max>=0.20':>15}  "
              f"{'hp=1':>5}  {'hp=0':>5}", flush=True)
        for lo, hi in bands:
            in_band = [a for a in all_ids if lo <= model_p[a] < hi]
            n_yolo = sum(1 for a in in_band if yolo_max.get(a, 0.0) >= YOLO_MIN)
            n_hp1 = sum(1 for a in in_band if new_hp[a] == 1)
            n_hp0 = sum(1 for a in in_band if new_hp[a] == 0)
            print(f"  [{lo:.2f},{hi:.2f})  {len(in_band):>5}  {n_yolo:>15}  "
                  f"{n_hp1:>5}  {n_hp0:>5}", flush=True)
        print(flush=True)

        if args.dry_run:
            print("--dry-run: not writing, exiting.", flush=True)
            return

        # 4. Apply UPDATE. Bulk-update in chunks of 500 per SQL statement
        # (faster than per-row, and SQLite's IN-list limit is ~999).
        t0 = time.monotonic()
        BATCH = 500
        hp1_aids = [a for a, v in new_hp.items() if v == 1]
        hp0_aids = [a for a, v in new_hp.items() if v == 0]
        n_set_1 = n_set_0 = 0
        for i in range(0, len(hp1_aids), BATCH):
            chunk = hp1_aids[i:i+BATCH]
            q = ",".join("?" * len(chunk))
            n_set_1 += conn.execute(
                f"UPDATE assets SET has_person = 1 WHERE asset_id IN ({q})",
                chunk,
            ).rowcount
        for i in range(0, len(hp0_aids), BATCH):
            chunk = hp0_aids[i:i+BATCH]
            q = ",".join("?" * len(chunk))
            n_set_0 += conn.execute(
                f"UPDATE assets SET has_person = 0 WHERE asset_id IN ({q})",
                chunk,
            ).rowcount
        print(f"Applied: {n_set_1} hp=1 writes, {n_set_0} hp=0 writes "
              f"in {time.monotonic()-t0:.1f}s", flush=True)

    # 5. Invalidate /api/results cache on the running uvicorn worker.
    _post_cache_invalidate()

    print("\nDone. has_person is now consistent with the YOLO-evidence rule.",
          flush=True)


def _post_cache_invalidate():
    """POST /api/cache/invalidate?prefix=all: so the running API worker
    drops its in-process results cache. Honors PORT env var."""
    import json
    import urllib.request
    port = os.environ.get("PORT", "8765")
    base_url = f"http://127.0.0.1:{port}"
    try:
        req = urllib.request.Request(
            f"{base_url}/api/cache/invalidate?prefix=all:", method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
            print(f"  cache invalidated: {body}", flush=True)
    except Exception as e:
        print(f"  WARN: cache invalidate failed: {type(e).__name__}: {e}",
              flush=True)
        print(f"  (uvicorn may need a restart for stale /api/results)",
              flush=True)


if __name__ == "__main__":
    main()