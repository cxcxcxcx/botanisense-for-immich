"""Apply BioCLIP classifier decisions to `assets.has_person`.

The classifier is much better at the use-case-relevant has_person
question than YOLO is (F1 0.96 vs 0.70 on 70 hand-labeled photos).

This script does a one-time update: for every asset with a BioCLIP
embedding, set has_person = 1 if the classifier says so, regardless
of the current value. This catches the cases where YOLO missed a
real person (stratum 7: 10/10 model-correct, 0/10 YOLO-correct).

We do NOT do the reverse (YOLO said yes → model says no → set to 0).
The 70-label sample had 0 cases where the model was right and YOLO
was wrong on use_case_relevant, so down-flipping isn't safe to
automate yet. If we want it later, gate it on:

  yolo_max < 0.65 AND model_p < 0.2

…and re-evaluate.

Usage:
  PYTHONPATH=. python scripts/person_apply_flips.py [--dry-run]

With --dry-run, prints the proposed changes without writing.
Without --dry-run, commits to the DB inside a single transaction
(commits on clean exit, rolls back on exception).

Cache invalidation: the API results cache for /api/results is
touched at the end so the next request re-reads from the DB.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np

import storage


DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change but don't write to DB")
    args = ap.parse_args()

    blob = joblib.load(MODEL_PATH)
    coef = blob["coef"]
    intercept = float(blob["intercept"])
    print(f"Model: trained_at={blob.get('trained_at')}  "
          f"n_pos={blob.get('n_pos')}  n_neg={blob.get('n_neg')}")

    with storage.connect(DB) as conn:
        # Get all asset_ids that have an embedding, plus their current has_person.
        # Embeddings live in the `embeddings` table; `assets.has_person` is what we update.
        all_ids, matrix = storage.all_embeddings(conn)
        id_to_idx = {a: i for i, a in enumerate(all_ids)}

        rows = conn.execute(
            "SELECT asset_id, has_person FROM assets WHERE asset_id IN ({})".format(
                ",".join("?" * len(all_ids))
            ),
            list(all_ids),
        ).fetchall()
        # Build a dict asset_id -> has_person
        hp = {r[0]: r[1] for r in rows}

        # Compute model scores for everyone
        p = sigmoid(matrix @ coef + intercept)
        model_p = {a: float(p[i]) for i, a in enumerate(all_ids)}

        # Determine flips: model says yes (p >= 0.5) AND current hp != 1
        to_flip_up = [a for a in all_ids if model_p[a] >= 0.5 and hp.get(a) != 1]

        # Summary stats
        n_total = len(all_ids)
        n_current_hp1 = sum(1 for a in all_ids if hp.get(a) == 1)
        n_model_says_yes = sum(1 for a in all_ids if model_p[a] >= 0.5)
        n_overlap = sum(1 for a in all_ids if model_p[a] >= 0.5 and hp.get(a) == 1)
        n_flip_up = len(to_flip_up)

        print(f"\nPopulation with embeddings: {n_total}")
        print(f"  current has_person=1:    {n_current_hp1}")
        print(f"  model says yes (p>=0.5): {n_model_says_yes}")
        print(f"  both say yes:            {n_overlap}")
        print(f"  WOULD FLIP up:           {n_flip_up}")

        # Show a sample of the flips with scores
        to_flip_up.sort(key=lambda a: -model_p[a])
        print(f"\nTop 10 highest-confidence flips:")
        cur = conn.execute(
            "SELECT asset_id, original_file_name FROM assets WHERE asset_id IN ({})".format(
                ",".join("?" * 10)
            ),
            to_flip_up[:10],
        ).fetchall()
        fn = {r[0]: r[1] for r in cur}
        for a in to_flip_up[:10]:
            print(f"  p={model_p[a]:.3f}  {a}  {fn.get(a, '?')}")

        # Show bottom 5 (lowest-confidence flips, near threshold)
        print(f"\nLowest-confidence flips (just over p=0.5):")
        for a in to_flip_up[-5:]:
            print(f"  p={model_p[a]:.3f}  {a}  {fn.get(a, '?')}")

        if args.dry_run:
            print("\n--dry-run: not committing")
            return

        if not to_flip_up:
            print("\nNo flips to make. Done.")
            return

        # Apply in batches to keep the UPDATE log readable
        BATCH = 500
        n_done = 0
        for i in range(0, len(to_flip_up), BATCH):
            chunk = to_flip_up[i:i+BATCH]
            qmarks = ",".join("?" * len(chunk))
            conn.execute(
                f"UPDATE assets SET has_person = 1 WHERE asset_id IN ({qmarks})",
                chunk,
            )
            n_done += len(chunk)
            print(f"  updated {n_done}/{len(to_flip_up)}")

        # storage.connect() auto-commits on clean exit; rolls back on exception.
        # Invalidate the running uvicorn worker's in-process /api/results
        # cache so the new flags show up immediately (no restart needed).
        _post_cache_invalidate()

    print(f"\nDone. Flipped {n_flip_up} rows from has_person=0 to has_person=1.")


def _post_cache_invalidate(base_url: str = "http://127.0.0.1:8765"):
    """POST /api/cache/invalidate so the running API worker drops its
    in-process results cache. Uses urllib to avoid a hard dependency on
    requests. Honors PORT env var if set.
    """
    import os
    port = os.environ.get("PORT", "8765")
    base_url = f"http://127.0.0.1:{port}"
    import json
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{base_url}/api/cache/invalidate?prefix=all:",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
            print(f"  cache invalidated: {body}")
    except Exception as e:
        print(f"  WARN: cache invalidate failed: {type(e).__name__}: {e}")
        print(f"  (the running uvicorn worker will still serve stale data "
              f"until restarted)")