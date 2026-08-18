"""Flip Plantae + classifier p<0.10 hp=1 → hp=0 (YOLO false positives).

After the 30-photo Plantae-band label review, we found:
  - 30 Plantae-band photos at classifier p in [0.10, 0.45): all 30 had
    actual_person=1. YOLO was correct in this band. Classifier was
    too cautious.
  - 176 Plantae-band photos at classifier p<0.10: spot-check confirmed
    these are pure-plant photos where YOLO fired person on arm/hand/
    jacket/etc. YOLO false positives.

Rule:
  Plantae score >= 0.5
  AND classifier p < 0.10
  AND currently has_person = 1
  → flip to hp = 0

This is conservative — we only flip when the classifier is *very*
confident there's no person (p<0.10), so we don't risk killing
borderline real-person photos in the [0.10, 0.45) band that the
labels showed are mostly correct.

Run with --dry-run to preview.
"""
from __future__ import annotations

import argparse
import os
from pathlib import Path

import joblib
import numpy as np

import storage


DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")

PLANTAE_THRESHOLD = 0.5
P_THRESHOLD = 0.10


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    blob = joblib.load(MODEL_PATH)
    coef = blob["coef"]
    intercept = float(blob["intercept"])

    with storage.connect(DB) as conn:
        all_ids, matrix = storage.all_embeddings(conn)
        id_to_idx = {a: i for i, a in enumerate(all_ids)}
        p = sigmoid(matrix @ coef + intercept)

        plantae_ids = {r[0] for r in conn.execute(
            "SELECT asset_id FROM taxon_scores WHERE taxon_name='Plantae' AND score>=?",
            (PLANTAE_THRESHOLD,))}
        cur = {r[0]: r[1] for r in conn.execute(
            "SELECT asset_id, has_person FROM assets"
        )}

        to_flip = [a for a in all_ids
                   if a in plantae_ids
                   and cur.get(a) == 1
                   and p[id_to_idx[a]] < P_THRESHOLD]
        print(f"Plantae>={PLANTAE_THRESHOLD} AND hp=1 AND classifier p<{P_THRESHOLD}:")
        print(f"  Found {len(to_flip)} rows to flip")

        # Show top 10 by lowest p (most confident)
        to_flip.sort(key=lambda a: p[id_to_idx[a]])
        for aid in to_flip[:10]:
            fn = conn.execute(
                "SELECT original_file_name FROM assets WHERE asset_id=?",
                (aid,)).fetchone()["original_file_name"]
            dets = conn.execute(
                "SELECT label, score FROM detections WHERE asset_id=? AND score >= 0.3",
                (aid,)).fetchall()
            max_person = max((d["score"] for d in dets if d["label"] == "person"), default=0)
            print(f"  p={p[id_to_idx[aid]]:.3f} person_max={max_person:.2f}  {fn}")

        if args.dry_run:
            print("\n--dry-run: not writing")
            return

        if not to_flip:
            print("\nNo rows to flip.")
            return

        # Apply in batches
        BATCH = 500
        for i in range(0, len(to_flip), BATCH):
            chunk = to_flip[i:i+BATCH]
            qmarks = ",".join("?" * len(chunk))
            conn.execute(
                f"UPDATE assets SET has_person = 0 WHERE asset_id IN ({qmarks})",
                chunk,
            )
            print(f"  flipped {min(i+BATCH, len(to_flip))}/{len(to_flip)}")
        # storage.connect auto-commits on exit

        # Bust API cache
        try:
            from urllib.request import Request, urlopen
            import json as _json
            port = os.environ.get("PORT", "8765")
            req = Request(f"http://127.0.0.1:{port}/api/cache/invalidate?prefix=all:",
                          method="POST")
            with urlopen(req, timeout=5) as resp:
                print(f"  cache invalidated: {_json.loads(resp.read().decode('utf-8'))}")
        except Exception as e:
            print(f"  WARN: cache invalidate failed: {type(e).__name__}: {e}")

    print(f"\nDone. Flipped {len(to_flip)} Plantae+p<0.10 hp=1 → hp=0.")


if __name__ == "__main__":
    main()