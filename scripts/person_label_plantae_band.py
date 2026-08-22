"""Sample photos from the Plantae+person band — the gap in our labeled set.

The original 70-photo stratified sample had ZERO photos in the band:
  hp=1 + Plantae score >= 0.5 + classifier p in [0.10, 0.45)

This is exactly the band where YOLO says person but the classifier
is biased low because of Plantae-as-negative training. We need to
know:
  - Is the classifier right or wrong on these?
  - Is the hybrid rule (p>=0.5 trust classifier, else trust YOLO)
    the right policy?

This script samples from that band into batches of ~10 for parallel
labelling agents. Per-batch output goes to data/label_sample/.

Stratification:
  band_a:  classifier p in [0.10, 0.20)   (very low — likely false positives)
  band_b:  classifier p in [0.20, 0.30)
  band_c:  classifier p in [0.30, 0.40)
  band_d:  classifier p in [0.40, 0.45)   (near threshold — interesting)

Sample size: 10 per band × 4 bands = 40 photos. Plenty to evaluate
the classifier bias.

Run with --per-band N to change sampling, --batch-size N to chunk.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path
import sys

import joblib
import numpy as np
import yaml

import storage


ROOT = Path("data/label_sample")
DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-band", type=int, default=10)
    ap.add_argument("--batch-size", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    blob = joblib.load(MODEL_PATH)
    coef, intercept = blob["coef"], float(blob["intercept"])

    with storage.connect(DB) as conn:
        all_ids, matrix = storage.all_embeddings(conn)
        id_to_idx = {a: i for i, a in enumerate(all_ids)}
        p = sigmoid(matrix @ coef + intercept)

        # Find hp=1 + Plantae >= 0.5 + classifier p in [0.10, 0.45)
        plantae_ids = {r[0] for r in conn.execute(
            "SELECT asset_id FROM taxon_scores WHERE taxon_name='Plantae' AND score>=0.5"
        )}
        cur = {r[0]: r[1] for r in conn.execute(
            "SELECT asset_id, has_person FROM assets"
        )}

        # Per-band lists
        bands = [
            ("plantae_band_a_p0.10-0.20", 0.10, 0.20),
            ("plantae_band_b_p0.20-0.30", 0.20, 0.30),
            ("plantae_band_c_p0.30-0.40", 0.30, 0.40),
            ("plantae_band_d_p0.40-0.45", 0.40, 0.45),
        ]
        for name, lo, hi in bands:
            ids = [a for a in all_ids if a in plantae_ids
                   and cur.get(a) == 1
                   and lo <= p[id_to_idx[a]] < hi]
            print(f"  {name}: {len(ids)} available", flush=True)

        # Sample
        rng = np.random.default_rng(args.seed)
        sampled = []
        for name, lo, hi in bands:
            ids = [a for a in all_ids if a in plantae_ids
                   and cur.get(a) == 1
                   and lo <= p[id_to_idx[a]] < hi]
            if len(ids) < args.per_band:
                picks = ids
            else:
                picks = list(rng.choice(len(ids), args.per_band, replace=False))
                picks = [ids[int(i)] for i in picks]
            for aid in picks:
                fn = conn.execute(
                    "SELECT original_file_name FROM assets WHERE asset_id=?",
                    (aid,)).fetchone()["original_file_name"]
                sampled.append({
                    "stratum": name,
                    "asset_id": aid,
                    "filename": fn,
                    "thumb_path": f"data/thumbs/{aid}.jpg",
                    "yolo_max": None,  # computed below
                    "yolo_person": 1,
                    "model_p": round(float(p[id_to_idx[aid]]), 4),
                    "actual_person": None,
                    "use_case_relevant": None,
                    "notes": None,
                    "sublabel": None,
                })

        # Compute yolo_max per row
        for r in sampled:
            row = conn.execute(
                "SELECT MAX(score) FROM detections WHERE asset_id=? AND label='person'",
                (r["asset_id"],),
            ).fetchone()
            r["yolo_max"] = round(float(row[0]), 4) if row and row[0] else None

        print(f"\nTotal sampled: {len(sampled)}", flush=True)

        # Carry over any existing labels
        existing: dict[str, dict] = {}
        for yf in ROOT.glob("batch_*.yaml"):
            for row in yaml.safe_load(yf.read_text()) or []:
                if row.get("actual_person") is not None:
                    existing[row["asset_id"]] = row
        for r in sampled:
            if r["asset_id"] in existing:
                ex = existing[r["asset_id"]]
                r["actual_person"] = ex.get("actual_person")
                r["use_case_relevant"] = ex.get("use_case_relevant")
                r["notes"] = ex.get("notes")
                r["sublabel"] = ex.get("sublabel")

        # Index rows sequentially
        for i, r in enumerate(sampled):
            r["i"] = 700 + i  # offset from earlier batches (00..06 used 0..69)

        # Snapshot existing batches before we touch anything — recovery
        # net so a buggy carry-forward can't silently destroy old labels.
        import shutil
        import time as _time
        old_batches = sorted(ROOT.glob("batch_*.yaml"))
        if old_batches:
            archive = ROOT.parent / f"label_sample_archive_{int(_time.time())}"
            archive.mkdir(parents=True, exist_ok=False)
            for ob in old_batches:
                shutil.copy2(ob, archive / ob.name)
            print(f"  archived {len(old_batches)} old batches → {archive}/", flush=True)

        # Write batches
        bs = args.batch_size
        for old in old_batches:
            old.unlink()
        for i in range(0, len(sampled), bs):
            chunk = sampled[i:i+bs]
            path = ROOT / f"batch_{i // bs:02d}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(chunk, f, sort_keys=False, allow_unicode=True)
            print(f"  wrote {path} ({len(chunk)} rows)", flush=True)

        print(f"\nDone. {len(list(ROOT.glob('batch_*.yaml')))} batches in {ROOT}/", flush=True)


if __name__ == "__main__":
    main()