"""Build a stratified sample using **reconstructed YOLO has_person** as the basis.

The current `assets.has_person` has been mutated by classifier flips,
Plantae-band safety nets, hand-with-plant heuristics, and threshold
bumps. The original stratified sample (70 photos) used the **YOLO-only**
has_person (max person score >= PERSON_MIN_CONF, currently 0.50).

This script rebuilds that original criterion from the **detections**
table alone — which is append-only and still has every YOLO run's
person scores — and uses it to pick stratified samples. We CAN'T
reproduce the exact same 70 asset_ids (random seed differs, pop is
slightly different), but the stratum definitions match what the
original sample used.

We also `carry-forward` any existing labels in `data/label_sample/` —
including the 32 Plantae-band labels — so no work is lost. Old batches
are auto-archived to `data/label_sample_archive_<ts>/` before unlink.

Strata (matching person_label_sample.py):
  1_yolo_only_disagree       YOLO hp=1, classifier model < 0.30
  2_borderline_hp1_low_model YOLO hp=1, 0.0 <= model < 0.3
  3_borderline_hp1_high_model YOLO hp=1, 0.7 <= model <= 1.0
  4_high_conf_hp1_untouched  YOLO hp=1, yolo_max >= 0.95, model >= 0.9,
                             AND NOT in training (face>=0.5 OR no hand overlap)
  5_hp0_plantae              YOLO hp=0, Plantae score >= 0.5
  6_hp0_model_agrees_no      YOLO hp=0, model < 0.2
  7_hp0_model_would_flip_yes YOLO hp=0, model >= 0.5

Run with --per-stratum N to change sampling.

Usage:
  python scripts/person_label_sample_yolo_basis.py --per-stratum 10
"""
from __future__ import annotations

import argparse
import shutil
import sqlite3
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml

import storage


DB = Path("data/state.db")
ROOT = Path("data/label_sample")
MODEL_PATH = Path("data/person_classifier.joblib")

PERSON_MIN_CONF = 0.50
FACE_MIN_CONF = 0.50
HAND_MIN_CONF = 0.05
CONTAINMENT_THRESHOLD = 0.85


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model():
    blob = joblib.load(MODEL_PATH)
    return blob["coef"], float(blob["intercept"])


def reconstructed_yolo_hp1(conn) -> set[str]:
    """Return asset_ids where YOLO detected person with conf >= PERSON_MIN_CONF.

    Reconstructs the **original** YOLO has_person=1 set from the
    append-only detections table. NOT the current `assets.has_person`
    (which has been mutated by classifier flips and Plantae safety nets).
    """
    rows = conn.execute("""
        SELECT a.asset_id,
               COALESCE(MAX(CASE WHEN d.label='person' THEN d.score ELSE 0 END), 0) AS max_person
        FROM assets a LEFT JOIN detections d ON d.asset_id=a.asset_id
        GROUP BY a.asset_id
    """).fetchall()
    return {r[0] for r in rows if float(r[1]) >= PERSON_MIN_CONF}


def has_face_or_no_hand_overlap(conn, asset_id: str) -> bool:
    """True if face>=0.50 OR no hand box overlaps a person box by >=0.85.

    Used by stratum 4 to exclude rows that were used in training.
    """
    dets = conn.execute(
        "SELECT label, score, x, y, w, h FROM detections WHERE asset_id=?",
        (asset_id,)).fetchall()
    faces = [(x, y, w, h) for l, s, x, y, w, h in dets if l == "face" and s >= FACE_MIN_CONF]
    if faces:
        return True
    hands = [(x, y, w, h) for l, s, x, y, w, h in dets if l == "hand" and s >= HAND_MIN_CONF]
    persons = [(x, y, w, h) for l, s, x, y, w, h in dets
               if l in ("person", "face", "human") and s >= PERSON_MIN_CONF]
    for hx, hy, hw, hh in hands:
        if hw * hh <= 0:
            continue
        for px, py, pw, ph in persons:
            ix1, iy1 = max(hx, px), max(hy, py)
            ix2, iy2 = min(hx + hw, px + pw), min(hy + hh, py + ph)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            if (ix2 - ix1) * (iy2 - iy1) / (hw * hh) >= CONTAINMENT_THRESHOLD:
                return False
    return True


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=10)
    ap.add_argument("--out-dir", default=str(ROOT))
    args = ap.parse_args()

    sys.stdout.reconfigure(line_buffering=True)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)

    with storage.connect(DB) as conn:
        coef, intercept = load_model()
        yolo_hp1 = reconstructed_yolo_hp1(conn)
        all_ids, matrix = storage.all_embeddings(conn)
        id_to_idx = {a: i for i, a in enumerate(all_ids)}
        p = sigmoid(matrix @ coef + intercept)
        id_to_p = {a: float(p[id_to_idx[a]]) for a in all_ids}

        # Build yolo_max lookup
        max_person_rows = conn.execute("""
            SELECT a.asset_id,
                   COALESCE(MAX(CASE WHEN d.label='person' THEN d.score ELSE 0 END), 0) AS mp
            FROM assets a LEFT JOIN detections d ON d.asset_id=a.asset_id
            GROUP BY a.asset_id
        """).fetchall()
        id_to_ymp = {r[0]: float(r[1]) for r in max_person_rows}

        # Strata (using reconstructed YOLO has_person + current classifier p)
        s1_ids = [a for a in yolo_hp1 if id_to_p.get(a, 1.0) < 0.30]
        s2_ids = [a for a in yolo_hp1 if 0.0 <= id_to_p.get(a, 0.0) < 0.3]
        s3_ids = [a for a in yolo_hp1 if 0.7 <= id_to_p.get(a, 0.0) <= 1.0]
        s4_candidates = [a for a in yolo_hp1
                         if id_to_ymp.get(a, 0) >= 0.95 and id_to_p.get(a, 0.0) >= 0.9]
        s4_ids = [a for a in s4_candidates if has_face_or_no_hand_overlap(conn, a)]
        plant_rows = conn.execute(
            "SELECT a.asset_id FROM assets a JOIN taxon_scores t ON t.asset_id=a.asset_id "
            "WHERE t.taxon_name='Plantae' AND t.score >= 0.5"
        ).fetchall()
        s5_ids = [a for (a,) in plant_rows if a not in yolo_hp1]
        s6_ids = [a for a in all_ids if a not in yolo_hp1 and id_to_p.get(a, 0.0) < 0.2]
        s7_ids = [a for a in all_ids if a not in yolo_hp1 and id_to_p.get(a, 0.0) >= 0.5]

        strata = {
            "1_yolo_only_disagree": (s1_ids, 1),
            "2_borderline_hp1_low_model": (s2_ids, 1),
            "3_borderline_hp1_high_model": (s3_ids, 1),
            "4_high_conf_hp1_untouched": (s4_ids, 1),
            "5_hp0_plantae": (s5_ids, 0),
            "6_hp0_model_agrees_no": (s6_ids, 0),
            "7_hp0_model_would_flip_yes": (s7_ids, 0),
        }
        print("Stratum sizes (reconstructed YOLO has_person):")
        for name, (ids, _) in strata.items():
            print(f"  {name}: {len(ids)} available")
        print()

        # Snapshot old batches first; this also serves as the source for
        # carry-forward so previous labels survive (the snapshot dir
        # holds the previous run's state, which is what we want to
        # inherit).
        old_batches = sorted(out_dir.glob("batch_*.yaml"))
        archive = None
        if old_batches:
            archive = out_dir.parent / f"label_sample_archive_{int(time.time())}"
            archive.mkdir(parents=True, exist_ok=False)
            for ob in old_batches:
                shutil.copy2(ob, archive / ob.name)
            print(f"  archived {len(old_batches)} old batches → {archive}/", flush=True)

        # Carry-forward existing labels from ANY archive dir, plus the
        # current dir. Archives are kept across runs as a safety net —
        # the most recent archive is the snapshot we just made (and is
        # full of empty-label rows), but earlier archives may hold real
        # labels. Merge them all, with later sources overwriting earlier
        # ones.
        existing: dict[str, dict] = {}
        carry_sources = sorted(out_dir.parent.glob("label_sample_archive_*"))
        if archive:
            # Put the snapshot we just made last so it wins on conflict.
            carry_sources = [a for a in carry_sources if a != archive] + [archive]
        for src in carry_sources:
            for yf in sorted(src.glob("batch_*.yaml")):
                for row in yaml.safe_load(yf.read_text()) or []:
                    if row.get("actual_person") is not None:
                        existing[row["asset_id"]] = row
        print(f"  carrying forward {len(existing)} already-labeled rows from "
              f"{len(carry_sources)} archive(s)")
        print()

        # Sample
        sampled: list[dict] = []
        for name, (ids, yolo_person) in strata.items():
            n = min(args.per_stratum, len(ids))
            if n == 0:
                print(f"  {name}: 0 available, skipping")
                continue
            picks = list(rng.choice(len(ids), n, replace=False))
            for i in picks:
                aid = ids[int(i)]
                mp = id_to_ymp.get(aid)
                row = {
                    "stratum": name,
                    "asset_id": aid,
                    "filename": conn.execute(
                        "SELECT original_file_name FROM assets WHERE asset_id=?",
                        (aid,)).fetchone()["original_file_name"],
                    "thumb_path": f"data/thumbs/{aid}.jpg",
                    "yolo_max": round(mp, 4) if mp is not None else None,
                    "yolo_person": yolo_person,
                    "model_p": round(float(id_to_p.get(aid, 0.0)), 4),
                    "actual_person": None,
                    "use_case_relevant": None,
                    "notes": None,
                    "sublabel": None,
                }
                if aid in existing:
                    ex = existing[aid]
                    row["actual_person"] = ex.get("actual_person")
                    row["use_case_relevant"] = ex.get("use_case_relevant")
                    row["notes"] = ex.get("notes")
                    row["sublabel"] = ex.get("sublabel")
                sampled.append(row)

        # Append carry-forward labels that AREN'T in the new sample —
        # e.g. Plantae-band labels from a previous run that the new
        # 7-stratum schema doesn't generate. Their stratum name still
        # starts with "plantae_band_" so PR-curve scripts can pick them
        # up via that prefix.
        n_added_carry = 0
        for aid, ex in existing.items():
            if any(r["asset_id"] == aid for r in sampled):
                continue  # already in new sample, already inherited labels
            sampled.append({
                "stratum": ex.get("stratum", "plantae_band_carry_forward"),
                "asset_id": aid,
                "filename": ex.get("filename"),
                "thumb_path": ex.get("thumb_path", f"data/thumbs/{aid}.jpg"),
                "yolo_max": ex.get("yolo_max"),
                "yolo_person": ex.get("yolo_person", 1),
                "model_p": ex.get("model_p"),
                "actual_person": ex.get("actual_person"),
                "use_case_relevant": ex.get("use_case_relevant"),
                "notes": ex.get("notes"),
                "sublabel": ex.get("sublabel"),
            })
            n_added_carry += 1
        if n_added_carry:
            print(f"  appended {n_added_carry} carry-forward rows (Plantae-band)", flush=True)

        # Skip rows whose thumb is missing
        before = len(sampled)
        sampled = [r for r in sampled if Path(r["thumb_path"]).exists()]
        print(f"  skipped {before - len(sampled)} rows with no thumb")

        # Index sequentially
        for i, r in enumerate(sampled):
            r["i"] = i

        # Write batches
        bs = args.batch_size
        for old in old_batches:
            old.unlink()
        for i in range(0, len(sampled), bs):
            chunk = sampled[i:i+bs]
            path = out_dir / f"batch_{i // bs:02d}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(chunk, f, sort_keys=False, allow_unicode=True)
            n_labeled = sum(1 for r in chunk if r.get("actual_person") is not None)
            print(f"  wrote {path} ({len(chunk)} rows, {n_labeled} already labeled)", flush=True)

        n_already = sum(1 for r in sampled if r.get("actual_person") is not None)
        n_todo = sum(1 for r in sampled if r.get("actual_person") is None)
        print(f"\nTotal sampled: {len(sampled)}, already labeled: {n_already}, "
              f"need labels: {n_todo}")


if __name__ == "__main__":
    main()