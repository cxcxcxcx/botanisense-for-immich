"""Stratified sample builder for manual person-flag labeling.

Outputs `data/label_sample/batch_NN.yaml` files, each containing up to
BATCH_SIZE photos for a single labeling agent to chew on. Each row
includes:

  - asset_id, filename, thumb_path
  - yolo_max (max person-detection score, or null for non-hp1 rows)
  - yolo_person (1/0 from current assets.has_person)
  - model_p (classifier output, or null if no embedding)
  - current_label (the value `label` already in the YAML, if any —
    so re-runs preserve partial work)

Stratification plan — covers the cases the user actually cares about:

  1. disagreement_yolo_only           (yolo=1, model says no)
  2. borderline_hp1_low_model         (yolo=1, model in [0.0, 0.3))
  3. borderline_hp1_high_model        (yolo=1, model in [0.7, 1.0])
  4. high_conf_hp1_untouched          (yolo=1, model in [0.9, 1.0], NOT in training)
  5. hp0_plantae                      (yolo=0, plantae score >= 0.5 — sanity)
  6. hp0_non_plantae_low_model        (yolo=0, model in [0.0, 0.2) — model flips these? no, they agree)
  7. hp0_non_plantae_high_model       (yolo=0, model in [0.5, 1.0]  — model would flip these TO hp=1)

Total: ~7 strata, ~10-12 photos each = ~70-84 photos, splittable into
~6-7 batches of 12.

Sampling uses fixed seed so re-running reproduces.

No DB writes. Just reads has_person, detections, taxon_scores,
embeddings, and writes YAML.
"""
from __future__ import annotations

import argparse
import sqlite3
from pathlib import Path

import joblib
import numpy as np
import yaml

import storage


CONTAINMENT = 0.85
PERSON_MIN = 0.50
FACE_MIN = 0.50
HAND_MIN = 0.05


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model():
    blob = joblib.load(Path("data/person_classifier.joblib"))
    return blob["coef"], float(blob["intercept"])


def has_face_or_no_hand_overlap(conn, aid: str) -> bool:
    """Mirrors training-positive filter — used to know if a high-conf
    hp=1 row was in the training set."""
    rows = conn.execute(
        "SELECT label, score, x, y, w, h FROM detections WHERE asset_id=?",
        (aid,),
    ).fetchall()
    has_face = any(r[0] == "face" and r[1] >= FACE_MIN for r in rows)
    if has_face:
        return True
    hands = [r for r in rows if r[0] == "hand" and r[1] >= HAND_MIN]
    persons = [r for r in rows if r[0] in ("person", "face", "human") and r[1] >= PERSON_MIN]
    for h in hands:
        _, _, hx, hy, hw, hh = h
        h_area = hw * hh
        if h_area <= 0:
            continue
        for p in persons:
            _, _, px, py, pw, ph = p
            ix1, iy1 = max(hx, px), max(hy, py)
            ix2, iy2 = min(hx + hw, px + pw), min(hy + hh, py + ph)
            if ix2 <= ix1 or iy2 <= iy1:
                continue
            inter = (ix2 - ix1) * (iy2 - iy1)
            if inter / h_area >= CONTAINMENT:
                return False  # overlapping hand -> not a positive
    return True  # no overlapping hand


def get_hp1_rows(conn) -> list[dict]:
    """All assets with has_person=1, max person score."""
    rows = conn.execute(
        """
        SELECT a.asset_id, a.original_file_name, MAX(d.score) AS max_score
        FROM assets a
        JOIN detections d ON d.asset_id=a.asset_id
        WHERE a.has_person = 1 AND d.label = 'person'
        GROUP BY a.asset_id
        """
    ).fetchall()
    return [{"asset_id": r[0], "filename": r[1], "yolo_max": r[2]} for r in rows]


def get_hp0_rows(conn) -> list[str]:
    return [r[0] for r in conn.execute(
        "SELECT asset_id FROM assets WHERE has_person = 0"
    ).fetchall()]


def score_model(conn, aids: list[str], coef, intercept) -> dict[str, float]:
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}
    out = {}
    X_parts, keep = [], []
    for a in aids:
        if a in id_to_idx:
            X_parts.append(matrix[id_to_idx[a]])
            keep.append(a)
    if not X_parts:
        return out
    X = np.stack(X_parts)
    p = sigmoid(X @ coef + intercept)
    for a, v in zip(keep, p):
        out[a] = float(v)
    return out


def filename_for(conn, aids: list[str]) -> dict[str, str]:
    if not aids:
        return {}
    qmarks = ",".join("?" * len(aids))
    rows = conn.execute(
        f"SELECT asset_id, original_file_name FROM assets WHERE asset_id IN ({qmarks})",
        aids,
    ).fetchall()
    return {r[0]: r[1] for r in rows}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-stratum", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--batch-size", type=int, default=12)
    ap.add_argument("--out-dir", default="data/label_sample")
    args = ap.parse_args()

    db = Path("data/state.db")
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    with storage.connect(db) as conn:
        coef, intercept = load_model()
        hp1 = get_hp1_rows(conn)
        hp0 = get_hp0_rows(conn)
        print(f"hp1 rows: {len(hp1)}, hp0 rows: {len(hp0)}")

        hp1_ids = [r["asset_id"] for r in hp1]
        model_p_hp1 = score_model(conn, hp1_ids, coef, intercept)
        hp0_ids = hp0
        model_p_hp0 = score_model(conn, hp0_ids, coef, intercept)

        # ---- Stratum 1: hp=1 + model says no (yolo_only) ----
        s1 = [r for r in hp1 if model_p_hp1.get(r["asset_id"], 1.0) < 0.30]

        # ---- Stratum 2: hp=1 + model low [0.0, 0.3) ----
        s2 = [r for r in hp1 if 0.0 <= model_p_hp1.get(r["asset_id"], 0.0) < 0.3]

        # ---- Stratum 3: hp=1 + model high [0.7, 1.0] ----
        s3 = [r for r in hp1 if 0.7 <= model_p_hp1.get(r["asset_id"], 0.0) <= 1.0]

        # ---- Stratum 4: hp=1 + yolo>=0.95 + model in [0.9, 1.0] + NOT in training ----
        s4_candidates = [r for r in hp1 if r["yolo_max"] >= 0.95 and model_p_hp1.get(r["asset_id"], 0.0) >= 0.9]
        s4 = []
        for r in s4_candidates:
            if has_face_or_no_hand_overlap(conn, r["asset_id"]):
                continue  # in training
            s4.append(r)

        # ---- Stratum 5: hp=0 + plantae >= 0.5 ----
        plant_ids = [r[0] for r in conn.execute(
            "SELECT a.asset_id FROM assets a JOIN taxon_scores t ON t.asset_id=a.asset_id "
            "WHERE a.has_person=0 AND t.taxon_name='Plantae' AND t.score >= 0.5"
        ).fetchall()]
        fn = filename_for(conn, plant_ids)
        s5 = [{"asset_id": a, "filename": fn[a], "yolo_max": None} for a in plant_ids if a in fn]

        # ---- Stratum 6: hp=0 + model low (model agrees with hp=0) ----
        s6_ids = [a for a in hp0_ids if model_p_hp0.get(a, 0.0) < 0.2]
        fn6 = filename_for(conn, s6_ids)
        s6 = [{"asset_id": a, "filename": fn6[a], "yolo_max": None} for a in s6_ids if a in fn6]

        # ---- Stratum 7: hp=0 + model high (model would flip to hp=1) ----
        s7_ids = [a for a in hp0_ids if model_p_hp0.get(a, 0.0) >= 0.5]
        fn7 = filename_for(conn, s7_ids)
        s7 = [{"asset_id": a, "filename": fn7[a], "yolo_max": None} for a in s7_ids if a in fn7]

        strata = {
            "1_yolo_only_disagree": s1,
            "2_borderline_hp1_low_model": s2,
            "3_borderline_hp1_high_model": s3,
            "4_high_conf_hp1_untouched": s4,
            "5_hp0_plantae": s5,
            "6_hp0_model_agrees_no": s6,
            "7_hp0_model_would_flip_yes": s7,
        }
        print("\nStratum sizes:")
        for name, rows in strata.items():
            print(f"  {name}: {len(rows)} available")

        # Sample
        sampled: list[dict] = []
        for name, rows in strata.items():
            n = min(args.per_stratum, len(rows))
            if n == 0:
                continue
            picks = rng.choice(len(rows), n, replace=False)
            for i in picks:
                r = rows[int(i)]
                aid = r["asset_id"]
                p = model_p_hp1.get(aid)
                if p is None:
                    p = model_p_hp0.get(aid)
                thumb = f"data/thumbs/{aid}.jpg"
                if not Path(thumb).exists():
                    print(f"  WARN no thumb for {aid}, skipping")
                    continue
                sampled.append({
                    "stratum": name,
                    "asset_id": aid,
                    "filename": r["filename"],
                    "thumb_path": thumb,
                    "yolo_max": round(r["yolo_max"], 4) if r["yolo_max"] is not None else None,
                    "yolo_person": 1 if name in ("1_yolo_only_disagree", "2_borderline_hp1_low_model",
                                                 "3_borderline_hp1_high_model", "4_high_conf_hp1_untouched") else 0,
                    "model_p": round(p, 4) if p is not None else None,
                })
        print(f"\nTotal sampled: {len(sampled)}")

        # Existing labels (for resume)
        existing: dict[str, dict] = {}
        for yf in out_dir.glob("batch_*.yaml"):
            for row in yaml.safe_load(yf.read_text()) or []:
                if row.get("actual_person") is not None or row.get("notes"):
                    existing[row["asset_id"]] = row
        print(f"Resuming with {len(existing)} already-labeled rows")

        # Carry labels forward onto new sample
        for s in sampled:
            if s["asset_id"] in existing:
                s["actual_person"] = existing[s["asset_id"]]["actual_person"]
                s["use_case_relevant"] = existing[s["asset_id"]].get("use_case_relevant")
                s["notes"] = existing[s["asset_id"]].get("notes")
                s["sublabel"] = existing[s["asset_id"]].get("sublabel")
            else:
                s["actual_person"] = None
                s["use_case_relevant"] = None
                s["notes"] = None
                s["sublabel"] = None

        # Split into batches
        n = len(sampled)
        bs = args.batch_size
        # Write batch_00.yaml, batch_01.yaml, ...
        old_batches = sorted(out_dir.glob("batch_*.yaml"))
        for old in old_batches:
            old.unlink()
        for i in range(0, n, bs):
            batch = sampled[i:i+bs]
            for j, row in enumerate(batch):
                row["i"] = i + j
            path = out_dir / f"batch_{i // bs:02d}.yaml"
            with open(path, "w") as f:
                yaml.safe_dump(batch, f, sort_keys=False, allow_unicode=True)
            print(f"  wrote {path} ({len(batch)} rows)")
        print(f"\nDone. {len(list(out_dir.glob('batch_*.yaml')))} batches in {out_dir}/")


if __name__ == "__main__":
    main()