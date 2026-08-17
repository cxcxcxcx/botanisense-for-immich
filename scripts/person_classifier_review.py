"""Stratified labeled review for the person classifier.

Builds a small sample of photos from across the model's prediction
spectrum, plus untouched holdout rows (high-conf hp=1 not in
training), and prints each one for manual labeling.

Sampling strategy:
  - 12 photos per model-confidence bucket x 6 buckets = 72 photos
    Buckets: [0,0.1), [0.1,0.3), [0.3,0.5), [0.5,0.7), [0.7,0.9), [0.9,1.0]
  - 12 high-conf hp=1 NOT used in training (max person score >= 0.95)
    so we can measure how the model does on the existing positives.
  - 12 hp=0 photos (no overlap with training) for the no-person test.

Total: 108 photos. Reviews via stdin — print 'p' for real person,
'n' for no person, 's' to skip.

After labeling, computes:
  - precision/recall of model vs labels
  - precision/recall of current YOLO has_person vs labels
  - confusion matrix for both

So we can directly answer: is the model better than YOLO at the
has_person decision?
"""
from __future__ import annotations

import sqlite3
from collections import Counter
from pathlib import Path

import joblib
import numpy as np
import yaml

import storage


CONTAINMENT = 0.85
PERSON_MIN = 0.50
FACE_MIN = 0.50


def load_model() -> tuple[np.ndarray, float]:
    path = Path("data/person_classifier.joblib")
    blob = joblib.load(path)
    return blob["coef"], blob["intercept"]


def sigmoid(x: float) -> float:
    return 1.0 / (1.0 + np.exp(-x))


def model_predict(coef: np.ndarray, intercept: float, X: np.ndarray) -> np.ndarray:
    """Return p(person) for each row."""
    return sigmoid(X @ coef + intercept)


def get_borderline_with_model(conn, coef, intercept) -> list[dict]:
    """Get all hp=1 rows in [0.50, 0.80), with model score + YOLO max."""
    sql = """
        SELECT a.asset_id, MAX(d.score) AS max_score
        FROM assets a JOIN detections d ON d.asset_id=a.asset_id
        WHERE a.has_person=1 AND d.label='person'
        GROUP BY a.asset_id
        HAVING MAX(d.score) >= 0.50 AND MAX(d.score) < 0.80
    """
    rows = conn.execute(sql).fetchall()
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}
    out = []
    for r in rows:
        aid = r[0]
        if aid not in id_to_idx:
            continue  # no embedding, model can't score
        X = matrix[id_to_idx[aid]]
        p = float(model_predict(coef, intercept, X.reshape(1, -1))[0])
        cur = conn.execute("SELECT original_file_name FROM assets WHERE asset_id=?", (aid,))
        fn = cur.fetchone()[0]
        out.append({
            "asset_id": aid,
            "filename": fn,
            "yolo_max": r[1],
            "model_p": p,
            "in_training": False,
        })
    return out


def get_high_conf_hp1(conn, coef, intercept) -> list[dict]:
    """hp=1 with max person score >= 0.95 NOT used in training."""
    sql = """
        SELECT a.asset_id, MAX(d.score) AS max_score
        FROM assets a JOIN detections d ON d.asset_id=a.asset_id
        WHERE a.has_person=1 AND d.label='person'
        GROUP BY a.asset_id
        HAVING MAX(d.score) >= 0.95
    """
    rows = conn.execute(sql).fetchall()
    candidates = [(r[0], r[1]) for r in rows]
    # Filter out training-set positives (face >= 0.50 OR no overlapping hand).
    aids = [c[0] for c in candidates]
    if not aids:
        return []
    qmarks = ",".join("?" * len(aids))
    det_rows = conn.execute(
        f"SELECT asset_id, label, score, x, y, w, h FROM detections "
        f"WHERE asset_id IN ({qmarks})",
        aids,
    ).fetchall()
    dets: dict[str, list] = {}
    for r in det_rows:
        dets.setdefault(r[0], []).append(r)
    out = []
    for aid, max_score in candidates:
        ds = dets.get(aid, [])
        has_face = any(d[1] == "face" and d[2] >= FACE_MIN for d in ds)
        hands = [d for d in ds if d[1] == "hand" and d[2] >= 0.05]
        persons = [d for d in ds if d[1] in ("person", "face", "human") and d[2] >= PERSON_MIN]
        overlapping = False
        for h in hands:
            _, _, _, hx, hy, hw, hh = h
            h_area = hw * hh
            if h_area <= 0:
                continue
            for p in persons:
                _, _, _, px, py, pw, ph = p
                ix1, iy1 = max(hx, px), max(hy, py)
                ix2, iy2 = min(hx + hw, px + pw), min(hy + hh, py + ph)
                if ix2 <= ix1 or iy2 <= iy1:
                    continue
                inter = (ix2 - ix1) * (iy2 - iy1)
                if inter / h_area >= CONTAINMENT:
                    overlapping = True
                    break
            if overlapping:
                break
        in_training = has_face or not overlapping
        if in_training:
            continue
        cur = conn.execute("SELECT original_file_name FROM assets WHERE asset_id=?", (aid,))
        fn = cur.fetchone()[0]
        out.append({
            "asset_id": aid,
            "filename": fn,
            "yolo_max": max_score,
            "model_p": None,  # will fill in below
            "in_training": False,
        })
    # Score
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}
    out2 = [o for o in out if o["asset_id"] in id_to_idx]
    if not out2:
        return out2
    X = np.stack([matrix[id_to_idx[o["asset_id"]]] for o in out2])
    p = model_predict(coef, intercept, X)
    for i, o in enumerate(out2):
        o["model_p"] = float(p[i])
    return out2


def get_hp0_unscored(conn, coef, intercept) -> list[dict]:
    """hp=0 rows not used in training (we used Plantae candidates as
    training negatives, but lots of hp=0 rows aren't plant candidates).
    """
    sql = """
        SELECT a.asset_id
        FROM assets a
        WHERE a.has_person = 0
          AND NOT EXISTS (
            SELECT 1 FROM taxon_scores t
            WHERE t.asset_id=a.asset_id AND t.taxon_name='Plantae'
              AND t.score >= 0.50
          )
        LIMIT 400
    """
    rows = conn.execute(sql).fetchall()
    aids = [r[0] for r in rows]
    if not aids:
        return []
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}
    X = np.stack([matrix[id_to_idx[a]] for a in aids if a in id_to_idx])
    p = model_predict(coef, intercept, X)
    out = []
    for i, aid in enumerate([a for a in aids if a in id_to_idx]):
        cur = conn.execute("SELECT original_file_name FROM assets WHERE asset_id=?", (aid,))
        fn = cur.fetchone()[0]
        out.append({
            "asset_id": aid,
            "filename": fn,
            "yolo_max": None,
            "model_p": float(p[i]),
            "in_training": False,
        })
    return out


def main():
    import sys
    db_path = Path("data/state.db")
    with storage.connect(db_path) as conn:
        coef, intercept = load_model()

        # Sample from each bucket
        borderline = get_borderline_with_model(conn, coef, intercept)
        high_conf_hp1 = get_high_conf_hp1(conn, coef, intercept)
        hp0 = get_hp0_unscored(conn, coef, intercept)

        # Bucket borderline by model_p
        buckets = [(0.0, 0.1), (0.1, 0.3), (0.3, 0.5), (0.5, 0.7), (0.7, 0.9), (0.9, 1.01)]
        per_bucket = 6
        sample = []
        rng = np.random.default_rng(42)
        for lo, hi in buckets:
            in_bucket = [b for b in borderline if lo <= b["model_p"] < hi]
            if len(in_bucket) >= per_bucket:
                picks = rng.choice(len(in_bucket), per_bucket, replace=False)
                sample.extend([in_bucket[i] for i in picks])
            else:
                sample.extend(in_bucket)

        # Add 12 high-conf hp=1 (untouched by training)
        if len(high_conf_hp1) >= 12:
            picks = rng.choice(len(high_conf_hp1), 12, replace=False)
            sample.extend([high_conf_hp1[i] for i in picks])
        else:
            sample.extend(high_conf_hp1)

        # Add 12 hp=0 (non-plantae)
        if len(hp0) >= 12:
            picks = rng.choice(len(hp0), 12, replace=False)
            sample.extend([hp0[i] for i in picks])
        else:
            sample.extend(hp0)

        print(f"Total sample: {len(sample)}")

        # Write a manifest for review (so we don't depend on running interactively)
        manifest = []
        for i, s in enumerate(sample):
            manifest.append({
                "i": i,
                "asset_id": s["asset_id"],
                "filename": s["filename"],
                "yolo_max": s["yolo_max"],
                "model_p": round(s["model_p"], 3),
                "label": None,  # user fills in
            })
        out_path = Path("data/person_classifier_review.yaml")
        with open(out_path, "w") as f:
            yaml.safe_dump(manifest, f, sort_keys=False)
        print(f"Manifest written to {out_path}")
        print(f"Open thumbs in: data/thumbs/<asset_id>.jpg")
        print(f"Then edit {out_path} to set 'label' (1=real person, 0=no person) for each row.")
        print()
        print("Sample preview (asset_id, filename, yolo_max, model_p):")
        for m in manifest:
            print(f"  [{m['i']:3d}]  {m['asset_id']}  yolo={m['yolo_max']}  model={m['model_p']}  {m['filename']}")


if __name__ == "__main__":
    main()