"""Retrain the person classifier with hand-labeled data.

Training data sources:
  1. Auto-derived positives: hp=1 + max(person) >= 0.90 + (face >= 0.50 OR
     no overlapping hand)
  2. Auto-derived negatives: hp=0 + Plantae >= 0.5
  3. Hand-labeled photos from data/label_sample/ — use use_case_relevant
     as the target (real use-case person vs not).

Output: data/person_classifier.joblib  (overwrites previous model)

Run with --dry-run to preview the new training-set sizes.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression

import storage


DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")
LABEL_DIR = Path("data/label_sample")

PERSON_MIN_CONF = 0.50
FACE_MIN_CONF = 0.50
HAND_MIN_CONF = 0.05
CONTAINMENT_THRESHOLD = 0.85
PLANTEAE_MIN = 0.5


def build_auto_training_set(conn):
    pos_sql = """
        SELECT a.asset_id
        FROM assets a
        WHERE a.has_person = 1
          AND EXISTS (
            SELECT 1 FROM detections d
            WHERE d.asset_id=a.asset_id AND d.label='person' AND d.score >= 0.90
          )
        GROUP BY a.asset_id
    """
    pos_candidates = [r[0] for r in conn.execute(pos_sql).fetchall()]
    pos_dets = {}
    if pos_candidates:
        qmarks = ",".join("?" * len(pos_candidates))
        rows = conn.execute(
            f"SELECT asset_id, label, score, x, y, w, h FROM detections "
            f"WHERE asset_id IN ({qmarks})", pos_candidates,
        ).fetchall()
        for r in rows:
            pos_dets.setdefault(r[0], []).append(r)
    pos_ids = []
    for aid in pos_candidates:
        ds = pos_dets.get(aid, [])
        has_face = any(d[1] == "face" and d[2] >= FACE_MIN_CONF for d in ds)
        if has_face:
            pos_ids.append(aid)
            continue
        hands = [d for d in ds if d[1] == "hand" and d[2] >= HAND_MIN_CONF]
        persons = [d for d in ds if d[1] in ("person", "face", "human") and d[2] >= PERSON_MIN_CONF]
        overlapping = False
        for h in hands:
            _, _, _, hx, hy, hw, hh = h
            h_area = hw * hh
            if h_area <= 0: continue
            for p in persons:
                _, _, _, px, py, pw, ph = p
                ix1, iy1 = max(hx, px), max(hy, py)
                ix2, iy2 = min(hx + hw, px + pw), min(hy + hh, py + ph)
                if ix2 <= ix1 or iy2 <= iy1: continue
                if (ix2 - ix1) * (iy2 - iy1) / h_area >= CONTAINMENT_THRESHOLD:
                    overlapping = True
                    break
            if overlapping: break
        if not overlapping:
            pos_ids.append(aid)

    neg_sql = """
        SELECT a.asset_id
        FROM assets a JOIN taxon_scores t ON t.asset_id=a.asset_id
        WHERE a.has_person=0 AND t.taxon_name='Plantae' AND t.score >= ?
        GROUP BY a.asset_id
    """
    neg_ids = [r[0] for r in conn.execute(neg_sql, (PLANTEAE_MIN,)).fetchall()]
    return set(pos_ids), set(neg_ids)


def build_hand_labeled_set():
    aids, y_use_case, y_actual = [], [], []
    for path in sorted(LABEL_DIR.glob("batch_*.yaml")):
        for row in yaml.safe_load(path.read_text()) or []:
            if row.get("actual_person") is None:
                continue
            aids.append(row["asset_id"])
            y_use_case.append(int(row["use_case_relevant"]))
            y_actual.append(int(row["actual_person"]))
    return aids, y_use_case, y_actual


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with storage.connect(DB) as conn:
        auto_pos, auto_neg = build_auto_training_set(conn)
        print(f"Auto-derived positives (high-conf YOLO person): {len(auto_pos)}")
        print(f"Auto-derived negatives (Plantae >= 0.5):       {len(auto_neg)}")

        hand_aids, hand_y_use_case, hand_y_actual = build_hand_labeled_set()
        print(f"Hand-labeled total:                          {len(hand_aids)}")

        all_ids, matrix = storage.all_embeddings(conn)
        id_to_idx = {a: i for i, a in enumerate(all_ids)}
        print(f"Total embedded assets: {len(all_ids)}")

        X_list, y_list = [], []
        sources = []
        for aid, y in zip(hand_aids, hand_y_use_case):
            if aid not in id_to_idx:
                continue
            X_list.append(matrix[id_to_idx[aid]])
            y_list.append(y)
            sources.append(("hand", aid))
        for aid in auto_pos:
            if aid in id_to_idx and aid not in set(hand_aids):
                X_list.append(matrix[id_to_idx[aid]])
                y_list.append(1)
                sources.append(("auto_pos", aid))
        for aid in auto_neg:
            if aid in id_to_idx and aid not in set(hand_aids):
                X_list.append(matrix[id_to_idx[aid]])
                y_list.append(0)
                sources.append(("auto_neg", aid))

        X = np.stack(X_list)
        y = np.array(y_list, dtype=int)
        print(f"\nTraining set: n={len(y)}, positives={y.sum()}, negatives={(1-y).sum()}")
        n_hand = sum(1 for s, _ in sources if s == "hand")
        n_auto_pos = sum(1 for s, _ in sources if s == "auto_pos")
        n_auto_neg = sum(1 for s, _ in sources if s == "auto_neg")
        print(f"  hand-labeled: {n_hand}")
        print(f"  auto-positive: {n_auto_pos}")
        print(f"  auto-negative: {n_auto_neg}")

        if args.dry_run:
            print("\n--dry-run: not training")
            return

        clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')
        clf.fit(X, y)

        X_hand = np.stack([matrix[id_to_idx[a]] for a in hand_aids if a in id_to_idx])
        y_hand_actual = np.array([hand_y_actual[i] for i, a in enumerate(hand_aids) if a in id_to_idx])
        y_hand_use_case = np.array([hand_y_use_case[i] for i, a in enumerate(hand_aids) if a in id_to_idx])

        p_hand = clf.predict_proba(X_hand)[:, 1]
        y_pred_05 = (p_hand >= 0.5).astype(int)

        print(f"\n=== Held-out eval on hand-labeled set (n={len(y_hand_actual)}) ===")
        for label, y_true in [("actual_person", y_hand_actual), ("use_case_relevant", y_hand_use_case)]:
            tp = ((y_pred_05 == 1) & (y_true == 1)).sum()
            tn = ((y_pred_05 == 0) & (y_true == 0)).sum()
            fp = ((y_pred_05 == 1) & (y_true == 0)).sum()
            fn = ((y_pred_05 == 0) & (y_true == 1)).sum()
            prec = tp / (tp + fp) if (tp + fp) else 0
            rec = tp / (tp + fn) if (tp + fn) else 0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
            acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0
            print(f"  vs {label}:")
            print(f"    precision={prec:.3f}  recall={rec:.3f}  f1={f1:.3f}  accuracy={acc:.3f}")
            print(f"    tp={tp}  tn={tn}  fp={fp}  fn={fn}")

        joblib.dump({
            "coef": clf.coef_.ravel(),
            "intercept": float(clf.intercept_[0]),
            "trained_at": str(np.datetime64('now')),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
            "n_hand_labeled": n_hand,
            "training_notes": "Retrained; use_case_relevant as target.",
        }, MODEL_PATH)
        print(f"\nSaved model to {MODEL_PATH}")


if __name__ == "__main__":
    main()