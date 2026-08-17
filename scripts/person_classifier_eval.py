"""Person classifier over BioCLIP-2 embeddings.

Bootstrap a training set from the confident ends of the current
has_person data, train a logistic regression on the 768-d BioCLIP-2
embeddings, and report per-quartile metrics on the borderline band.

This script does NOT touch the DB. It produces a classification
report and a small per-asset prediction table for the borderline
band so we can audit what the model says.

Training set (auto-derived, no manual labels):

  Positives:
    has_person=1 AND
    max(person score) >= 0.90 AND
    (face score >= 0.50 OR no hand box is contained in any person box)

    Both clauses argue the person is real — face is a hard label,
    and no-overlapping-hand rules out the hand-silhouette false
    positive. YOLO high-confidence alone is not enough because
    some hand silhouettes score >= 0.90.

  Negatives:
    has_person=0 AND
    Plantae score >= 0.50 (it's a plantae candidate — almost never
    has people in it, very high precision).

  Borderline (held out / audited):
    has_person=1 AND 0.50 <= max(person score) < 0.80.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

import storage


CONTAINMENT = 0.85
PERSON_MIN = 0.50
FACE_MIN = 0.50
PLANTEAE_MIN = 0.50


def build_training_set(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray, np.ndarray]:
    """Return (asset_ids, X, y) for the labeled training set.

    X is the L2-normalized BioCLIP-2 embedding for each asset.
    y is 1 for positive, 0 for negative.
    """
    # Candidate positives: hp=1 rows with at least one person >= 0.90.
    # We then post-filter in Python to add the face / no-hand-overlap tests
    # (containment is computed per-asset from per-row bbox arrays).
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

    # Fetch all detections for these candidates in one go (small set).
    pos_dets: dict[str, list[tuple]] = {}
    if pos_candidates:
        qmarks = ",".join("?" * len(pos_candidates))
        det_rows = conn.execute(
            f"SELECT asset_id, label, score, x, y, w, h FROM detections "
            f"WHERE asset_id IN ({qmarks})",
            pos_candidates,
        ).fetchall()
        for r in det_rows:
            pos_dets.setdefault(r[0], []).append(r)

    pos_ids = []
    for aid in pos_candidates:
        dets = pos_dets.get(aid, [])
        has_face = any(d[1] == "face" and d[2] >= FACE_MIN for d in dets)
        if has_face:
            pos_ids.append(aid)
            continue
        # No face — check hand-person overlap
        hands = [d for d in dets if d[1] == "hand" and d[2] >= 0.05]
        persons = [d for d in dets if d[1] in ("person", "face", "human") and d[2] >= PERSON_MIN]
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
                containment = inter / h_area
                if containment >= CONTAINMENT:
                    overlapping = True
                    break
            if overlapping:
                break
        if not overlapping:
            pos_ids.append(aid)
    print(f"Positives after filtering: {len(pos_ids)} (candidates: {len(pos_candidates)})")

    # Negatives: plantae candidates (very high precision negatives)
    neg_sql = """
        SELECT a.asset_id
        FROM assets a
        JOIN taxon_scores t ON t.asset_id = a.asset_id
        WHERE a.has_person = 0
          AND t.taxon_name = 'Plantae'
          AND t.score >= ?
        GROUP BY a.asset_id
    """
    neg_ids = [r[0] for r in conn.execute(neg_sql, (PLANTEAE_MIN,)).fetchall()]
    print(f"Negatives: {len(neg_ids)}")

    # Load embeddings
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}

    pos_idx = [id_to_idx[a] for a in pos_ids if a in id_to_idx]
    neg_idx = [id_to_idx[a] for a in neg_ids if a in id_to_idx]
    print(f"Positives with embedding: {len(pos_idx)}")
    print(f"Negatives with embedding: {len(neg_idx)}")

    X_pos = matrix[pos_idx]
    X_neg = matrix[neg_idx]
    X = np.vstack([X_pos, X_neg])
    y = np.concatenate([np.ones(len(pos_idx)), np.zeros(len(neg_idx))]).astype(np.int32)
    asset_ids = [pos_ids[i] for i in range(len(pos_idx))] + [neg_ids[i] for i in range(len(neg_idx))]
    return asset_ids, X, y


def get_borderline(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    """Assets currently flagged hp=1 with max person score in [0.50, 0.80)."""
    sql = """
        SELECT a.asset_id, MAX(d.score) AS max_score
        FROM assets a JOIN detections d ON d.asset_id=a.asset_id
        WHERE a.has_person=1 AND d.label='person'
        GROUP BY a.asset_id
        HAVING MAX(d.score) >= 0.50 AND MAX(d.score) < 0.80
    """
    rows = conn.execute(sql).fetchall()
    borderline_ids = [r[0] for r in rows]
    print(f"Borderline hp=1 rows: {len(borderline_ids)}")

    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {aid: i for i, aid in enumerate(all_ids)}
    idx = [id_to_idx[a] for a in borderline_ids if a in id_to_idx]
    return [borderline_ids[i] for i in range(len(idx))], matrix[idx]


def main():
    db_path = Path("data/state.db")
    with storage.connect(db_path) as conn:

        print("=== Building training set ===")
        asset_ids, X, y = build_training_set(conn)
        print(f"Training set: {X.shape}, positives={y.sum()}, negatives={(1-y).sum()}")

        # Hold out 20% for evaluation
        X_tr, X_te, y_tr, y_te, ids_tr, ids_te = train_test_split(
            X, y, asset_ids, test_size=0.2, random_state=42, stratify=y
        )
        print(f"Train: {X_tr.shape[0]}, Test: {X_te.shape[0]}")

        print("\n=== Training logistic regression ===")
        clf = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                 class_weight='balanced')
        clf.fit(X_tr, y_tr)
        y_pred = clf.predict(X_te)
        y_proba = clf.predict_proba(X_te)[:, 1]
        print(f"Test accuracy: {accuracy_score(y_te, y_pred):.4f}")
        print("Confusion matrix:")
        print(confusion_matrix(y_te, y_pred))
        print(classification_report(y_te, y_pred, target_names=["no_person", "person"]))

        # Refit on full training data for production predictions
    # Use class_weight='balanced' to compensate for ~23:1 imbalance.
        print("\n=== Refit on full training set ===")
        clf_full = LogisticRegression(C=1.0, max_iter=1000, random_state=42,
                                       class_weight='balanced')
        clf_full.fit(X, y)

        # Persist model coefficients for fast inference later (no
        # re-training needed). Just (coef, intercept) since it's a
        # linear model — the dot product + sigmoid is all we need.
        import joblib
        model_path = Path("data/person_classifier.joblib")
        joblib.dump({
            "coef": clf_full.coef_.ravel(),
            "intercept": float(clf_full.intercept_[0]),
            "trained_at": str(np.datetime64('now')),
            "n_pos": int(y.sum()),
            "n_neg": int((1 - y).sum()),
        }, model_path)
        print(f"Saved model to {model_path}")

        print("\n=== Predict on borderline band [0.50, 0.80) ===")
        bl_ids, bl_X = get_borderline(conn)
        if len(bl_ids) > 0:
            bl_proba = clf_full.predict_proba(bl_X)[:, 1]
            bl_pred = (bl_proba >= 0.5).astype(int)

            # Bucket predictions
            buckets = [(0.0, 0.2), (0.2, 0.4), (0.4, 0.6), (0.6, 0.8), (0.8, 1.01)]
            print("Prediction confidence distribution on borderline band:")
            for lo, hi in buckets:
                count = ((bl_proba >= lo) & (bl_proba < hi)).sum()
                print(f"  [{lo:.2f}, {hi:.2f}): {count}")

            # How many would the model flip to hp=0?
            n_flip = (bl_pred == 0).sum()
            n_keep = (bl_pred == 1).sum()
            print(f"\nModel says: flip to hp=0: {n_flip}, keep hp=1: {n_keep}")

            # Show 10 most confident "should be hp=0" predictions
            flip_idx = np.argsort(bl_proba)[:10]
            print("\n10 MOST CONFIDENT 'flip to hp=0' predictions:")
            for i in flip_idx:
                aid = bl_ids[i]
                cur = conn.execute("SELECT original_file_name FROM assets WHERE asset_id=?", (aid,))
                fn = cur.fetchone()[0]
                print(f"  {aid}  p(no_person)={1-bl_proba[i]:.3f}  {fn}")

            # Show 10 most confident "keep hp=1" predictions
            keep_idx = np.argsort(bl_proba)[::-1][:10]
            print("\n10 MOST CONFIDENT 'keep hp=1' predictions:")
            for i in keep_idx:
                aid = bl_ids[i]
                cur = conn.execute("SELECT original_file_name FROM assets WHERE asset_id=?", (aid,))
                fn = cur.fetchone()[0]
                print(f"  {aid}  p(person)={bl_proba[i]:.3f}  {fn}")


if __name__ == "__main__":
    main()