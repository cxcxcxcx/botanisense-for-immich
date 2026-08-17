"""Train/holdout experiment for the person classifier.

We have 70 manually labeled photos across 7 strata. Split them in half
(stratified — each half keeps the same stratum mix), train a logistic
regression on half-1's labels, evaluate on half-2 (and vice versa).

If the held-out half stays high-F1, we know:
  - The human labels are learnable (not noise)
  - BioCLIP-2 embeddings carry the signal
  - We can confidently scale up labeling + retrain with all70 labels

Reports:
  - Per-half metrics (precision, recall, F1, accuracy) on its held-out half
  - A "blended" estimate (average across both halves)
  - Comparison to the current production model on the same 70 photos
"""
from __future__ import annotations

from pathlib import Path
from collections import defaultdict
from itertools import groupby

import joblib
import numpy as np
import yaml
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix

import storage


ROOT = Path("data/label_sample")
DB = Path("data/state.db")


def load_batches() -> list[dict]:
    out = []
    for path in sorted(ROOT.glob("batch_*.yaml")):
        for r in yaml.safe_load(path.read_text()) or []:
            r["_batch"] = path.stem
            out.append(r)
    return out


def get_features(conn, aids: list[str]) -> dict[str, np.ndarray]:
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}
    out = {}
    for a in aids:
        if a in id_to_idx:
            out[a] = matrix[id_to_idx[a]]
    return out


def evaluate(y_true, y_pred):
    return {
        "n": len(y_true),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "accuracy": accuracy_score(y_true, y_pred),
        "cm": confusion_matrix(y_true, y_pred).tolist(),
    }


def stratified_split(rows, seed: int):
    """Split rows by stratum: each stratum is shuffled, then alternate rows
    into half-A and half-B. Each half ends up with ~50% of each stratum.
    """
    rng = np.random.default_rng(seed)
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_strat[r["stratum"]].append(r)
    a, b = [], []
    for strat, lst in sorted(by_strat.items()):
        idxs = rng.permutation(len(lst))
        for i, idx in enumerate(idxs):
            if i % 2 == 0:
                a.append(lst[int(idx)])
            else:
                b.append(lst[int(idx)])
    return a, b


def train_and_eval(train_rows, test_rows, X_train_map, X_test_map, target_field):
    """Train logistic regression on train_rows[target_field], eval on test_rows."""
    pairs = [(r["asset_id"], r[target_field]) for r in train_rows
             if r.get(target_field) is not None and r["asset_id"] in X_train_map]
    if not pairs:
        return None
    aids_tr = [p[0] for p in pairs]
    y_tr = np.array([int(p[1]) for p in pairs], dtype=int)

    X_tr = np.stack([X_train_map[a] for a in aids_tr])

    if len(np.unique(y_tr)) < 2:
        # Can't train a binary classifier on one class — return degenerate
        return None

    clf = LogisticRegression(C=1.0, max_iter=2000, random_state=42, class_weight='balanced')
    clf.fit(X_tr, y_tr)

    test_pairs = [(r["asset_id"], r[target_field]) for r in test_rows
                  if r.get(target_field) is not None and r["asset_id"] in X_test_map]
    aids_te = [p[0] for p in test_pairs]
    y_te = np.array([int(p[1]) for p in test_pairs], dtype=int)
    if not aids_te:
        return None
    X_te = np.stack([X_test_map[a] for a in aids_te])
    y_pred = clf.predict(X_te)
    return evaluate(y_te.tolist(), y_pred.tolist()), clf


def main():
    rows = load_batches()
    labeled = [r for r in rows if r.get("actual_person") is not None]
    print(f"Labeled photos: {len(labeled)}\n")

    with storage.connect(DB) as conn:
        # Get features for all labeled photos
        aids = [r["asset_id"] for r in labeled]
        X_map = get_features(conn, aids)

    # Run experiment across multiple seeds for robustness
    print("=== Two-fold cross-validation (stratified by stratum) ===\n")
    for target in ("actual_person", "use_case_relevant"):
        if target == "use_case_relevant" and not all(r.get("use_case_relevant") is not None for r in labeled):
            continue
        print(f"--- Target: {target} ---")
        all_metrics = []
        for seed in (42, 7, 99, 17, 31):
            a, b = stratified_split(labeled, seed)
            # Half A trains, half B tests
            res_b, _ = train_and_eval(a, b, X_map, X_map, target)
            # Half B trains, half A tests
            res_a, _ = train_and_eval(b, a, X_map, X_map, target)
            if res_a is None or res_b is None:
                print(f"  seed={seed}: SKIPPED (degenerate class)")
                continue
            all_metrics.append(("a", res_a))
            all_metrics.append(("b", res_b))
            print(f"  seed={seed}: A->B f1={res_b['f1']:.3f}  B->A f1={res_a['f1']:.3f}")

        if not all_metrics:
            print("  no valid runs\n")
            continue

        # Aggregate: each fold's metrics are weighted by n in that fold
        total_n = sum(m["n"] for _, m in all_metrics)
        blended = {
            "precision": sum(m["precision"] * m["n"] for _, m in all_metrics) / total_n,
            "recall": sum(m["recall"] * m["n"] for _, m in all_metrics) / total_n,
            "f1": sum(m["f1"] * m["n"] for _, m in all_metrics) / total_n,
            "accuracy": sum(m["accuracy"] * m["n"] for _, m in all_metrics) / total_n,
        }
        print(f"  BLENDED across all folds:")
        for k, v in blended.items():
            print(f"    {k}: {v:.3f}")
        print()

    # Compare to production model on the same 70 photos
    print("=== Production model on the same 70 photos (for comparison) ===\n")
    prod = joblib.load(Path("data/person_classifier.joblib"))
    coef, intercept = prod["coef"], float(prod["intercept"])

    def sig(x): return 1.0 / (1.0 + np.exp(-x))

    aids = [r["asset_id"] for r in labeled if r["asset_id"] in X_map]
    X = np.stack([X_map[a] for a in aids])
    p = sig(X @ coef + intercept)
    y_pred_person = (p >= 0.5).astype(int)

    y_actual = np.array([int(r["actual_person"]) for r in labeled if r["asset_id"] in X_map])
    y_use_case = np.array([int(r["use_case_relevant"]) for r in labeled if r["asset_id"] in X_map])

    m_ap = evaluate(y_actual.tolist(), y_pred_person.tolist())
    m_uc = evaluate(y_use_case.tolist(), y_pred_person.tolist())
    print(f"  Production model vs actual_person:")
    for k in ("precision","recall","f1","accuracy"):
        print(f"    {k}: {m_ap[k]:.3f}")
    print(f"  Production model vs use_case_relevant:")
    for k in ("precision","recall","f1","accuracy"):
        print(f"    {k}: {m_uc[k]:.3f}")


if __name__ == "__main__":
    main()