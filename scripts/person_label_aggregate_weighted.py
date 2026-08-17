"""Re-weighted metrics for the person-flag labeling.

The plain aggregator treats every labeled photo equally, but our
sample is stratified so the disagreement bands are over-represented.
That makes the classifier's precision look better than it really is.

This script:
  1. Counts the actual population size of each stratum (in the DB).
  2. Computes per-row weight = (population of stratum) / (sample in stratum).
  3. Recomputes precision / recall / F1 / accuracy as weighted sums.

The metrics that come out of this are an estimate of the
classifier's real performance on the deployment population,
*assuming* the within-stratum accuracy generalizes from our sample.

This is NOT a fully held-out test. For that, you'd want a random
sample from the full library. But it is more honest than the
equal-weighted aggregator because it down-weights the
over-sampled disagreement strata and up-weights the
under-sampled easy strata.
"""
from __future__ import annotations

import sqlite3
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import yaml

import storage

ROOT = Path("data/label_sample")
DB = Path("data/state.db")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_model():
    blob = joblib.load(Path("data/person_classifier.joblib"))
    return blob["coef"], float(blob["intercept"])


def load_batches() -> list[dict]:
    out = []
    for path in sorted(ROOT.glob("batch_*.yaml")):
        for r in yaml.safe_load(path.read_text()) or []:
            r["_batch"] = path.stem
            out.append(r)
    return out


def population_sizes(conn, coef, intercept) -> dict[str, int]:
    """Return {stratum_name: N_in_full_library}."""
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}

    # hp=1 ids
    hp1_ids = [r[0] for r in conn.execute(
        "SELECT asset_id FROM assets WHERE has_person=1"
    ).fetchall()]
    hp1_in_idx = [id_to_idx[a] for a in hp1_ids if a in id_to_idx]
    p1 = sigmoid(matrix[hp1_in_idx] @ coef + intercept)

    # hp=0 ids
    hp0_ids = [r[0] for r in conn.execute(
        "SELECT asset_id FROM assets WHERE has_person=0"
    ).fetchall()]
    hp0_in_idx = [id_to_idx[a] for a in hp0_ids if a in id_to_idx]
    p0 = sigmoid(matrix[hp0_in_idx] @ coef + intercept)

    n_plantae = conn.execute(
        "SELECT COUNT(DISTINCT a.asset_id) FROM assets a "
        "JOIN taxon_scores t ON t.asset_id=a.asset_id "
        "WHERE a.has_person=0 AND t.taxon_name='Plantae' AND t.score>=0.5"
    ).fetchone()[0]

    return {
        "1_yolo_only_disagree": int((p1 < 0.5).sum()),
        "2_borderline_hp1_low_model": int(((p1 >= 0.0) & (p1 < 0.3)).sum()),
        "3_borderline_hp1_high_model": int(((p1 >= 0.7) & (p1 <= 1.0)).sum()),
        "4_high_conf_hp1_untouched": int((p1 >= 0.9).sum()),  # upper bound; "untouched" filter is stricter
        "5_hp0_plantae": n_plantae,
        "6_hp0_model_agrees_no": int((p0 < 0.2).sum()),
        "7_hp0_model_would_flip_yes": int((p0 >= 0.5).sum()),
    }


def model_says_positive(p):
    if p is None: return None
    return 1 if p >= 0.5 else 0


def weighted_metrics(rows: list[dict], pop: dict[str, int],
                    sample_size: dict[str, int],
                    pred_field: str, truth_field: str) -> dict:
    """Compute weighted precision / recall / F1 / accuracy."""
    # Group rows by stratum to compute per-stratum weights
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_strat[r["stratum"]].append(r)

    tp = tn = fp = fn = 0.0
    for strat, lst in by_strat.items():
        if not lst:
            continue
        n_pop = pop.get(strat, 0)
        n_samp = sample_size.get(strat, len(lst))
        w_per_row = n_pop / n_samp if n_samp else 0.0
        for r in lst:
            pred = r[pred_field]
            truth = r[truth_field]
            if pred is None or truth is None:
                continue
            if pred == 1 and truth == 1: tp += w_per_row
            elif pred == 0 and truth == 0: tn += w_per_row
            elif pred == 1 and truth == 0: fp += w_per_row
            elif pred == 0 and truth == 1: fn += w_per_row

    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {
        "precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
        "tp": tp, "tn": tn, "fp": fp, "fn": fn,
    }


def main():
    rows = load_batches()
    labeled = [r for r in rows if r.get("actual_person") is not None]
    print(f"Total rows: {len(rows)}, labeled: {len(labeled)}\n")

    with storage.connect(DB) as conn:
        coef, intercept = load_model()
        pop = population_sizes(conn, coef, intercept)

    sample_size: dict[str, int] = defaultdict(int)
    for r in labeled:
        sample_size[r["stratum"]] += 1
    sample_size = dict(sample_size)

    print("Stratum weighting:")
    print(f"  {'stratum':<32}  {'sample':>6}  {'pop':>6}  {'weight/row':>11}")
    for s in sorted(pop):
        n_s = sample_size.get(s, 0)
        n_p = pop[s]
        w = n_p / n_s if n_s else 0.0
        print(f"  {s:<32}  {n_s:>6}  {n_p:>6}  {w:>11.1f}")
    print()

    # Attach predictions
    for r in labeled:
        r["yolo_pred"] = int(r["yolo_person"])
        r["model_pred"] = model_says_positive(r.get("model_p"))

    print("=== Population-weighted metrics ===\n")

    print("YOLO vs actual_person:")
    m = weighted_metrics(labeled, pop, sample_size, "yolo_pred", "actual_person")
    for k in ("tp","tn","fp","fn"):
        print(f"  {k}: {m[k]:.0f}")
    for k in ("precision","recall","f1","accuracy"):
        print(f"  {k}: {m[k]:.3f}")
    print()

    print("Classifier vs actual_person:")
    m = weighted_metrics(labeled, pop, sample_size, "model_pred", "actual_person")
    for k in ("tp","tn","fp","fn"):
        print(f"  {k}: {m[k]:.0f}")
    for k in ("precision","recall","f1","accuracy"):
        print(f"  {k}: {m[k]:.3f}")
    print()

    if all(r.get("use_case_relevant") is not None for r in labeled):
        print("YOLO vs use_case_relevant:")
        m = weighted_metrics(labeled, pop, sample_size, "yolo_pred", "use_case_relevant")
        for k in ("tp","tn","fp","fn"):
            print(f"  {k}: {m[k]:.0f}")
        for k in ("precision","recall","f1","accuracy"):
            print(f"  {k}: {m[k]:.3f}")
        print()

        print("Classifier vs use_case_relevant:")
        m = weighted_metrics(labeled, pop, sample_size, "model_pred", "use_case_relevant")
        for k in ("tp","tn","fp","fn"):
            print(f"  {k}: {m[k]:.0f}")
        for k in ("precision","recall","f1","accuracy"):
            print(f"  {k}: {m[k]:.3f}")

    # Confidence intervals via bootstrap on rows (still stratified within stratum)
    print("\n=== Bootstrap 95% CIs (stratified, weighted) ===\n")
    rng = np.random.default_rng(42)
    N_BOOT = 1000

    def bootstrap_metric(pred_field, truth_field):
        # Per-stratum resampling with replacement, weighted by pop
        strata = list(set(r["stratum"] for r in labeled))
        boot_metrics = []
        for _ in range(N_BOOT):
            tp = tn = fp = fn = 0.0
            for strat in strata:
                lst = [r for r in labeled if r["stratum"] == strat]
                if not lst:
                    continue
                n_samp = sample_size.get(strat, len(lst))
                w_per_row = pop.get(strat, 0) / n_samp if n_samp else 0.0
                # resample
                idxs = rng.integers(0, len(lst), size=len(lst))
                for i in idxs:
                    r = lst[int(i)]
                    pred = r[pred_field]; truth = r[truth_field]
                    if pred is None or truth is None:
                        continue
                    if pred == 1 and truth == 1: tp += w_per_row
                    elif pred == 0 and truth == 0: tn += w_per_row
                    elif pred == 1 and truth == 0: fp += w_per_row
                    else: fn += w_per_row
            prec = tp / (tp + fp) if (tp + fp) else 0.0
            rec = tp / (tp + fn) if (tp + fn) else 0.0
            f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
            boot_metrics.append((prec, rec, f1))
        boot_metrics = np.array(boot_metrics)
        lo = np.percentile(boot_metrics, 2.5, axis=0)
        hi = np.percentile(boot_metrics, 97.5, axis=0)
        return lo, hi

    for label, pf, tf in [
        ("YOLO vs actual_person", "yolo_pred", "actual_person"),
        ("Classifier vs actual_person", "model_pred", "actual_person"),
        ("YOLO vs use_case_relevant", "yolo_pred", "use_case_relevant"),
        ("Classifier vs use_case_relevant", "model_pred", "use_case_relevant"),
    ]:
        lo, hi = bootstrap_metric(pf, tf)
        print(f"  {label}:")
        print(f"    precision 95% CI: [{lo[0]:.3f}, {hi[0]:.3f}]")
        print(f"    recall    95% CI: [{lo[1]:.3f}, {hi[1]:.3f}]")
        print(f"    f1        95% CI: [{lo[2]:.3f}, {hi[2]:.3f}]")


if __name__ == "__main__":
    main()