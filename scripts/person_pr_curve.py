"""Population-weighted Precision-Recall curve for person-flag classifier.

Sweeps classifier thresholds and reports precision / recall / F1
weighted by each row's stratum population. The sample is stratified
so the disagreement bands are over-represented; the per-stratum
weight = (population of stratum) / (sample in stratum) gives a
less-biased estimate of deployment performance.

Strategies:
  A) Pure classifier: hp=1 iff classifier p >= threshold
  B) Hybrid: classifier p >= threshold → hp=1
             else trust YOLO (per-row `yolo_person` field, which was
             set from reconstructed original YOLO has_person)
             + Plantae safety net: drop rows that look like
             YOLO-hand/jacket false positives (classifier very confident no
             AND on Plantae disagreement band)

Run with --target {actual_person, use_case_relevant}.

Output: tabular text + bootstrap 95% CIs.

Stratum population sizes are computed from the DB using
reconstructed YOLO has_person — see person_label_sample_yolo_basis.py
for the reconstruction. This way PR estimates are interpretable as
"if we set this rule on the full library, what would we see".
"""
from __future__ import annotations

import argparse
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

import joblib
import numpy as np
import yaml

import storage


ROOT = Path("data/label_sample")
DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")

PERSON_MIN_CONF = 0.50
PLANTAE_MIN = 0.50


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def load_labeled(target: str) -> list[dict]:
    rows = []
    for p in sorted(ROOT.glob("batch_*.yaml")):
        rows.extend(yaml.safe_load(p.read_text()) or [])
    labeled = [r for r in rows if r.get(target) is not None]
    for r in labeled:
        r["_truth"] = int(r[target])
        r["_p"] = float(r["model_p"]) if r.get("model_p") is not None else None
        r["_yolo"] = int(r.get("yolo_person", 0))
        # Plantae-band = old "plantae_band_*" strata OR new hp=1 with low p
        # We treat all plantae_band_* rows as needing safety net
        r["_plantae_band"] = "plantae_band" in (r.get("stratum") or "")
    return labeled


def reconstructed_yolo_hp1(conn) -> set[str]:
    rows = conn.execute("""
        SELECT a.asset_id,
               COALESCE(MAX(CASE WHEN d.label='person' THEN d.score ELSE 0 END), 0) AS mp
        FROM assets a LEFT JOIN detections d ON d.asset_id=a.asset_id
        GROUP BY a.asset_id
    """).fetchall()
    return {r[0] for r in rows if float(r[1]) >= PERSON_MIN_CONF}


def population_sizes(conn) -> dict[str, int]:
    """Return {stratum_name: N_in_full_library} matching the
    7-stratum sampling of person_label_sample_yolo_basis.py."""
    blob = joblib.load(MODEL_PATH)
    coef, intercept = blob["coef"], float(blob["intercept"])
    all_ids, matrix = storage.all_embeddings(conn)
    id_to_idx = {a: i for i, a in enumerate(all_ids)}
    p = sigmoid(matrix @ coef + intercept)

    yolo_hp1 = reconstructed_yolo_hp1(conn)

    plant_ids = {r[0] for r in conn.execute(
        "SELECT a.asset_id FROM assets a JOIN taxon_scores t ON t.asset_id=a.asset_id "
        "WHERE t.taxon_name='Plantae' AND t.score >= ?",
        (PLANTAE_MIN,)).fetchall()}

    # All asset ids we know about
    s1 = set()
    s2 = set()
    s3 = set()
    s4_candidates = set()
    s5 = set()
    s6 = set()
    s7 = set()
    for aid in all_ids:
        pa = float(p[id_to_idx[aid]])
        in_yolo_hp1 = aid in yolo_hp1
        if in_yolo_hp1:
            if pa < 0.30:
                s1.add(aid)
                s2.add(aid)
            elif 0.7 <= pa <= 1.0:
                s3.add(aid)
                if pa >= 0.9:
                    s4_candidates.add(aid)
        else:
            if aid in plant_ids:
                s5.add(aid)
            if pa < 0.2:
                s6.add(aid)
            elif pa >= 0.5:
                s7.add(aid)

    return {
        "1_yolo_only_disagree": len(s1),
        "2_borderline_hp1_low_model": len(s2),
        "3_borderline_hp1_high_model": len(s3),
        "4_high_conf_hp1_untouched": len(s4_candidates),
        "5_hp0_plantae": len(s5),
        "6_hp0_model_agrees_no": len(s6),
        "7_hp0_model_would_flip_yes": len(s7),
        # Plantae-band is a sub-band; estimate as ~40% of the 681
        # hp=1+Plantae borderline rows from PERSON_CLASSIFIER.md.
        # The original "681 YOLO-only" matches hp=1 AND classifier <0.5
        # AND YOLO says person.
        "plantae_band_*": 681,
    }


def weighted_metrics(rows: list[dict], pop: dict[str, int],
                     sample_size: dict[str, int],
                     pred_field: str, truth_field: str) -> dict:
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_strat[r["stratum"]].append(r)

    tp = tn = fp = fn = 0.0
    for strat, lst in by_strat.items():
        n_pop = pop.get(strat, 0)
        n_samp = sample_size.get(strat, len(lst))
        w = n_pop / n_samp if n_samp else 0.0
        for r in lst:
            pred = r[pred_field]
            truth = r[truth_field]
            if pred is None or truth is None:
                continue
            if pred == 1 and truth == 1: tp += w
            elif pred == 0 and truth == 0: tn += w
            elif pred == 1 and truth == 0: fp += w
            elif pred == 0 and truth == 1: fn += w
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc,
            "tp": tp, "tn": tn, "fp": fp, "fn": fn}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default="use_case_relevant",
                    choices=["actual_person", "use_case_relevant"])
    ap.add_argument("--hybrid-plantae-p-min", type=float, default=0.15,
                    help="hybrid safety: drop if classifier p < this on Plantae band")
    ap.add_argument("--yolo-min", type=float, default=0.20,
                    help="YOLO-evidence rule: classifier positive only counts if yolo_person_max >= this")
    ap.add_argument("--n-boot", type=int, default=500)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    rows = load_labeled(args.target)
    if not rows:
        print(f"No rows labeled with {args.target}", file=sys.stderr)
        return
    p_vals = [r["_p"] for r in rows if r["_p"] is not None]
    if not p_vals:
        print("No classifier probabilities in labels", file=sys.stderr)
        return

    with storage.connect(DB) as conn:
        pop = population_sizes(conn)
    sample_size: dict[str, int] = defaultdict(int)
    for r in rows:
        sample_size[r["stratum"]] += 1
    sample_size = dict(sample_size)

    print(f"=== Stratified population vs sample sizes ===")
    print(f"  {'stratum':<32}  {'sample':>6}  {'pop':>8}  {'weight/row':>11}")
    for s in sorted(pop):
        n_s = sample_size.get(s, 0)
        n_p = pop[s]
        w = n_p / n_s if n_s else 0.0
        print(f"  {s:<32}  {n_s:>6}  {n_p:>8}  {w:>11.1f}")
    print()

    def pure_pred(t: float, rows=rows):
        return [1 if (p is not None and p >= t) else 0 for p in [r["_p"] for r in rows]]

    def hybrid_pred(t: float, rows=rows):
        out = []
        for r in rows:
            if r["_p"] is None:
                out.append(r["_yolo"])
                continue
            if r["_p"] >= t:
                out.append(1)
            elif r["_plantae_band"] and r["_p"] < args.hybrid_plantae_p_min:
                out.append(0)
            else:
                out.append(r["_yolo"])
        return out

    def yolo_evidence_pred(t: float, yolo_min: float, rows=rows):
        """Pure classifier BUT requires both:
          - classifier p >= t
          - YOLO person_max >= yolo_min
        YOLO acts as evidence gate: if YOLO truly saw nothing (low max),
        don't trust the model's positive. yolo_min=0.20 preserves
        wedding/family portraits (yolo ~0.27) while killing pure_plant
        misfires (yolo ~0.0)."""
        out = []
        for r in rows:
            p = r["_p"]
            ym = r.get("yolo_max") or 0.0
            if p is None:
                out.append(r["_yolo"])
                continue
            out.append(1 if (p >= t and ym >= yolo_min) else 0)
        return out

    def attach(rows, preds):
        rs = [dict(r) for r in rows]
        for r, p in zip(rs, preds):
            r["_pred"] = p
        return rs

    thresholds = [round(t, 2) for t in
                  [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.35, 0.40, 0.45,
                   0.50, 0.55, 0.60, 0.70, 0.80, 0.90]]

    print(f"=== Pure classifier PR curve (target={args.target}) ===")
    print(f"  n={len(rows)}  sample p range: [{min(p_vals):.3f}, {max(p_vals):.3f}]")
    print(f"  {'thresh':>6}  {'prec':>5}  {'rec':>5}  {'f1':>5}  {'tp':>8}  {'fp':>8}  {'fn':>8}")
    for t in thresholds:
        m = weighted_metrics(attach(rows, pure_pred(t)), pop, sample_size, "_pred", "_truth")
        print(f"  {t:>6.2f}  {m['precision']:>5.3f}  {m['recall']:>5.3f}  {m['f1']:>5.3f}  "
              f"{m['tp']:>8.1f}  {m['fp']:>8.1f}  {m['fn']:>8.1f}")

    print(f"\n=== Hybrid PR curve (target={args.target}) ===")
    print(f"  pure threshold T, fall back to YOLO if p<T")
    print(f"  safety: Plantae-band rows with p<{args.hybrid_plantae_p_min} → 0")
    print(f"  {'thresh':>6}  {'prec':>5}  {'rec':>5}  {'f1':>5}  {'tp':>8}  {'fp':>8}  {'fn':>8}")
    for t in thresholds:
        m = weighted_metrics(attach(rows, hybrid_pred(t)), pop, sample_size, "_pred", "_truth")
        print(f"  {t:>6.2f}  {m['precision']:>5.3f}  {m['recall']:>5.3f}  {m['f1']:>5.3f}  "
              f"{m['tp']:>8.1f}  {m['fp']:>8.1f}  {m['fn']:>8.1f}")

    print(f"\n=== YOLO-evidence rule (target={args.target}) ===")
    print(f"  hp=1 iff (classifier p >= T AND yolo_person_max >= {args.yolo_min})")
    print(f"  YOLO acts as a 'did it see anything' gate. Low yolo_min ({args.yolo_min})")
    print(f"  preserves wedding/family TPs where YOLO saw ~0.25 but missed shape.")
    print(f"  {'thresh':>6}  {'prec':>5}  {'rec':>5}  {'f1':>5}  {'tp':>8}  {'fp':>8}  {'fn':>8}")
    for t in thresholds:
        m = weighted_metrics(attach(rows, yolo_evidence_pred(t, args.yolo_min)),
                             pop, sample_size, "_pred", "_truth")
        print(f"  {t:>6.2f}  {m['precision']:>5.3f}  {m['recall']:>5.3f}  {m['f1']:>5.3f}  "
              f"{m['tp']:>8.1f}  {m['fp']:>8.1f}  {m['fn']:>8.1f}")

    # Bootstrap CIs for both strategies at a few thresholds
    print(f"\n=== Bootstrap 95% CIs (per-stratum resampling) ===")
    rng = np.random.default_rng(args.seed)
    strata = list(set(r["stratum"] for r in rows))

    def boot_curve(pred_fn, thresholds):
        out = {th: [] for th in thresholds}
        for _ in range(args.n_boot):
            sampled = []
            for s in strata:
                lst = [r for r in rows if r["stratum"] == s]
                if not lst:
                    continue
                idxs = rng.integers(0, len(lst), size=len(lst))
                sampled.extend(lst[int(i)] for i in idxs)
            for th in thresholds:
                preds = pred_fn(th, sampled)
                rs = attach(sampled, preds)
                m = weighted_metrics(rs, pop, sample_size, "_pred", "_truth")
                out[th].append((m["precision"], m["recall"], m["f1"]))
        return out

    boot_pure = boot_curve(pure_pred, thresholds)
    print("  Pure classifier:")
    for t in thresholds:
        arr = np.array(boot_pure[t])
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        print(f"    thresh={t:.2f}: prec=[{lo[0]:.3f},{hi[0]:.3f}] "
              f"rec=[{lo[1]:.3f},{hi[1]:.3f}] f1=[{lo[2]:.3f},{hi[2]:.3f}]")

    boot_hyb = boot_curve(hybrid_pred, thresholds)
    print("  Hybrid:")
    for t in thresholds:
        arr = np.array(boot_hyb[t])
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        print(f"    thresh={t:.2f}: prec=[{lo[0]:.3f},{hi[0]:.3f}] "
              f"rec=[{lo[1]:.3f},{hi[1]:.3f}] f1=[{lo[2]:.3f},{hi[2]:.3f}]")

    boot_yolo = boot_curve(lambda t, s: yolo_evidence_pred(t, args.yolo_min, s), thresholds)
    print(f"  YOLO-evidence (yolo_min={args.yolo_min}):")
    for t in thresholds:
        arr = np.array(boot_yolo[t])
        lo = np.percentile(arr, 2.5, axis=0)
        hi = np.percentile(arr, 97.5, axis=0)
        print(f"    thresh={t:.2f}: prec=[{lo[0]:.3f},{hi[0]:.3f}] "
              f"rec=[{lo[1]:.3f},{hi[1]:.3f}] f1=[{lo[2]:.3f},{hi[2]:.3f}]")


if __name__ == "__main__":
    main()