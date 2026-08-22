"""Show model-vs-label disagreements for human review.

For each threshold T, classifies each labeled photo as:
  - TP: classifier says yes (p >= T), label is use_case_relevant=1
  - TN: classifier says no (p < T), label is use_case_relevant=0
  - FP: classifier says yes, label is 0  ← false positive (model is wrong)
  - FN: classifier says no, label is 1  ← false negative (model missed)

Outputs three tables you can eyeball:
  1. ALL FP+FN at T (with thumb path + model_p + label)
  2. Most-confident FPs (classifier very high but label says no)
  3. Most-regrettable FNs (classifier very low but label says yes)

Usage:
  python scripts/person_label_vs_model.py --threshold 0.15
  python scripts/person_label_vs_model.py --threshold 0.50
  python scripts/person_label_vs_model.py --threshold 0.15 --target actual_person
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

ROOT = Path("data/label_sample")


def load_labeled(target: str) -> list[dict]:
    rows = []
    for p in sorted(ROOT.glob("batch_*.yaml")):
        rows.extend(yaml.safe_load(p.read_text()) or [])
    return [r for r in rows if r.get(target) is not None]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--threshold", type=float, default=0.15,
                    help="classifier probability threshold")
    ap.add_argument("--target", default="use_case_relevant",
                    choices=["actual_person", "use_case_relevant"])
    ap.add_argument("--show", type=int, default=40,
                    help="max rows to show in the table")
    args = ap.parse_args()

    rows = load_labeled(args.target)
    if not rows:
        print(f"No rows labeled with {args.target}", file=sys.stderr)
        return

    n = len(rows)
    print(f"Threshold T = {args.target} >= T → predicted positive")
    print(f"Target: {args.target} (=1 means real/use-case person)")
    print(f"Sample: {n} rows from {ROOT}/")
    print()

    # Per-row pred + outcome
    tp, tn, fp, fn = [], [], [], []
    for r in rows:
        p = r.get("model_p")
        truth = int(r[args.target])
        if p is None:
            continue
        pred = int(p >= args.threshold)
        row = {
            "asset_id": r["asset_id"],
            "thumb": r["thumb_path"],
            "filename": r.get("filename"),
            "model_p": p,
            "yolo_person": r.get("yolo_person"),
            "yolo_max": r.get("yolo_max"),
            "stratum": r["stratum"],
            "truth": truth,
            "use_case_relevant": r.get("use_case_relevant"),
            "notes": r.get("notes"),
            "sublabel": r.get("sublabel"),
        }
        if pred == 1 and truth == 1:
            tp.append(row)
        elif pred == 0 and truth == 0:
            tn.append(row)
        elif pred == 1 and truth == 0:
            fp.append(row)
        elif pred == 0 and truth == 1:
            fn.append(row)

    print(f"=== Confusion at T={args.threshold} ===")
    print(f"  TP: {len(tp):>3}")
    print(f"  TN: {len(tn):>3}")
    print(f"  FP: {len(fp):>3}    <-- model says yes, label says no  (false positives)")
    print(f"  FN: {len(fn):>3}    <-- model says no,  label says yes (false negatives)")
    prec = len(tp) / (len(tp) + len(fp)) if (len(tp) + len(fp)) else 0
    rec = len(tp) / (len(tp) + len(fn)) if (len(tp) + len(fn)) else 0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0
    print(f"  precision = {prec:.3f}  recall = {rec:.3f}  F1 = {f1:.3f}")
    print()

    def show(label: str, rows: list[dict], sort_key: str, ascending: bool):
        if not rows:
            print(f"--- {label}: 0 rows ---")
            return
        rows_sorted = sorted(rows, key=lambda r: r[sort_key], reverse=not ascending)
        print(f"--- {label} ({len(rows_sorted)} total, showing top {min(args.show, len(rows_sorted))} sorted by {sort_key}) ---")
        print(f"{'model_p':>8}  {'yolo_p':>6}  {'yolo_max':>8}  {'truth':>5}  "
              f"{'ucr':>3}  {'sublabel':<14}  {'notes':<28}  {'thumb':<54}  {'stratum'}")
        for r in rows_sorted[:args.show]:
            thumb_short = r["thumb"] if len(r["thumb"]) < 54 else "..." + r["thumb"][-51:]
            print(f"{r['model_p']:>8.3f}  {int(r.get('yolo_person') or 0):>6}  "
                  f"{(r.get('yolo_max') or 0):>8.3f}  {r['truth']:>5}  "
                  f"{int(r.get('use_case_relevant') or 0):>3}  "
                  f"{(r.get('sublabel') or '-'):<14}  "
                  f"{(r.get('notes') or '-'):<28}  "
                  f"{thumb_short:<54}  {r['stratum']}")
        print()

    # FP: highest model_p first (most confident wrong predictions)
    show("FALSE POSITIVES (model says yes, label says no — possible to drop)",
         fp, "model_p", ascending=False)
    # FN: lowest model_p first (most embarrassing misses)
    show("FALSE NEGATIVES (model says no, label says yes — possible to recover)",
         fn, "model_p", ascending=True)


if __name__ == "__main__":
    main()