"""Aggregate manual person-flag labels across batches.

Reads `data/label_sample/batch_*.yaml`, computes:

  - per-stratum counts of actual_person / use_case_relevant
  - confusion matrices for YOLO vs labels (actual_person, use_case_relevant)
  - confusion matrices for the classifier vs labels
  - precision / recall for both
  - top failure cases grouped by `notes` value

So we can answer the user's actual question: is the classifier better
than YOLO at the *use-case-relevant has_person* decision?

If you add a new `notes` tag in the prompt, it shows up in the
failure-case table automatically — no script edit needed.
"""
from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import yaml

ROOT = Path("data/label_sample")


def load_batches() -> list[dict]:
    out = []
    for path in sorted(ROOT.glob("batch_*.yaml")):
        rows = yaml.safe_load(path.read_text()) or []
        for r in rows:
            r["_batch"] = path.stem
            out.append(r)
    return out


def confmat(preds: list[int], labels: list[int]) -> dict[str, int]:
    p, l = preds, labels
    return {
        "tp": sum(1 for x, y in zip(p, l) if x == 1 and y == 1),
        "tn": sum(1 for x, y in zip(p, l) if x == 0 and y == 0),
        "fp": sum(1 for x, y in zip(p, l) if x == 1 and y == 0),
        "fn": sum(1 for x, y in zip(p, l) if x == 0 and y == 1),
    }


def metrics(cm: dict[str, int]) -> dict[str, float]:
    tp, tn, fp, fn = cm["tp"], cm["tn"], cm["fp"], cm["fn"]
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    acc = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) else 0.0
    return {"precision": prec, "recall": rec, "f1": f1, "accuracy": acc}


def model_says_positive(p: float | None) -> int | None:
    if p is None:
        return None
    return 1 if p >= 0.5 else 0


def main():
    rows = load_batches()
    labeled = [r for r in rows if r.get("actual_person") is not None]
    print(f"Total rows: {len(rows)}, labeled: {len(labeled)}\n")
    if not labeled:
        print("No labels yet — nothing to do.")
        return

    # Per-stratum summary
    by_strat: dict[str, list[dict]] = defaultdict(list)
    for r in labeled:
        by_strat[r["stratum"]].append(r)
    print("Per-stratum breakdown:")
    print(f"  {'stratum':<32}  {'n':>3}  {'actual_person=1':>15}  {'use_case=1':>11}")
    for strat, lst in sorted(by_strat.items()):
        ap = sum(1 for r in lst if r.get("actual_person") == 1)
        uc = sum(1 for r in lst if r.get("use_case_relevant") == 1)
        print(f"  {strat:<32}  {len(lst):>3}  {ap:>15}  {uc:>11}")

    # Confusion matrices
    print("\n=== Comparing classifiers against actual_person ===")
    yolo_preds = [int(r["yolo_person"]) for r in labeled]
    actual = [int(r["actual_person"]) for r in labeled]
    model_preds = [model_says_positive(r.get("model_p")) for r in labeled]
    model_preds_clean = [p for p in model_preds if p is not None]
    actual_clean = [a for p, a in zip(model_preds, actual) if p is not None]
    if not model_preds_clean:
        print("No model scores — skipping model matrix.")
        return

    yolo_cm = confmat(yolo_preds, actual)
    model_cm = confmat(model_preds_clean, actual_clean)

    print("\nYOLO vs actual_person:")
    for k, v in yolo_cm.items():
        print(f"  {k}: {v}")
    print("  metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics(yolo_cm).items()))

    print("\nClassifier vs actual_person:")
    for k, v in model_cm.items():
        print(f"  {k}: {v}")
    print("  metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics(model_cm).items()))

    if any(r.get("use_case_relevant") is not None for r in labeled):
        print("\n=== Comparing classifiers against use_case_relevant ===")
        uc_actual = [int(r["use_case_relevant"]) for r in labeled]
        yolo_cm_uc = confmat(yolo_preds, uc_actual)
        model_cm_uc = confmat(model_preds_clean, [a for p, a in zip(model_preds, uc_actual) if p is not None])
        print("\nYOLO vs use_case_relevant:")
        for k, v in yolo_cm_uc.items():
            print(f"  {k}: {v}")
        print("  metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics(yolo_cm_uc).items()))

        print("\nClassifier vs use_case_relevant:")
        for k, v in model_cm_uc.items():
            print(f"  {k}: {v}")
        print("  metrics: " + ", ".join(f"{k}={v:.3f}" for k, v in metrics(model_cm_uc).items()))

    # Failure cases by notes
    print("\n=== Failure cases (notes) ===")
    failures = []
    for r in labeled:
        actual_v = r.get("actual_person")
        use_case_v = r.get("use_case_relevant")
        # YOLO says yes, actual no, OR model says yes, actual no
        # OR use_case_relevant=0 even when actual_person=1
        yolo_p = int(r["yolo_person"])
        model_p = model_says_positive(r.get("model_p"))
        if actual_v == 0 and (yolo_p == 1 or model_p == 1):
            failures.append((r, "yolo_or_model_says_yes_but_no_person", yolo_p, model_p))
        if actual_v == 1 and use_case_v == 0:
            failures.append((r, "person_but_not_use_case_relevant", yolo_p, model_p))
    note_counter: Counter = Counter()
    for r, kind, _, _ in failures:
        note_counter[(kind, r.get("notes") or "(no note)", r.get("sublabel") or "(no sublabel)")] += 1
    print(f"Total failures: {len(failures)}")
    for (kind, note, sublabel), count in note_counter.most_common():
        print(f"  [{count:>3}] {kind:<35}  notes={note!r:<25}  sublabel={sublabel!r}")

    # Detailed per-stratum failure mode listing (so reviewer can see what the model is missing)
    print("\n=== Per-stratum agreement summary (actual_person) ===")
    for strat, lst in sorted(by_strat.items()):
        if not lst:
            continue
        yp = [int(r["yolo_person"]) for r in lst]
        mp = [model_says_positive(r.get("model_p")) for r in lst if r.get("model_p") is not None]
        ap = [int(r["actual_person"]) for r in lst]
        ap_m = [int(r["actual_person"]) for r in lst if r.get("model_p") is not None]
        y_correct = sum(1 for a, b in zip(yp, ap) if a == b)
        m_correct = sum(1 for a, b in zip(mp, ap_m) if a == b)
        print(f"  {strat:<32}  yolo_correct={y_correct}/{len(yp)}  model_correct={m_correct}/{len(mp)}")


if __name__ == "__main__":
    main()