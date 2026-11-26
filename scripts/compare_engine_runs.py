#!/usr/bin/env python3
"""Compare two pipeline runs over the same Immich query (e.g. Linux torch
teacher vs macOS Core ML) using their state.db files — gate 2 of
docs/mac_coreml_plan.md.

  python scripts/compare_engine_runs.py --ref linux/state.db --test mac/state.db \
      --out out/step3/report.json

Whole-photo level (assets in both DBs): embedding cosine, kingdom triage
(argmax of the stored kingdom taxon_scores) and the Plantae >= 0.35
eligibility decision, species top-1 agreement and top-1-in-ref-top-5.
Box level: detections are matched per asset by label + IoU (both runs
detect independently), then the same numbers on bbox embeddings and
bbox species predictions.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from collections import defaultdict

import numpy as np

PLANTAE_MIN = 0.35


def _open(path: str) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _vecs(conn, sql: str) -> dict:
    return {r[0]: np.frombuffer(r[1], dtype="<f4") for r in conn.execute(sql)}


def _preds(conn, table: str, key: str) -> dict:
    out: dict = defaultdict(list)
    for r in conn.execute(
        f"SELECT {key} AS k, species FROM {table} WHERE rank='species' ORDER BY {key}, position"
    ):
        out[r["k"]].append(r["species"])
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    return float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b)))


def _stats(cos: list[float]) -> dict:
    a = np.asarray(cos)
    if not len(a):
        return {"n": 0}
    return {"n": int(len(a)), "mean": round(float(a.mean()), 6),
            "min": round(float(a.min()), 6), "p01": round(float(np.quantile(a, 0.01)), 6)}


def _pred_agreement(ref: dict, test: dict, keys) -> dict:
    keys = [k for k in keys if ref.get(k) and test.get(k)]
    top1 = sum(ref[k][0] == test[k][0] for k in keys)
    in5 = sum(test[k][0] in ref[k][:5] for k in keys)
    n = len(keys) or 1
    return {"n": len(keys), "top1_agree": round(top1 / n, 4),
            "top1_in_ref_top5": round(in5 / n, 4),
            "flips": [{"key": k, "ref": ref[k][0], "test": test[k][0]}
                      for k in keys if ref[k][0] != test[k][0]][:40]}


def _kingdoms(conn) -> dict:
    out: dict = {}
    for r in conn.execute(
        "SELECT asset_id, taxon_name, score FROM taxon_scores WHERE rank='kingdom'"
    ):
        out.setdefault(r["asset_id"], {})[r["taxon_name"]] = r["score"]
    return out


def _iou(a, b) -> float:
    ax2, ay2, bx2, by2 = a[0] + a[2], a[1] + a[3], b[0] + b[2], b[1] + b[3]
    iw = max(0.0, min(ax2, bx2) - max(a[0], b[0]))
    ih = max(0.0, min(ay2, by2) - max(a[1], b[1]))
    inter = iw * ih
    union = a[2] * a[3] + b[2] * b[3] - inter
    return inter / union if union > 0 else 0.0


def _detections(conn) -> dict:
    out: dict = defaultdict(list)
    for r in conn.execute("SELECT id, asset_id, label, x, y, w, h FROM detections"):
        out[r["asset_id"]].append((r["id"], r["label"], (r["x"], r["y"], r["w"], r["h"])))
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--iou", type=float, default=0.9)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    ref, test = _open(args.ref), _open(args.test)

    ra = {r[0] for r in ref.execute("SELECT asset_id FROM assets")}
    ta = {r[0] for r in test.execute("SELECT asset_id FROM assets")}
    common = sorted(ra & ta)
    rep: dict = {"assets": {"ref": len(ra), "test": len(ta), "common": len(common)}}

    # Whole photo.
    re_, te_ = (_vecs(c, "SELECT asset_id, vector FROM embeddings") for c in (ref, test))
    rep["photo_embedding_cos"] = _stats(
        [_cos(re_[a], te_[a]) for a in common if a in re_ and a in te_])
    rk, tk = _kingdoms(ref), _kingdoms(test)
    kk = [a for a in common if rk.get(a) and tk.get(a)]
    arg = lambda d: max(d, key=d.get)  # noqa: E731
    elig = lambda d: d.get("Plantae", 0.0) >= PLANTAE_MIN  # noqa: E731
    rep["kingdom_triage"] = {
        "n": len(kk),
        "argmax_agree": round(sum(arg(rk[a]) == arg(tk[a]) for a in kk) / max(len(kk), 1), 4),
        "plantae_eligible_agree": round(
            sum(elig(rk[a]) == elig(tk[a]) for a in kk) / max(len(kk), 1), 4),
        "max_abs_plantae_delta": round(max(
            (abs(rk[a].get("Plantae", 0) - tk[a].get("Plantae", 0)) for a in kk), default=0), 6),
        "disagree": [{"asset": a, "ref": arg(rk[a]), "test": arg(tk[a])}
                     for a in kk if arg(rk[a]) != arg(tk[a])],
    }
    rep["photo_species"] = _pred_agreement(
        _preds(ref, "predictions", "asset_id"), _preds(test, "predictions", "asset_id"), common)

    # Boxes: match per asset by label + IoU.
    rd, td = _detections(ref), _detections(test)
    pairs, n_ref_boxes = [], 0
    for a in common:
        used = set()
        n_ref_boxes += len(rd.get(a, []))
        for rid, rl, rb in rd.get(a, []):
            best, bi = None, args.iou
            for tid, tl, tb in td.get(a, []):
                if tl != rl or tid in used:
                    continue
                i = _iou(rb, tb)
                if i >= bi:
                    best, bi = tid, i
            if best is not None:
                used.add(best)
                pairs.append((rid, best))
    n_test_boxes = sum(len(td.get(a, [])) for a in common)
    rep["boxes"] = {"ref": n_ref_boxes, "test": n_test_boxes, "matched": len(pairs),
                    "iou_min": args.iou}
    rbe = _vecs(ref, "SELECT detection_id, vector FROM bbox_embeddings")
    tbe = _vecs(test, "SELECT detection_id, vector FROM bbox_embeddings")
    rep["bbox_model_version"] = {
        "ref": [r[0] for r in ref.execute("SELECT DISTINCT model_version FROM bbox_embeddings")],
        "test": [r[0] for r in test.execute("SELECT DISTINCT model_version FROM bbox_embeddings")],
    }
    rep["bbox_embedding_cos"] = _stats(
        [_cos(rbe[r], tbe[t]) for r, t in pairs if r in rbe and t in tbe])
    rp, tp = _preds(ref, "bbox_predictions", "detection_id"), _preds(test, "bbox_predictions", "detection_id")
    tp_by_ref = {r: tp[t] for r, t in pairs if t in tp}
    rep["bbox_species"] = _pred_agreement(rp, tp_by_ref, [r for r, _ in pairs])

    txt = json.dumps(rep, indent=1)
    if args.out:
        with open(args.out, "w") as f:
            f.write(txt + "\n")
    print(txt)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
