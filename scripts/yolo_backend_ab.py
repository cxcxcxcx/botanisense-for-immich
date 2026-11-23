#!/usr/bin/env python3
"""A/B two trained-detector ONNX exports on the same library photos.

Read-only: previews come from Immich, nothing is written to any DB. It
runs both backends through the exact product path (fetch_thumbs decode
cap → detect → pipeline._merge_overlapping_boxes → tier split) and
reports, per asset and in aggregate:

  * detection counts per label after the merge, and
  * crop-set agreement: the boxes that would actually get a BioCLIP
    crop (score >= BBOX_BIOCLIP_MIN, label in plant/flower/tree),
    matched greedily by IoU >= 0.5 across backends.

Usage:
  .venv/bin/python scripts/yolo_backend_ab.py \
      --a obj6 --b rtmdet --db data/state.db \
      --assets b19a9b47-...,aecaae90-... --random 40
"""

from __future__ import annotations

import argparse
import csv
import os
import random
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

import pipeline  # noqa: E402
import storage  # noqa: E402
from immich_client import ImmichClient  # noqa: E402

BACKENDS = {"obj6": "yolo_obj6", "rtmdet": "yolo_rtmdet"}
CROP_LABELS = ("plant", "flower", "tree")


def _backend(name: str):
    import importlib

    if name not in BACKENDS:
        raise SystemExit(f"unknown backend {name!r}; known {sorted(BACKENDS)}")
    return importlib.import_module(BACKENDS[name])


def _connection(db: Path) -> tuple[str, str]:
    with storage.connect(db) as conn:
        saved = storage.get_connection_settings(conn) or {}
    base = os.environ.get("IMMICH_BASE_URL") or str(saved.get("base_url") or "")
    key = os.environ.get("IMMICH_API_KEY") or str(saved.get("api_key") or "")
    if not base or not key:
        raise SystemExit("No Immich connection (env or saved settings).")
    return base, key


def _asset_ids(db: Path, *, pinned: list[str], n_random: int, seed: int) -> list[str]:
    with storage.connect(db) as conn:
        picked = []
        for aid in pinned:
            row = conn.execute(
                "SELECT asset_id FROM assets WHERE asset_id = ?", (aid,)
            ).fetchone()
            if row is None:
                # Accept the 8-char prefix used in bug reports.
                row = conn.execute(
                    "SELECT asset_id FROM assets WHERE asset_id LIKE ?", (aid + "%",)
                ).fetchone()
            picked.append(row[0] if row else aid)
        pool = [
            r[0] for r in conn.execute(
                "SELECT asset_id FROM assets "
                "WHERE yolo_scanned_at IS NOT NULL ORDER BY asset_id"
            )
        ]
    picked = [a for a in picked if a in pool]
    rng = random.Random(seed)
    extra = [a for a in rng.sample(pool, min(n_random, len(pool))) if a not in picked]
    return picked + extra


def _iou(a: dict, b: dict) -> float:
    ax0, ay0 = a["x"], a["y"]
    ax1, ay1 = ax0 + a["w"], ay0 + a["h"]
    bx0, by0 = b["x"], b["y"]
    bx1, by1 = bx0 + b["w"], by0 + b["h"]
    iw = max(0.0, min(ax1, bx1) - max(ax0, bx0))
    ih = max(0.0, min(ay1, by1) - max(ay0, by0))
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _crops(dets: list[dict]) -> list[dict]:
    return [
        d for d in dets
        if d["label"] in CROP_LABELS and d["score"] >= pipeline.BBOX_BIOCLIP_MIN
    ]


def _match(ref: list[dict], other: list[dict], iou_thr: float = 0.5):
    """Greedy count of ref boxes that have an IoU>=thr partner in other."""
    used: set[int] = set()
    hit = same_label = 0
    for a in sorted(ref, key=lambda d: -d["score"]):
        best, best_i = 0.0, -1
        for i, b in enumerate(other):
            if i in used:
                continue
            v = _iou(a, b)
            if v > best:
                best, best_i = v, i
        if best >= iou_thr:
            used.add(best_i)
            hit += 1
            same_label += int(other[best_i]["label"] == a["label"])
    return hit, same_label


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", default="obj6")
    ap.add_argument("--b", default="rtmdet")
    ap.add_argument("--db", default=os.environ.get("DB_PATH", "data/state.db"))
    ap.add_argument("--assets", default="", help="comma-separated ids or prefixes")
    ap.add_argument("--random", type=int, default=40, help="extra random assets")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--csv", default="", help="write per-asset rows here")
    args = ap.parse_args()

    db_path = Path(args.db)
    pinned = [a.strip() for a in args.assets.split(",") if a.strip()]
    ids = _asset_ids(db_path, pinned=pinned, n_random=args.random, seed=args.seed)
    print(f"assets: {len(ids)} ({len(pinned)} pinned)")

    base_url, api_key = _connection(db_path)
    client = ImmichClient(base_url=base_url, api_key=api_key, timeout=60.0)
    mods = {name: _backend(name) for name in (args.a, args.b)}

    # One decode per asset at each backend's cap; both trained detectors
    # declare 960, so the A/B shares the decode when they agree.
    caps = {name: int(getattr(m, "DECODE_PREVIEW_MAX_SIZE", 448))
            for name, m in mods.items()}
    cap = max(caps.values())
    images, dead = pipeline.fetch_thumbs(
        client, [(a, a) for a in ids], "yolo-ab", max_size=cap
    )
    ids = [a for a in ids if a in images]
    print(f"fetched {len(ids)} previews at {cap}px ({len(dead)} dead)")

    rows = []
    agg = {n: {"dets": 0, "crops": 0, "sec": 0.0,
               "labels": {}} for n in mods}
    for name, mod in mods.items():
        mod.get_model()

    for aid in ids:
        pil = images[aid]
        res = {}
        for name, mod in mods.items():
            if caps[name] != cap:
                one, dead1 = pipeline.fetch_thumbs(
                    client, [(aid, aid)], "yolo-ab", max_size=caps[name]
                )
                pil_use = one.get(aid)
                if pil_use is None:
                    continue
            else:
                pil_use = pil
            t = time.monotonic()
            dets = mod.detect([pil_use], conf=args.conf)[0]
            agg[name]["sec"] += time.monotonic() - t
            dets = pipeline._merge_overlapping_boxes(dets)
            res[name] = dets
            agg[name]["dets"] += len(dets)
            agg[name]["crops"] += len(_crops(dets))
            for d in dets:
                agg[name]["labels"][d["label"]] = (
                    agg[name]["labels"].get(d["label"], 0) + 1
                )
        ca, cb = _crops(res[args.a]), _crops(res[args.b])
        hit_b, lab_b = _match(ca, cb)
        hit_a, lab_a = _match(cb, ca)
        rows.append({
            "asset": aid,
            f"{args.a}_dets": len(res[args.a]),
            f"{args.b}_dets": len(res[args.b]),
            f"{args.a}_crops": len(ca),
            f"{args.b}_crops": len(cb),
            f"{args.a}_crops_matched_in_{args.b}": f"{hit_b}/{len(ca)}",
            f"{args.b}_crops_matched_in_{args.a}": f"{hit_a}/{len(cb)}",
            f"{args.a}_matched_label_agree": f"{lab_b}/{hit_b}" if hit_b else "-",
            f"{args.b}_matched_label_agree": f"{lab_a}/{hit_a}" if hit_a else "-",
        })

    n = max(len(ids), 1)
    for name in mods:
        a = agg[name]
        labels = ", ".join(f"{k}:{v}" for k, v in sorted(a["labels"].items()))
        print(f"\n[{name}] dets={a['dets']} crops={a['crops']} "
              f"{a['sec']/n*1000:.0f} ms/img  ({labels})")

    if rows:
        keys = list(rows[0])
        for r in rows[:10]:
            print("  " + " ".join(f"{k}={r[k]}" for k in keys[1:]))
        if args.csv:
            with open(args.csv, "w", newline="") as fh:
                w = csv.DictWriter(fh, fieldnames=keys)
                w.writeheader()
                w.writerows(rows)
            print(f"\nwrote {args.csv} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
