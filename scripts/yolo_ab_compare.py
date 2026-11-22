#!/usr/bin/env python3
"""Side-by-side detector A/B on a sample of library photos (read-only).

Runs two gemini1-contract ONNX detectors (same 6-class / 42-channel output,
so both go through yolo_gemini1 unchanged) over the same previews, applies
the product's class-agnostic merge, and renders one row per photo:

    A boxes | B boxes      + per-photo counts of the crops BioCLIP would get

Sample = the 53 reviewed photos in data/labeling/bbox_review_v1/manifest.csv
(includes the TRAINING_DATA_NOTES §3 cases) + --random extra assets drawn
with a fixed seed from the target DB. The DB is opened read-only; nothing
is written to it. Output goes under the :8780 static server's root:

    data/labeling/bbox_review_v1/<--out>/index.html   → :8780/<--out>/

Usage:
  .venv/bin/python scripts/yolo_ab_compare.py \
      --a data/models/gemini1_seg_best.fp16.onnx \
      --b data/models/2026_11_19_obj4_control.fp16.onnx \
      --db data/tg/state.db --random 100
"""

from __future__ import annotations

import argparse
import csv
import html
import json
import os
import random
import sqlite3
import sys
from pathlib import Path

from dotenv import load_dotenv
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")
os.environ["YOLO_BACKEND"] = "gemini1"

import pipeline  # noqa: E402
import storage  # noqa: E402
import yolo_gemini1  # noqa: E402
from immich_client import ImmichClient  # noqa: E402

REVIEW = ROOT / "data/labeling/bbox_review_v1"
# Same rule as bbox_backfill_mac.py: what gets a BioCLIP crop + embedding.
CROP_LABELS = ("plant", "flower", "tree")
CROP_MIN = 0.25
RGB = {
    "plant": (40, 220, 40), "pot": (40, 120, 255), "flower": (255, 80, 200),
    "tree": (20, 130, 20), "person": (230, 60, 60), "hand": (255, 160, 40),
}


def _font(size: int):
    try:
        return ImageFont.truetype("/usr/share/fonts/TTF/DejaVuSans-Bold.ttf", size)
    except OSError:
        return ImageFont.load_default()


def _sample(db: Path, n_random: int, seed: int) -> list[tuple[str, str]]:
    picked: list[tuple[str, str]] = []
    with (REVIEW / "manifest.csv").open() as fh:
        for row in csv.DictReader(fh):
            picked.append((row["asset_id"], "reviewed: " + row["reason"]))
    seen = {a for a, _ in picked}
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    ids = [r[0] for r in conn.execute("SELECT asset_id FROM assets ORDER BY asset_id")]
    conn.close()
    pool = [a for a in ids if a not in seen]
    random.Random(seed).shuffle(pool)
    picked += [(a, "random") for a in pool[:n_random]]
    return picked


def _run(onnx: Path, images: list[Image.Image], conf: float) -> list[list[dict]]:
    yolo_gemini1.unload_model()
    os.environ["YOLO_GEMINI1_ONNX"] = str(onnx)
    out = []
    for i in range(0, len(images), 8):
        for dets in yolo_gemini1.detect(images[i:i + 8], conf=conf):
            out.append(pipeline._merge_overlapping_boxes(dets))
    yolo_gemini1.unload_model()
    return out


def _crops(dets: list[dict]) -> list[dict]:
    return [d for d in dets if d["label"] in CROP_LABELS and d["score"] >= CROP_MIN]


def _draw(img: Image.Image, dets: list[dict], title: str) -> Image.Image:
    im = img.convert("RGB").copy()
    W, H = im.size
    d = ImageDraw.Draw(im)
    f = _font(max(14, H // 40))
    for det in sorted(dets, key=lambda x: x["score"]):
        c = RGB.get(det["label"], (255, 255, 0))
        x0, y0 = det["x"] * W, det["y"] * H
        x1, y1 = x0 + det["w"] * W, y0 + det["h"] * H
        weak = det["label"] in CROP_LABELS and det["score"] < CROP_MIN
        d.rectangle([x0, y0, x1, y1], outline=c, width=1 if weak else 3)
        d.text((x0 + 3, y0 + 2), f"{det['label']} {det['score']:.2f}", fill=c, font=f,
               stroke_width=2, stroke_fill=(0, 0, 0))
    d.text((8, 8), title, fill=(255, 255, 255), font=_font(max(18, H // 28)),
           stroke_width=3, stroke_fill=(0, 0, 0))
    return im


def _stats(dets: list[dict]) -> dict:
    crops = _crops(dets)
    areas = sorted(c["w"] * c["h"] for c in crops)
    return {
        "crops": len(crops),
        "pots": sum(1 for d in dets if d["label"] == "pot" and d["score"] >= CROP_MIN),
        "max_crop_area": round(areas[-1], 3) if areas else 0.0,
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--a", required=True, type=Path, help="baseline ONNX")
    ap.add_argument("--b", required=True, type=Path, help="candidate ONNX")
    ap.add_argument("--c", type=Path, default=None, help="optional third ONNX")
    ap.add_argument("--a-name", default="A: gemini1 (8771 now)")
    ap.add_argument("--b-name", default="B: 2026-11-19-obj4-control")
    ap.add_argument("--c-name", default="C: 2026-11-20-obj6-control-5c")
    ap.add_argument("--db", default="data/tg/state.db", type=Path)
    ap.add_argument("--random", type=int, default=100)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05)
    ap.add_argument("--out", default="yolo_ab_obj4")
    args = ap.parse_args()

    page = REVIEW / args.out
    (page / "img").mkdir(parents=True, exist_ok=True)
    sample = _sample(args.db, args.random, args.seed)

    with storage.connect(args.db) as conn:
        saved = storage.get_connection_settings(conn) or {}
    client = ImmichClient(
        base_url=os.environ.get("IMMICH_BASE_URL") or str(saved.get("base_url") or ""),
        api_key=os.environ.get("IMMICH_API_KEY") or str(saved.get("api_key") or ""),
        timeout=60.0,
    )
    images, dead = pipeline.fetch_thumbs(
        client, [(a, a) for a, _ in sample], "yolo-ab",
        max_size=yolo_gemini1.DECODE_PREVIEW_MAX_SIZE,
    )
    sample = [(a, why) for a, why in sample if a in images]
    imgs = [images[a] for a, _ in sample]
    print(f"sample={len(sample)} dead={len(dead)}", flush=True)

    specs = [("a", args.a, args.a_name), ("b", args.b, args.b_name)]
    if args.c:
        specs.append(("c", args.c, args.c_name))
    results = {side: _run(onnx, imgs, args.conf) for side, onnx, _ in specs}

    rows, records = [], []
    for idx, (aid, why) in enumerate(sample):
        img = imgs[idx]
        img_stats = {side: _stats(results[side][idx]) for side, _, _ in specs}
        n = len(specs)
        strip = Image.new("RGB", (img.width * n + 8 * (n - 1), img.height), (30, 30, 30))
        for k, (side, _, title) in enumerate(specs):
            strip.paste(_draw(img, results[side][idx], title), (k * (img.width + 8), 0))
        strip.thumbnail((1600, 1600))
        strip.save(page / "img" / f"{aid}.jpg", quality=82)
        sa, sb = img_stats["a"], img_stats["b"]
        changed = sa["crops"] != sb["crops"] or abs(sa["max_crop_area"] - sb["max_crop_area"]) > 0.1
        if args.c:
            sc = img_stats["c"]
            changed = changed or sb["crops"] != sc["crops"] or abs(sb["max_crop_area"] - sc["max_crop_area"]) > 0.1
        records.append({"asset_id": aid, "why": why, **img_stats, "changed": changed})
        parts = " · ".join(
            f'{side.upper()} {st["crops"]} crops / {st["max_crop_area"]:.0%}'
            for side, st in img_stats.items()
        )
        rows.append(
            f'<div class="r{" ch" if changed else ""}"><p><b>{aid[:8]}</b> · {html.escape(why)} · '
            f'{parts} · <a href="http://192.168.84.113:2283/photos/{aid}" target="_blank">immich</a></p>'
            f'<img loading="lazy" src="img/{aid}.jpg"></div>'
        )

    def tot(side: str, key: str) -> int:
        return sum(r[side][key] for r in records)

    sides = [side for side, _, _ in specs]
    summary = {
        **{side: str(onnx) for side, onnx, _ in specs},
        "n": len(records),
        "crops": {side: tot(side, "crops") for side in sides},
        "pots": {side: tot(side, "pots") for side in sides},
        "no_crop_images": {side: sum(r[side]["crops"] == 0 for r in records) for side in sides},
        "changed": sum(r["changed"] for r in records),
    }
    (page / "results.json").write_text(json.dumps({"summary": summary, "rows": records}, indent=1))
    (page / "index.html").write_text(
        "<!doctype html><meta charset=utf-8><title>YOLO A/B</title><style>"
        "body{background:#111;color:#ddd;font:14px sans-serif;margin:12px}"
        "img{max-width:100%}.r{margin:0 0 24px}a{color:#8cf}"
        "#only:checked~.r:not(.ch){display:none}</style>"
        f"<h2>{' vs '.join(html.escape(t) for _, _, t in specs)}</h2>"
        f"<p>n={summary['n']} · BioCLIP crops (plant/flower/tree ≥{CROP_MIN}) "
        + " · ".join(
            f"{side.upper()} {summary['crops'][side]} (no crop "
            f"{summary['no_crop_images'][side]}, pot {summary['pots'][side]})"
            for side in sides) + f" · changed {summary['changed']}. "
        "Thin boxes = below the crop threshold.</p>"
        '<input type=checkbox id=only><label for=only> only changed</label>'
        + "".join(rows)
    )
    print(json.dumps(summary), flush=True)
    print(f"page: http://192.168.84.113:8780/{args.out}/", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
