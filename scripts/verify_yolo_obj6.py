#!/usr/bin/env python3
"""Deploy-time golden gate for the obj6 ONNX detection backend.

Checks the parts this repo ADDED on top of the already-gated export:
letterbox, numpy per-class NMS, and the letterbox→source coordinate
restore. Runs yolo_obj6.detect() end-to-end on the 8 golden source
images and matches boxes against the references stored in the export's
golden.npz (greedy same-class IoU>=0.5, the mobile_golden_gate.py
protocol):

    python scripts/verify_yolo_obj6.py \
        [--golden data/models/obj6_golden] \
        [--onnx   data/models/2026_11_20_obj6_control_5c.fp16.onnx]

Exit 0 iff every image matches >=95% of ref boxes with mean conf
|delta| <= 0.02.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import json
import os

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

import yolo_obj6 as ydet  # noqa: E402


def box_iou(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """(N,4)/(M,4) xyxy IoU matrix."""
    aa = np.clip(a[:, 2] - a[:, 0], 0, None) * np.clip(a[:, 3] - a[:, 1], 0, None)
    ab = np.clip(b[:, 2] - b[:, 0], 0, None) * np.clip(b[:, 3] - b[:, 1], 0, None)
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(ref_b, ref_c, new_b, new_c):
    """Greedy same-class IoU>=0.5 matching — mobile_golden_gate protocol."""
    pairs: list[tuple[int, int]] = []
    if not len(ref_b) or not len(new_b):
        return pairs
    iou = box_iou(ref_b, new_b)
    order = np.dstack(np.unravel_index(np.argsort(-iou, axis=None), iou.shape))[0]
    used: set[int] = set()
    for i, j in order:
        if iou[i, j] < 0.5:
            break
        if int(j) not in used and ref_c[i] == new_c[j]:
            pairs.append((int(i), int(j)))
            used.add(int(j))
    return pairs


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--golden", default="data/models/obj6_golden")
    ap.add_argument("--onnx", default="data/models/2026_11_20_obj6_control_5c.fp16.onnx")
    ap.add_argument(
        "--source-dir", default=None,
        help="original images the golden was built from (preferred over "
             "golden/img/*.jpg, which are lossy re-encodes); defaults to "
             "<golden>/src when that directory exists")
    args = ap.parse_args()

    golden_dir = Path(args.golden)
    if not golden_dir.is_absolute():
        golden_dir = REPO / golden_dir
    onnx = Path(args.onnx)
    if not onnx.is_absolute():
        onnx = REPO / onnx

    # detect() resolves its graph from YOLO_OBJ6_ONNX; without this the gate
    # would silently run the default model against the wrong export's golden
    # (2026-11-20: obj6 golden vs the retired 6-class graph → 'pot' label
    # not in names).
    os.environ["YOLO_OBJ6_ONNX"] = str(onnx)

    if args.source_dir:
        source_dir = Path(args.source_dir)
    else:
        default_src = golden_dir / "src"
        source_dir = default_src if default_src.is_dir() else None

    g = np.load(golden_dir / "golden.npz", allow_pickle=True)
    # The golden dir carries the class table it was exported with
    # (5 for the 2026-11-20 obj6 export).
    gmanifest = json.loads((golden_dir / "manifest.json").read_text())
    names = list(gmanifest.get("names") or ydet.NAMES)
    ydet.CLASS_TABLES[len(names)] = tuple(names)


    ratios, conf_ds, n_ref, n_new = [], [], 0, 0
    for i, stem in enumerate(g["stems"]):
        # golden/img/*.jpg are cv2.imwrite RE-ENCODED copies of the source
        # images, but the reference boxes were computed from the originals
        # (export_golden.py: cv2.imread -> letterbox -> predict -> imwrite).
        # Decoding the copy moves pixels by up to ~30 counts and drops boxes
        # near the conf floor (2026-11-20: obj6 min ratio .909 on the copy).
        # Prefer --source-dir originals; fall back to the copy.
        img_path = (source_dir / f"{stem}.jpg") if source_dir else None
        if not (img_path and img_path.exists()):
            img_path = golden_dir / "img" / f"{stem}.jpg"
        # detect() returns normalized source-space xywh; the golden ref
        # boxes are letterbox-space xyxy — undo via the stored gain/pad.
        from PIL import Image

        with Image.open(img_path) as im:
            dets = ydet.detect([im.convert("RGB")], conf=0.25)
        gain, padw, padh = float(g["gain"][i]), float(g["padw"][i]), float(g["padh"][i])
        h0, w0 = (int(v) for v in g["orig_hw"][i])
        ref_b = np.asarray(g["boxes"][i], dtype=np.float64)
        ref_c = np.asarray(g["cls"][i], dtype=int)
        # ref letterbox xyxy → source-normalized xyxy
        ref_src = np.stack(
            [
                (ref_b[:, 0] - padw) / gain / w0,
                (ref_b[:, 1] - padh) / gain / h0,
                (ref_b[:, 2] - padw) / gain / w0,
                (ref_b[:, 3] - padh) / gain / h0,
            ],
            axis=1,
        ) if len(ref_b) else np.zeros((0, 4))
        new_b = np.array(
            [[d["x"], d["y"], d["x"] + d["w"], d["y"] + d["h"]] for d in dets[0]],
            dtype=np.float64,
        )
        new_c = np.array([names.index(d["label"]) for d in dets[0]], dtype=int)
        pairs = match(ref_src, ref_c, new_b, new_c)
        conf_ds += [
            abs(float(g["conf"][i][ri]) - dets[0][ni]["score"]) for ri, ni in pairs
        ]
        ratios.append(len(pairs) / max(1, len(ref_b)))
        n_ref += len(ref_b)
        n_new += len(new_b)
        print(
            f"{stem}: ref={len(ref_b):3d} new={len(new_b):3d} "
            f"match={len(pairs):3d} ({ratios[-1]:.2f})"
        )

    ok = min(ratios) >= 0.95 and (np.mean(conf_ds) if conf_ds else 0.0) <= 0.02
    print(
        f"\nmatch ratio {np.mean(ratios):.3f} (min {min(ratios):.3f}), "
        f"conf |d| {np.mean(conf_ds) if conf_ds else 0:.4f}, "
        f"ref/new dets {n_ref}/{n_new}"
    )
    print("GATE:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
