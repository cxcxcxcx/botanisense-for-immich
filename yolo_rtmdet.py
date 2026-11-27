"""Trained potted-plant detector — RTMDet-Ins-tiny ONNX backend.

The default detection backend (YOLO_BACKEND unset or `rtmdet`). Same
interface as the opt-in `yolo_obj6.py`:

    get_model() / unload_model() / detect(images, conf=...) -> [[{…}], …]

Why this one is the default: the obj6 export is an Ultralytics YOLO26
weight, so it carries AGPL-3.0 into anything that ships it. This
detector is an
RTMDet-Ins-tiny (mmdet, Apache-2.0) trained on the same obj6 labels —
the same 5 classes, an export that is numerically identical to the
scored checkpoint (scripts/verify_rtmdet_onnx.py in the modeling repo
reproduces its COCO numbers exactly), at a slightly lower accuracy and
~2× the parameters/bytes (see docs/detector_backends.md).

Model contract (mmdet test pipeline, replicated here in numpy/cv2 —
NOT the ultralytics letterbox):
  * BGR uint8, Resize(960, keep_ratio, cv2 INTER_LINEAR):
    gain = min(960/h, 960/w), new_w = int(w*gain + 0.5) (round-half-up);
  * Pad right/bottom to 960×960 with 114 — no centering, so no pad offset;
  * (x − mean) / std with the mmdet BGR mean [103.53, 116.28, 123.675]
    and std [57.375, 57.12, 58.395]; no /255.

Graph outputs (static 960×960):
  boxes  (1,18900,4) float32 — xyxy in letterbox pixels, clamped [0,960]
  scores (1,18900,5) float32 — sigmoid class scores

Decode (mmdet test_cfg, class-aware):
  per-level top-k 1000 (18900 = 120²+60²+30² rows, strides 8/16/32, so
  level row ranges are fixed) → score filter → per-class NMS IoU 0.60 →
  max_per_img 100 → rescale into the source image. The score filter is
  set to the calibrated operating point (SCORE_THR below), not mmdet's
  0.05 COCO-sweep floor.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_log = logging.getLogger("pipeline")

# Same reasoning as yolo_obj6: the detector's native input is 960, so
# fetch previews at 960 instead of the 448 BioCLIP draft.
DECODE_PREVIEW_MAX_SIZE = 960

NAMES = ("plant", "person", "hand", "flower", "tree")
IMGSZ = 960
MEAN = np.array([103.53, 116.28, 123.675], dtype=np.float32)
STD = np.array([57.375, 57.12, 58.395], dtype=np.float32)
# mmdet's test_cfg floor is 0.05, but that is the COCO-sweep threshold,
# not an operating point: RTMDet's scores here are much flatter than the
# Ultralytics model's (median 0.11 vs 0.59 on the same photos), so 0.05
# yields ~100 boxes/image of mostly noise. The operating point is picked
# to match the production obj6 export's crop budget: on the 81-image
# obj6 val split (GT-matched at IoU>=0.5), 0.35 gives 6.0 boxes/image —
# the same as the obj6 export's shipped 0.25 floor — while beating it on
# every quality metric at that budget:
#
#   thr   P      R      F1     boxes/img  images with >=1 GT crop matched
#   .25  .355   .651   .460    11.4       76/77
#   .30  .460   .594   .519     8.0       74/77
#   .35  .549   .531   .540     6.0       70/77   <- shipped
#   .40  .620   .481   .542     4.8       67/77
#   obj6 @ .25 (production):    .534/.515/.524, 6.0/img, 66/77
#
# Raise it if BioCLIP crop cost matters more than recall; lower it to
# 0.25 for maximum coverage at ~2x the crops. Like yolo_obj6's
# CONF_FLOOR, the caller's `conf` is floored here.
SCORE_THR = 0.35
NMS_PRE = 1000
NMS_IOU = 0.60
MAX_DET = 100
LEVELS = ((0, 14400), (14400, 18000), (18000, 18900))  # strides 8/16/32

# Baked into the image at /app/models/ — 20.5 MB fp32. Resolved relative to the CWD like
# face.py's YuNet ONNX.
DEFAULT_MODEL_PATH = "models/obj6_rtmdet_ep97.onnx"

_LOCK = threading.Lock()
_SESSION = None


def _model_path() -> Path:
    return Path(os.environ.get("YOLO_RTMDET_ONNX", DEFAULT_MODEL_PATH))


def get_model():
    """Lazy singleton for the onnxruntime session (fp32 graph — fp16 was
    converted and benchmarked but buys no CPU speed in ORT)."""
    global _SESSION
    if _SESSION is None:
        with _LOCK:
            if _SESSION is None:
                import onnxruntime as ort

                path = _model_path()
                if not path.exists():
                    raise FileNotFoundError(
                        f"RTMDet ONNX not found at {path} — set "
                        f"YOLO_RTMDET_ONNX (see docs/detector_backends.md)"
                    )
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = min(8, os.cpu_count() or 8)
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                _log.info("[yolo] loading RTMDet ONNX %s", path)
                _SESSION = ort.InferenceSession(
                    str(path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _SESSION


def unload_model() -> None:
    global _SESSION
    with _LOCK:
        _SESSION = None
    import gc

    gc.collect()
    # The next stage (bbox BioCLIP) loads ~5.6 GB right after this call.
    # gc alone leaves the detector's tensors in glibc arenas — measured
    # 222 MB resident vs 153 after malloc_trim on a 16-image detect
    # (out/memprobe/unload_yolo.py), and up to ~2 GB in a real run.
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def letterbox(img: Image.Image) -> tuple[np.ndarray, int, int]:
    """RGB PIL → (1,3,960,960) float32 normalized BGR, plus the resized
    (new_w, new_h) that letterbox pixels divide by to become normalized
    0..1 source coordinates. Padding is right/bottom only, so no offset."""
    import cv2

    arr = np.asarray(img.convert("RGB"))[:, :, ::-1]  # RGB → BGR
    h0, w0 = arr.shape[:2]
    gain = min(IMGSZ / h0, IMGSZ / w0)
    new_w = int(w0 * gain + 0.5)
    new_h = int(h0 * gain + 0.5)
    if (new_w, new_h) != (w0, h0):
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    padded = np.full((IMGSZ, IMGSZ, 3), 114, dtype=np.uint8)
    padded[:new_h, :new_w] = arr
    chw = padded.transpose(2, 0, 1).astype(np.float32)
    chw = (chw - MEAN[:, None, None]) / STD[:, None, None]
    return np.ascontiguousarray(chw[None]), new_w, new_h


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> np.ndarray:
    """Greedy NMS over (x1, y1, x2, y2) boxes; returns kept indices,
    highest score first."""
    x0, y0, x1, y1 = boxes.T
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        inter = (
            np.clip(np.minimum(x1[i], x1[rest]) - np.maximum(x0[i], x0[rest]), 0, None)
            * np.clip(np.minimum(y1[i], y1[rest]) - np.maximum(y0[i], y0[rest]), 0, None)
        )
        union = areas[i] + areas[rest] - inter
        order = rest[inter / np.where(union > 0, union, 1e-9) <= iou]
    return np.asarray(keep)


def _decode(
    boxes: np.ndarray, scores: np.ndarray, conf: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """mmdet's test_cfg order: per-level top-k over (prior, class) rows →
    score filter → per-class NMS → max_per_img. Rows of `scores` are
    row-major (prior-major), so a flat index maps back as
    prior = idx // nc, class = idx % nc."""
    parts_b, parts_s, parts_l = [], [], []
    for lo, hi in LEVELS:
        b = boxes[0, lo:hi]
        s = scores[0, lo:hi]
        flat = s.reshape(-1)
        if flat.size > NMS_PRE:
            idx = np.argpartition(-flat, NMS_PRE)[:NMS_PRE]
        else:
            idx = np.arange(flat.size)
        idx = idx[flat[idx] > conf]
        parts_b.append(b[idx // len(NAMES)])
        parts_s.append(flat[idx])
        parts_l.append((idx % len(NAMES)).astype(np.int64))
    if not parts_b:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    b = np.concatenate(parts_b)
    s = np.concatenate(parts_s)
    lab = np.concatenate(parts_l)
    keep: list[int] = []
    for c in range(len(NAMES)):
        sel = np.nonzero(lab == c)[0]
        if sel.size:
            keep.extend(sel[_nms(b[sel], s[sel], NMS_IOU)].tolist())
    if not keep:
        return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int64)
    keep_arr = np.asarray(keep)
    order = s[keep_arr].argsort()[::-1][:MAX_DET]
    keep_arr = keep_arr[order]
    return b[keep_arr], s[keep_arr], lab[keep_arr]


def detect(images: list[Image.Image], *, conf: float = 0.05) -> list[list[dict]]:
    """Run the detector on each image. `conf` is floored at the
    calibrated operating point SCORE_THR — see its comment for why this
    model's scores cannot use the pipeline's 0.05 default."""
    if not images:
        return []
    session = get_model()
    input_name = session.get_inputs()[0].name
    conf = max(float(conf), SCORE_THR)

    out: list[list[dict]] = []
    for img in images:
        x, new_w, new_h = letterbox(img)
        res = session.run(None, {input_name: x})
        boxes, scores, labels = _decode(
            np.asarray(res[0], dtype=np.float32),
            np.asarray(res[1], dtype=np.float32),
            conf,
        )
        dets: list[dict] = []
        for (bx0, by0, bx1, by1), sc, lb in zip(boxes, scores, labels):
            # letterbox pixels → normalized source coords (the image was
            # resized to new_w×new_h before padding, so dividing by the
            # resized size lands on the original frame).
            nx0 = min(1.0, max(0.0, bx0 / new_w))
            ny0 = min(1.0, max(0.0, by0 / new_h))
            nx1 = min(1.0, max(0.0, bx1 / new_w))
            ny1 = min(1.0, max(0.0, by1 / new_h))
            if nx1 <= nx0 or ny1 <= ny0:
                continue
            dets.append(
                {
                    "label": NAMES[int(lb)],
                    "score": float(sc),
                    "x": float(nx0),
                    "y": float(ny0),
                    "w": float(nx1 - nx0),
                    "h": float(ny1 - ny0),
                }
            )
        out.append(dets)
    return out
