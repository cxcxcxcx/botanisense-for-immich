"""gemini1_seg potted-plant detector — ONNX backend for the yolo stage.

Serves the trained YOLO26-seg model (weights gemini1_seg_best, labels
from the GEMINI1 annotation standard: plant / pot / person / hand /
flower / tree) through onnxruntime, as a drop-in replacement for the
YOLO-World open-vocab module (`yolo.py`). Same interface:

    get_model() / unload_model() / detect(images, conf=...) -> [[{…}], …]

with each detection `{"label", "score", "x", "y", "w", "h"}` normalized
to the image size (x/y = top-left), exactly what pipeline.py's tier
split (plant vs PERSON_LABELS vs HAND_LABEL) already consumes. The class
names were chosen to line up: "person" lands in PERSON_LABELS, "hand"
in HAND_LABEL, everything else (plant/pot/flower/tree) feeds the
bbox-bioclip crop stage.

Model/export contract (plant_yolo docs/MOBILE_EXPORT.md, golden
manifest): letterbox to 960×960 (gain = min(960/h, 960/w), centered,
pad to stride 32, BGR→RGB CHW, /255 — done here, not in the graph).
Outputs: `output0` (1, 42, 18900) = 4 xywh (center format, PIXELS in
the letterboxed frame) + 6 per-class sigmoid scores + 32 mask
coefficients, pre-NMS; `output1` (1, 32, 240, 240) mask prototypes.
We only need boxes — the 32 coefficients are sliced off and the
prototypes are never touched.

Decode constants (conf 0.25 / IoU 0.70) come from the training/golden
manifest — the exported model's score distribution is calibrated for
them, so `detect(conf=...)` floors the caller's value at 0.25 instead
of honoring a lower threshold.
"""

from __future__ import annotations

import logging
import os
import threading
from pathlib import Path

import numpy as np
from PIL import Image

_log = logging.getLogger("pipeline")

# Detection decode wants the model's native input size — pipeline's
# _decode_preview defaults to the 448 BioCLIP draft, which would blur
# small boxes away when upsampled to 960.
DECODE_PREVIEW_MAX_SIZE = 960

IMGSZ = 960
STRIDE = 32
NAMES = ("plant", "pot", "person", "hand", "flower", "tree")
CONF_FLOOR = 0.25
NMS_IOU = 0.70
MAX_DET = 300

DEFAULT_MODEL_PATH = "data/models/gemini1_seg_best.fp16.onnx"

_LOCK = threading.Lock()
_SESSION = None


def _model_path() -> Path:
    return Path(
        os.environ.get("YOLO_GEMINI1_ONNX", DEFAULT_MODEL_PATH)
    )


def get_model():
    """Lazy singleton for the onnxruntime session. First call loads the
    fp16 graph (fp32 IO — ORT casts the input internally)."""
    global _SESSION
    if _SESSION is None:
        with _LOCK:
            if _SESSION is None:
                import onnxruntime as ort

                path = _model_path()
                if not path.exists():
                    raise FileNotFoundError(
                        f"gemini1 ONNX not found at {path} — set "
                        f"YOLO_GEMINI1_ONNX (see docs/mobile_engine_ab.md)"
                    )
                opts = ort.SessionOptions()
                opts.intra_op_num_threads = min(8, os.cpu_count() or 8)
                opts.inter_op_num_threads = 1
                opts.graph_optimization_level = (
                    ort.GraphOptimizationLevel.ORT_ENABLE_ALL
                )
                _log.info("[yolo] loading gemini1_seg ONNX %s", path)
                _SESSION = ort.InferenceSession(
                    str(path),
                    sess_options=opts,
                    providers=["CPUExecutionProvider"],
                )
    return _SESSION


def unload_model() -> None:
    """Drop the session. Cheap for a 5.6 MB graph — kept for interface
    parity with yolo.py so the stage's finally block needs no branches."""
    global _SESSION
    with _LOCK:
        _SESSION = None
    import gc

    gc.collect()


def letterbox(img: Image.Image) -> tuple[np.ndarray, float, float, float]:
    """RGB PIL → (1,3,960,960) float32 letterboxed, plus the undo
    parameters (gain, pad_w, pad_h) mapping letterbox pixels back to
    source pixels: x_src = (x_box - pad_w) / gain.

    This is ultralytics LetterBox((960,960), auto=False) verbatim —
    cv2 INTER_LINEAR resize + 114-gray BORDER_CONSTANT padding split
    ±0.1px around center — because the export's golden tensors were
    produced by exactly that path. A PIL-resize/zero-pad letterbox
    looks reasonable but disagrees with the exported graph's scores
    enough to drop ~15% of detections near the conf floor (the golden
    gate exists to catch precisely this)."""
    import cv2

    arr = np.asarray(img.convert("RGB"))
    h0, w0 = arr.shape[:2]
    gain = min(IMGSZ / h0, IMGSZ / w0)
    new_w, new_h = int(round(w0 * gain)), int(round(h0 * gain))
    pad_w = (IMGSZ - new_w) / 2
    pad_h = (IMGSZ - new_h) / 2
    if (w0, h0) != (new_w, new_h):
        arr = cv2.resize(arr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(pad_h - 0.1)), int(round(pad_h + 0.1))
    left, right = int(round(pad_w - 0.1)), int(round(pad_w + 0.1))
    arr = cv2.copyMakeBorder(
        arr, top, bottom, left, right,
        cv2.BORDER_CONSTANT, value=(114, 114, 114),
    )
    chw = arr.transpose(2, 0, 1).astype(np.float32) / 255.0
    return np.ascontiguousarray(chw[None]), gain, pad_w, pad_h


def _nms(boxes: np.ndarray, scores: np.ndarray, iou: float) -> list[int]:
    """Greedy NMS over (x0, y0, x1, y1) letterbox-pixel boxes. Returns
    kept indices, highest score first."""
    x0, y0, x1, y1 = boxes[:, 0], boxes[:, 1], boxes[:, 2], boxes[:, 3]
    areas = np.clip(x1 - x0, 0, None) * np.clip(y1 - y0, 0, None)
    order = scores.argsort()[::-1]
    keep: list[int] = []
    while order.size:
        i = order[0]
        keep.append(int(i))
        if order.size == 1:
            break
        rest = order[1:]
        ix0 = np.maximum(x0[i], x0[rest])
        iy0 = np.maximum(y0[i], y0[rest])
        ix1 = np.minimum(x1[i], x1[rest])
        iy1 = np.minimum(y1[i], y1[rest])
        inter = np.clip(ix1 - ix0, 0, None) * np.clip(iy1 - iy0, 0, None)
        union = areas[i] + areas[rest] - inter
        iou_rest = inter / np.where(union > 0, union, 1e-9)
        order = rest[iou_rest < iou]
    return keep


def detect(images: list[Image.Image], *, conf: float = CONF_FLOOR) -> list[list[dict]]:
    """Run the detector on each image. Thresholds are floored at the
    model's calibrated CONF_FLOOR — gemini1's class scores were gated
    at 0.25 during training/export, and the pipeline's default 0.05
    would pull in a long tail of unstable boxes."""
    if not images:
        return []
    session = get_model()
    input_name = session.get_inputs()[0].name
    conf = max(float(conf), CONF_FLOOR)

    out: list[list[dict]] = []
    for img in images:
        x, gain, pad_w, pad_h = letterbox(img)
        raw_pred = session.run(None, {input_name: x})[0]  # (1, 42, 18900)
        pred = np.asarray(raw_pred, dtype=np.float32)[0]  # (42, 18900)
        xywh = pred[:4]                 # center xywh, letterbox pixels
        cls_scores = pred[4 : 4 + len(NAMES)]
        # pred[4 + len(NAMES):] = 32 mask coefficients — boxes only here

        cls_id = cls_scores.argmax(axis=0)
        best = cls_scores.max(axis=0)
        m = best >= conf
        if not m.any():
            out.append([])
            continue
        xywh, best, cls_id = xywh[:, m], best[m], cls_id[m]
        # center xywh → corner xyxy (letterbox space)
        boxes = np.stack(
            [
                xywh[0] - xywh[2] / 2,
                xywh[1] - xywh[3] / 2,
                xywh[0] + xywh[2] / 2,
                xywh[1] + xywh[3] / 2,
            ],
            axis=1,
        )
        # Class-aware NMS: one pass per class, then union the keeps
        # (cross-class overlaps are pipeline._merge_overlapping_boxes's
        # job, same as the YOLO-World path).
        keep: list[int] = []
        for c in range(len(NAMES)):
            idx_c = np.nonzero(cls_id == c)[0]
            if idx_c.size:
                # _nms indexes into the subset — map back to global rows.
                keep.extend(int(idx_c[j]) for j in _nms(
                    boxes[idx_c], best[idx_c], NMS_IOU
                ))
        keep = sorted(set(keep))[:MAX_DET]

        w_img, h_img = img.size
        dets: list[dict] = []
        for i in keep:
            bx0, by0, bx1, by1 = boxes[i]
            # letterbox pixels → source pixels → normalized 0..1
            sx0 = (bx0 - pad_w) / gain
            sy0 = (by0 - pad_h) / gain
            sx1 = (bx1 - pad_w) / gain
            sy1 = (by1 - pad_h) / gain
            x = max(0.0, sx0 / w_img)
            y = max(0.0, sy0 / h_img)
            w = min(1.0, sx1 / w_img) - x
            h = min(1.0, sy1 / h_img) - y
            if w <= 0 or h <= 0:
                continue
            dets.append(
                {
                    "label": NAMES[cls_id[i]],
                    "score": float(best[i]),
                    "x": x,
                    "y": y,
                    "w": w,
                    "h": h,
                }
            )
        out.append(dets)
    return out
