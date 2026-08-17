"""YOLO-World plant detection.

Wraps `ultralytics.YOLOWorld` to find plant-flavored bounding boxes in
each thumbnail. The class set is locked at module import time — YOLO-World
bakes the open-vocab embeddings when `set_classes(...)` is called, and
re-setting them for every batch is wasteful.

Outputs are normalized (x, y, w, h) where (x, y) is the top-left corner
and w/h describe the box size. All four lie in [0, 1].
"""

from __future__ import annotations

import threading

import numpy as np
from PIL import Image

# PyTorch 2.6+ defaults `torch.load(weights_only=True)`, which rejects the
# ultralytics classes used in YOLO-World checkpoints. The cleanest workaround
# is the `TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD=1` environment variable. Note
# that `bioclip/predict.py` sets `TORCH_FORCE_WEIGHTS_ONLY_LOAD=true` at
# import time; we explicitly override both to avoid a conflict.
import os as _os
_os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"
_os.environ["TORCH_FORCE_WEIGHTS_ONLY_LOAD"] = "0"
_os.environ.setdefault("ULTRALYTICS_SAFE_LOAD", "0")

# Phrases tuned for the Immich dataset (a personal photo library of plants).
# The last three ("person", "face", "human") feed the person-detection
# pipeline stage — YOLO-World treats them as additional open-vocab
# phrases alongside the plants. Semantically overlapping on purpose so
# YOLO can pick whichever phrase matches best for a given photo.
# storage.PERSON_LABELS mirrors these for the persistence filter.
# "hand" is added so YOLO can label hands explicitly when only a hand
# is visible (reaching into frame, etc.) — but it is NOT in
# PERSON_LABELS, so it doesn't trigger has_person=1. That way
# photos with just a hand get has_person=0 instead of a false
# positive on "person".
PLANT_LABELS: list[str] = [
    "plant",
    "leaf",
    "flower",
    "tree",
    "shrub",
    "succulent",
    "cactus",
    "herb",
    "grass",
    "stem",
    "branch",
    "bloom",
    "petal",
    "seedling",
    "person",
    "face",
    "human",
    "hand",
]

_LOCK = threading.Lock()
_MODEL = None


def get_model():
    """Lazy singleton. First call downloads yolov8s-world.pt (~338 MB)."""
    global _MODEL
    if _MODEL is None:
        with _LOCK:
            if _MODEL is None:
                from ultralytics import YOLOWorld

                _MODEL = YOLOWorld("yolov8s-world.pt")
                _MODEL.set_classes(PLANT_LABELS)
    return _MODEL


def unload_model() -> None:
    """Best-effort: drop the YOLO weights from memory."""
    global _MODEL
    with _LOCK:
        if _MODEL is not None:
            try:
                del _MODEL.model
                del _MODEL.ckpt
            except Exception:
                pass
        _MODEL = None
    import gc
    gc.collect()


def detect(
    images: list[Image.Image],
    *,
    conf: float = 0.05,
) -> list[list[dict]]:
    """Run YOLO-World on each image, returning per-image lists of detections.
    Each detection is `{"label", "score", "x", "y", "w", "h"}` normalized
    to the image size."""
    if not images:
        return []
    model = get_model()
    # ultralytics takes PIL images directly via `predict(source=...)`.
    results = model.predict(images, conf=conf, verbose=False)

    out: list[list[dict]] = []
    for img, res in zip(images, results):
        h, w = res.orig_shape  # (height, width)
        boxes = res.boxes
        if boxes is None or len(boxes) == 0:
            out.append([])
            continue
        xywhn = boxes.xywhn.cpu().numpy()  # (cx, cy, w, h) normalized
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)
        dets: list[dict] = []
        for (cx, cy, bw, bh), score, cls in zip(xywhn, confs, clss):
            x = float(cx - bw / 2)
            y = float(cy - bh / 2)
            dets.append(
                {
                    "label": PLANT_LABELS[cls] if 0 <= cls < len(PLANT_LABELS) else str(cls),
                    "score": float(score),
                    "x": max(0.0, x),
                    "y": max(0.0, y),
                    "w": min(1.0, float(bw)),
                    "h": min(1.0, float(bh)),
                }
            )
        out.append(dets)
    return out