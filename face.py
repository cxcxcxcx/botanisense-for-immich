"""YuNet face detection — person-stage rescue.

YOLO-World's "face" prompt is dead weight in practice: it never fires on
this library's photos at any resolution or prompt configuration (verified
empirically — a face-only prompt at imgsz=1920 on a frame-filling face
crop returns zero boxes). That meant storage.has_real_person()'s face
escape hatch could never fire, so real-people photos whose hands fall
inside their person boxes were suppressed as hand silhouettes.

This module wraps OpenCV's YuNet detector (cv2.FaceDetectorYN) — a 232 KB
ONNX that finds those faces at the pipeline's own 448px decode in ~3 ms.

Model: face_detection_yunet_2023mar.onnx, from github.com/opencv/opencv_zoo
(Apache-2.0). Committed at the repo root and baked into the Docker image:
it loads by bare filename from the CWD (same convention as
the detector ONNX in models/) because /app is read-only at runtime, so a
download-on-first-use path is not viable.

Outputs are normalized (x, y, w, h) where (x, y) is the top-left corner
and w/h describe the box size, matching yolo.detect()'s shape.
"""

from __future__ import annotations

import threading

from PIL import Image

from storage import PERSON_MIN_CONF

MODEL_FILE = "face_detection_yunet_2023mar.onnx"

_LOCK = threading.Lock()
_DETECTOR = None


def get_detector():
    """Lazy singleton. Keeps `import cv2` (~100 MB RSS) out of any
    process that never runs the person rescue."""
    global _DETECTOR
    if _DETECTOR is None:
        with _LOCK:
            if _DETECTOR is None:
                import cv2

                # Input size is a placeholder — setInputSize() is called
                # per image inside detect_faces (YuNet requires the size
                # to match the frame it is fed).
                _DETECTOR = cv2.FaceDetectorYN.create(
                    MODEL_FILE,
                    "",
                    (320, 320),
                    score_threshold=PERSON_MIN_CONF,
                    nms_threshold=0.3,
                    top_k=5000,
                )
    return _DETECTOR


def detect_faces(
    img: Image.Image,
    *,
    score_threshold: float = PERSON_MIN_CONF,
) -> list[dict]:
    """Run YuNet on one PIL image, returning detections in the same
    normalized shape as yolo.detect(): `{"label", "score", "x", "y",
    "w", "h"}`. `label` is always "face" (a storage.PERSON_LABELS
    member, so persisted rows feed has_real_person()'s face hatch and
    are skipped by the bbox-bioclip stage).
    """
    import cv2
    import numpy as np

    det = get_detector()
    arr = cv2.cvtColor(np.asarray(img), cv2.COLOR_RGB2BGR)
    h, w = arr.shape[:2]
    det.setInputSize((w, h))
    _, faces = det.detect(arr)

    out: list[dict] = []
    if faces is None:
        return out
    for f in faces:
        score = float(f[-1])
        if score < score_threshold:
            continue
        out.append(
            {
                "label": "face",
                "score": score,
                "x": max(0.0, float(f[0]) / w),
                "y": max(0.0, float(f[1]) / h),
                "w": min(1.0, float(f[2]) / w),
                "h": min(1.0, float(f[3]) / h),
            }
        )
    return out
