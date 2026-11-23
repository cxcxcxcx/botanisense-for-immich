"""Unit tests for the RTMDet ONNX backend's decode/preprocess contract.

The graph itself is exercised by the modeling repo's parity gate
(scripts/verify_rtmdet_onnx.py, which reproduces the trained
checkpoint's COCO eval exactly). These tests pin the parts that are
easy to break locally: the row ranges of the three pyramid levels, the
row-major flat-index mapping (prior = idx // nc), the mmdet
right/bottom-pad preprocess, and the calibrated score floor.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import yolo_rtmdet as m  # noqa: E402


def test_levels_cover_the_whole_prior_tensor():
    assert m.LEVELS[0][0] == 0
    assert m.LEVELS[-1][1] == 18900
    assert [hi - lo for lo, hi in m.LEVELS] == [14400, 3600, 900]
    # consecutive, no overlap
    for (_, hi), (lo, _) in zip(m.LEVELS, m.LEVELS[1:]):
        assert hi == lo


def test_decode_maps_flat_rows_to_prior_and_class():
    """A high score at flat row idx must carry the box at prior
    idx // nc and the label idx % nc — not the level-relative index."""
    boxes = np.zeros((1, 18900, 4), dtype=np.float32)
    scores = np.zeros((1, 18900, 5), dtype=np.float32)
    nc = len(m.NAMES)
    # level 1 (strides 16) starts at 14400; 2nd prior, class "flower"
    prior = 14400 + 1
    cls = m.NAMES.index("flower")
    boxes[0, prior] = [10.0, 20.0, 30.0, 40.0]
    scores[0, prior, cls] = 0.9
    # level 2, first prior, class "tree"
    prior2 = 18000
    c2 = m.NAMES.index("tree")
    boxes[0, prior2] = [1.0, 2.0, 3.0, 4.0]
    scores[0, prior2, c2] = 0.8

    b, s, lab = m._decode(boxes, scores, conf=0.05)
    got = {(tuple(np.round(box, 1)), int(l)) for box, l in zip(b, lab)}
    assert (tuple(np.round(boxes[0, prior], 1)), cls) in got
    assert (tuple(np.round(boxes[0, prior2], 1)), c2) in got
    assert len(b) == 2


def test_decode_applies_score_floor():
    boxes = np.zeros((1, 18900, 4), dtype=np.float32)
    scores = np.zeros((1, 18900, 5), dtype=np.float32)
    scores[0, 0, 0] = 0.34
    scores[0, 1, 0] = 0.36
    b, s, lab = m._decode(boxes, scores, conf=0.35)
    assert len(b) == 1 and s[0] == pytest.approx(0.36)


def test_decode_empty_when_nothing_passes():
    boxes = np.zeros((1, 18900, 4), dtype=np.float32)
    scores = np.zeros((1, 18900, 5), dtype=np.float32)
    b, s, lab = m._decode(boxes, scores, conf=0.35)
    assert b.shape == (0, 4) and s.size == 0 and lab.size == 0


def test_decode_per_class_nms_keeps_one_of_two_duplicates():
    boxes = np.zeros((1, 18900, 4), dtype=np.float32)
    scores = np.zeros((1, 18900, 5), dtype=np.float32)
    boxes[0, 0] = [0.0, 0.0, 100.0, 100.0]
    boxes[0, 1] = [2.0, 2.0, 102.0, 102.0]  # IoU > 0.6 with row 0
    scores[0, 0, 0] = 0.9
    scores[0, 1, 0] = 0.8
    b, s, lab = m._decode(boxes, scores, conf=0.35)
    assert len(b) == 1 and s[0] == pytest.approx(0.9)


def test_decode_caps_at_max_det():
    boxes = np.zeros((1, 18900, 4), dtype=np.float32)
    scores = np.zeros((1, 18900, 5), dtype=np.float32)
    # disjoint boxes, all class 0, above the floor
    for i in range(200):
        x = float(i % 20) * 10.0
        y = float(i // 20) * 10.0
        boxes[0, i] = [x, y, x + 5.0, y + 5.0]
        scores[0, i, 0] = 0.9 - i * 1e-4
    b, s, lab = m._decode(boxes, scores, conf=0.05)
    assert len(b) == m.MAX_DET == 100


def _solid(w, h, color=(200, 50, 50)):
    from PIL import Image

    return Image.new("RGB", (w, h), color)


def test_letterbox_pads_right_bottom_and_normalizes_bgr():
    pytest.importorskip("cv2")
    img = _solid(480, 960, (10, 20, 30))  # 1:2 portrait, gain 1.0
    x, new_w, new_h = m.letterbox(img)
    assert x.shape == (1, 3, m.IMGSZ, m.IMGSZ)
    assert (new_w, new_h) == (480, 960)
    # right half is the 114 pad, in every channel
    for c in range(3):
        pad = (114.0 - m.MEAN[c]) / m.STD[c]
        assert x[0, c, 100, 700] == pytest.approx(pad, abs=1e-4)
    # pixel 0,0 is RGB(10,20,30) → BGR(30,20,10), normalized
    for c, bgr in enumerate((30.0, 20.0, 10.0)):
        assert x[0, c, 0, 0] == pytest.approx((bgr - m.MEAN[c]) / m.STD[c], abs=1e-4)


def test_detect_floors_conf_at_the_calibrated_operating_point(monkeypatch):
    """The mmdet 0.05 test floor is a COCO-sweep value; on this export it
    emits ~100 mostly-noise boxes per image. detect() must never go
    below SCORE_THR regardless of the pipeline's 0.05 default."""
    calls = []

    class FakeSession:
        def get_inputs(self):
            return [type("I", (), {"name": "images"})()]

        def run(self, _names, feed):
            calls.append(feed)
            b = np.zeros((1, 18900, 4), dtype=np.float32)
            s = np.zeros((1, 18900, 5), dtype=np.float32)
            # one box at 0.10 (above 0.05, below the floor) and one at 0.90
            b[0, 0] = [0, 0, 50, 50]
            s[0, 0, 0] = 0.10
            b[0, 1] = [100, 100, 150, 150]
            s[0, 1, 0] = 0.90
            return [b, s]

    monkeypatch.setattr(m, "get_model", lambda: FakeSession())
    out = m.detect([_solid(64, 64)], conf=0.05)
    assert m.SCORE_THR == pytest.approx(0.35)
    assert [d["score"] for d in out[0]] == [pytest.approx(0.90)]
    assert calls  # session was actually used
