"""Unit tests for storage.has_real_person() and the YuNet rescue helper.

Regression context (d844f1f6): a photo of two people — each with a hand
inside their own person silhouette — was stamped has_person=0 because

1. HAND_MIN_AREA existed as a constant but was never applied, so tiny
   incidental hands (~0.2-0.6% of frame) counted as suppression signals
   and defeated the has_separate_person hatch (every person box contained
   its own hand), and
2. YOLO-World's "face" prompt never fires, so the has_face hatch could
   never rescue anything.
"""

from __future__ import annotations

import sys
from pathlib import Path
from unittest import mock

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402
from storage import has_real_person  # noqa: E402


def _person(score=0.8, x=0.0, y=0.0, w=1.0, h=1.0):
    return {"label": "person", "score": score, "x": x, "y": y, "w": w, "h": h}


def _hand(score=0.6, x=0.1, y=0.1, w=0.2, h=0.2):
    return {"label": "hand", "score": score, "x": x, "y": y, "w": w, "h": h}


def _face(score=0.9, x=0.4, y=0.4, w=0.2, h=0.2):
    return {"label": "face", "score": score, "x": x, "y": y, "w": w, "h": h}


# --- HAND_MIN_AREA gating ------------------------------------------------


def test_small_incidental_hand_does_not_suppress():
    """A hand sliver inside a real body silhouette (0.61% of frame, the
    d844f1f6 geometry) is not a suppression signal — even fully
    contained, even with no other escape hatch available."""
    dets = [
        _person(0.9, x=0.512, y=0.385, w=0.179, h=0.615),
        _person(0.62, x=0.255, y=0.306, w=0.29, h=0.694),
        _hand(0.098, x=0.441, y=0.585, w=0.066, h=0.092),  # 0.61% area
        _hand(0.054, x=0.658, y=0.709, w=0.022, h=0.062),  # 0.13% area
    ]
    assert has_real_person(dets) is True


def test_big_contained_hand_still_suppresses():
    """The hand-silhouette false positive the rule targets: a >=2%-of-
    frame hand fully inside a person box, no face, no second person."""
    dets = [_person(0.7), _hand(0.6, x=0.1, y=0.1, w=0.2, h=0.2)]  # 4% area
    assert has_real_person(dets) is False


def test_big_sticking_out_hand_never_suppresses():
    dets = [
        _person(0.8, x=0.0, y=0.0, w=0.4, h=1.0),
        _hand(0.6, x=0.7, y=0.4, w=0.2, h=0.2),
    ]
    assert has_real_person(dets) is True


# --- face hatch ----------------------------------------------------------


def test_face_rescues_big_hand_suppression():
    dets = [_person(0.7), _hand(0.6, x=0.1, y=0.1, w=0.2, h=0.2), _face(0.91)]
    assert has_real_person(dets) is True


def test_sub_threshold_face_does_not_rescue():
    dets = [_person(0.7), _hand(0.6, x=0.1, y=0.1, w=0.2, h=0.2), _face(0.3)]
    assert has_real_person(dets) is False


def test_no_person_box_is_false_regardless():
    # A hand alone is never person evidence...
    assert has_real_person([_hand(0.9)]) is False
    assert has_real_person([]) is False
    # ...but a face IS: "face" is a PERSON_LABELS member (the existence
    # test includes it), so a detected face counts as person evidence
    # and skips suppression entirely (hand disjoint from the face box).
    assert has_real_person([_hand(0.9), _face(0.9)]) is True


# --- pipeline._augment_with_yunet_faces gating ---------------------------


def test_rescue_not_consulted_without_person_box():
    """Gate 1: YuNet never runs on photos with no person evidence —
    this is what makes foliage false positives structurally impossible."""
    dets = [{"label": "flower", "score": 0.9, "x": 0, "y": 0, "w": 1, "h": 1}]
    with mock.patch.object(pipeline.face, "detect_faces") as df:
        assert pipeline._augment_with_yunet_faces(dets, _fake_img()) is dets
    assert df.call_count == 0


def test_rescue_not_consulted_when_already_real():
    """Gate 2: has_real_person() True → no YuNet call, no duplicate rows."""
    dets = [_person(0.8, x=0.0, y=0.0, w=0.4, h=1.0),
            _hand(0.6, x=0.7, y=0.4, w=0.2, h=0.2)]
    with mock.patch.object(pipeline.face, "detect_faces") as df:
        assert pipeline._augment_with_yunet_faces(dets, _fake_img()) is dets
    assert df.call_count == 0


def test_rescue_appends_face_inside_person_box_only():
    """Suppressed photo + YuNet: faces whose CENTER is inside a person
    box are appended; the input list is not mutated; the augmented list
    flips has_real_person() via the face hatch."""
    dets = [_person(0.7, x=0.3, y=0.3, w=0.4, h=0.4),
            _hand(0.6, x=0.35, y=0.35, w=0.2, h=0.2)]
    inside = _face(0.91, x=0.4, y=0.4, w=0.2, h=0.2)     # center (0.5, 0.5)
    outside = _face(0.95, x=0.0, y=0.0, w=0.04, h=0.04)  # center (0.02, 0.02)
    with mock.patch.object(
        pipeline.face, "detect_faces", lambda img, **k: [inside, outside]
    ):
        aug = pipeline._augment_with_yunet_faces(dets, _fake_img())
    assert aug == dets + [inside]
    assert has_real_person(aug) is True


def test_rescue_noop_when_no_face_lands_inside():
    dets = [_person(0.7, x=0.3, y=0.3, w=0.4, h=0.4),
            _hand(0.6, x=0.35, y=0.35, w=0.2, h=0.2)]
    with mock.patch.object(
        pipeline.face, "detect_faces", lambda img, **k: [_face(0.95, x=0.0, y=0.0, w=0.04, h=0.04)]
    ):
        assert pipeline._augment_with_yunet_faces(dets, _fake_img()) is dets


def _fake_img():
    from PIL import Image

    return Image.new("RGB", (8, 8))
