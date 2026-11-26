"""coreml_engine: engine selection, static-batch chunking, golden gate, and
the pipeline's fallback to the torch teacher — all without coremltools,
so this runs on Linux CI."""

from __future__ import annotations

import json

import numpy as np
import pytest

import coreml_engine as ce


@pytest.fixture
def pkg(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"batch": 16}))
    return tmp_path


def test_select_engine_passes_through_other_engines(pkg):
    assert ce.select_engine("teacher", system="Darwin", package_dir=pkg,
                            have_coremltools=True) == "teacher"
    assert ce.select_engine("mobile-v0", system="Darwin", package_dir=pkg,
                            have_coremltools=True) == "mobile-v0"


@pytest.mark.parametrize("engine", ["auto", "coreml"])
def test_select_engine_picks_coreml_on_mac_with_package(engine, pkg):
    assert ce.select_engine(engine, system="Darwin", package_dir=pkg,
                            have_coremltools=True) == "coreml"


@pytest.mark.parametrize("engine", ["auto", "coreml"])
@pytest.mark.parametrize("system,have_ct,has_pkg", [
    ("Linux", True, True),       # Docker / Linux: never Core ML
    ("Darwin", False, True),     # no coremltools installed
    ("Darwin", True, False),     # package not downloaded
])
def test_select_engine_falls_back_to_teacher(engine, system, have_ct, has_pkg, pkg, tmp_path):
    d = pkg if has_pkg else tmp_path / "missing"
    assert ce.select_engine(engine, system=system, package_dir=d,
                            have_coremltools=have_ct) == "teacher"


@pytest.mark.parametrize("n", [1, 15, 16, 17, 33])
def test_encode_batched_pads_tail_and_keeps_order(n):
    calls = []

    def predict(chunk):
        calls.append(chunk.shape)
        # Row i's embedding = its first pixel, so order and slicing are visible.
        return np.repeat(chunk[:, :1, 0, 0], 4, axis=1)

    x = np.arange(n, dtype=np.float32).reshape(n, 1, 1, 1) + 1
    x = np.broadcast_to(x, (n, 3, 2, 2)).copy()
    out = ce.encode_batched(predict, x, 16)
    assert out.shape == (n, 4)
    np.testing.assert_array_equal(out[:, 0], np.arange(1, n + 1))
    assert all(s[0] == 16 for s in calls)          # static batch every call
    assert len(calls) == -(-n // 16)


def test_golden_gate_thresholds():
    rng = np.random.default_rng(0)
    ref = rng.normal(size=(32, 768)).astype(np.float32)
    ok, mean, low = ce.golden_gate(ref * 3.0, ref)  # scale-invariant
    assert ok and mean == pytest.approx(1.0, abs=1e-6)
    noisy = ref + rng.normal(scale=0.3, size=ref.shape).astype(np.float32)
    ok, mean, low = ce.golden_gate(noisy, ref)
    assert not ok and low < ce.COS_MIN_GATE


def test_resolve_classifier_falls_back_when_coreml_fails(monkeypatch):
    import pipeline

    sentinel = object()
    monkeypatch.setenv("ENGINE", "coreml")
    monkeypatch.setattr(ce, "select_engine", lambda engine: "coreml")
    monkeypatch.setattr(pipeline, "_load_coreml_engine", lambda: None)
    monkeypatch.setitem(pipeline.__dict__, "TreeOfLifeClassifier", lambda: sentinel)
    monkeypatch.setattr(pipeline, "_clf_cache", None)
    assert pipeline._resolve_classifier() is sentinel


def test_load_coreml_engine_rejects_failed_gate(monkeypatch):
    import pipeline

    class Bad:
        def self_check(self):
            return False, 0.99, 0.95

        def version_key(self):
            return "x"

    monkeypatch.setattr(ce, "CoreMLEngine", lambda: Bad())
    assert pipeline._load_coreml_engine() is None

    def boom():
        raise RuntimeError("no ANE")

    monkeypatch.setattr(ce, "CoreMLEngine", boom)
    assert pipeline._load_coreml_engine() is None
