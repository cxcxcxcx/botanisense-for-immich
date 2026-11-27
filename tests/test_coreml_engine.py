"""coreml_engine: engine selection, static-batch chunking, golden gate, and
the pipeline's fallback to the torch teacher — all without coremltools,
so this runs on Linux CI."""

from __future__ import annotations

import json

import numpy as np
import pytest

import coreml_engine as ce


@pytest.fixture(autouse=True)
def _isolated_package_dir(tmp_path, monkeypatch):
    # record_loaded() writes next to the package dir — keep it out of data/.
    monkeypatch.setenv("COREML_PACKAGE", str(tmp_path / "models" / "bioclip-2-coreml"))


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


def test_load_coreml_engine_records_fallback_reason(monkeypatch):
    import pipeline

    def boom():
        raise RuntimeError("no ANE")

    monkeypatch.setattr(ce, "CoreMLEngine", boom)
    assert pipeline._load_coreml_engine() is None
    loaded = ce.read_loaded()
    assert loaded["engine"] == "teacher"
    assert "no ANE" in loaded["fallback_reason"]


def _write_files(root, files):
    for rel, data in files.items():
        f = root / rel
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_bytes(data)


def _fake_snapshot(files):
    def snapshot_download(repo, *, revision, local_dir, allow_patterns):
        assert repo == ce.HF_REPO and revision == ce.HF_REVISION
        root = ce.Path(local_dir)
        _write_files(root, files)
        (root / ".cache").mkdir(exist_ok=True)
        sha = ce.sha256_dir(root / "encoder_b16.mlpackage")
        (root / "manifest.json").write_text(
            json.dumps({"file": "encoder_b16.mlpackage", "sha256": sha})
        )
    return snapshot_download


def test_fetch_package_verifies_then_moves_into_place(tmp_path, monkeypatch):
    import huggingface_hub

    files = {"encoder_b16.mlpackage/Manifest.json": b"{}",
             "encoder_b16.mlpackage/Data/w.bin": b"weights"}
    monkeypatch.setattr(huggingface_hub, "snapshot_download", _fake_snapshot(files))
    _write_files(tmp_path / "stage", files)
    want = ce.sha256_dir(tmp_path / "stage" / "encoder_b16.mlpackage")

    dest = tmp_path / "pkg"
    ce.fetch_package(dest, expected_sha256=want)
    assert (dest / "manifest.json").exists()
    assert not (dest / ".cache").exists()
    assert not (tmp_path / "pkg.partial").exists()

    # A corrupted / unexpected download never lands in `dest`.
    dest2 = tmp_path / "pkg2"
    with pytest.raises(RuntimeError, match="sha256 mismatch"):
        ce.fetch_package(dest2, expected_sha256="0" * 64)
    assert not dest2.exists() and not (tmp_path / "pkg2.partial").exists()


def test_ensure_package_only_downloads_on_a_mac_that_can_use_it(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(ce.subprocess, "run", lambda *a, **k: calls.append((a, k)))
    d = tmp_path / "missing"
    assert ce.ensure_package(d, system="Linux", have_coremltools=True) is False
    assert ce.ensure_package(d, system="Darwin", have_coremltools=False) is False
    monkeypatch.setenv("COREML_AUTO_DOWNLOAD", "0")
    assert ce.ensure_package(d, system="Darwin", have_coremltools=True) is False
    assert calls == []


def test_ensure_package_fetches_in_child_without_offline_flag(tmp_path, monkeypatch):
    d = tmp_path / "pkg"
    seen = {}

    def fake_run(cmd, *, env, **kw):
        seen["cmd"], seen["env"] = cmd, env
        d.mkdir()
        (d / "manifest.json").write_text("{}")
        return ce.subprocess.CompletedProcess(cmd, 0, "", "")

    monkeypatch.setenv("HF_HUB_OFFLINE", "1")
    monkeypatch.setattr(ce.subprocess, "run", fake_run)
    assert ce.ensure_package(d, system="Darwin", have_coremltools=True) is True
    assert seen["cmd"][1:] == ["-m", "coreml_engine", "fetch", str(d)]
    assert "HF_HUB_OFFLINE" not in seen["env"]
    # Already present → no second download.
    monkeypatch.setattr(ce.subprocess, "run", lambda *a, **k: pytest.fail("refetched"))
    assert ce.ensure_package(d, system="Darwin", have_coremltools=True) is True
