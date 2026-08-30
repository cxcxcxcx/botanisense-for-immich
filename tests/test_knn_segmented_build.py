"""Segmented builds must equal a single-pass build.

`build_faiss_index(resume=True)` used to re-stream the source from row
0, so a second segment re-added every row already in the index. The
out-of-tree `/tmp/runner.py` carried its own state.json offset tracker
to work around that — and having two build paths is exactly how the v1
index ended up trained on unnormalized vectors while adding normalized
ones. These tests pin the folded-in behaviour so the workaround script
never needs to exist again.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

pytest.importorskip("faiss")

N_ROWS = 600


@pytest.fixture
def source(tmp_path):
    """A parquet shard with the columns build_faiss_index expects."""
    import knn
    rng = np.random.default_rng(7)
    embs = rng.standard_normal((N_ROWS, knn.EMB_DIM)).astype(np.float32)
    # Unnormalized, like the real TreeOfLife-200M export (norms ~40-57).
    embs *= 45.0
    cols = {
        "uuid": [f"u{i}" for i in range(N_ROWS)],
        "kingdom": ["Plantae"] * N_ROWS,
        "phylum": ["p"] * N_ROWS,
        "class": ["c"] * N_ROWS,
        "order": ["o"] * N_ROWS,
        "family": ["f"] * N_ROWS,
        "genus": ["g"] * N_ROWS,
        "species": [f"s{i % 10}" for i in range(N_ROWS)],
        "scientific_name": [f"sci{i % 10}" for i in range(N_ROWS)],
        "common_name": [f"cn{i % 10}" for i in range(N_ROWS)],
        "identifier": [f"https://example.test/{i}.jpg" for i in range(N_ROWS)],
        "source_dataset": ["gbif"] * N_ROWS,
        "emb": [e.tolist() for e in embs],
    }
    p = tmp_path / "src.parquet"
    pq.write_table(pa.table(cols), p)
    return p


def _build(knn, source, out_dir, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    kw.setdefault("index_type", "flat")
    return knn.build_faiss_index(
        parquet_glob=str(source),
        index_path=out_dir / "tol200m_bioclip2.faiss",
        labels_path=out_dir / "labels.parquet",
        kingdoms=["Plantae"],
        version_tag="test",
        **kw,
    )


def test_build_produces_valid_index(source, tmp_path):
    """A build off unnormalized source must still yield a normalized
    index — this is the invariant v1 violated."""
    import knn, faiss
    out = tmp_path / "idx"
    res = _build(knn, source, out)
    assert res["n_rows"] == N_ROWS
    assert res["complete"] is True
    assert res["health"]["ok"] is True, res["health"]

    idx = faiss.read_index(str(out / "tol200m_bioclip2.faiss"))
    assert knn.validate_index(idx)["ok"] is True


def test_segment_cap_stops_early_and_records_cursor(source, tmp_path, monkeypatch):
    import knn
    # The cap is enforced at batch granularity, so the source has to
    # span more than one batch for it to trip at all.
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 200)
    out = tmp_path / "idx"
    res = _build(knn, source, out, max_segment_rows=200)
    assert res["complete"] is False
    assert res["segment_rows"] == 200
    state = json.loads((out / "state.json").read_text())
    assert state["next_offset_rows"] == res["next_offset_rows"] == 200


def test_segment_cap_larger_than_source_completes(source, tmp_path):
    """A cap the stream never reaches is not an early stop."""
    import knn
    out = tmp_path / "idx"
    res = _build(knn, source, out, max_segment_rows=10 * N_ROWS)
    assert res["complete"] is True
    assert res["segment_rows"] == N_ROWS


def test_resume_does_not_duplicate_rows(source, tmp_path, monkeypatch):
    """The regression that mattered: segment 2 must not re-add
    segment 1's rows."""
    import knn
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 200)
    out = tmp_path / "idx"

    first = _build(knn, source, out, max_segment_rows=200)
    assert first["complete"] is False
    added = first["segment_rows"]
    assert 0 < added < N_ROWS

    second = _build(knn, source, out, resume=True)
    # Total across both segments is each source row exactly once.
    assert first["segment_rows"] + second["segment_rows"] == N_ROWS
    assert second["next_offset_rows"] == N_ROWS
    assert second["complete"] is True


def test_fresh_build_clears_a_stale_cursor(source, tmp_path):
    """A leftover state.json from an earlier run must not make a
    from-scratch build skip rows."""
    import knn
    out = tmp_path / "idx"
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text(json.dumps({"next_offset_rows": 500}))

    res = _build(knn, source, out)
    assert res["n_rows"] == N_ROWS, "fresh build honoured a stale offset"


def test_resume_with_corrupt_state_refuses(source, tmp_path):
    """Rather than guess an offset and duplicate rows, fail."""
    import knn
    out = tmp_path / "idx"
    out.mkdir(parents=True, exist_ok=True)
    (out / "state.json").write_text("{not json")
    with pytest.raises(ValueError, match="unreadable"):
        _build(knn, source, out, resume=True)


def test_checkpoint_advances_the_resume_cursor(source, tmp_path, monkeypatch):
    """A checkpoint writes the index AND the cursor together.

    They used to disagree: the index was written at every checkpoint but
    state.json only at end-of-build, so a killed build left rows in the
    index and no cursor — and --resume then re-added every one of them.
    """
    import knn
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 100)
    monkeypatch.setattr(knn, "CHECKPOINT_EVERY", 1)
    out = tmp_path / "idx"

    seen: list[int] = []
    real_write = knn.faiss.write_index

    def spy(idx, path):
        real_write(idx, path)
        sp = out / "state.json"
        if sp.exists():
            seen.append(json.loads(sp.read_text())["next_offset_rows"])

    monkeypatch.setattr(knn.faiss, "write_index", spy)
    _build(knn, source, out, ivf_train_size=300, index_type="ivf")

    assert seen, "no checkpoint wrote a cursor"
    # The cursor advances during the build, not only at the end.
    assert max(seen) > 0
    assert seen == sorted(seen), f"cursor went backwards: {seen}"


def test_checkpoints_do_not_leak_files(source, tmp_path, monkeypatch):
    """Checkpointing overwrites one canonical file. It used to also
    write a .ckpt.<chunk> per checkpoint and never delete them — 391
    files / 355 GB on a 116M-row build."""
    import knn
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 100)
    monkeypatch.setattr(knn, "CHECKPOINT_EVERY", 1)
    out = tmp_path / "idx"
    _build(knn, source, out, ivf_train_size=300, index_type="ivf")

    strays = list(out.glob("*.ckpt*"))
    assert strays == [], f"checkpoint files leaked: {strays}"
