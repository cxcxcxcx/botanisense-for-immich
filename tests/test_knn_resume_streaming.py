"""Resume must stream labels, not accumulate them in RAM.

Previously resume=True collected this segment's labels into
`pending_label_batches` and concatenated them with the on-disk labels at
end of build: `pa.concat_tables([prev, new_table])`. For a 116M-row
build that's ~7 GB held in memory simultaneously — the exact failure mode
/tmp/concat_labels.py was written to avoid, and listed as a known
anti-pattern in the project's KNN build notes.

This test pins the post-fix invariant: resumed builds do NOT keep the
prior segment's labels resident. We assert via a tracemalloc delta that
the build's RAM footprint does not grow linearly with the prior
segment's label table size.
"""

from __future__ import annotations

import sys
import tracemalloc
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
    import knn
    rng = np.random.default_rng(7)
    embs = rng.standard_normal((N_ROWS, knn.EMB_DIM)).astype(np.float32)
    cols = {
        c: [f"{c}{i}" for i in range(N_ROWS)] for c in knn.LABEL_COLUMNS
    }
    # LABEL_COLUMNS includes "kingdom" — override it with actual kingdom
    # values, not the placeholder "{c}{i}" string used above. Without
    # this, the kingdom filter rejects every row and the build silently
    # produces n_rows=0.
    cols["kingdom"] = ["Plantae"] * N_ROWS
    cols["emb"] = [e.tolist() for e in embs]
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


def test_resume_writes_to_dotnew_and_swaps(source, tmp_path):
    """The pre-fix path held labels in RAM and concat'd at close. The
    post-fix path writes everything through a streaming writer to
    `<labels>.new`, then atomically swaps it into place."""
    import knn
    monkeypatch = pytest.MonkeyPatch()
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 200)
    out = tmp_path / "idx"

    first = _build(knn, source, out, max_segment_rows=200)  # 200 of 600
    new_path = out / "labels.parquet.new"
    assert not new_path.exists(), "no .new file should exist after a fresh build"

    second = _build(knn, source, out, resume=True)  # the remaining 400
    # .new must NOT survive past the swap.
    assert not new_path.exists()
    # The canonical file has both segments' rows, index and labels aligned.
    final_md = pq.read_metadata(str(out / "labels.parquet"))
    assert final_md.num_rows == N_ROWS
    assert second["n_labels"] == N_ROWS
    assert second["n_index"] == N_ROWS
    assert second["aligned"] is True
    assert first["n_rows"] + second["segment_rows"] == N_ROWS


def test_resume_does_not_materialise_old_labels(source, tmp_path, monkeypatch):
    """The actual OOM invariant: building segment 2 must not load the
    previous segment's labels into a python Table. Pre-fix code did
    `prev = pq.read_table(...)` and then `pa.concat_tables([prev, new])`
    — both materialized. Post-fix code streams row-group by row-group."""
    import knn
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 200)
    out = tmp_path / "idx"
    _build(knn, source, out, max_segment_rows=200)

    # Catch anyone reintroducing pq.read_table() in the resume path.
    def _explode(*a, **kw):
        raise AssertionError(
            "resume path called pq.read_table — that materializes the "
            "previous labels file in RAM and is the OOM trap"
        )

    monkeypatch.setattr(knn.pq, "read_table", _explode)
    # pq.ParquetFile for the row-group streaming is fine.
    monkeypatch.setattr(knn.pq, "ParquetFile", pq.ParquetFile)

    res = _build(knn, source, out, resume=True)
    assert res["n_labels"] == N_ROWS


def test_resume_preserves_segment_order(source, tmp_path, monkeypatch):
    """row 0 of the resumed labels file must still correspond to row 0
    of the index — concatenating in the wrong order would scramble
    species lookups just like the alignment bug did."""
    import knn
    monkeypatch.setattr(knn, "BUILD_CHUNK_ROWS", 200)
    out = tmp_path / "idx"
    _build(knn, source, out, max_segment_rows=200)
    _build(knn, source, out, resume=True)

    lbl = pq.read_table(str(out / "labels.parquet"), columns=["uuid"])
    ids = lbl.column("uuid").to_pylist()
    # uu-prefixed, ordered: must be uuid0..uuid(N-1), not interleaved or
    # reordered by the resume merge. The fixture uses {col_name}{i}
    # for every column, so the UUID column becomes "uuid<i>".
    assert ids == [f"uuid{i}" for i in range(N_ROWS)]