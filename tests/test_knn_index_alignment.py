"""Index/labels alignment must be checked, not assumed.

classify_via_knn maps FAISS ids straight into labels.parquet row offsets
(`labels.take(I)`). So the index and the labels file must have identical
row counts AND identical row order — otherwise every neighbour resolves
to the wrong species, quietly.

This is not hypothetical. The v1 index shipped with ntotal=117,070,179
against labels.num_rows=116,218,519 (851,660 Animalia rows leaked into
the tail) and was repaired by a hand-written /tmp/trim_index.py. Nothing
in the library noticed, because nothing looked.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

faiss = pytest.importorskip("faiss")

DIM = 64


def _write_index_and_labels(d: Path, n_index: int, n_labels: int):
    """Write a normalized flat index with n_index rows and a labels
    parquet with n_labels rows."""
    import knn_io as knn
    rng = np.random.default_rng(1)
    m = rng.standard_normal((n_index, knn.EMB_DIM)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    idx = faiss.IndexFlatIP(knn.EMB_DIM)
    idx.add(m)
    faiss.write_index(idx, str(d / "tol200m_bioclip2.faiss"))

    cols = {c: [f"{c}{i}" for i in range(n_labels)] for c in knn.LABEL_COLUMNS}
    pq.write_table(pa.table(cols), d / "labels.parquet")
    (d / "INDEX_VERSION.txt").write_text("vtest\nn_rows=%d\n" % n_labels)


def test_aligned_index_passes(tmp_path):
    import knn_io as knn
    _write_index_and_labels(tmp_path, 50, 50)
    _idx, _lbl, meta = knn._load_index(tmp_path)
    assert meta["health"]["ok"] is True, meta["health"]
    assert meta["n_index"] == meta["n_labels"] == 50


def test_index_longer_than_labels_is_flagged(tmp_path):
    """The exact v1 failure: extra rows at the tail of the index."""
    import knn_io as knn
    _write_index_and_labels(tmp_path, 60, 50)
    _idx, _lbl, meta = knn._load_index(tmp_path)
    assert meta["health"]["ok"] is False
    assert "misaligned" in meta["health"]["reason"].lower()
    assert "60" in meta["health"]["reason"]
    assert "50" in meta["health"]["reason"]


def test_labels_longer_than_index_is_flagged(tmp_path):
    """The other direction is equally wrong — it means rows were added to
    labels that never made it into the index."""
    import knn_io as knn
    _write_index_and_labels(tmp_path, 50, 60)
    _idx, _lbl, meta = knn._load_index(tmp_path)
    assert meta["health"]["ok"] is False


def test_alignment_failure_outranks_a_passing_norm_check(tmp_path):
    """A normalized index with misaligned labels must still fail. The
    norm probe alone would have said 'ok'."""
    import knn_io as knn
    _write_index_and_labels(tmp_path, 60, 50)
    idx = faiss.read_index(str(tmp_path / "tol200m_bioclip2.faiss"))
    assert knn.validate_index(idx)["ok"] is True, "norms are fine here"
    _i, _l, meta = knn._load_index(tmp_path)
    assert meta["health"]["ok"] is False, "alignment must override"
