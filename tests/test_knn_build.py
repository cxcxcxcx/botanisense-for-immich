"""Tests for the two-pass build: FAISS index + labels.parquet.

Each test pins an invariant that a shipped build actually violated:

- Source embeddings are unnormalized (norms ~40-57 in TreeOfLife-200M).
  v1 trained IVF on raw vectors but added normalized ones, so every
  entry decoded to roughly its coarse centroid and scores were
  arbitrary. The built index must be L2-normalized.
- FAISS id `i` and labels.parquet row `i` are the same source row.
  classify_via_knn does `labels.take(I)` — a count or order mismatch
  resolves every neighbour to the wrong species, silently.
- Labels are streamed through one ParquetWriter. The old resume path
  did `pq.read_table()` + `pa.concat_tables()` and held ~7 GB of labels
  in RAM on a 116M-row build.
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

pytest.importorskip("faiss")

N_ROWS = 600


def _write_source(path: Path, *, kingdoms=None):
    """A parquet shard with the columns build_faiss_index expects.

    Embeddings are unnormalized (scaled to norm ~45) to mirror the real
    TreeOfLife-200M export. `kingdoms` overrides the per-row kingdom;
    default is all-Plantae.
    """
    import knn
    rng = np.random.default_rng(7)
    embs = rng.standard_normal((N_ROWS, knn.EMB_DIM)).astype(np.float32)
    embs *= 45.0
    if kingdoms is None:
        kingdoms = ["Plantae"] * N_ROWS
    cols = {
        "uuid": [f"u{i}" for i in range(N_ROWS)],
        "kingdom": kingdoms,
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
    pq.write_table(pa.table(cols), path)


def _build(knn, source, out_dir, **kw):
    out_dir.mkdir(parents=True, exist_ok=True)
    # The production default (200K) would never trigger IVF training on
    # a 600-row fixture.
    kw.setdefault("ivf_train_size", 300)
    return knn.build_faiss_index(
        parquet_glob=str(source),
        index_path=out_dir / "tol200m_bioclip2.faiss",
        labels_path=out_dir / "labels.parquet",
        kingdoms=["Plantae"],
        version_tag="test",
        **kw,
    )


def test_build_produces_valid_aligned_index(tmp_path):
    import knn
    import faiss
    source = tmp_path / "src.parquet"
    _write_source(source)
    out = tmp_path / "idx"

    res = _build(knn, source, out)

    assert res["n_rows"] == N_ROWS
    assert res["n_labels"] == N_ROWS
    assert res["aligned"] is True
    assert res["health"]["ok"] is True, res["health"]

    idx = faiss.read_index(str(out / "tol200m_bioclip2.faiss"))
    assert idx.ntotal == N_ROWS
    assert knn.validate_index(idx)["ok"] is True

    md = pq.read_metadata(str(out / "labels.parquet"))
    assert md.num_rows == N_ROWS


def test_label_rows_keep_source_order(tmp_path):
    """Row i of labels.parquet must be source row i — labels.take(I)
    depends on it."""
    import knn
    source = tmp_path / "src.parquet"
    _write_source(source)
    out = tmp_path / "idx"
    _build(knn, source, out)

    uuids = (
        pq.read_table(str(out / "labels.parquet"), columns=["uuid"])
        .column("uuid")
        .to_pylist()
    )
    assert uuids == [f"u{i}" for i in range(N_ROWS)]


def test_kingdom_filter_excludes_other_kingdoms(tmp_path):
    """The filter runs per-batch (pyarrow 25's scanner filter silently
    returns 0 rows when the emb list<float> column is projected)."""
    import knn
    import faiss
    kingdoms = ["Plantae" if i % 2 == 0 else "Animalia" for i in range(N_ROWS)]
    source = tmp_path / "src.parquet"
    _write_source(source, kingdoms=kingdoms)
    out = tmp_path / "idx"

    res = _build(knn, source, out)

    assert res["n_rows"] == N_ROWS // 2
    idx = faiss.read_index(str(out / "tol200m_bioclip2.faiss"))
    assert idx.ntotal == N_ROWS // 2
    label_kingdoms = set(
        pq.read_table(str(out / "labels.parquet"), columns=["kingdom"])
        .column("kingdom")
        .to_pylist()
    )
    assert label_kingdoms == {"Plantae"}


def test_build_streams_labels_without_materialising_them(tmp_path, monkeypatch):
    """pq.read_table() in the build path loads the whole labels file
    into RAM — the 7 GB concat OOM. Labels must go through a streaming
    ParquetWriter only."""
    import knn
    source = tmp_path / "src.parquet"
    _write_source(source)
    out = tmp_path / "idx"

    def _explode(*a, **kw):
        raise AssertionError(
            "build called pq.read_table — that materializes labels in RAM"
        )

    monkeypatch.setattr(knn.pq, "read_table", _explode)
    monkeypatch.setattr(knn.pq, "read_metadata", _explode)

    res = _build(knn, source, out)
    assert res["n_labels"] == N_ROWS


def test_misaligned_passes_raise(tmp_path, monkeypatch):
    """The end-of-build guard: if the two passes ever disagree on row
    count, refuse to return a usable-looking index."""
    import knn
    source = tmp_path / "src.parquet"
    _write_source(source)
    out = tmp_path / "idx"

    def _lie(**kw):
        return N_ROWS + 1

    monkeypatch.setattr(knn, "_write_labels", _lie)
    with pytest.raises(RuntimeError, match="MISALIGNED"):
        _build(knn, source, out)
