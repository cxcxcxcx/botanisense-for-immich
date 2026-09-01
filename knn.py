"""Build a FAISS index from a parquet glob.

CLI entry: `python -m knn --parquet-glob PATTERN --index-dir DIR [--kingdoms ...]`
(this module's `main()`).

Public surface:

  build_faiss_index(parquet_glob, index_path, labels_path, *,
                    kingdoms=None, version_tag=...) -> dict
    Walks all parquet files matching `parquet_glob`, optionally filters by
    kingdom, and writes a FAISS IndexIVFPQ index plus an aligned
    labels.parquet + INDEX_VERSION.txt. Returns a stats dict
    {"n_rows", "dim", "n_labels", "aligned"}.

The read/query path lives in `knn_io.py` (kept separate so the build's
chunk loop and rlimit setup aren't dragged into every
`from knn_io import classify_via_knn`). Shared constants
(LABEL_COLUMNS, EMB_DIM, BUILD_CHUNK_ROWS) are imported from `knn_io`.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import os as _os
# Force pyarrow's default memory pool to system malloc instead of
# mimalloc. Mimalloc allocates 1 GB aligned mmap regions per thread
# (~6 GB total on a 4-thread machine), which dominates the build's
# RLIMIT_DATA usage and shows up as anonymous 1 GB mappings in
# /proc/self/maps. System malloc returns regular brk-based heap which
# RLIMIT_DATA counts more cheaply. Must be set BEFORE pyarrow import.
_os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

# Cap glibc malloc arenas. With 8 FAISS OMP threads + pyarrow threads,
# glibc by default creates one ~heap arena per thread and does not return
# freed memory from them, so the heap climbs with every batch until it
# hits RLIMIT_DATA (observed: 8 GB at 22M vectors, ~320 B/vector retained,
# none of it live). Two arenas bound the heap to ~1.6 GB steady with no
# throughput loss (34K rows/s either way). Must be set before faiss/numpy
# spawn worker threads.
_os.environ.setdefault("MALLOC_ARENA_MAX", "2")
# RLIMIT_DATA caps the brk data segment's VIRTUAL high-water (VmData),
# not RSS — free() rarely moves the sbrk break down, so freed-but-not-
# returned heap accumulates against the cap even when RSS is low (build
# died at VmData=8 GB with RSS=2.4 GB). Two knobs make glibc give memory
# back: trim the top of the heap aggressively, and serve allocations
# >=128 KB via mmap (munmap on free fully releases them, outside
# RLIMIT_DATA). The 77 MB/batch buffers then never touch brk.
_os.environ.setdefault("MALLOC_TRIM_THRESHOLD_", "131072")
_os.environ.setdefault("MALLOC_MMAP_THRESHOLD_", "131072")

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from knn_io import (  # noqa: F401
    LABEL_COLUMNS, EMB_DIM, BUILD_CHUNK_ROWS,
)
from knn_io import validate_index  # used at end of build

_log = logging.getLogger("knn")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


# Columns we extract from the source parquet. Order matches the index
# row order (FAISS ids 0..N correspond to labels rows 0..N).
# `emb` is the 768-d BioCLIP-2 vector (float16 in the source, cast to
# float32 for FAISS).


def build_faiss_index(
    *,
    parquet_glob: str,
    index_path: Path,
    labels_path: Path,
    kingdoms: Iterable[str] | None = None,
    version_tag: str,
    ivf_nlist: int | None = None,
    ivf_train_size: int = 200_000,
) -> dict:
    """Build a FAISS IVFPQ index and aligned labels.parquet.

    Two passes over the same parquet files in the same order with the
    same kingdom filter:
      Pass 1: read embeddings → build FAISS IVFPQ → write index
      Pass 2: read label columns → write labels.parquet
    Alignment invariant: row `i` of labels.parquet == FAISS id `i`.
    Verified by raising if the row counts diverge at end-of-build.

    No resume, no segments. If the build crashes, run it again from
    scratch — the cost is one extra parquet read of an already-page-
    cached directory.
    """
    index_path = Path(index_path)
    labels_path = Path(labels_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    # FAISS OpenMP cap. 16 threads oversubscribes against pyarrow and
    # loses 1.7× to 8 (measured, see [[knn-build-checkpoint-thrashing]]).
    try:
        faiss.omp_set_num_threads(max(1, min(8, (_os.cpu_count() or 4) - 2)))
    except Exception:
        pass

    by_kingdom: Counter[str] = Counter()
    t0 = time.monotonic()

    n_rows = _build_faiss_index(
        parquet_glob=parquet_glob,
        index_path=index_path,
        kingdoms=kingdoms,
        ivf_nlist=ivf_nlist,
        ivf_train_size=ivf_train_size,
        by_kingdom=by_kingdom,
    )

    n_labels = _write_labels(
        parquet_glob=parquet_glob,
        labels_path=labels_path,
        kingdoms=kingdoms,
    )

    (index_path.parent / "INDEX_VERSION.txt").write_text(
        f"{version_tag}\n"
        f"parquet_glob={parquet_glob}\n"
        f"kingdoms={(list(kingdoms) if kingdoms else 'ALL')}\n"
        f"n_rows={n_rows}\n"
        f"dim={EMB_DIM}\n"
        f"built_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )

    if n_rows != n_labels:
        raise RuntimeError(
            f"index/labels MISALIGNED: index has {n_rows:,} vectors but "
            f"labels.parquet has {n_labels:,} rows. Species lookups will "
            f"return the wrong row. Do not use this index."
        )

    health = {"ok": None, "max_sim": None, "reason": "untrained IVF (build too small)"}
    saved = faiss.read_index(str(index_path))
    if saved.is_trained:
        health = validate_index(saved)
        if health["ok"] is False:
            _log.error("[knn.build] INDEX FAILED VALIDATION: %s", health["reason"])

    return {
        "n_rows": n_rows,
        "dim": EMB_DIM,
        "n_labels": n_labels,
        "by_kingdom": dict(by_kingdom),
        "aligned": True,
        "version": version_tag,
        "elapsed_seconds": time.monotonic() - t0,
        "health": health,
    }


def _filtered_batches(parquet_glob, kingdoms, *, columns=None, chunk_rows: int = 25_000):
    """Yield filtered RecordBatches from parquet_glob.

    Two layers of kingdom filtering:
      - EARLY: scanner(filter=kingdom.isin(...)) uses parquet row-group
        statistics to skip whole non-matching shards without reading
        their columns. The corpus is kingdom-clustered (all Animalia
        files first), so this prunes ~half the files and ~half the I/O.
      - LATE: per-batch pac.is_in below catches rows in mixed row groups
        whose statistics straddle a kingdom boundary.
    `columns` projects at the parquet level (the emb column is ~99% of a
    row's bytes; pass 2 must not read it). Verified on pyarrow 25:
    columns+filter together return rows correctly here (the old
    zero-row bug does not reproduce for this dataset).
    """
    import glob as _glob
    import pyarrow.dataset as _pads
    import pyarrow.compute as _pac
    files = sorted(_glob.glob(parquet_glob))
    if not files:
        raise FileNotFoundError(f"No parquet files matched glob: {parquet_glob}")
    kingdom_allowed = set(kingdoms) if kingdoms else None
    kingdom_value_set = pa.array(sorted(kingdom_allowed)) if kingdom_allowed else None
    for f in files:
        try:
            ds = _pads.dataset(f, format="parquet")
            kwargs = {"columns": columns, "batch_size": chunk_rows}
            if kingdom_allowed is not None:
                kwargs["filter"] = _pac.field("kingdom").isin(kingdom_value_set)
            scanner = ds.scanner(**kwargs)
        except Exception as e:
            _log.warning(f"[knn.build] could not read {f}: {e}")
            continue
        for batch in scanner.to_batches():
            if kingdom_allowed is not None:
                # Late filter for mixed row groups the stats-based early
                # filter could not fully exclude.
                keep = _pac.is_in(batch.column("kingdom"), value_set=kingdom_value_set)
                if not _pac.any(keep).as_py():
                    continue
                batch = batch.filter(keep)
            yield batch
        del scanner, ds


def _mem_mb() -> tuple[float, float]:
    """(RSS, VmData) in MB from /proc. VmData is the brk data-segment
    size — what RLIMIT_DATA actually caps — so it is the number that
    matters for an impending OOM, while RSS shows true residency."""
    rss = vdata = -1.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith("VmRSS:"):
                    rss = int(line.split()[1]) / 1024
                elif line.startswith("VmData:"):
                    vdata = int(line.split()[1]) / 1024
    except OSError:
        pass
    return rss, vdata


def _l2_normalize(mat: np.ndarray) -> np.ndarray:
    """L2-normalize rows in place. Source embeddings are unnormalized
    (TreeOfLife-200M norms ~40-57); FAISS inner-product needs unit
    vectors. v1 trained on raw vectors but added normalized ones — never
    split these two steps."""
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    np.divide(mat, norms, out=mat)
    return mat


def _sample_training_vectors(parquet_glob, kingdoms, train_size, *, seed=1234):
    """Pass 1: random sample of normalized vectors for IVF k-means.

    Picks row groups whose parquet kingdom stats overlap `kingdoms`
    (so we don't read Animalia shards), shuffles them with a fixed
    seed, and reads row groups until `train_size` vectors are gathered.
    Sampling across the whole Plantae+Fungi range — not the first N
    rows — keeps the centroids representative of the full corpus.
    """
    import glob as _glob
    import pyarrow.compute as _pac
    import pyarrow.parquet as _pq

    allowed = set(kingdoms) if kingdoms else None
    value_set = pa.array(sorted(allowed)) if allowed else None
    candidates: list[tuple[str, int]] = []
    for f in sorted(_glob.glob(parquet_glob)):
        pf = _pq.ParquetFile(f)
        for r in range(pf.metadata.num_row_groups):
            if allowed is None:
                candidates.append((f, r))
                continue
            rg = pf.metadata.row_group(r)
            ci = next(
                j for j in range(rg.num_columns)
                if rg.column(j).path_in_schema == "kingdom"
            )
            st = rg.column(ci).statistics
            if st.min in allowed or st.max in allowed:
                candidates.append((f, r))

    rng = np.random.default_rng(seed)
    rng.shuffle(candidates)

    chunks: list[np.ndarray] = []
    have = 0
    for f, r in candidates:
        table = _pq.ParquetFile(f).read_row_group(r, columns=["kingdom", "emb"])
        if allowed is not None:
            table = table.filter(_pac.is_in(table.column("kingdom"), value_set=value_set))
        if table.num_rows == 0:
            continue
        emb_flat = table.column("emb").combine_chunks().flatten()
        mat = np.asarray(emb_flat, dtype=np.float32).reshape(-1, EMB_DIM)
        _l2_normalize(mat)
        chunks.append(mat)
        have += mat.shape[0]
        if have >= train_size:
            break
    if not chunks:
        raise RuntimeError("training sample found 0 rows — check kingdom names / glob")
    return np.ascontiguousarray(np.vstack(chunks)[:train_size], dtype=np.float32)


def _build_faiss_index(
    *, parquet_glob, index_path, kingdoms,
    ivf_nlist, ivf_train_size, by_kingdom,
) -> int:
    """Pass 1: stream embeddings → build FAISS IVFPQ → write index."""
    if ivf_nlist is None:
        ivf_nlist = max(64, int(round(ivf_train_size ** 0.5)))
    ivf = faiss.IndexIVFPQ(
        faiss.IndexFlatIP(EMB_DIM),
        EMB_DIM, ivf_nlist, 16, 8, faiss.METRIC_INNER_PRODUCT,
    )
    ivf.nprobe = 64
    _log.info(f"[knn.build] IVF nlist={ivf_nlist}")

    # Delete stale .ivfdata from previous OnDiskInvertedLists runs.
    stale = Path(str(index_path) + ".ivfdata")
    if stale.exists():
        stale.unlink()

    # Pass 1: random sample across the PF range → k-means train.
    t_train0 = time.monotonic()
    train_mat = _sample_training_vectors(parquet_glob, kingdoms, ivf_train_size)
    _log.info(f"[knn.build] training IVF on {train_mat.shape[0]:,} sampled vectors")
    ivf.train(train_mat)
    del train_mat
    _log.info(
        f"[knn.build] IVF trained in {time.monotonic()-t_train0:.1f}s"
    )

    # Pass 2: full scan of PF rows → normalize → add. The sampled
    # training rows are part of this stream and get added here (training
    # itself never adds), so every source row is in the index exactly
    # once and FAISS id i aligns with labels row i.
    n_rows = 0
    t0 = time.monotonic()
    last_log = t0
    for batch in _filtered_batches(
        parquet_glob, kingdoms, columns=["kingdom", "emb"],
        chunk_rows=BUILD_CHUNK_ROWS,
    ):
        for k in batch.column("kingdom").to_pylist():
            if k:
                by_kingdom[k] += 1

        emb_col = batch.column("emb")
        flat = np.asarray(emb_col.flatten(), dtype=np.float32).reshape(-1, EMB_DIM)
        del emb_col
        _l2_normalize(flat)
        ivf.add(flat)
        n_rows += flat.shape[0]
        now = time.monotonic()
        if now - last_log >= 20:
            last_log = now
            rate = n_rows / max(now - t0, 1e-9)
            rss, vdata = _mem_mb()
            _log.info(
                f"[knn.build] {ivf.ntotal:,} vectors in index "
                f"({rate:,.0f} rows/s, RSS={rss:.0f} MB, VmData={vdata:.0f} MB)"
            )

    faiss.write_index(ivf, str(index_path))
    _log.info(
        f"[knn.build] FAISS pass done: {ivf.ntotal:,} vectors -> {index_path}"
    )
    return n_rows


def _write_labels(*, parquet_glob, labels_path, kingdoms) -> int:
    """Pass 2: stream label columns → write labels.parquet."""
    writer = None
    n_rows = 0
    t0 = time.monotonic()
    last_log = t0
    for batch in _filtered_batches(
        parquet_glob, kingdoms, columns=list(LABEL_COLUMNS),
        chunk_rows=BUILD_CHUNK_ROWS,
    ):
        labels_batch = batch  # already projected to LABEL_COLUMNS
        if writer is None:
            writer = pq.ParquetWriter(
                str(labels_path), labels_batch.schema, compression="zstd"
            )
        writer.write_table(pa.Table.from_batches([labels_batch]))
        n_rows += labels_batch.num_rows
        now = time.monotonic()
        if now - last_log >= 20:
            last_log = now
            rate = n_rows / max(now - t0, 1e-9)
            _log.info(f"[knn.build] labels: {n_rows:,} rows ({rate:,.0f} rows/s)")
    if writer is not None:
        writer.close()
    _log.info(f"[knn.build] labels pass done: {n_rows:,} rows -> {labels_path}")
    return n_rows


def main() -> int:
    """CLI entry: build an index from a parquet glob."""
    import argparse

    ap = argparse.ArgumentParser(description="Build a FAISS KNN index from TreeOfLife-200M-Embeddings parquet files.")
    ap.add_argument("--parquet-glob", required=True, help="Glob pattern (quoted) to the source parquet files.")
    ap.add_argument("--index-dir", required=True, help="Output directory; writes tol200m_bioclip2.faiss, labels.parquet, INDEX_VERSION.txt.")
    ap.add_argument("--tag", required=True, help="Version tag written to INDEX_VERSION.txt.")
    ap.add_argument(
        "--kingdoms",
        default=None,
        help="Comma-separated kingdoms to keep (e.g. 'Plantae,Fungi'). Default: ALL.",
    )
    ap.add_argument(
        "--ivf-nlist",
        type=int,
        default=None,
        help="IVF coarse-quantizer cell count. Defaults to sqrt(train_size).",
    )
    ap.add_argument(
        "--ivf-train-size",
        type=int,
        default=200_000,
        help="Vectors for IVF k-means training. Default 200K. Top-1 "
             "recall is bounded by c=cos(query, nearest_centroid), which "
             "depends on corpus diversity and nlist, not on train_size "
             "alone — see faiss-index-sanity-checks memory note §6.",
    )
    args = ap.parse_args()
    import resource
    # RLIMIT_DATA is NOT set by default. On this kernel it counts
    # anonymous mmap as well as brk, and importing faiss reserves ~2.7 GB
    # of VmData virtual up front (RSS ~0.1 GB) with add growing it
    # ~174 B/vector — so an 8 GB cap kills the build around 35M vectors
    # while RSS is only ~2 GB. RSS is the real pressure and stays
    # ~5.5 GB at 116M. Opt in with _KNN_BUILD_MEM_LIMIT_BYTES=<bytes>
    # only on a host whose RLIMIT_DATA semantics you have verified.
    _env_cap = _os.environ.get("_KNN_BUILD_MEM_LIMIT_BYTES", "").strip().lower()
    if _env_cap not in ("", "0", "unlimited"):
        try:
            resource.setrlimit(
                resource.RLIMIT_DATA,
                (int(_env_cap), int(_env_cap)),
            )
        except (ValueError, OSError) as _e:
            _log.warning(f"[knn.build] could not set RLIMIT_DATA: {_e}")

    try:
        soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
        if soft == resource.RLIM_INFINITY:
            _log.info("[knn.build] RLIMIT_DATA = unlimited (watch RSS, not VmData)")
        else:
            _log.info(
                f"[knn.build] RLIMIT_DATA = {soft / 1024**3:.1f} GB "
                f"(hard={hard / 1024**3:.1f} GB)"
            )
    except (ValueError, OSError):
        _log.warning("[knn.build] could not read RLIMIT_DATA")

    kingdoms = (
        [k.strip() for k in args.kingdoms.split(",") if k.strip()]
        if args.kingdoms
        else None
    )
    stats = build_faiss_index(
        parquet_glob=args.parquet_glob,
        index_path=Path(args.index_dir) / "tol200m_bioclip2.faiss",
        labels_path=Path(args.index_dir) / "labels.parquet",
        kingdoms=kingdoms,
        version_tag=args.tag,
        ivf_nlist=args.ivf_nlist,
        ivf_train_size=args.ivf_train_size,
    )
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
