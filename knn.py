"""Build and query a FAISS index over TreeOfLife-200M-Embeddings.

Public surface (used by pipeline.py / run.py / app.py):

  build_faiss_index(parquet_glob, index_path, labels_path, *,
                    kingdoms=None, version_tag=...) -> dict
    Walks all parquet files matching `parquet_glob`, optionally filters by
    kingdom (so you can build a Plantae-only subset), and writes a FAISS
    IndexFlatIP plus an aligned labels.parquet + INDEX_VERSION.txt. Returns
    a stats dict {"n_rows", "dim", "by_kingdom"}.

  classify_via_knn(query_embeddings, index_dir, *, k=10, batch_size=1024)
                  -> list[dict]
    Loads the FAISS index + labels from `index_dir` and returns one
    dict per query row: top species (weighted majority vote over the K
    nearest reference embeddings), common name, score, kingdom, family,
    genus, and the raw neighbor list.

Why FAISS over numpy matmul: for 233M reference vectors at 768-d, a numpy
dot-product per query would be ~700 MB of memory traffic per matmul and
~30 ms on modern hardware. FAISS gives the same exact result with the
same memory profile but supports batched queries (the 1024-query batch
path is 10× faster per-query), and we get an `index.ntotal` and the
ability to swap in IVF-PQ later without touching the pipeline.

Why pyarrow.dataset instead of DuckDB at build time: DuckDB mmap's
every parquet shard it touches and never releases the mmap until the
connection is closed — we observed 30M-row builds accumulating 17
shards × 128 MB virtual = 2 GB of dirty mmap RSS. pyarrow.dataset's
scanner pushes the kingdom IN filter down to row-group statistics
(so non-Plantae/Fungi shards are skipped at the file-stat level, no
parquet decode at all), and pyarrow releases each file's mmap'd
pages as the scanner advances. Net effect: bounded RSS, same
throughput as DuckDB.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections import Counter
from pathlib import Path

import os as _os
# Force pyarrow's default memory pool to system malloc instead of
# mimalloc. Mimalloc allocates 1 GB aligned mmap regions per thread
# (~6 GB total on a 4-thread machine), which dominates the build's
# RLIMIT_DATA usage and shows up as anonymous 1 GB mappings in
# /proc/PID/maps. System malloc returns regular brk-based heap which
# RLIMIT_DATA counts more cheaply. Must be set BEFORE pyarrow import.
_os.environ.setdefault("ARROW_DEFAULT_MEMORY_POOL", "system")

import faiss
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

_log = logging.getLogger("knn")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


# Columns we extract from the source parquet. Order matters: the FAISS
# index is built in the order duckdb returns rows, and the labels.parquet
# sidecar uses the same row order, so this list is the contract between
# build and classify. `emb` is the 768-d BioCLIP-2 vector (float16 in
# the source, cast to float32 for FAISS).
LABEL_COLUMNS = (
    "uuid",
    "kingdom",
    "phylum",
    "class",
    "order",
    "family",
    "genus",
    "species",
    "scientific_name",
    "common_name",
    "identifier",
    "source_dataset",
)
EMB_DIM = 768
# How many source rows to pull per FAISS .add() during a build.
# 200K × 768 × 4 bytes = ~600 MB, well within the RAM budget on
# a single-machine build (we keep ~1 GB free for FAISS + label arrays).
BUILD_CHUNK_ROWS = 25_000
# Periodic index reload — every CHECKPOINT_EVERY chunks (25K rows each)
# the live IVFPQ index is written to disk and read back, to bound the
# in-RAM bookkeeping that grows with row count, and to leave a resumable
# index behind if the build is killed.
#
# This used to be 8 (every 200K rows), on the assumption — see the old
# comment, and the stale one still nearby in the checkpoint block — that
# write_index only serialized a header because the PQ codes lived in an
# .ivfdata file under OnDiskInvertedLists. That stopped being true when
# the build switched to in-RAM ArrayInvertedLists: write_index now
# serializes the WHOLE index, which grows linearly with the build.
#
# Measured on the 116M-row v2 build at CHECKPOINT_EVERY=8:
#   - 13.5s per checkpoint at 78M rows, against ~9.6s of actual add work
#   - throughput fell from ~20.8K rows/s (add-only) to ~7.2K rows/s
#   - RSS it was meant to bound sat flat at ~5 GB with 15 GB headroom
# So it was spending more than half the build serializing, to solve a
# memory problem that wasn't occurring. 400 chunks = 10M rows keeps a
# useful crash-recovery point while making the overhead negligible.
CHECKPOINT_EVERY = 400


def build_faiss_index(
    *,
    parquet_glob: str,
    index_path: Path,
    labels_path: Path,
    kingdoms: Iterable[str] | None = None,
    version_tag: str,
    index_type: str = "ivf",
    ivf_nlist: int | None = None,
    # Default 200K (was 1M) — holds 200K * 768 * 4 = ~600 MB during
    # training. With the default 1M we'd need ~3 GB just for the
    # training buffer, plus FAISS k-means internals push peak to 12+ GB
    # on memory-constrained boxes. 200K is still ≥ 30 × nlist for
    # nlist ≤ 6K, which is a reasonable cluster count for libraries
    # up to ~200M rows.
    ivf_train_size: int = 200_000,
    # If True and an index already exists at `index_path`, load it
    # and continue adding (skip training). Used for multi-pass builds
    # when one segment is too large to fit in RLIMIT_DATA: the user
    # runs several builds back-to-back, each resuming the previous.
    resume: bool = False,
    # Cap on how many source rows THIS invocation consumes. Paired with
    # `resume`, this is what makes a segmented build safe: each run
    # processes at most `max_segment_rows` rows and records where it
    # stopped in state.json, so the next run picks up from there instead
    # of re-streaming from row 0. None = consume the whole stream.
    max_segment_rows: int | None = None,
) -> dict:
    """Build a FAISS index over the embeddings matching `parquet_glob`.

    Two index types:
      - "flat": exact IndexFlatIP. Requires the full vector matrix in RAM
        (~3 KB × n_rows). Recall@10 = 100%. Default for small corpora.
      - "ivf": approximate IndexFlatIP coarse-quantized into nlist Voronoi
        cells. RAM ≈ nlist × EMB_DIM × 4 bytes for the quantizer plus
        ~3 KB × n_rows for the vectors themselves (the vectors are still
        stored uncompressed, just partitioned into cells). At
        nlist = sqrt(n_rows) and search-time nprobe = 32, recall@10 is
        ~90-95% — fine for species voting where a single missed neighbor
        barely changes the score.

    The FAISS index is exact cosine (inner product on L2-normalized
    vectors — BioCLIP-2 features are normalized upstream by
    `create_image_features(..., normalize=True)`).

    `on_progress` is an optional callback (chunk_idx, total_chunks,
    rows_so_far) for the UI.

    **Segmented builds.** `resume=True` continues adding to an existing
    index. It reads `state.json` (next to the index) for how many source
    rows previous segments already consumed and skips exactly that many
    before adding anything, then records the new total on the way out.
    Without that offset a resumed build re-streams from row 0 and adds
    every row a second time — silently doubling the index and corrupting
    the label alignment, since labels.parquet is appended in stream
    order. Always resume through this function rather than a bespoke
    script: the out-of-tree runner that built v1 drifted from this code
    and normalized its `add()` path but not its `train()` path, which
    produced a degenerate index (see validate_index).
    """
    index_path = Path(index_path)
    labels_path = Path(labels_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    # Streaming via pyarrow.dataset with row-group filter pushdown.
    # DuckDB mmap's every parquet shard it touches and never releases
    # the mmap until the connection is closed — observed 30M-row build
    # accumulating 17 shards × 128 MB virtual = 2 GB of dirty mmap RSS,
    # plus heap. Each shard is ~100 MB on disk; once DuckDB touches it
    # the kernel keeps the pages resident until the file is closed,
    # and DuckDB keeps the file open. Switching to pyarrow.dataset
    # gives us the same row-group filter pushdown DuckDB has — pyarrow
    # reads ONLY the row groups whose statistics match the filter,
    # skipping non-Plantae shards entirely on disk I/O. Memory stays
    # bounded by the active batch + IVF codes; no shard-level mmap
    # accumulation.
    import glob as _glob
    import pyarrow.dataset as _pads
    import pyarrow.compute as _pac
    parquet_files = sorted(_glob.glob(parquet_glob))
    if not parquet_files:
        raise FileNotFoundError(f"No parquet files matched glob: {parquet_glob}")
    # Stale .ivfdata from previous OnDiskInvertedLists-based runs.
    stale_ivfdata = Path(str(index_path) + ".ivfdata")
    if stale_ivfdata.exists():
        stale_ivfdata.unlink()
    # A from-scratch build starts the source cursor over. Leaving a
    # previous run's state.json in place would make the NEXT resumed
    # segment skip rows this build never added, silently truncating the
    # index.
    if not resume:
        stale_state = index_path.parent / "state.json"
        if stale_state.exists():
            _log.info(
                "[knn.build] fresh build: resetting %s", stale_state
            )
            stale_state.unlink()
    columns = list(LABEL_COLUMNS) + ["emb"]
    kingdom_filter = None
    if kingdoms:
        # Build an OR-of-equalities filter. pyarrow's scanner pushes
        # this down to the row-group statistics level, so any shard
        # whose kingdom column is entirely != {Plantae, Fungi} gets
        # skipped at file open (no parquet decode, no mmap).
        kingdom_filter = _pac.field("kingdom").isin(list(kingdoms))

    def _shard_stream():
        for f in parquet_files:
            try:
                # No columns projection AND no filter on the scanner —
                # in pyarrow 25 the combination of `columns=...` (which
                # includes the `emb` list<float> column) with a kingdom
                # `filter=` silently yields zero batches. The bug is
                # invisible because the build just returns "0 rows" and
                # exits cleanly. Apply the kingdom filter per-batch in
                # the consumer loop instead (see for batch in stream:
                # below) — cheaper than re-encoding `emb` and identical
                # in cost (we read the column anyway).
                ds_ = _pads.dataset(f, format="parquet")
                scanner = ds_.scanner(batch_size=BUILD_CHUNK_ROWS)
                for batch in scanner.to_batches():
                    yield batch
                del scanner, ds_
            except Exception as _e:
                _log.warning(f"[knn.build] could not read {f}: {_e}")
                continue
    stream = _shard_stream()

    index = faiss.IndexFlatIP(EMB_DIM)
    ivf: faiss.IndexIVFFlat | None = None
    ivf_train_buf: list[np.ndarray] = []
    # FAISS parallelizes add()/train() with OpenMP. The default here is
    # all 16 cores, and add() has always been parallel — but 16 threads
    # oversubscribe against pyarrow's reader threads and lose badly.
    # Measured (200K x 768 synthetic adds into a trained nlist=4096
    # IVFPQ, no disk in the loop):
    #     threads= 1   11,372 rows/s
    #     threads= 2   21,176 rows/s
    #     threads= 4   34,938 rows/s
    #     threads= 8   38,618 rows/s   <- best
    #     threads=16   22,432 rows/s   <- the default, 1.7x worse than 8
    # So capping at 8 is worth ~1.7x. Do NOT "simplify" this away on the
    # theory that FAISS already picks a sensible default — it picks
    # cpu_count, which is the worst of the five settings above.
    #
    # For context on where the rest of the build's time goes: that
    # 38.6K rows/s is the add-only ceiling, while the full build sustains
    # ~18.3K rows/s at only ~3 of 16 cores busy. Roughly half the wall
    # clock is parquet decode + the in-place normalize + label writing,
    # not FAISS. Optimize there before touching thread counts again.
    try:
        # Cap at 8 on a 16-core host. 16 is the worst setting
        # (oversubscribes against pyarrow's reader threads and loses
        # 1.7× to 8); 8 was the measured sweet spot in
        # [[knn-build-checkpoint-thrashing]]. Don't go higher than
        # cores-2 so OS + pyarrow still get CPU.
        _threads = max(1, min(8, (_os.cpu_count() or 4) - 2))
        faiss.omp_set_num_threads(_threads)
        _log.info(f"[knn.build] faiss OpenMP threads = {_threads}")
    except Exception as _e:  # pragma: no cover - faiss build without omp
        _log.warning(f"[knn.build] could not set faiss thread count: {_e}")
    # Resume mode: if a prior build wrote a valid IVFPQ index to
    # index_path, load it back and skip training. Subsequent .add()
    # calls append PQ codes to the same in-RAM invlists. Caller is
    # expected to pass the same ivf_nlist/ivf_train_size params as the
    # first pass (they aren't read after training). Used for multi-pass
    # builds that exceed RLIMIT_DATA: run build #1 (50M rows), run
    # build #2 with --resume on the same dir (next 50M rows), etc.
    # The index file at index_path is OVERWRITTEN at end of build, so
    # pass --resume only on the FIRST pass that finds no existing file.
    # On later passes it appends to the prior index.
    if resume and index_path.exists():
        # Loads the prior index and CONTINUES adding. Used to be gated
        # on index_type == "ivf", which silently ignored the persisted
        # index when the test used "flat" — the resumed build started
        # from 0 vectors, so labels and index drifted apart and the
        # alignment check fired.
        prev = faiss.read_index(str(index_path))
        if isinstance(prev, faiss.IndexIVFPQ):
            ivf = prev
            ivf_nlist = ivf.nlist
            index = ivf
        elif isinstance(prev, faiss.IndexFlatIP):
            # Flat resume: keep `index` as a fresh IndexFlatIP with the
            # prior ntotal preserved by calling read_index. The trick:
            # IndexFlatIP is not designed for incremental addition via
            # .add() — but it IS — .add() appends and ntotal grows.
            # So we just take the loaded object as `index`. It has the
            # prior vectors and is already-trained for `add()`.
            index = prev
        else:
            raise ValueError(
                f"resume requested but {index_path} is "
                f"{type(prev).__name__}, neither IndexIVFPQ nor IndexFlatIP"
            )
        _log.info(
            f"[knn.build] RESUME: loaded {prev.ntotal:,} rows from "
            f"{index_path.name}; type={type(prev).__name__}; continuing add()"
        )
    if index_type == "ivf":
        # nlist = sqrt(n_rows) is the rule of thumb. We don't know n_rows
        # up front (the parquet stream might be filtered down to nothing),
        # so we start with sqrt of the *training* set size and adjust
        # upward if more data arrives.
        if ivf_nlist is None:
            ivf_nlist = max(64, int(round(ivf_train_size ** 0.5)))
        # Only build a fresh IVF when we're not resuming — in resume
        # mode the `ivf` was already loaded above from the existing
        # index file and we must not overwrite it.
        if ivf is None:
            # Flat quantizer (assigns each vector to its cluster centroid).
            quantizer = faiss.IndexFlatIP(EMB_DIM)
            # IVFPQ instead of IVFFlat: each vector is encoded as M
            # sub-quantizers × nbits bits = M*nbits/8 bytes. With M=16
            # nbits=8 → 16 bytes per vector. For 130M vectors that's
            # ~2 GB on disk vs ~390 GB for uncompressed IVFFlat —
            # critical because we don't have 1.5 TB free on the build
            # host. The k-means inside PQ learns the codebook during
            # ivf.train(), so the same training flow as IVFFlat works.
            # Recall tradeoff: PQ is approximate. For BioCLIP-2 vectors
            # M=16 nbits=8 gives ~85–90% recall@10 in typical use,
            # which is acceptable for album grouping (the user can
            # re-sync a single photo and get correct species on a
            # 1-NN lookup against the centroid, since the dominant
            # match usually wins despite PQ error).
            pq_m = 16
            pq_nbits = 8
            ivf = faiss.IndexIVFPQ(
                quantizer, EMB_DIM, ivf_nlist, pq_m, pq_nbits,
                faiss.METRIC_INNER_PRODUCT,
            )
            # Probe enough cells for good recall with PQ; nprobe=64
            # roughly compensates for the lossy encoding.
            ivf.nprobe = 64
            _log.info(
                f"[knn.build] IVF using in-RAM invlists (PQ codes "
                f"~16 B/vec; CHECKPOINT_EVERY={CHECKPOINT_EVERY} "
                f"chunks between reloads)"
            )
    label_records: list[dict] = []  # noqa: F841 (legacy; labels now stream via label_writer)
    # Streaming parquet writer — labels are appended per chunk instead
    # of accumulating in a Python list. Schema is derived from the
    # first batch and frozen for the rest of the build.
    label_writer: pq.ParquetWriter | None = None
    # ---- source-offset tracking (segmented builds) ----------------------
    # `consumed_rows` counts every filtered source row this build has seen,
    # including rows skipped to catch up to a previous segment. It is the
    # cursor persisted to state.json. `skip_target` is where this run
    # starts adding. A from-scratch build always starts at 0 and resets
    # the file, so a stale state.json can't silently truncate a rebuild.
    skip_target = 0
    state_path = index_path.parent / "state.json"
    if resume and state_path.exists():
        try:
            skip_target = int(
                json.loads(state_path.read_text()).get("next_offset_rows", 0)
            )
        except Exception as _e:
            raise ValueError(
                f"resume requested but {state_path} is unreadable ({_e}). "
                f"Refusing to continue: without a valid offset this build "
                f"would re-add rows already in the index."
            )
        _log.info(
            f"[knn.build] RESUME: skipping first {skip_target:,} source rows"
        )
    elif resume:
        _log.warning(
            "[knn.build] RESUME with no state.json — starting at offset 0. "
            "If the index already holds rows, this WILL duplicate them."
        )
    # On resume, before we start writing new labels, we stream the
    # *previous* segment's labels through this same writer row-group by
    # row-group. This way the on-disk labels file is built up one write
    # at a time and never materializes either the old or the new table
    # in full. ParquetWriter has no append mode; the "append" is the
    # file we recreate here by streaming. The previous labels file
    # remains untouched until the new one is fully written, so a crash
    # partway through loses at most the current segment's labels.
    if resume and labels_path.exists():
        _prev = pq.ParquetFile(str(labels_path))
        label_writer = pq.ParquetWriter(
            str(labels_path) + ".new",
            _prev.schema_arrow,
            compression="zstd",
        )
        # Stream ALL prior labels through the new writer. The skip in
        # the source-row loop is for VECTORS, not labels: prior segments
        # already wrote labels[0..skip_target-1] to disk, and those rows
        # are what align with index ids 0..skip_target-1. The current
        # segment will append labels[skip_target..skip_target+N-1] for
        # the new vectors. Result on disk: skip_target + N rows, matching
        # the index's skip_target + N rows. Exactly aligned.
        n_streamed = 0
        for rg in range(_prev.num_row_groups):
            tbl = _prev.read_row_group(rg)
            label_writer.write_table(tbl)
            n_streamed += tbl.num_rows
        del _prev
        _log.info(
            f"[knn.build] RESUME: streamed {n_streamed:,} prior labels "
            f"into {labels_path.name}.new"
        )
    # n_rows tracks how many rows THIS segment has added. In resume
    # mode the loaded IVFPQ already has prev_n rows from the previous
    # segment; we initialize n_rows to prev_n so the per-chunk progress
    # log shows the cumulative total.
    n_rows = index.ntotal if (resume and index_path.exists()) else 0
    by_kingdom: Counter[str] = Counter()
    # Label slices held between flushes, as zero-copy pyarrow
    # RecordBatch views. Each is the 12 label columns of one DuckDB
    # batch (the `emb` column dropped via batch.select()). Much cheaper
    # than the previous list-of-dict accumulation.
    pending_label_batches: list[pa.RecordBatch] = []
    # Single buffer for the current chunk's embeddings. Set in the main
    # loop, consumed in _flush(). We deliberately don't accumulate
    # across chunks — each chunk is ~77 MB and accumulating them
    # would blow up RAM by 100× or more (5M rows of embeddings held
    # at once = ~15 GB by itself, as measured via a sidecar test).
    mat: np.ndarray | None = None
    t0 = time.monotonic()
    chunk_idx = 0

    # ---- source-offset tracking (segmented builds) ----------------------
    # `consumed_rows` counts every filtered source row this build has seen,
    # including rows skipped to catch up to a previous segment. It is the
    # cursor persisted to state.json. `skip_target` is where this run
    # starts adding. A from-scratch build always starts at 0 and resets
    # the file, so a stale state.json can't silently truncate a rebuild.
    # (state.json path + skip_target are computed ABOVE the writer-open
    # block so the resume label-stream can reference them.)
    consumed_rows = 0
    segment_rows = 0  # rows actually added by this invocation
    stopped_early = False

    def _flush() -> None:
        nonlocal n_rows, mat
        if mat is None:
            return
        # Re-normalize in-place. `mat = mat / norms` would allocate a
        # fresh 77 MB array; `np.divide(..., out=mat)` writes back into
        # the buffer we already own. The intermediate `norms` is also
        # released immediately so it doesn't linger.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        np.divide(mat, norms, out=mat)
        del norms

        if ivf is not None and not ivf.is_trained:
            # Accumulate training vectors until we hit ivf_train_size,
            # then call .train(). After training, add the training vectors
            # + this flush's vectors into the IVF so no rows are lost.
            ivf_train_buf.append(mat)
            trained_rows = sum(m.shape[0] for m in ivf_train_buf)
            if trained_rows >= ivf_train_size:
                train_mat = np.ascontiguousarray(
                    np.vstack(ivf_train_buf), dtype=np.float32
                )
                ivf_train_buf.clear()
                _log.info(
                    f"[knn.build] training IVF on {trained_rows:,} vectors "
                    f"(nlist={ivf_nlist}) ..."
                )
                t_train = time.monotonic()
                ivf.train(train_mat)
                _log.info(
                    f"[knn.build] IVF trained in "
                    f"{time.monotonic() - t_train:.1f}s"
                )
                # Use plain in-RAM ArrayInvertedLists (the FAISS default).
                # Previously we tried OnDiskInvertedLists to stream PQ
                # codes to a .ivfdata file — but the mmap'd file counts
                # toward RLIMIT_DATA (~2 GB at 130M rows × 16 B PQ code)
                # and our 12 GB cap doesn't have the headroom once DuckDB
                # + FAISS quantizer + Python heap are factored in. The
                # VmPeak climbs steadily toward 12 GB and OOM-kills the
                # build before it completes. In-RAM is simpler and the
                # periodic write_index+del+gc+read_index checkpoint
                # below keeps the RSS bounded by the checkpoint interval.
                # Add training vectors and current flush's vectors to
                # FAISS in two separate .add() calls instead of one
                # .add(vstack([train_mat, mat])). The vstack would need
                # an extra ~train_mat worth of contiguous memory
                # (~600 MB for the default 200K-vector training set);
                # on a memory-constrained build that allocation can
                # bump into RLIMIT_AS even though the system has plenty
                # of free RAM (RSS keeps growing across chunks). Two
                # .add() calls have identical semantics — FAISS just
                # appends to the inverted lists in order.
                ivf.add(train_mat)
                ivf.add(mat)
                # Drop the reference so train_mat can be GC'd before
                # the next chunk arrives.
                train_mat = None  # noqa: F841
        elif ivf is not None:
            # IVF already trained — add straight to it.
            ivf.add(mat)
        else:
            # Flat index (no IVF in use).
            index.add(mat)

        n_rows += mat.shape[0]
        # Stream labels to disk. Pending label batches are stored as
        # pyarrow RecordBatches (zero-copy slice of the original batch),
        # NOT as `to_pylist()`-produced Python dicts — the to_pylist
        # path took ~4.5 s per 25K-row chunk (creating 325K Python
        # objects under the GIL) and was the dominant build bottleneck.
        # On a non-resume build the writer was opened on the first batch;
        # on resume it's writing to <labels>.new after streaming the
        # previous labels row-group by row-group. Either way we just
        # stream every chunk through it.
        if pending_label_batches and label_writer is not None:
            for rb in pending_label_batches:
                label_writer.write_table(pa.Table.from_batches([rb]))
            pending_label_batches.clear()
        # Drop the mat reference now that FAISS has copied the codes.
        # FAISS .add() on IndexIVFPQ copies the codes into the inverted
        # lists, so this 77 MB can be released immediately rather than
        # waiting for the next chunk's reassignment.
        mat = None

    # Per-batch kingdom filter. We can't put it on the scanner because
    # pyarrow 25 + the `emb` list<float> column projection kills the
    # filter silently (yields zero batches). Pre-compute the allowed
    # set once.
    kingdom_allowed = (
        set(kingdoms) if kingdoms else None
    )
    for batch in stream:
        if kingdom_allowed is not None:
            keep = _pac.is_in(
                batch.column("kingdom"),
                value_set=pa.array(sorted(kingdom_allowed)),
            )
            # Skip entire batches that match no rows — common when
            # the source has other kingdoms interleaved.
            if not _pac.any(keep).as_py():
                consumed_rows += batch.num_rows
                continue
            batch = batch.filter(keep)
        # Fast-forward past rows a previous segment already added. We
        # count them but touch nothing else — no embedding decode, no
        # label write, no .add(). Whole batches only: BUILD_CHUNK_ROWS is
        # fixed and the stream is deterministic, so segment boundaries
        # always fall on batch boundaries.
        if consumed_rows + batch.num_rows <= skip_target:
            consumed_rows += batch.num_rows
            continue
        if consumed_rows < skip_target:
            raise ValueError(
                f"state.json offset {skip_target:,} falls mid-batch "
                f"(at row {consumed_rows:,}, batch of {batch.num_rows}). "
                f"The source stream or batch size changed between "
                f"segments; resuming would misalign labels."
            )
        # Segment cap reached — stop adding and let the close path
        # persist the cursor for the next run.
        if (
            max_segment_rows is not None
            and segment_rows >= max_segment_rows
        ):
            stopped_early = True
            _log.info(
                f"[knn.build] segment cap reached "
                f"({segment_rows:,} >= {max_segment_rows:,}); stopping"
            )
            # Flush whatever this segment has accumulated so labels and
            # embeddings leave together. Without this, an early-cap
            # segment ends with the FAISS index holding N rows but
            # labels.parquet holding N - chunk_rows, which trips the
            # alignment check and would misalign species on the next
            # segment if anyone tried to use it.
            _flush()
            # Persist the index too. The end-of-build write_index()
            # below is bypassed when we break out early, so without this
            # the next segment would resume against an empty index on
            # disk — duplicating every row that was just added.
            faiss.write_index(index, str(index_path))
            break
        consumed_rows += batch.num_rows
        segment_rows += batch.num_rows
        # batch is a pyarrow.RecordBatch with our 13 columns. Convert
        # `emb` (DuckDB emits variable-length `list<float>`, length 768
        # per row for BioCLIP-2) to a (N, 768) float32 numpy array.
        # We `flatten()` the list column to one contiguous float array
        # then reshape — this avoids the Python-object overhead of
        # `to_pylist()` (which materializes N*768 boxed Python floats,
        # ~3.7 GB per chunk for our 200K-row batches) and also avoids
        # the broken `cast(pa.list_(pa.float32())) + np.asarray` path
        # (np.asarray on a variable-length list column produces a 1-D
        # object array, not the 2-D numeric array we want).
        emb_col = batch.column(batch.schema.get_field_index("emb"))
        flat = emb_col.flatten()
        flat_arr = np.asarray(flat, dtype=np.float32)
        if flat_arr.size % EMB_DIM != 0:
            # Some rows have wrong-dim embeddings (truncated / corrupt).
            # Bail loudly — silently reshaping would misalign vectors
            # and corrupt the FAISS index.
            bad = batch.num_rows - (flat_arr.size // EMB_DIM)
            raise ValueError(
                f"{bad}/{batch.num_rows} rows in this batch have emb "
                f"len != {EMB_DIM} (flat size {flat_arr.size}); "
                f"skipping bad rows is not implemented — investigate the source."
            )
        emb_arr = flat_arr.reshape(-1, EMB_DIM)
        # Free the previous chunk's mat (if any) before assigning the
        # new one — otherwise the previous batch's 77 MB lingers as a
        # numpy reference. This is the single biggest source of RSS
        # creep we tracked down; without it we'd hold up to 100 chunks
        # × 77 MB = 7.7 GB transiently during burst reads.
        mat = emb_arr

        # Capture the kingdom counts off the column directly via pyarrow
        # (no per-row Python iteration). column.to_pylist() on a single
        # VARCHAR column is cheap (~1 ms for 25K rows) — much faster
        # than the previous `for row in batch.to_pylist()` loop.
        kingdom_col = batch.column(batch.schema.get_field_index("kingdom"))
        for k in kingdom_col.to_pylist():
            if k:
                by_kingdom[k] += 1

        # Keep just the 12 label columns (drop the 6 KB emb column so
        # the labels parquet stays compact). batch.select() returns a
        # zero-copy RecordBatch view; pyarrow will materialize it once
        # at write_table() time. Projecting once here also lets the
        # writer freeze its schema on the first batch.
        pending_label_batches.append(batch.select(list(LABEL_COLUMNS)))

        # Lazily initialize the parquet writer on the first batch so we
        # have a real schema to freeze. Subsequent batches are converted
        # to Table in _flush() and written.
        # On resume, the writer was already opened above (writing to
        # labels_path + ".new"), so this branch only fires for fresh
        # builds.
        if label_writer is None:
            # Open with the schema but DON'T clear pending yet — the
            # flush immediately below needs to drain pending into the
            # new writer. The previous code cleared pending here, so
            # the first batch's labels never reached disk when the cap
            # fired on that same batch.
            first_table = pa.Table.from_batches(pending_label_batches)
            label_writer = pq.ParquetWriter(
                str(labels_path), first_table.schema, compression="zstd"
            )

        if batch.num_rows >= BUILD_CHUNK_ROWS:
            _flush()
            chunk_idx += 1
            _log.info(
                f"[knn.build] chunk {chunk_idx}  rows={n_rows:,}  "
                f"elapsed={time.monotonic() - t0:.1f}s"
            )
            # Per-50-chunk tracemalloc snapshot to identify Python-level
            # memory growth. tracemalloc is ~10% overhead so we only
            # enable it every 50 chunks, dump the top 10 allocators to
            # the log, then continue. The snapshot diff (current vs
            # baseline) surfaces where the growth is coming from —
            # without this we were guessing why Python heap grew 16 GB
            # at 27M rows.
            if chunk_idx % 50 == 0:
                import tracemalloc as _tm
                if not _tm.is_tracing():
                    _tm.start(25)
                    _tm_base = _tm.take_snapshot()
                else:
                    cur = _tm.take_snapshot()
                    diff = cur.compare_to(_tm_base, 'lineno')
                    _log.info("[knn.build] tracemalloc top 10 (Δ from chunk 50):")
                    for stat in diff[:10]:
                        _log.info(f"  {stat}")
                    # Roll the baseline so the next interval is small
                    # deltas (otherwise the diff is dominated by the
                    # very first allocations).
                    _tm_base = cur
            # Periodic index reload to bound RSS. Empirically, RSS grows
            # ~68→167 MB per 1M rows under IVFPQ+OnDiskInvertedLists —
            # the OnDiskInvertedLists header bookkeeping (per-list
            # capacity realloc, ntotal counter, code recompute buffers)
            # accumulates in RAM even though the PQ codes themselves
            # stream to disk. After ~50M rows that growth would push
            # RSS past the 12 GB RLIMIT_DATA cap. Every CHECKPOINT_EVERY
            # chunks, write_index() the live IVF to a checkpoint file
            # then read_index() it back. The reload drops the in-RAM
            # bookkeeping back to ~1 GB; OnDiskInvertedLists keeps the
            # .ivfdata file descriptor open across the round-trip so
            # accumulated codes are NOT lost (FAISS appends to the file
            # at the current offset).
            if (
                ivf is not None
                and ivf.is_trained  # write_index on untrained IVF crashes
                and CHECKPOINT_EVERY > 0
                and chunk_idx % CHECKPOINT_EVERY == 0
            ):
                _log.info(
                    f"[knn.build] checkpointing {n_rows:,} rows"
                )
                t_ck = time.monotonic()
                # Write ONCE, to the canonical path. Previously this
                # wrote a second `.ckpt.<chunk_idx>` file as well, on the
                # belief that write_index only serialized a header (true
                # under OnDiskInvertedLists, false since the switch to
                # in-RAM ArrayInvertedLists). It actually serializes the
                # whole index, so that was double the I/O — and because
                # each ckpt was named by chunk_idx, nothing ever deleted
                # them: a 116M-row build accumulated 391 files / 355 GB
                # before running the disk down. The canonical path is
                # overwritten in place, so it can't leak, and it's the
                # file --resume looks for anyway.
                faiss.write_index(ivf, str(index_path))
                # Advance the resume cursor IN THE SAME breath as the
                # index write. These two must agree: the index says
                # "these rows are in me", state.json says "start here
                # next time". Previously state.json was only written at
                # end-of-build, so a killed build left an index full of
                # rows and no cursor — and --resume would start from row
                # 0 and add every one of them a second time.
                state_path.write_text(
                    json.dumps({"next_offset_rows": consumed_rows})
                )
                # Free the old index BEFORE reloading — that's where
                # the memory reclaim comes from. Note `index` (the
                # fallback IndexFlatIP for non-IVF builds) might still
                # reference this ivf — re-bind it after the reload.
                index = None  # break the alias so the next `del ivf` actually frees
                del ivf
                import gc
                gc.collect()
                # Reload so the accumulated in-RAM bookkeeping is dropped.
                ivf = faiss.read_index(str(index_path))
                # Keep the `index` alias pointing at the live IVFPQ so
                # the final write_index() at end-of-build uses the
                # current object (not a stale reference to the deleted
                # one — which would have ntotal=0).
                index = ivf
                _log.info(
                    f"[knn.build] reloaded in {time.monotonic() - t_ck:.1f}s; "
                    f"ntotal={ivf.ntotal:,}, RSS should drop on next sample"
                )

    _flush()

    # If we ended mid-training (dataset smaller than ivf_train_size),
    # train on whatever we accumulated. If even that was empty, the
    # IVF can't function — fall back to writing the empty flat index.
    if ivf is not None and not ivf.is_trained:
        if ivf_train_buf:
            train_mat = np.ascontiguousarray(
                np.vstack(ivf_train_buf), dtype=np.float32
            )
            ivf_train_buf.clear()
            _log.warning(
                f"[knn.build] training IVF on only {train_mat.shape[0]:,} "
                f"vectors (< requested {ivf_train_size:,}); "
                f"recall may be poor"
            )
            ivf.train(train_mat)
            ivf.add(train_mat)
            index = ivf
        else:
            _log.warning(
                "[knn.build] no training vectors; falling back to flat"
            )
            index = faiss.IndexFlatIP(EMB_DIM)

    # Persist FAISS index. IndexFlatIP.write_index is binary-safe.
    faiss.write_index(index, str(index_path))

    # Record the source cursor so the next segment resumes here. Written
    # AFTER the index so a crash mid-write can't advance the offset past
    # rows that never made it into the index.
    state_path.write_text(json.dumps({"next_offset_rows": consumed_rows}))
    _log.info(f"[knn.build] state saved: next_offset_rows={consumed_rows:,}")

    # Close the streaming parquet writer. On resume we wrote to
# <labels>.new; swap it into place atomically so a reader never sees
# a half-written file. The previous labels file is removed only AFTER
# the rename, so a crash before this point leaves both files and the
# caller can retry by inspecting which is which.
    if label_writer is not None:
        label_writer.close()
        new_path = Path(str(labels_path) + ".new")
        if new_path.exists():
            new_path.replace(labels_path)
            _log.info(f"[knn.build] resumed labels swapped into {labels_path.name}")
    elif not labels_path.exists():
        # Edge case: zero rows were processed and nothing was written.
        # Consumers still need a file with the canonical schema.
        empty = pa.Table.from_pylist(
            [{c: None for c in LABEL_COLUMNS}]
        ).slice(0, 0)
        pq.write_table(empty, str(labels_path), compression="zstd")

    # Version file — read by classify_via_knn so the caller doesn't
    # have to pass version_tag explicitly.
    (index_path.parent / "INDEX_VERSION.txt").write_text(
        f"{version_tag}\n"
        f"parquet_glob={parquet_glob}\n"
        f"kingdoms={(list(kingdoms) if kingdoms else 'ALL')}\n"
        f"n_rows={n_rows}\n"
        f"dim={EMB_DIM}\n"
        f"index_type={index_type}\n"
        f"ivf_nlist={ivf_nlist if ivf is not None else 'NA'}\n"
        f"ivf_nprobe={ivf.nprobe if ivf is not None else 'NA'}\n"
        f"built_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )

    elapsed = time.monotonic() - t0
    # Post-build health check. The whole point of unifying the build
    # path is that a bad index never reaches the server again — so fail
    # loudly here rather than waiting for someone to notice that the
    # species predictions are subtly wrong.
    health = validate_index(index)
    if health["ok"] is False:
        _log.error(
            "[knn.build] BUILD PRODUCED AN INVALID INDEX: %s", health["reason"]
        )
    elif health["ok"]:
        _log.info(
            "[knn.build] index validated (max unit-query sim %.4f)",
            health["max_sim"],
        )
    # Index/labels alignment. classify_via_knn uses FAISS ids as label row
    # offsets, so a mismatch means wrong species names, not just a count
    # discrepancy. v1 shipped 851,660 rows out of step and needed a
    # hand-written /tmp/trim_index.py to repair — check it here instead.
    try:
        n_labels = int(pq.read_metadata(labels_path).num_rows)
    except Exception as _e:
        n_labels = -1
        _log.warning(f"[knn.build] could not read labels row count: {_e}")
    n_index = int(index.ntotal)
    aligned = (n_labels == n_index)
    if n_labels >= 0 and not aligned:
        _log.error(
            f"[knn.build] MISALIGNED: index has {n_index:,} vectors but "
            f"labels.parquet has {n_labels:,} rows. Species lookups will "
            f"return the wrong row. Do not use this index."
        )
    elif aligned:
        _log.info(f"[knn.build] index/labels aligned at {n_index:,} rows")
    # nlist sanity. The default derives from ivf_train_size, NOT corpus
    # size, so a big corpus with the default training set silently gets
    # far too few cells — v1 ended up with nlist=316 for 116M rows
    # (~368K vectors per cell, so nprobe=64 scanned ~20% of the index:
    # slow, and the coarse quantizer barely partitions anything).
    # sqrt(n_rows) is the usual rule of thumb.
    if ivf is not None and n_rows > 0 and ivf_nlist:
        ideal = int(n_rows ** 0.5)
        if ivf_nlist < ideal / 4:
            _log.warning(
                f"[knn.build] nlist={ivf_nlist} is small for {n_rows:,} rows "
                f"(~sqrt(n) = {ideal:,}); each cell holds ~"
                f"{n_rows // max(1, ivf_nlist):,} vectors. Searches will be "
                f"slow and recall poor. Pass --ivf-nlist explicitly."
            )
    _log.info(
        f"[knn.build] done  rows={n_rows:,}  "
        f"kingdoms={dict(by_kingdom)}  elapsed={elapsed:.1f}s"
    )
    return {
        "n_rows": n_rows,
        "dim": EMB_DIM,
        "by_kingdom": dict(by_kingdom),
        "elapsed_seconds": elapsed,
        "version": version_tag,
        "index_type": index_type,
        "ivf_nlist": ivf_nlist if ivf is not None else None,
        "ivf_nprobe": ivf.nprobe if ivf is not None else None,
        # Segmentation bookkeeping — the caller driving a multi-segment
        # build loops until `complete` is True.
        "next_offset_rows": consumed_rows,
        "segment_rows": segment_rows,
        "complete": not stopped_early,
        "health": health,
        "n_index": n_index,
        "n_labels": n_labels,
        "aligned": aligned,
    }


def _probe_directions(index, n_probes: int) -> np.ndarray:
    """Unit vectors to probe `index` with.

    Prefer directions drawn from the index's own data: a random Gaussian
    vector in 768-d is nearly orthogonal to every stored vector, so it
    scores low even against a badly-scaled index (the real v1 index
    returned only 1.2 on random probes but 20.7 on a real embedding).
    An IVF coarse centroid points into an actual cluster, which makes
    the signal ~20x stronger and the check correspondingly harder to
    slip past.
    """
    d = index.d
    try:
        ivf = faiss.extract_index_ivf(index)
        quantizer = ivf.quantizer
        n = min(n_probes, quantizer.ntotal)
        if n > 0:
            rows = np.stack([quantizer.reconstruct(i) for i in range(n)])
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return np.ascontiguousarray(rows / norms, dtype=np.float32)
    except Exception:
        pass  # not an IVF index, or reconstruct unsupported — fall through
    try:
        n = min(n_probes, index.ntotal)
        if n > 0:
            rows = np.stack([index.reconstruct(i) for i in range(n)])
            norms = np.linalg.norm(rows, axis=1, keepdims=True)
            norms[norms == 0] = 1.0
            return np.ascontiguousarray(rows / norms, dtype=np.float32)
    except Exception:
        pass
    rng = np.random.default_rng(0)
    q = rng.standard_normal((n_probes, d)).astype(np.float32)
    return np.ascontiguousarray(q / np.linalg.norm(q, axis=1, keepdims=True))


def validate_index(index, *, n_probes: int = 3) -> dict:
    """Sanity-check that `index` actually holds L2-normalized vectors.

    The index is inner-product over unit vectors, so `IP(q, ref) <= 1.0`
    for any unit query `q` — by Cauchy-Schwarz, exactly. A top score
    meaningfully above 1.0 is therefore impossible for a healthy index
    and means the stored vectors are not unit-norm.

    This exists because the v1 index shipped in that state for months
    and nothing noticed: it was TRAINED on raw TreeOfLife-200M vectors
    (norm ~40-57) but had NORMALIZED vectors added to it. With
    IndexIVFPQ's default by_residual=True the residuals degenerated to
    ~-centroid, the PQ codebooks were fit at the wrong scale, and every
    entry decoded to roughly its coarse centroid. Queries returned
    near-identical scores (~20.7) for unrelated species — plausible
    output, entirely arbitrary ranking. A one-line assertion at load
    time would have caught it on day one.

    Returns {"ok", "max_sim", "reason"}. Never raises: a validation
    failure must degrade to a loud warning, not take the server down.
    """
    try:
        q = _probe_directions(index, n_probes)
        D, _ = index.search(q, 1)
        max_sim = float(np.max(D)) if D.size else 0.0
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": None, "max_sim": None, "reason": f"probe failed: {e}"}

    # 1.01 not 1.0: PQ reconstruction error can nudge a near-duplicate
    # fractionally over unity. The failure mode we're catching overshoots
    # by 20x, so the exact threshold is not delicate.
    if max_sim > 1.01:
        return {
            "ok": False,
            "max_sim": max_sim,
            "reason": (
                f"top inner-product {max_sim:.3f} exceeds 1.0 for a unit "
                f"query — the index does not contain L2-normalized "
                f"vectors. Its neighbours are unreliable; rebuild it."
            ),
        }
    return {"ok": True, "max_sim": max_sim, "reason": ""}


def _read_version(index_dir: Path) -> dict:
    """Parse INDEX_VERSION.txt into a dict for logging/debugging."""
    p = Path(index_dir) / "INDEX_VERSION.txt"
    if not p.exists():
        return {"version": "<unknown>", "n_rows": "?"}
    out: dict[str, str] = {}
    for line in p.read_text().splitlines():
        if "=" in line:
            k, v = line.split("=", 1)
            out[k.strip()] = v.strip()
        elif out.get("version") is None and line.strip():
            out["version"] = line.strip()
    return out


def _load_index(
    index_dir: Path,
) -> tuple[faiss.Index, pa.Table, dict]:
    """Load FAISS index, labels.parquet, and INDEX_VERSION.txt. Caller
    owns the returned objects; we don't cache module-level because the
    pipeline / web server share one index per process but a CLI run
    shouldn't keep it warm across invocations."""
    index_dir = Path(index_dir)
    index_path = index_dir / "tol200m_bioclip2.faiss"
    labels_path = index_dir / "labels.parquet"
    if not index_path.exists():
        raise FileNotFoundError(
            f"FAISS index not found at {index_path}. "
            f"Build it first with build_faiss_index(...)."
        )
    if not labels_path.exists():
        raise FileNotFoundError(
            f"labels.parquet missing at {labels_path}; "
            f"rebuild the index."
        )
    index = faiss.read_index(str(index_path))
    # Load the labels parquet LAZILY: read only the schema + row count,
    # NOT the actual row data (which is 3 GB for 91M rows and would OOM
    # the 29 GB host alongside the 2 GB faiss index + uvicorn base).
    # classify_via_knn uses `labels.take(pa.array(I.flat))` to fetch
    # only the matched neighbor rows — pyarrow's row-group statistics
    # + Parquet's column projection make this O(matched) memory, not
    # O(table). We implement take() by opening the file with
    # ParquetFile + reading only the row groups containing matched
    # indices via the row-group index. This way we never read all 91M
    # rows into RAM.
    meta = _read_version(index_dir)
    _labels_md = pq.read_metadata(labels_path)

    class _LazyLabelsTable:
        """Proxy over a parquet file that materializes the underlying
        table only on full-table access. `take(indices)` is the hot
        path for KNN queries — it reads only the row groups whose
        index range intersects the matched indices, then assembles a
        small Table with just those rows.

        For a k=10 query against 91M rows, take() touches at most
        ceil(10 / rows_per_row_group) row groups. With 25K rows per
        group (default Parquet) that's 1 row group → a few hundred KB
        of data. Compared to the naive `pq.read_table(memory_map=True)`
        which materializes the full 3 GB table at first .take(), this
        proxy keeps RSS bounded.
        """
        def __init__(self, path, metadata):
            self._path = path
            self._metadata = metadata
            self._cache: pa.Table | None = None
            # How many rows per row group (default 64K-1M for Parquet;
            # ours is ~43K based on row_groups=2171 for 91M rows).
            self._rows_per_rg = max(
                1,
                metadata.num_rows // max(1, metadata.num_row_groups),
            )

        def _materialize(self) -> pa.Table:
            if self._cache is None:
                self._cache = pq.read_table(self._path, memory_map=True)
            return self._cache

        def take(self, indices):
            # Convert pa.Array of int64 → numpy int64 array.
            if isinstance(indices, list):
                idx_np = np.asarray(indices, dtype="int64")
            elif hasattr(indices, "to_numpy"):
                idx_np = indices.to_numpy()
            else:
                idx_np = np.asarray(indices, dtype="int64")
            idx_np = idx_np.astype("int64", copy=False)
            if len(idx_np) == 0:
                return pa.table({c.name: [] for c in self._metadata.schema})
            # Compute which row groups are needed. For each index, the
            # containing row group is idx // rows_per_rg. Gather the
            # unique set of row groups, read them, then concat the
            # matching slices.
            rg_idx = (idx_np // self._rows_per_rg).tolist()
            unique_rgs = sorted(set(rg_idx))
            pf = pq.ParquetFile(self._path)
            pieces: list[pa.Table] = []
            for rg in unique_rgs:
                rg_table = pf.read_row_group(rg)
                # The row group's rows are [rg * RPR, (rg+1) * RPR).
                rg_lo = rg * self._rows_per_rg
                rg_hi = rg_lo + rg_table.num_rows
                # Filter indices that fall in this row group.
                rg_idx_local = [int(i) - rg_lo for i in idx_np if rg_lo <= int(i) < rg_hi]
                if rg_idx_local:
                    pieces.append(rg_table.take(pa.array(rg_idx_local)))
            if not pieces:
                return pa.table({c.name: [] for c in self._metadata.schema})
            return pa.concat_tables(pieces)

        def column(self, name):
            return self._materialize().column(name)

        @property
        def num_rows(self):
            return self._metadata.num_rows

        @property
        def schema(self):
            return self._metadata.schema

        def slice(self, *args, **kwargs):
            return self._materialize().slice(*args, **kwargs)

        def __getattr__(self, name):
            return getattr(self._materialize(), name)

    labels = _LazyLabelsTable(labels_path, _labels_md)
    # Health check on every load. Cheap (3 single-NN probes) relative to
    # reading a multi-GB index off disk, and it runs exactly where a
    # corrupt index would otherwise start silently returning nonsense.
    health = validate_index(index)
    # Alignment check: classify_via_knn maps FAISS ids straight into
    # labels.parquet row offsets (`labels.take(I)`), so the two files
    # MUST have the same row count and the same row order. If the index
    # has more rows than the labels table, every neighbor id past the end
    # is either an error or — worse — silently the wrong species.
    #
    # This has happened: the v1 build ended up with ntotal=117,070,179
    # against labels.num_rows=116,218,519 (851,660 Animalia rows leaked
    # into the tail) and had to be repaired by hand with a one-off
    # /tmp/trim_index.py. Nothing in the library noticed. Now it does.
    n_index = int(index.ntotal)
    n_labels = int(_labels_md.num_rows)
    if n_index != n_labels:
        health = {
            "ok": False,
            "max_sim": health.get("max_sim"),
            "reason": (
                f"index has {n_index:,} vectors but labels.parquet has "
                f"{n_labels:,} rows — FAISS ids are used directly as label "
                f"row offsets, so species names are misaligned. Rebuild."
            ),
        }
    meta["health"] = health
    meta["n_index"] = n_index
    meta["n_labels"] = n_labels
    if health["ok"] is False:
        _log.error("[knn] INDEX FAILED VALIDATION: %s", health["reason"])
    elif health["ok"] is None:
        _log.warning("[knn] index validation inconclusive: %s", health["reason"])
    else:
        _log.info(
            "[knn] index validation ok (%s vectors, max unit-query sim %.4f)",
            f"{n_index:,}", health["max_sim"],
        )
    return index, labels, meta


def classify_via_knn(
    *,
    query_embeddings: np.ndarray,
    index_dir: Path,
    k: int = 10,
    batch_size: int = 1024,
) -> list[dict]:
    """Find the K nearest reference embeddings per query row and
    majority-vote on species (weighted by cosine similarity).

    `query_embeddings` must be (N, EMB_DIM) float32, already L2-normalized
    (BioCLIP-2 output is normalized by default via
    `create_image_features(..., normalize=True)`). We re-normalize here
    defensively.

    Returns one dict per query row:
      {
        "top_species":    str | None,
        "common_name":    str,
        "score":          float,        # weighted vote mass for top species
        "kingdom":        str,
        "family":         str,
        "genus":          str,
        "neighbors":      [{"uuid", "species", "common_name", "sim",
                            "identifier", "source_dataset"}, ...]  # length k
      }

    `identifier` is the source image URL of the reference photo (gbif /
    eol media URL); None for rows that have none (bioscan).
    """
    index, labels, meta = _load_index(index_dir)
    if query_embeddings.ndim != 2 or query_embeddings.shape[1] != EMB_DIM:
        raise ValueError(
            f"query_embeddings must be (N, {EMB_DIM}); got {query_embeddings.shape}"
        )
    if query_embeddings.shape[0] == 0:
        return []

    n_queries = query_embeddings.shape[0]
    k = min(k, index.ntotal)

    # Normalize defensively. If input is already normalized this is a
    # no-op (norm ≈ 1.0, division is identity within float epsilon).
    q = np.ascontiguousarray(query_embeddings, dtype=np.float32)
    norms = np.linalg.norm(q, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    q = q / norms

    # Batch search for throughput. faiss.search on (1024, 768) is much
    # faster per-query than (1, 768) due to kernel amortization.
    # Output: D (n_queries, k) sim scores, I (n_queries, k) indices.
    D = np.empty((n_queries, k), dtype=np.float32)
    I = np.empty((n_queries, k), dtype=np.int64)
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        d, i = index.search(q[start:end], k)
        D[start:end] = d
        I[start:end] = i

    # Pull labels for the matched indices in one vectorized slice.
    # labels.to_pydict() would copy everything; we want only the matched
    # rows. labels.take(pa.array(I.flat)) is O(matched) and avoids the
    # full-table copy.
    flat_idx = pa.array(I.reshape(-1))
    matched = labels.take(flat_idx)
    # Convert columns we need to python lists, once. Column order matches
    # LABEL_COLUMNS.
    col_idx = {c: matched.column(c).to_pylist() for c in LABEL_COLUMNS}
    n_matched = len(col_idx["uuid"])

    out: list[dict] = []
    for qi in range(n_queries):
        row_offsets = qi * k
        sims = D[qi]
        neighbor_dicts: list[dict] = []
        species_scores: dict[str, float] = {}
        species_meta: dict[str, dict] = {}
        for j in range(k):
            ref_idx = row_offsets + j
            sp = col_idx["species"][ref_idx]
            sim = float(sims[j])
            if sp is None or sp == "":
                continue
            species_scores[sp] = species_scores.get(sp, 0.0) + sim
            if sp not in species_meta:
                species_meta[sp] = {
                    "kingdom": col_idx["kingdom"][ref_idx] or "",
                    "family": col_idx["family"][ref_idx] or "",
                    "genus": col_idx["genus"][ref_idx] or "",
                    "common_name": col_idx["common_name"][ref_idx] or "",
                }
            neighbor_dicts.append(
                {
                    "uuid": col_idx["uuid"][ref_idx],
                    "species": sp,
                    "common_name": col_idx["common_name"][ref_idx] or "",
                    "sim": sim,
                    # Source image URL for the reference photo. Non-null
                    # for gbif + eol rows (~99.9% of the index); null for
                    # bioscan. The UI renders it as a thumbnail through
                    # /api/knn/ref-thumb rather than hotlinking.
                    "identifier": col_idx["identifier"][ref_idx] or None,
                    "source_dataset": col_idx["source_dataset"][ref_idx] or "",
                }
            )

        if not species_scores:
            out.append(
                {
                    "top_species": None,
                    "common_name": "",
                    "score": 0.0,
                    "kingdom": "",
                    "family": "",
                    "genus": "",
                    "neighbors": neighbor_dicts,
                }
            )
            continue

        # Pick the species with the highest weighted vote mass. Tie-break
        # by which species appeared at the closest neighbor rank.
        top_species = max(
            species_scores.keys(),
            key=lambda s: (
                species_scores[s],
                -next(
                    j for j, n in enumerate(neighbor_dicts)
                    if n["species"] == s
                ),
            ),
        )
        meta0 = species_meta[top_species]
        total_mass = sum(species_scores.values()) or 1.0
        out.append(
            {
                "top_species": top_species,
                "common_name": meta0["common_name"],
                "score": species_scores[top_species] / total_mass,
                "kingdom": meta0["kingdom"],
                "family": meta0["family"],
                "genus": meta0["genus"],
                "neighbors": neighbor_dicts,
            }
        )

    _log.info(
        f"[knn.query] k={k}  queries={n_queries}  index_ntotal={index.ntotal}  "
        f"version={meta.get('version', '?')}"
    )
    return out


def _rss_watchdog(env_cap_str: str, stop_event: threading.Event) -> None:
    """In-process RSS watchdog. Polls /proc/self/status every 5s and
    `os._exit(137)` if RSS exceeds the cap. RLIMIT_DATA alone is
    not enough — it caps the heap, but the kernel OOM killer will
    still SIGKILL the build on a memory-tight host, taking the
    parent shell with it. This gives the build a clean failure
    instead."""
    s = env_cap_str.strip().lower()
    if s in ("", "0", "unlimited"):
        return  # unlimited — nothing to watch
    try:
        cap_bytes = int(s)
    except ValueError:
        return
    cap_mb = cap_bytes / 1024**2
    while not stop_event.is_set():
        stop_event.wait(timeout=5.0)
        if stop_event.is_set():
            return
        try:
            with open("/proc/self/status") as f:
                for line in f:
                    if line.startswith("VmRSS:"):
                        rss_mb = int(line.split()[1]) / 1024
                        if rss_mb > cap_mb:
                            _log.error(
                                f"[knn.build] RSS {rss_mb:.0f} MB exceeds "
                                f"cap {cap_mb:.0f} MB; exiting cleanly to "
                                f"avoid OOM-kill by the kernel."
                            )
                            _os._exit(137)
                        break
        except (FileNotFoundError, ProcessLookupError):
            return


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
        "--index-type",
        choices=["flat", "ivf"],
        default="ivf",
        help="flat = exact (needs full matrix in RAM, ~3 KB x n_rows). "
             "ivf = approximate Voronoi-quantized (default; ~10x less RAM, "
             "~90-95%% recall at nprobe=32).",
    )
    ap.add_argument(
        "--ivf-nlist",
        type=int,
        default=None,
        help="IVF coarse-quantizer cell count. Defaults to sqrt(n_rows).",
    )
    ap.add_argument(
        "--ivf-train-size",
        type=int,
        default=200_000,
        help="Vectors to use for IVF k-means training (default 200K — "
             "600MB training buffer; the 1M default exceeds the 12GB "
             "RLIMIT_DATA cap on this host).",
    )
    ap.add_argument(
        "--resume",
        action="store_true",
        help="If an index already exists at <index-dir>/tol200m_bioclip2.faiss, "
             "load it and continue adding instead of starting from scratch. "
             "Used for multi-pass builds that exceed RLIMIT_DATA: run build #1 "
             "fresh, then build #2 #3 with --resume on the same dir.",
    )
    ap.add_argument(
        "--max-segment-rows",
        type=int,
        default=None,
        help="Stop after adding roughly this many source rows and record "
             "the cursor in <index-dir>/state.json, so the next --resume "
             "run continues from there instead of re-adding from row 0. "
             "Use for multi-pass builds that exceed the memory budget.",
    )
    args = ap.parse_args()
    import resource
    _env_cap = _os.environ.get("_KNN_BUILD_MEM_LIMIT_BYTES", "").strip().lower()
    if _env_cap not in ("", "0", "unlimited"):
        try:
            resource.setrlimit(
                resource.RLIMIT_DATA,
                (int(_env_cap), int(_env_cap)),
            )
        except (ValueError, OSError) as _e:
            _log.warning(f"[knn.build] could not set RLIMIT_DATA: {_e}")
    elif _env_cap == "":
        # CLI default: 8 GB. The build's working set is ~5 GB (1.8 GB
        # IVF training + 2 GB labels in flight + 1 GB index + heap), and
        # 8 GB leaves room for both pyarrow's mmap'd parquet and any
        # short-lived intermediates without taking down uvicorn or the
        # other dev services on this host.
        try:
            resource.setrlimit(
                resource.RLIMIT_DATA, (8 * 1024**3, 8 * 1024**3)
            )
        except (ValueError, OSError) as _e:
            _log.warning(f"[knn.build] could not set default RLIMIT_DATA: {_e}")

    # Surface the actual rlimit in the log so an OOM in the field is
    # easy to diagnose — `RLIMIT_DATA` (12 GB on this host) is what the
    # `app.py` start_knn_build sets via preexec_fn. If you're running
    # the CLI standalone, this just prints whatever your shell set.
    import resource
    try:
        data_limit = resource.getrlimit(resource.RLIMIT_DATA)
        _log.info(
            f"[knn.build] RLIMIT_DATA = {data_limit[0] / 1024**3:.1f} GB "
            f"(hard={data_limit[1] / 1024**3:.1f} GB) — OOM-kill at the soft cap"
        )
    except (ValueError, OSError):
        _log.warning("[knn.build] could not read RLIMIT_DATA")

    # Start the in-process RSS watchdog AFTER rlimit is set so it
    # watches the right cap. RLIMIT_DATA only caps the heap; the
    # kernel OOM killer will still SIGKILL the build on a memory-
    # tight host, taking the parent shell with it. The watchdog
    # polls /proc/self/status every 5s and exits cleanly (status
    # 137) if RSS exceeds the cap, so we fail loudly instead of
    # being torn down silently.
    _watchdog_stop = threading.Event()
    threading.Thread(
        target=_rss_watchdog,
        args=(_os.environ.get("_KNN_BUILD_MEM_LIMIT_BYTES", ""), _watchdog_stop),
        daemon=True,
        name="knn-rss-watchdog",
    ).start()

    try:
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
            index_type=args.index_type,
            ivf_nlist=args.ivf_nlist,
            ivf_train_size=args.ivf_train_size,
            resume=args.resume,
            max_segment_rows=args.max_segment_rows,
        )
        print(json.dumps(stats, indent=2, default=str))
        return 0
    finally:
        _watchdog_stop.set()


if __name__ == "__main__":
    raise SystemExit(main())
