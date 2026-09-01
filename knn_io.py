"""Read and query a FAISS TreeOfLife-200M-Embeddings index.

Two index layouts are supported:

1. **BotaniSense-built**: `tol200m_bioclip2.faiss` + `labels.parquet` +
   `INDEX_VERSION.txt` in a directory. L2-normalized vectors, METRIC_IP.
   The returned scores are already cosine.
2. **Official** (imageomics/bioclip-image-search-lite): `faiss/index.index`
   + `duckdb/metadata.duckdb`. L2-normalized vectors, METRIC_L2. The
   returned scores are squared L2 distances: d = ||q − x||² = 2 − 2·cos,
   so we convert with cos = 1 − d/2 before ranking/voting. PQ makes d a
   slight overestimate, so the converted value is a lower bound on the
   true cosine (measured: d=0.204 → implied 0.898 vs re-encoded 0.938).

Detection is automatic from which files are present in the dir.
"""


from __future__ import annotations

import functools
import json
import logging
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


def _search_st(index, q: np.ndarray, k: int):
    """index.search with OMP pinned to 1 thread, restored after.

    Batched IndexIVFPQ.search with >=2 OpenMP threads over the mmap'd
    official index returns corrupt results: negative squared distances
    (d = -0.014, impossible for L2) and wrong neighbors (a plant query
    "matching" brittle stars at implied cos = 1.0). Single-query search
    is unaffected, and single-threaded batch search is bit-identical to
    per-row search — the index data is fine, the parallel scan is not.

    nprobe=1 keeps single-query latency ~40ms, so serializing costs
    ~40ms per reclassified asset — negligible at our batch sizes.
    """
    prev = faiss.omp_get_max_threads()
    faiss.omp_set_num_threads(1)
    try:
        return index.search(q, k)
    finally:
        faiss.omp_set_num_threads(prev)


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


def validate_index(index, *, n_probes: int = 3, normalized: bool = True) -> dict:
    """Sanity-check the loaded index.

    For normalized indexes (BotaniSense-built), `IP(q, ref) <= 1.0` for
    any unit query `q` by Cauchy-Schwarz; we probe with unit vectors
    derived from IVF coarse centroids (random Gaussian directions are
    nearly orthogonal to all stored 768-d vectors and miss real bugs).

    For L2-metric indexes (imageomics official), scores are squared
    distances, not inner products — they're unbounded above, so the
    ≤1.0 assertion doesn't apply. Setting `normalized=False` skips it
    and only checks that the probe returns finite, varying scores (not
    all-identical, which was the v1 symptom).

    Returns {"ok", "max_sim", "reason"}. Never raises: a validation
    failure must degrade to a loud warning, not take the server down.
    """
    try:
        q = _probe_directions(index, n_probes)
        D, _ = _search_st(index, q, 1)
        max_sim = float(np.max(D)) if D.size else 0.0
    except Exception as e:  # pragma: no cover - defensive
        return {"ok": None, "max_sim": None, "reason": f"probe failed: {e}"}

    if not normalized:
        # Official index: METRIC_L2, so scores are squared distances
        # (unbounded). We only sanity-check that scores vary across
        # probes — the v1 bug was byte-identical 20.7 returns; that's
        # caught here.
        if D.size >= 2 and np.ptp(D) < 1e-3:
            return {
                "ok": False,
                "max_sim": max_sim,
                "reason": (
                    f"top inner-product {max_sim:.3f} has near-zero range "
                    f"across probes — index may be returning identical "
                    f"scores (v1 corruption symptom)."
                ),
            }
        return {"ok": True, "max_sim": max_sim, "reason": ""}

    # Normalized layout: max sim should be ≤ 1.0 by Cauchy-Schwarz.
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
) -> tuple[faiss.Index, "object", dict]:
    """Load FAISS index + label source. Returns (faiss_index, labels_proxy, meta).

    Two layouts supported:
      - BotaniSense-built:  faiss/index from `tol200m_bioclip2.faiss`,
                            labels from `labels.parquet`
      - Official imageomics: faiss/index from `faiss/index.index`,
                            labels from `duckdb/metadata.duckdb`

    `labels_proxy` is a duck-typed table-like object that supports
    `.take(indices_array)` -> pa.Table. The two layout paths return
    different proxy classes; consumers use `.take()` only.
    """
    index_dir = Path(index_dir)
    # Detect which layout is on disk.
    custom_index = index_dir / "tol200m_bioclip2.faiss"
    custom_labels = index_dir / "labels.parquet"
    official_index = index_dir / "faiss" / "index.index"
    official_duckdb = index_dir / "duckdb" / "metadata.duckdb"

    if custom_index.exists() and custom_labels.exists():
        return _load_custom_index(custom_index, custom_labels, index_dir)
    if official_index.exists() and official_duckdb.exists():
        return _load_official_index(official_index, official_duckdb, index_dir)
    raise FileNotFoundError(
        f"No recognised index layout in {index_dir}: need either "
        f"(tol200m_bioclip2.faiss + labels.parquet) or "
        f"(faiss/index.index + duckdb/metadata.duckdb)"
    )


def _load_custom_index(index_path, labels_path, index_dir):
    """BotaniSense-built layout: L2-normalized FAISS + labels.parquet."""
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
    return _finalize_load(index, labels, int(_labels_md.num_rows), meta)


@functools.lru_cache(maxsize=4)
def _load_official_url_prefixes(duckdb_path: str) -> dict[int, str]:
    """Read url_prefixes table once and cache by duckdb path.

    411 rows total, so the dict is tiny (~50 KB). Cached because every
    classify_via_knn call would otherwise re-connect to DuckDB and refetch.
    """
    import duckdb
    conn = duckdb.connect(duckdb_path, read_only=True)
    rows = conn.execute("SELECT prefix_id, prefix FROM url_prefixes").fetchall()
    conn.close()
    return {int(pid): str(prefix) for pid, prefix in rows}


def _load_official_index(index_path, duckdb_path, index_dir):
    """imageomics official index layout: faiss/index.index + duckdb/metadata.duckdb.

    Differences from custom:
      - metric is METRIC_L2 (not IP): returned distances are squared L2
        over unit vectors, d = 2 − 2·cos → cos = 1 − d/2 (converted in
        classify_via_knn). Vectors themselves ARE L2-normalized — the
        official build uses create_image_features(normalize=True).
      - id 0..N-1 maps directly to metadata.id (verified at load time)
      - URL = url_prefixes.prefix[url_prefix_id] + identifier_suffix
      - no 'scientific_name' column
    """
    if not index_path.exists():
        raise FileNotFoundError(f"FAISS index not found at {index_path}")
    if not duckdb_path.exists():
        raise FileNotFoundError(f"DuckDB metadata not found at {duckdb_path}")

    # Mmap the FAISS index. ~5.8 GB; with IO_FLAG_MMAP the OS pages it in
    # on demand instead of loading all of it into RSS.
    index = faiss.read_index(str(index_path), faiss.IO_FLAG_MMAP)
    meta = {
        "version": "imageomics-official",
        "index_path": str(index_path),
        "duckdb_path": str(duckdb_path),
    }

    # Lazy DuckDB labels proxy. Uses DuckDB's native fast take().
    class _DuckDBLabelsTable:
        def __init__(self, db_path, n_rows):
            import duckdb
            self._path = db_path
            self._n_rows = n_rows
            self._conn = None

        def _conn_lazy(self):
            if self._conn is None:
                import duckdb
                # read_only=True so we don't accidentally write, and so
                # multiple processes can attach without lock contention.
                self._conn = duckdb.connect(self._path, read_only=True)
            return self._conn

        @property
        def num_rows(self): return self._n_rows

        @property
        def schema(self):
            return self._conn_lazy().execute(
                "SELECT column_name, data_type FROM information_schema.columns "
                "WHERE table_name='metadata' ORDER BY ordinal_position"
            ).fetchall()

        def take(self, indices):
            # Convert pa.Array or list to Python list of ints.
            if hasattr(indices, "to_pylist"):
                idx_list = indices.to_pylist()
            elif isinstance(indices, list):
                idx_list = [int(i) for i in indices]
            else:
                idx_list = [int(i) for i in indices]
            if not idx_list:
                return pa.table({})
            conn = self._conn_lazy()
            # Inline the IDs into the SQL — DuckDB parameterizes lists
            # fine but a VALUES clause is faster for big batches.
            ids_sql = ",".join(str(int(i)) for i in idx_list)
            rows = conn.execute(
                f"SELECT id, uuid, kingdom, phylum, \"class\", \"order\", "
                f"family, genus, species, common_name, "
                f"source_dataset, url_prefix_id, identifier_suffix "
                f"FROM metadata WHERE id IN ({ids_sql})"
            ).fetchall()
            cols = ["uuid", "kingdom", "phylum", "class", "order", "family",
                    "genus", "species", "common_name", "source_dataset",
                    "url_prefix_id", "identifier_suffix"]
            # id column dropped; rows are in arbitrary order from DuckDB,
            # so re-order by the requested ids.
            by_id = {int(r[0]): r[1:] for r in rows}
            data = {c: [] for c in cols}
            for i in idx_list:
                r = by_id.get(int(i))
                if r is None:
                    for c in cols: data[c].append(None)
                else:
                    for c, v in zip(cols, r):
                        data[c].append(v)
            # `class` is a Python keyword — rename in the result.
            result = {c: data[c] for c in cols}
            return pa.table(result)

        def column(self, name):
            # Full-materialize — used only for diagnostics.
            conn = self._conn_lazy()
            rows = conn.execute(f"SELECT {name} FROM metadata").fetchall()
            return pa.array([r[0] for r in rows])

        def __getattr__(self, name):
            # Anything else just opens + delegates.
            return getattr(self._conn_lazy(), name)

    # The id-to-row mapping is identity (id == row offset) for the
    # official DuckDB: rows are inserted in id order. We still sanity
    # check at load time via a single count query, not a full scan.
    import duckdb
    n_meta = duckdb.connect(str(duckdb_path), read_only=True).execute(
        "SELECT max(id) + 1 FROM metadata"
    ).fetchone()[0]

    labels = _DuckDBLabelsTable(str(duckdb_path), int(n_meta))
    return _finalize_load(index, labels, int(n_meta), meta, normalized=False)


def _finalize_load(index, labels, n_labels, meta, normalized: bool = True):
    """Shared post-load: health probe + alignment check + log + return."""
    health = validate_index(index)
    n_index = int(index.ntotal)
    if n_index != n_labels:
        health = {
            "ok": False,
            "max_sim": health.get("max_sim"),
            "reason": (
                f"index has {n_index:,} vectors but labels has "
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
            "[knn] index validation ok (%s vectors, max probe sim %.4f)",
            f"{n_index:,}", health["max_sim"] if health["max_sim"] is not None else 0.0,
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
    is_official = meta.get("version") == "imageomics-official"
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

    # Official layout: METRIC_L2 over L2-normalized vectors. The scores
    # come back as squared L2 distances, best-first (ascending):
    #   d = ||q − x||² = |q|² + |x|² − 2·q·x = 2 − 2·cos   (unit vectors)
    # so cosine = 1 − d/2. PQ reconstruction inflates d slightly, making
    # the converted value a small underestimate of the true cosine
    # (measured: d=0.204 → 0.898 implied vs 0.938 re-encoded ground truth).
    #
    # Custom layout: METRIC_IP over L2-normalized vectors — the returned
    # scores are already cosine. In both cases the query is unit-norm
    # (matching the stored vectors); for L2 this matters doubly since L2
    # is not scale-invariant — a raw-norm query would route to different
    # IVF cells entirely.
    # Batch search for throughput. faiss.search on (1024, 768) is much
    # faster per-query than (1, 768) due to kernel amortization.
    # Output: D (n_queries, k) sim scores, I (n_queries, k) indices.
    D = np.empty((n_queries, k), dtype=np.float32)
    I = np.empty((n_queries, k), dtype=np.int64)
    for start in range(0, n_queries, batch_size):
        end = min(start + batch_size, n_queries)
        d, i = _search_st(index, q[start:end], k)
        D[start:end] = d
        I[start:end] = i
    if is_official:
        # Squared-L2 → cosine. Clip guards the degenerate d≈0 → cos≈1
        # float edge; values stay in [-1, 1] otherwise.
        D = np.clip(1.0 - D / 2.0, -1.0, 1.0)

    # Pull labels for the matched indices.
    flat_idx = pa.array(I.reshape(-1))
    matched = labels.take(flat_idx)

    # The label-schema columns differ between layouts:
    #   custom:   uuid, kingdom..species, scientific_name, common_name,
    #             identifier, source_dataset
    #   official: uuid, kingdom..species, common_name, source_dataset,
    #             url_prefix_id, identifier_suffix
    if is_official:
        cols = {
            # DuckDB returns uuid as a Python uuid.UUID object (custom
            # layout returns str). Coerce to str here so downstream
            # json.dumps works without modification.
            "uuid": [str(u) if u is not None else None for u in matched.column("uuid").to_pylist()],
            "kingdom": matched.column("kingdom").to_pylist(),
            "phylum": matched.column("phylum").to_pylist(),
            "class": matched.column("class").to_pylist(),
            "order": matched.column("order").to_pylist(),
            "family": matched.column("family").to_pylist(),
            "genus": matched.column("genus").to_pylist(),
            "species": matched.column("species").to_pylist(),
            "common_name": matched.column("common_name").to_pylist(),
            "source_dataset": matched.column("source_dataset").to_pylist(),
            "prefix_id": matched.column("url_prefix_id").to_pylist(),
            "identifier_suffix": matched.column("identifier_suffix").to_pylist(),
        }
        url_prefixes = _load_official_url_prefixes(meta["duckdb_path"])
        col_idx = cols
        col_idx["identifier"] = [
            (url_prefixes.get(pid, "") + suf) if pid is not None and suf else None
            for pid, suf in zip(col_idx["prefix_id"], col_idx["identifier_suffix"])
        ]
    else:
        cols = {c: matched.column(c).to_pylist() for c in LABEL_COLUMNS}
        col_idx = cols

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
                    # Source image URL. Custom layout: pre-built
                    # 'identifier' column. Official: reconstructed from
                    # url_prefixes.prefix[id] + identifier_suffix.
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

