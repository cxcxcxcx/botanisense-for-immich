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

Why duckdb at build time: the parquet files have ~16 columns. duckdb's
predicate pushdown (page indexes on kingdom/phylum/etc.) means we read
only the columns + row groups matching the WHERE clause during a subset
build (e.g. Plantae only). Without duckdb, pyarrow would materialize
every column for every row group.
"""

from __future__ import annotations

import json
import logging
import time
from collections import Counter
from pathlib import Path
from typing import Iterable

import duckdb
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
# How many source rows to pull per duckdb fetch + FAISS .add() during a
# build. 200K × 768 × 4 bytes = ~600 MB, well within the RAM budget on
# a single-machine build (we keep ~1 GB free for FAISS + label arrays).
BUILD_CHUNK_ROWS = 200_000


def _quoted_columns() -> str:
    """Return LABEL_COLUMNS + emb as a SELECT list. `class` and `order`
    are SQL reserved words so they must be double-quoted."""
    cols = []
    for c in LABEL_COLUMNS + ("emb",):
        if c in ("class", "order"):
            cols.append(f'"{c}"')
        else:
            cols.append(c)
    return ", ".join(cols)


def _duckdb_filter_sql(
    parquet_glob: str,
    *,
    kingdoms: Iterable[str] | None,
) -> tuple[str, list]:
    """Return (sql, params) selecting LABEL_COLUMNS + emb with optional
    kingdom filter. We select the emb column explicitly (rather than
    SELECT *) so the projection is columnar and only the embedding column
    + the requested labels are decoded from the row group."""
    params: list = []
    where = ""
    if kingdoms:
        placeholders = ", ".join("?" for _ in kingdoms)
        where = f"WHERE kingdom IN ({placeholders})"
        params = list(kingdoms)
    return (
        f"SELECT {_quoted_columns()} "
        f"FROM read_parquet('{parquet_glob}') "
        f"{where}",
        params,
    )


def build_faiss_index(
    *,
    parquet_glob: str,
    index_path: Path,
    labels_path: Path,
    kingdoms: Iterable[str] | None = None,
    version_tag: str,
    on_progress=None,
) -> dict:
    """Build a FAISS IndexFlatIP over the embeddings matching `parquet_glob`.

    The FAISS index is exact cosine (inner product on L2-normalized
    vectors — BioCLIP-2 features are normalized upstream by
    `create_image_features(..., normalize=True)`). Memory: ~700 MB for
    233M × 768 float32.

    `on_progress` is an optional callback (chunk_idx, total_chunks,
    rows_so_far) for the UI.
    """
    index_path = Path(index_path)
    labels_path = Path(labels_path)
    index_path.parent.mkdir(parents=True, exist_ok=True)
    labels_path.parent.mkdir(parents=True, exist_ok=True)

    sql, params = _duckdb_filter_sql(parquet_glob, kingdoms=kingdoms)
    con = duckdb.connect()
    # Streamed execution: rel.to_arrow() yields RecordBatches incrementally
    # instead of materializing all 350 GB at once. The first batch arrives
    # while the rest are still being decoded.
    rel = con.execute(sql, params)
    # Chunked fetch: 50K-row arrow batches (matches the parquet row
    # group size for efficient columnar decoding). We accumulate several
    # of these before adding to FAISS to amortize the .add() cost.
    # duckdb>=1.4 deprecated fetch_record_batch in favor of
    # fetch_arrow_reader(batch_size=). Try the new name first.
    stream = None
    for getter in (
        lambda: rel.fetch_arrow_reader(batch_size=50_000),
        lambda: rel.fetch_record_batch(rows_per_batch=50_000),
    ):
        try:
            stream = getter()
            break
        except (AttributeError, TypeError):
            continue
    if stream is None:
        raise RuntimeError(
            "duckdb.Relation exposes neither fetch_arrow_reader nor "
            "fetch_record_batch — please upgrade duckdb (>=0.8)."
        )

    index = faiss.IndexFlatIP(EMB_DIM)
    label_records: list[dict] = []
    n_rows = 0
    by_kingdom: Counter[str] = Counter()
    pending_labels: list[dict] = []
    pending_embs: list[np.ndarray] = []
    t0 = time.monotonic()
    chunk_idx = 0

    def _flush() -> None:
        nonlocal n_rows
        if not pending_embs:
            return
        mat = np.ascontiguousarray(np.vstack(pending_embs), dtype=np.float32)
        # BioCLIP-2 embeddings are already L2-normalized, but re-normalize
        # defensively in case a future source parquet ships unnormalized
        # vectors. Tiny cost (one norm + divide per row) and keeps the
        # contract "IndexFlatIP == cosine" honest.
        norms = np.linalg.norm(mat, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        mat = mat / norms
        index.add(mat)
        n_rows += mat.shape[0]
        label_records.extend(pending_labels)
        pending_labels.clear()
        pending_embs.clear()

    for batch in stream:
        # batch is a pyarrow.RecordBatch with our 13 columns. Convert
        # `emb` (list<float16>) → numpy float32 (N, 768) by going through
        # pyarrow's fixed-shape tensor view; if that's not available we
        # fall back to a numpy .to_pylist() stack.
        emb_col = batch.column(batch.schema.get_field_index("emb"))
        try:
            emb_arr = np.asarray(
                emb_col.cast(pa.list_(pa.float32())),
                dtype=np.float32,
            ).reshape(-1, EMB_DIM)
        except Exception:
            # Fallback for older pyarrow versions that don't allow the
            # list_ cast cheaply: go through python list. ~3x slower
            # but still works.
            raw = emb_col.to_pylist()
            emb_arr = np.asarray(raw, dtype=np.float32).reshape(-1, EMB_DIM)
        pending_embs.append(emb_arr)

        # Build label records from the same row order. Strings are kept
        # as Python objects (not numpy) to preserve None for NULL.
        chunk_labels = batch.to_pylist()
        # to_pylist returns one dict per row in row order, which matches
        # the emb_arr rows exactly. Drop the `emb` key to keep the
        # labels parquet compact.
        for row in chunk_labels:
            rec = {k: row.get(k) for k in LABEL_COLUMNS}
            pending_labels.append(rec)
            k = rec.get("kingdom")
            if k:
                by_kingdom[k] += 1

        if len(pending_labels) >= BUILD_CHUNK_ROWS:
            _flush()
            chunk_idx += 1
            _log.info(
                f"[knn.build] chunk {chunk_idx}  rows={n_rows:,}  "
                f"elapsed={time.monotonic() - t0:.1f}s"
            )
            if on_progress is not None:
                on_progress(chunk_idx, -1, n_rows)

    _flush()

    # Persist FAISS index. IndexFlatIP.write_index is binary-safe.
    faiss.write_index(index, str(index_path))

    # Persist labels in the SAME row order as the FAISS index. Parquet
    # with ZSTD compression: ~250 MB for 233M rows of short strings.
    label_table = pa.Table.from_pylist(label_records)
    pq.write_table(label_table, str(labels_path), compression="zstd")

    # Version file — read by classify_via_knn so the caller doesn't
    # have to pass version_tag explicitly.
    (index_path.parent / "INDEX_VERSION.txt").write_text(
        f"{version_tag}\n"
        f"parquet_glob={parquet_glob}\n"
        f"kingdoms={(list(kingdoms) if kingdoms else 'ALL')}\n"
        f"n_rows={n_rows}\n"
        f"dim={EMB_DIM}\n"
        f"built_at={time.strftime('%Y-%m-%dT%H:%M:%S')}\n"
    )

    elapsed = time.monotonic() - t0
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
    }


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
    # mmap the labels parquet so we don't pay the full RAM cost upfront.
    labels = pq.read_table(str(labels_path), memory_map=True)
    meta = _read_version(index_dir)
    return index, labels, meta


def _emb_column_to_numpy(
    emb_col: pa.ChunkedArray, n: int
) -> np.ndarray:
    """Convert a pyarrow chunked array of list<float16/32> into an
    (n, EMB_DIM) float32 contiguous numpy array."""
    emb_arr = np.empty((n, EMB_DIM), dtype=np.float32)
    offset = 0
    for chunk in emb_col.chunks:
        m = len(chunk)
        try:
            sub = np.asarray(
                chunk.cast(pa.list_(pa.float32())),
                dtype=np.float32,
            ).reshape(m, EMB_DIM)
        except Exception:
            sub = np.asarray(chunk.to_pylist(), dtype=np.float32).reshape(
                m, EMB_DIM
            )
        emb_arr[offset : offset + m] = sub
        offset += m
    return np.ascontiguousarray(emb_arr)


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
        "neighbors":      [{"uuid", "species", "common_name", "sim"}, ...]  # length k
      }
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
    args = ap.parse_args()

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
    )
    print(json.dumps(stats, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())