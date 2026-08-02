"""Shared pipeline: Immich smart-search → BioCLIP embed+predict →
DB → k-means cluster → YOLO-World detect.

Memory note: BioCLIP-2 (~340 MB weights) and YOLOv8s-world (~340 MB weights)
are both resident during the first run on uncached assets. To keep peak RAM
manageable on small machines, all heavy passes happen in chunks of
`batch_size` images. PIL decodes, embeddings, predictions, and detections
are flushed back to SQLite between chunks so we never hold the full
gallery in memory at once.
"""

from __future__ import annotations

import gc
import logging
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

# Heavy imports (torch/bioclip/sklearn-via-clusters/ultralytics-via-yolo)
# are deferred to run_pipeline() so that `import pipeline` stays cheap and
# the FastAPI browse-only server doesn't pay ~3 GB of model weights at
# startup. See run_pipeline() for the deferred-import block.

import storage
from immich_client import ImmichAsset, ImmichClient

# PEP 562 module-level __getattr__ — lazy-loads the heavy bits
# (torch/bioclip/sklearn-via-clusters/ultralytics-via-yolo) the first
# time anything looks them up by name. Keeps `import pipeline` cheap
# for the browse-only FastAPI server, while still allowing
# `pipeline.TreeOfLifeClassifier`, `pipeline.clusters`, `pipeline.yolo`,
# `pipeline.Rank` to work for both test patches (mock.patch.object) and
# the deferred body of run_pipeline().
_LAZY_ATTRS = ("TreeOfLifeClassifier", "Rank", "clusters", "yolo")


def __getattr__(name: str):  # PEP 562
    if name == "TreeOfLifeClassifier":
        from bioclip.predict import TreeOfLifeClassifier
        globals()["TreeOfLifeClassifier"] = TreeOfLifeClassifier
        return TreeOfLifeClassifier
    if name == "Rank":
        from bioclip import Rank
        globals()["Rank"] = Rank
        return Rank
    if name == "clusters":
        import clusters as _clusters
        globals()["clusters"] = _clusters
        return _clusters
    if name == "yolo":
        import yolo as _yolo
        globals()["yolo"] = _yolo
        return _yolo
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


THUMBNAIL_SIZE = "preview"
DEFAULT_BATCH_SIZE = 16
# YOLO bboxes with score >= this are cropped and re-classified with
# BioCLIP so we can see if a tight crop gets a higher-confidence species
# than the full image. Lower = more crops = more BioCLIP work.

# Per-stage progress logging. Goes to stderr so the user sees it in the
# server terminal. Format: HH:MM:SS [stage] message. We keep messages
# terse; the goal is to fill the silence between chunks of work.
_log = logging.getLogger("pipeline")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False


# Shared pipeline state. Thread-safe: the pipeline thread writes under
# _state_lock; the FastAPI thread reads via /api/status. Keys:
#   status:        "idle" | "running" | "done" | "error"
#   stage:         "" | "search" | "species" | "cluster" | "kingdom" | "yolo" | "bbox-bioclip" | "stitch"
#   chunk_idx/total:  progress within the current stage (1-based)
#   rate:          items/sec estimated for the current stage
#   eta_seconds:   0 if unknown
#   current_run_id: int | None
#   query, n_new, n_cached, n_requested, started_at, finished_at
#   last_error: str | None
import threading
_state_lock = threading.Lock()
_state: dict = {
    "status": "idle",
    "stage": "",
    "chunk_idx": 0,
    "chunk_total": 0,
    "rate": 0.0,
    "eta_seconds": 0.0,
    "current_run_id": None,
    "query": "",
    "n_new": 0,
    "n_cached": 0,
    "n_requested": 0,
    "started_at": None,
    "finished_at": None,
    "last_error": None,
}

# File log path — set by app.py on startup so the log lives next to the DB.
_log_file_path: Path | None = None
_file_handler: logging.FileHandler | None = None


def attach_log_file(path: Path) -> None:
    """Install a file handler so logs persist beyond the terminal scrollback.
    Safe to call multiple times — replaces any prior file handler."""
    global _log_file_path, _file_handler
    if _file_handler is not None:
        _log.removeHandler(_file_handler)
        _file_handler.close()
    _log_file_path = path
    path.parent.mkdir(parents=True, exist_ok=True)
    _file_handler = logging.FileHandler(path, encoding="utf-8")
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    _log.addHandler(_file_handler)


def reset_state() -> None:
    with _state_lock:
        _state.update(
            {
                "status": "running",
                "stage": "",
                "chunk_idx": 0,
                "chunk_total": 0,
                "rate": 0.0,
                "eta_seconds": 0.0,
                "current_run_id": None,
                "started_at": time.time(),
                "finished_at": None,
                "last_error": None,
            }
        )


def finish_state(*, run_id: int, n_new: int, n_cached: int,
                 n_requested: int, query: str) -> None:
    with _state_lock:
        _state.update(
            {
                "status": "done",
                "stage": "",
                "current_run_id": run_id,
                "n_new": n_new,
                "n_cached": n_cached,
                "n_requested": n_requested,
                "query": query,
                "finished_at": time.time(),
                "eta_seconds": 0.0,
            }
        )


def error_state(message: str) -> None:
    with _state_lock:
        _state.update(
            status="error", last_error=message, finished_at=time.time()
        )


def set_state(**updates) -> None:
    with _state_lock:
        _state.update(updates)


def get_state() -> dict:
    with _state_lock:
        return dict(_state)


def read_recent_log_lines(max_lines: int = 200) -> list[str]:
    """Return the last `max_lines` lines from the file log (or empty if
    no file handler is attached). Cheap O(n) tail — fine for a few hundred
    lines; the file is small in practice."""
    if _log_file_path is None or not _log_file_path.exists():
        return []
    try:
        # Read last ~64KB then split — adequate for the volumes we expect.
        size = _log_file_path.stat().st_size
        with _log_file_path.open("rb") as f:
            if size > 64 * 1024:
                f.seek(size - 64 * 1024)
            data = f.read().decode("utf-8", errors="replace")
        lines = data.splitlines()
        return lines[-max_lines:]
    except OSError:
        return []


def _fmt_eta(seconds: float) -> str:
    if seconds < 0 or seconds != seconds:  # NaN guard
        return "?"
    if seconds < 60:
        return f"{seconds:.0f}s"
    m, s = divmod(int(seconds), 60)
    if m < 60:
        return f"{m}m{s:02d}s"
    h, m = divmod(m, 60)
    return f"{h}h{m:02d}m"


def _chunk_log(
    stage: str,
    chunk_idx: int,
    total_chunks: int,
    *,
    n_items: int,
    elapsed: float,
    note: str = "",
) -> None:
    """Log a chunk progress line with ETA. Safe to call on every chunk
    (called once per chunk). chunk_idx is 1-based for human readability."""
    if total_chunks <= 0:
        return
    done = chunk_idx
    per_chunk = elapsed / done if done > 0 else 0.0
    remaining = max(0, total_chunks - done) * per_chunk
    rate = n_items / elapsed if elapsed > 0 else 0.0
    tail = f"  {note}" if note else ""
    _log.info(
        f"[{stage}] chunk {done}/{total_chunks}  items={n_items}  "
        f"rate={rate:.1f}/s  elapsed={_fmt_eta(elapsed)}  "
        f"eta={_fmt_eta(remaining)}{tail}"
    )
    # Mirror into the shared state so the UI can show the same numbers.
    cumulative_items = chunk_idx * max(1, n_items)  # rough; refined per-stage
    set_state(
        stage=stage,
        chunk_idx=chunk_idx,
        chunk_total=total_chunks,
        rate=rate,
        eta_seconds=remaining,
    )
BBOX_BIOCLIP_MIN = 0.15
# Top-K to ask BioCLIP for on each crop. Stored in full; UI shows 3.
BBOX_BIOCLIP_TOP_K = 5


@dataclass
class Prediction:
    species: str
    common_name: str
    score: float
    kingdom: str = ""
    family: str = ""
    genus: str = ""


@dataclass
class ImageResult:
    asset_id: str
    original_file_name: str
    original_url: str
    thumbnail_url: str
    cluster_id: int | None = None
    top_predictions: list[Prediction] = field(default_factory=list)
    detections: list[dict] = field(default_factory=list)
    taxon_scores: dict[str, float] = field(default_factory=dict)


@dataclass
class RunResult:
    run_id: int
    query: str
    n_new: int
    n_cached: int
    n_requested: int
    n_clusters: int
    results: list[ImageResult] = field(default_factory=list)


def _chunks(items: list, size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def _group_predictions(rows: list[dict], k: int) -> dict[str, list[dict]]:
    by_file: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_file[row["file_name"]].append(row)
    return {
        f: sorted(rs, key=lambda r: r["score"], reverse=True)[:k]
        for f, rs in by_file.items()
    }


def _read_predictions_for(
    asset_id: str, conn, *, rank: str, top_k: int
) -> list[Prediction]:
    rows = storage.fetch_predictions(conn, asset_id=asset_id, rank=rank)[:top_k]
    return [
        Prediction(
            species=p["species"],
            common_name=p["common_name"],
            score=p["score"],
            kingdom=p.get("kingdom", ""),
            family=p.get("family", ""),
            genus=p.get("genus", ""),
        )
        for p in rows
    ]


# Streaming search-then-decide. The pipeline asks Immich for pages of
# PLANT_PAGE_SIZE, scores each asset's Plantae confidence (cache hit
# OR BioCLIP kingdom-rank predict), and stops at the boundary where
# decision batches stop being mostly plantae. This avoids paying for
# BioCLIP work on photos Immich returned that aren't plants.
PLANT_BATCH_SIZE = 32            # decision batch size (2× DEFAULT_BATCH_SIZE)
PLANT_PAGE_SIZE = 500            # Immich page size
PLANT_MIN_PROB = 0.60            # Plantae > this counts as "is a plant"
PLANT_BATCH_MAJORITY = 0.30      # 30% of the batch must be plantae to continue


def _score_batch_for_plantae(
    clf,
    paths: list[str],
) -> dict[str, float]:
    """Run BioCLIP kingdom-rank predict on `paths` and return
    {file_path: Plantae_score}. For files where Plantae is not in the
    top-K, score is 0.0. Cached scores should be checked before calling
    this; we don't cache here because the caller (discover_plantae_assets)
    persists the rows after we get them back."""
    if not paths:
        return {}
    # BioCLIP needs the Rank enum, not a string. Resolve lazily via the
    # PEP 562 __getattr__ on this module so the import cost stays
    # deferred until the first live run.
    import sys as _sys
    _Rank = _sys.modules[__name__].Rank
    raw = clf.predict(images=paths, rank=_Rank.KINGDOM, k=5)
    out: dict[str, float] = {}
    for row in raw:
        if row.get("kingdom") != "Plantae":
            continue
        f = row["file_name"]
        score = float(row["score"])
        if score > out.get(f, 0.0):
            out[f] = score
    return out


def discover_plantae_assets(
    conn,
    client,
    clf,
    *,
    query: str,
    n_samples: int,
    base_url: str,
    api_key: str,
    thumb_dir: Path,
    decision_batch_size: int = PLANT_BATCH_SIZE,
) -> list[ImmichAsset]:
    """Stream-search Immich and return assets up to the plantae boundary.

    Algorithm:
      1. Fetch pages of PLANT_PAGE_SIZE from Immich (size=500, page=1, 2…).
      2. For each page, walk through it in decision batches of
         `decision_batch_size` (caller passes 2× the inference batch so each
         decision step is a whole number of GPU chunks).
      3. For each batch, ensure thumbs exist (download if needed) and score
         every asset's Plantae confidence — cache hit reads the stored row,
         cache miss runs BioCLIP kingdom-rank predict on the batch and
         persists the result.
      4. If >= PLANT_BATCH_MAJORITY of the batch are Plantae > PLANT_MIN_PROB,
         include the plantae assets and continue to the next batch.
      5. If < PLANT_BATCH_MAJORITY, include EVERY asset from this batch
         (plantae and non-plantae) so the user can review the boundary
         with their Plantae scores visible, then return.

    Hard cap at n_samples. Status updates are pushed to the shared state
    so the UI can show progress.
    """
    accepted: list[ImmichAsset] = []
    seen_ids: set[str] = set()
    page = 1
    batch_size = decision_batch_size
    page_size = PLANT_PAGE_SIZE

    set_state(stage="search", query=query, n_requested=n_samples)

    _log.info(
        f"[search] streaming Immich smart-search for query={query!r}, "
        f"target={n_samples} photos"
    )

    while len(accepted) < n_samples:
        try:
            payload = client.search_text(query, size=page_size, page=page)
        except Exception as e:
            msg = _format_search_error(e)
            _log.error(f"[search] page {page}: {msg}")
            error_state(msg)
            raise _PipelineAbort(msg) from e
        items = client.list_search_assets(payload)
        if not items:
            _log.info(
                f"[search] no more results from Immich after page {page - 1}"
            )
            break
        n_batches = (len(items) + batch_size - 1) // batch_size
        _log.info(
            f"[search] page {page} from Immich: got {len(items)} photos to score "
            f"(target={n_samples}, run so far={len(accepted)})"
        )

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            batch_idx = batch_start // batch_size + 1

            # Gather cache hits and download preds in one pass.
            plantae_by_path: dict[str, float] = {}
            to_predict_paths: list[str] = []
            to_predict_assets: list[ImmichAsset] = []
            for asset in batch:
                if asset.id in seen_ids:
                    continue
                # Ensure the asset row exists before we write any
                # taxon_scores (FK on taxon_scores.asset_id). The full
                # URL fields are filled in later by run_pipeline's stitch.
                original_url = f"{base_url}/api/assets/{asset.id}/original"
                thumbnail_url = (
                    f"{base_url}/api/assets/{asset.id}/thumbnail"
                    f"?size={THUMBNAIL_SIZE}"
                )
                storage.upsert_asset(
                    conn,
                    asset_id=asset.id,
                    original_file_name=asset.original_file_name,
                    original_url=original_url,
                    thumbnail_url=thumbnail_url,
                )
                if storage.taxon_scores_have_asset(
                    conn, asset_id=asset.id, rank="kingdom"
                ):
                    cached = storage.fetch_taxon_scores(
                        conn, asset_ids=[asset.id], rank="kingdom"
                    )
                    plantae_by_path[str(thumb_dir / f"{asset.id}.jpg")] = (
                        cached.get(asset.id, {}).get("Plantae", 0.0)
                    )
                else:
                    p = thumb_dir / f"{asset.id}.jpg"
                    if not p.exists():
                        thumb_bytes = client.download_bytes(
                            f"/api/assets/{asset.id}/thumbnail"
                            f"?size={THUMBNAIL_SIZE}"
                        )
                        p.write_bytes(thumb_bytes)
                    to_predict_paths.append(str(p))
                    to_predict_assets.append(asset)

            # Run BioCLIP kingdom-rank on the misses. Persist each row.
            if to_predict_paths:
                scores = _score_batch_for_plantae(clf, to_predict_paths)
                for asset, p in zip(to_predict_assets, to_predict_paths):
                    all_scores: dict[str, float] = {}
                    # Re-run a fuller predict per asset is wasteful; just
                    # record Plantae we have. The full kingdom loop later
                    # will fill in the rest.
                    plantae = scores.get(p, 0.0)
                    if plantae > 0:
                        all_scores["Plantae"] = plantae
                    if all_scores:
                        storage.upsert_taxon_scores(
                            conn,
                            asset_id=asset.id,
                            rank="kingdom",
                            scores=all_scores,
                        )
                    plantae_by_path[p] = plantae

            # Build ordered list of (asset, plantae) for the batch.
            scored: list[tuple[ImmichAsset, float]] = []
            for asset in batch:
                p = str(thumb_dir / f"{asset.id}.jpg")
                scored.append((asset, plantae_by_path.get(p, 0.0)))

            n_plant = sum(1 for _, s in scored if s > PLANT_MIN_PROB)
            majority = n_plant / len(scored) if scored else 0.0
            threshold_pct = int(PLANT_BATCH_MAJORITY * 100)

            if majority >= PLANT_BATCH_MAJORITY:
                # Continue: include plantae in this batch.
                n_added = 0
                for asset, s in scored:
                    if asset.id in seen_ids:
                        continue
                    if s > PLANT_MIN_PROB:
                        accepted.append(asset)
                        seen_ids.add(asset.id)
                        n_added += 1
                _log.info(
                    f"[search]   page {page}, batch {batch_idx}/{n_batches}: "
                    f"{n_plant} of {len(scored)} are plants (>= {threshold_pct}%) "
                    f"-> keep going, added {n_added} (run={len(accepted)})"
                )
            else:
                # Boundary batch: < PLANT_BATCH_MAJORITY plantae. Include
                # EVERY asset from this batch — plantae and non-plantae
                # alike — so the user can review the entire batch in
                # Immich search order, with their Plantae scores visible
                # via the per-asset badge. Then stop searching.
                n_added = 0
                for asset, s in scored:
                    if asset.id in seen_ids:
                        continue
                    accepted.append(asset)
                    seen_ids.add(asset.id)
                    n_added += 1
                _log.info(
                    f"[search]   page {page}, batch {batch_idx}/{n_batches}: "
                    f"only {n_plant} of {len(scored)} are plants "
                    f"(< {threshold_pct}%) -> boundary, adding all {n_added} "
                    f"(run={len(accepted)}, stopping search)"
                )
                return accepted

            if len(accepted) >= n_samples:
                _log.info(
                    f"[search] reached target of {n_samples}, stopping"
                )
                return accepted[:n_samples]

        page += 1
        if page > 50:  # safety net
            _log.warning(
                f"[search] hit 50-page safety cap, stopping "
                f"(Immich search isn't ending — try a more specific query)"
            )
            break

    _log.info(
        f"[search] done streaming: returning {len(accepted)} photos "
        f"(target was {n_samples})"
    )
    return accepted[:n_samples]


def run_pipeline(
    *,
    base_url: str,
    api_key: str,
    query: str,
    n_samples: int,
    top_k: int,
    db_path: str | Path,
    classifier: "TreeOfLifeClassifier | None" = None,
    rank: str = "species",
    yolo_conf: float = 0.05,
    batch_size: int = DEFAULT_BATCH_SIZE,
) -> RunResult:
    """Run a smart-search on Immich for `query`, embed + predict any assets
    not yet cached, recompute clusters, run YOLO-World on uncached images,
    persist everything, and return the ordered result list for this run.

    Heavy passes run in chunks of `batch_size` so peak memory stays small.

    Each cache check below is explicit: if a row exists for the (asset,
    rank) pair, we skip the work. Add new caches here so a freshly-cached
    asset still gets the new feature on the next run.
    """
    # Touch the lazy attributes — PEP 562 __getattr__ on this module loads
    # them on first access. Once loaded, they live in module globals and
    # subsequent runs reuse them without re-importing.
    import sys as _sys
    _self = _sys.modules[__name__]
    _TreeOfLifeClassifier = _self.TreeOfLifeClassifier
    _Rank = _self.Rank
    _clusters = _self.clusters
    _yolo = _self.yolo

    storage.init_db(db_path)
    rank_name = rank.name.lower() if hasattr(rank, "name") else str(rank)
    # BioCLIP's predict() expects a `Rank` enum (it does `rank.value`),
    # not the lowercase string we accept as the public parameter form.
    # Resolve once and use the enum everywhere internally.
    rank_enum = _Rank[rank_name.upper()]
    kingdom_rank = _Rank.KINGDOM

    reset_state()
    set_state(query=query, n_requested=n_samples, stage="search")

    client = ImmichClient(base_url=base_url, api_key=api_key)
    try:
        with storage.connect(db_path) as conn:
            clf = classifier or _TreeOfLifeClassifier()

            run_id = storage.create_run(
                conn,
                query=query,
                n_requested=n_samples,
                rank=rank_name,
                top_k=top_k,
            )
            # Persistent thumb cache: keyed by asset_id, lives next to
            # the DB so cache-hit re-runs don't re-download thumbs.
            # Crop files (per-detection) still go in `tmp` since they're
            # only needed for the duration of one predict() call.
            db_path = Path(db_path)
            thumb_dir = db_path.parent / "thumbs"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            tmp = Path(tempfile.mkdtemp(prefix="immich_bioclip_"))

            # Stream-search Immich and stop at the plantae boundary.
            # Returns up to n_samples assets. Each asset's Plantae
            # confidence is already cached in taxon_scores, so the
            # downstream kingdom loop will skip them.
            # The plantae-decision batch is 2x the inference batch so
            # each decision step is always a whole number of GPU chunks
            # — no awkward tail batch.
            decision_batch_size = batch_size * 2

            assets = discover_plantae_assets(
                conn,
                client,
                clf,
                query=query,
                n_samples=n_samples,
                base_url=base_url,
                api_key=api_key,
                thumb_dir=thumb_dir,
                decision_batch_size=decision_batch_size,
            )

            needs_bioclip: list[tuple[ImmichAsset, Path | None]] = []
            needs_yolo: list[tuple[ImmichAsset, Path | None]] = []
            ordered_ids: list[str] = []

            for asset in assets:
                original_url = f"{base_url}/api/assets/{asset.id}/original"
                thumbnail_url = (
                    f"{base_url}/api/assets/{asset.id}/thumbnail"
                    f"?size={THUMBNAIL_SIZE}"
                )
                storage.upsert_asset(
                    conn,
                    asset_id=asset.id,
                    original_file_name=asset.original_file_name,
                    original_url=original_url,
                    thumbnail_url=thumbnail_url,
                )

                cached_preds = storage.fetch_predictions(
                    conn, asset_id=asset.id, rank=rank_name
                )
                preds_ready = len(cached_preds) >= top_k

                ordered_ids.append(asset.id)
                if not preds_ready:
                    # Defer download until the chunk loop reads it.
                    needs_bioclip.append((asset, None))
                    needs_yolo.append((asset, None))

            n_new = len(needs_bioclip)
            n_cached = len(assets) - n_new

            _log.info(
                f"query={query!r}  n_requested={n_samples}  "
                f"n_new={n_new}  n_cached={n_cached}  "
                f"rank={rank_name}  top_k={top_k}  batch_size={batch_size}"
            )

            # ---- BioCLIP: embed + predict for uncached assets ----
            # Iterate by index throughout so we never read from a stale
            # slice. We mutate needs_bioclip/need_yolo in place as we go.
            species_total_chunks = (len(needs_bioclip) + batch_size - 1) // batch_size

            if needs_bioclip:
                species_t0 = time.monotonic()
                _log.info(
                    f"[species] starting  {len(needs_bioclip)} assets, "
                    f"{species_total_chunks} chunks of {batch_size}"
                )
                i = 0
                while i < len(needs_bioclip):
                    j = min(i + batch_size, len(needs_bioclip))
                    pil_images: list[Image.Image] = []
                    file_paths: list[str] = []
                    chunk_indices: list[int] = []  # indices we just processed

                    for k in range(i, j):
                        asset, p = needs_bioclip[k]
                        if p is None:
                            p = thumb_dir / f"{asset.id}.jpg"
                            if not p.exists():
                                _log.info(
                                    f"[species] downloading thumb {asset.id}"
                                )
                                thumb_bytes = client.download_bytes(
                                    f"/api/assets/{asset.id}/thumbnail"
                                    f"?size={THUMBNAIL_SIZE}"
                                )
                                p.write_bytes(thumb_bytes)
                            with Image.open(p) as im:
                                im.verify()
                            needs_bioclip[k] = (asset, p)
                            # Mirror the populated path into needs_yolo.
                            for jj, (a2, p2) in enumerate(needs_yolo):
                                if a2.id == asset.id and p2 is None:
                                    needs_yolo[jj] = (a2, p)
                                    break
                        pil_images.append(Image.open(p).convert("RGB"))
                        file_paths.append(str(p))
                        chunk_indices.append(k)

                    feats = clf.create_image_features(pil_images, normalize=True)
                    feats_np = feats.detach().cpu().numpy().astype("<f4")
                    del feats
                    raw = clf.predict(images=file_paths, rank=rank_enum, k=top_k)
                    preds_by_file = _group_predictions(raw, top_k)

                    for k, vec in zip(chunk_indices, feats_np):
                        asset, p = needs_bioclip[k]   # re-read; `p` is populated
                        storage.upsert_embedding(
                            conn, asset_id=asset.id, vector=vec
                        )
                        rows = preds_by_file.get(str(p), [])
                        if not rows:
                            rows = _rows_for_image(preds_by_file, str(p))
                        storage.replace_predictions(
                            conn,
                            asset_id=asset.id,
                            rank=rank_name,
                            rows=rows,
                        )

                    # Free the PIL decodes — yolo.detect will reopen them.
                    for im in pil_images:
                        try:
                            im.close()
                        except Exception:
                            pass
                    del pil_images, feats_np, raw, preds_by_file
                    gc.collect()

                    _chunk_log(
                        "species",
                        (i // batch_size) + 1,
                        species_total_chunks,
                        n_items=len(chunk_indices),
                        elapsed=time.monotonic() - species_t0,
                    )
                    i = j

            if needs_bioclip:
                _log.info(
                    f"[species] done  {len(needs_bioclip)} assets in "
                    f"{_fmt_eta(time.monotonic() - species_t0)}"
                )

            # ---- Clustering: re-fit across all embedded assets ----
            _log.info("[cluster] re-fitting k-means on all embedded assets")
            cluster_t0 = time.monotonic()
            asset_to_cluster, n_clusters = _clusters.compute_clusters(conn)
            storage.set_cluster_ids(conn, asset_to_cluster=asset_to_cluster)
            _log.info(
                f"[cluster] done  n_clusters={n_clusters}  "
                f"elapsed={_fmt_eta(time.monotonic() - cluster_t0)}"
            )

            # ---- Kingdom-level classification ----
            # A dedicated predict() at Rank.KINGDOM, k=KINGDOM_TOP_K so we
            # always have a Plantae/Fungi/Animalia score per image even
            # when the species-rank top-K happens to be all one kingdom.
            # Iterates over EVERY asset in the run (not just the ones
            # missing species-rank predictions), with per-asset cache
            # skipping via taxon_scores_have_asset. Thumb downloads are
            # served from the persistent thumb_dir so cache-hit re-runs
            # don't re-fetch.
            KINGDOM_TOP_K = 5
            kingdom_assets: list[ImmichAsset] = []
            kingdom_paths: list[Path] = []
            for asset in assets:
                if storage.taxon_scores_have_asset(
                    conn, asset_id=asset.id, rank="kingdom"
                ):
                    continue
                p = thumb_dir / f"{asset.id}.jpg"
                if not p.exists():
                    _log.info(f"[kingdom] downloading thumb {asset.id}")
                    thumb_bytes = client.download_bytes(
                        f"/api/assets/{asset.id}/thumbnail"
                        f"?size={THUMBNAIL_SIZE}"
                    )
                    p.write_bytes(thumb_bytes)
                kingdom_assets.append(asset)
                kingdom_paths.append(p)

            kingdom_total_chunks = (
                (len(kingdom_assets) + batch_size - 1) // batch_size
            )
            kingdom_t0 = time.monotonic()
            if kingdom_assets:
                _log.info(
                    f"[kingdom] starting  {len(kingdom_assets)} assets, "
                    f"{kingdom_total_chunks} chunks of {batch_size}"
                )
            else:
                _log.info("[kingdom] no work — all assets already cached")

            i = 0
            while i < len(kingdom_assets):
                j = min(i + batch_size, len(kingdom_assets))
                pil_images: list[Image.Image] = []
                file_paths: list[str] = []
                chunk_indices: list[int] = []
                for k in range(i, j):
                    p = kingdom_paths[k]
                    pil_images.append(Image.open(p).convert("RGB"))
                    file_paths.append(str(p))
                    chunk_indices.append(k)
                if not pil_images:
                    i = j
                    continue
                raw_kingdom = clf.predict(
                    images=file_paths, rank=kingdom_rank, k=KINGDOM_TOP_K
                )
                kingdom_by_file: dict[str, dict[str, float]] = {}
                for row in raw_kingdom:
                    f = row["file_name"]
                    d = kingdom_by_file.setdefault(f, {})
                    prev = d.get(row["kingdom"])
                    if prev is None or row["score"] > prev:
                        d[row["kingdom"]] = float(row["score"])
                for k, fp in zip(chunk_indices, file_paths):
                    asset = kingdom_assets[k]
                    scores = kingdom_by_file.get(fp) or {}
                    if not scores:
                        for k2, v in kingdom_by_file.items():
                            if Path(k2).name == Path(fp).name:
                                scores = v
                                break
                    storage.upsert_taxon_scores(
                        conn,
                        asset_id=asset.id,
                        rank="kingdom",
                        scores=scores,
                    )
                for im in pil_images:
                    try:
                        im.close()
                    except Exception:
                        pass
                del pil_images, raw_kingdom, kingdom_by_file
                gc.collect()
                _chunk_log(
                    "kingdom",
                    (i // batch_size) + 1,
                    kingdom_total_chunks,
                    n_items=len(chunk_indices),
                    elapsed=time.monotonic() - kingdom_t0,
                )
                i = j
            if kingdom_total_chunks:
                _log.info(
                    f"[kingdom] done  {len(kingdom_assets)} assets in "
                    f"{_fmt_eta(time.monotonic() - kingdom_t0)}"
                )
            del kingdom_assets, kingdom_paths

            # ---- YOLO-World: detect on uncached assets (chunked) ----
            # Detection rows are persisted as we go; the bbox BioCLIP
            # pass below queries the detections table directly so it
            # handles cache-hit assets without needing tuples handed
            # off from here.
            yolo_total_chunks = (len(needs_yolo) + batch_size - 1) // batch_size
            yolo_t0 = time.monotonic()
            if needs_yolo:
                _log.info(
                    f"[yolo] starting  {len(needs_yolo)} assets, "
                    f"{yolo_total_chunks} chunks of {batch_size}"
                )
                yolo_model_loaded = False
                try:
                    i = 0
                    while i < len(needs_yolo):
                        j = min(i + batch_size, len(needs_yolo))
                        todo: list[tuple[int, ImmichAsset, Path]] = []
                        for k in range(i, j):
                            asset, p = needs_yolo[k]
                            if p is None:
                                # BioCLIP loop didn't reach this asset (e.g.
                                # already-cached prediction). Skip YOLO too.
                                continue
                            if not storage.detections_have_asset(conn, asset.id):
                                todo.append((k, asset, p))
                        if not todo:
                            i = j
                            continue
                        # Lazy: don't import YOLOWorld until we actually
                        # need it.
                        if not yolo_model_loaded:
                            _log.info("[yolo] loading YOLOv8s-world weights")
                            _yolo.get_model()
                            yolo_model_loaded = True

                        pil_images = [Image.open(p).convert("RGB") for _, _, p in todo]
                        detections = _yolo.detect(pil_images, conf=yolo_conf)
                        n_dets = 0
                        n_merged = 0
                        for (_k, asset, p), dets in zip(todo, detections):
                            # Class-agnostic IoU merge: when YOLO returns
                            # multiple boxes for the same region under
                            # different classes (e.g. "plant" + "leaf"),
                            # collapse them so we only BioCLIP-classify
                            # one crop per region.
                            before = len(dets)
                            dets = _merge_overlapping_boxes(dets)
                            n_merged += before - len(dets)
                            new_ids = storage.replace_detections(
                                conn, asset_id=asset.id, detections=dets
                            )
                            n_dets += len(new_ids)
                        for im in pil_images:
                            try:
                                im.close()
                            except Exception:
                                pass
                        _chunk_log(
                            "yolo",
                            (i // batch_size) + 1,
                            yolo_total_chunks,
                            n_items=len(todo),
                            elapsed=time.monotonic() - yolo_t0,
                            note=f"{n_dets} new detections ({n_merged} merged)",
                        )
                        del pil_images, detections
                        gc.collect()
                        i = j
                finally:
                    # Best-effort: free YOLO from memory when done.
                    if yolo_model_loaded:
                        _log.info("[yolo] unloading YOLO weights")
                        _yolo.unload_model()
            if needs_yolo:
                _log.info(
                    f"[yolo] done  {len(needs_yolo)} assets in "
                    f"{_fmt_eta(time.monotonic() - yolo_t0)}"
                )

            # ---- Bbox BioCLIP: re-classify each crop ----
            # Iterates over EVERY detection across the run (not just the
            # ones collected during the YOLO pass), filtering by score
            # >= BBOX_BIOCLIP_MIN and skipping cached bbox_predictions.
            # Thumbs come from the persistent thumb_dir. Decoupled from
            # the species-rank cache so cached assets also get bbox
            # classification on the next run.
            all_bbox_targets: list[tuple[Path, int, dict]] = []
            for asset in assets:
                dets = storage.fetch_detections(
                    conn, asset_id=asset.id, rank=rank_name
                )
                for d in dets:
                    if d["score"] < BBOX_BIOCLIP_MIN:
                        continue
                    if storage.bbox_predictions_have_detection(
                        conn,
                        detection_id=d["id"],
                        rank=rank_name,
                        need=BBOX_BIOCLIP_TOP_K,
                    ):
                        continue
                    p = thumb_dir / f"{asset.id}.jpg"
                    if not p.exists():
                        _log.info(f"[bbox-bioclip] downloading thumb {asset.id}")
                        thumb_bytes = client.download_bytes(
                            f"/api/assets/{asset.id}/thumbnail"
                            f"?size={THUMBNAIL_SIZE}"
                        )
                        p.write_bytes(thumb_bytes)
                    all_bbox_targets.append((p, d["id"], d))

            bbox_total_chunks = (
                (len(all_bbox_targets) + batch_size - 1) // batch_size
            )
            bbox_t0 = time.monotonic()
            if all_bbox_targets:
                _log.info(
                    f"[bbox-bioclip] starting  {len(all_bbox_targets)} detections, "
                    f"{bbox_total_chunks} chunks of {batch_size}"
                )
            else:
                _log.info("[bbox-bioclip] no work — all detections already cached")

            # Chunk so we don't blow up memory if there are many bboxes.
            if all_bbox_targets:
                i = 0
                while i < len(all_bbox_targets):
                    j = min(i + batch_size, len(all_bbox_targets))
                    chunk = all_bbox_targets[i:j]

                    # One thumbnail open per asset; we crop multiple
                    # bboxes from the same image in the inner loop.
                    crops: list[Image.Image] = []
                    crop_paths: list[str] = []
                    crop_meta: list[tuple[int, Path]] = []  # (det_id, src_path)
                    by_src: dict[Path, Image.Image] = {}
                    try:
                        for src, det_id, d in chunk:
                            if src not in by_src:
                                by_src[src] = Image.open(src).convert("RGB")
                            full = by_src[src]
                            W, H = full.size
                            x0 = max(0, int(d["x"] * W))
                            y0 = max(0, int(d["y"] * H))
                            x1 = min(W, int((d["x"] + d["w"]) * W))
                            y1 = min(H, int((d["y"] + d["h"]) * H))
                            if x1 <= x0 or y1 <= y0:
                                continue
                            crop = full.crop((x0, y0, x1, y1))
                            crop_path = tmp / f"bbox_{det_id}.jpg"
                            crop.save(crop_path, format="JPEG", quality=90)
                            crops.append(crop)
                            crop_paths.append(str(crop_path))
                            crop_meta.append((det_id, src))
                    finally:
                        for im in by_src.values():
                            try:
                                im.close()
                            except Exception:
                                pass

                    if not crops:
                        i = j
                        continue

                    # We don't need the embeddings here — clf.predict
                    # below recomputes them internally. Just classify.
                    raw = clf.predict(
                        images=crop_paths, rank=rank_enum, k=BBOX_BIOCLIP_TOP_K
                    )
                    preds_by_file = _group_predictions(raw, BBOX_BIOCLIP_TOP_K)

                    for (det_id, _src), crop_path in zip(crop_meta, crop_paths):
                        rows = preds_by_file.get(str(crop_path), [])
                        if not rows:
                            rows = _rows_for_image(preds_by_file, str(crop_path))
                        storage.replace_bbox_predictions(
                            conn,
                            detection_id=det_id,
                            rank=rank_name,
                            rows=rows,
                        )

                    n_crops = len(crops)
                    for im in crops:
                        try:
                            im.close()
                        except Exception:
                            pass
                    _chunk_log(
                        "bbox-bioclip",
                        (i // batch_size) + 1,
                        bbox_total_chunks,
                        n_items=n_crops,
                        elapsed=time.monotonic() - bbox_t0,
                    )
                    del crops, raw, preds_by_file
                    gc.collect()
                    i = j
            if bbox_total_chunks:
                _log.info(
                    f"[bbox-bioclip] done  {len(all_bbox_targets)} detections "
                    f"in {_fmt_eta(time.monotonic() - bbox_t0)}"
                )
            del all_bbox_targets

            # Load per-asset taxon scores for the run's display.
            asset_to_taxon: dict[str, dict[str, float]] = {}
            if ordered_ids:
                taxon_rows = storage.fetch_taxon_scores(
                    conn, asset_ids=ordered_ids, rank="kingdom"
                )
                asset_to_taxon = taxon_rows

            # ---- Stitch the run's results in original search order ----
            final_results: list[ImageResult] = []
            for asset in assets:
                preds = _read_predictions_for(
                    asset.id, conn, rank=rank_name, top_k=top_k
                )
                dets = storage.fetch_detections(
                    conn, asset_id=asset.id, rank=rank_name
                )
                original_url = f"{base_url}/api/assets/{asset.id}/original"
                thumbnail_url = (
                    f"{base_url}/api/assets/{asset.id}/thumbnail"
                    f"?size={THUMBNAIL_SIZE}"
                )
                final_results.append(
                    ImageResult(
                        asset_id=asset.id,
                        original_file_name=asset.original_file_name,
                        original_url=original_url,
                        thumbnail_url=thumbnail_url,
                        cluster_id=asset_to_cluster.get(asset.id),
                        top_predictions=preds,
                        detections=dets,
                        taxon_scores=asset_to_taxon.get(asset.id, {}),
                    )
                )

            storage.insert_search_results(
                conn, run_id=run_id, ordered_asset_ids=ordered_ids
            )
            conn.execute(
                "UPDATE runs SET n_clusters = ? WHERE id = ?",
                (n_clusters, run_id),
            )
            storage.finish_run(
                conn, run_id=run_id, n_new=n_new, n_cached=n_cached
            )
            conn.commit()

            set_state(stage="done", chunk_idx=0, chunk_total=0, eta_seconds=0.0)
            finish_state(
                run_id=run_id,
                n_new=n_new,
                n_cached=n_cached,
                n_requested=n_samples,
                query=query,
            )
            return RunResult(
                run_id=run_id,
                query=query,
                n_new=n_new,
                n_cached=n_cached,
                n_requested=n_samples,
                n_clusters=n_clusters,
                results=final_results,
            )
    finally:
        client.close()


def _run_pipeline_safe(**kwargs) -> RunResult | None:
    """Wrapper that converts pipeline exceptions into error_state() so the
    UI can surface them instead of just hanging the request."""
    try:
        return run_pipeline(**kwargs)
    except Exception as e:  # noqa: BLE001 — surface anything to the UI
        _log.exception("pipeline failed")
        error_state(f"{type(e).__name__}: {e}")
        return None


class _PipelineAbort(Exception):
    """Internal exception used to unwind the pipeline cleanly after we
    have already populated error_state() with a user-facing message.
    Caught by _run_pipeline_safe so the request returns normally."""


def _format_search_error(e: Exception) -> str:
    """Best-effort extraction of Immich's error message from an exception
    raised during the smart-search call. We want the user to see what
    Immich actually rejected, not just 'HTTPStatusError'.

    Immich error bodies take several shapes:
      - {"message": "..."}
      - {"error": "...", "statusCode": 400}
      - [{"error": "Bad Request", "message": "size must be ≤ 1000"}]
      - ["size must not be greater than 1000"]  (list of strings)
      - [["size must not be greater than 1000"]]  (nested, seen in the wild)
    """
    import httpx

    def _flatten(obj) -> str:
        """Recursively pull all leaf-ish strings out of an error body."""
        if obj is None:
            return ""
        if isinstance(obj, str):
            return obj
        if isinstance(obj, dict):
            return (
                obj.get("message")
                or obj.get("error")
                or _flatten(list(obj.values()))
            )
        if isinstance(obj, (list, tuple)):
            parts = [_flatten(x) for x in obj]
            return "; ".join(p for p in parts if p)
        return str(obj)

    if isinstance(e, httpx.HTTPStatusError):
        try:
            body = e.response.json()
        except Exception:
            body = None
        msg = _flatten(body).strip()
        if not msg:
            try:
                msg = (e.response.text or "").strip()
            except Exception:
                msg = ""
        if not msg:
            msg = "(no body)"
        return f"Immich smart-search {e.response.status_code}: {msg}"
    return f"{type(e).__name__}: {e}"


def _merge_overlapping_boxes(
    detections: list[dict], iou_threshold: float = 0.5
) -> list[dict]:
    """Class-agnostic IoU merge. YOLO may return multiple boxes for the
    same region under different classes (e.g. "plant" + "leaf"); we want
    one entry per region, not a stack of overlapping crops. Sort by score
    desc, then greedily merge each box into the first already-kept box
    whose IoU >= threshold. The kept box takes the union coords (so we
    don't clip a slightly tighter second box) and the higher-scoring
    label. Returns a new list; input is not mutated."""
    if not detections:
        return []
    # Normalize to a (x0,y0,x1,y1) view for IoU + union math.
    boxes = []
    for d in detections:
        x0 = float(d["x"])
        y0 = float(d["y"])
        x1 = x0 + float(d["w"])
        y1 = y0 + float(d["h"])
        boxes.append({**d, "_x0": x0, "_y0": y0, "_x1": x1, "_y1": y1})
    boxes.sort(key=lambda b: b["score"], reverse=True)

    kept: list[dict] = []
    for b in boxes:
        merged = False
        for k in kept:
            ix0 = max(b["_x0"], k["_x0"])
            iy0 = max(b["_y0"], k["_y0"])
            ix1 = min(b["_x1"], k["_x1"])
            iy1 = min(b["_y1"], k["_y1"])
            if ix1 <= ix0 or iy1 <= iy0:
                continue
            inter = (ix1 - ix0) * (iy1 - iy0)
            union = (
                (k["_x1"] - k["_x0"]) * (k["_y1"] - k["_y0"])
                + (b["_x1"] - b["_x0"]) * (b["_y1"] - b["_y0"])
                - inter
            )
            iou = inter / union if union > 0 else 0.0
            if iou >= iou_threshold:
                # Take the union; higher-score label wins (we sorted, so
                # `k` already has the higher class score).
                k["_x0"] = min(k["_x0"], b["_x0"])
                k["_y0"] = min(k["_y0"], b["_y0"])
                k["_x1"] = max(k["_x1"], b["_x1"])
                k["_y1"] = max(k["_y1"], b["_y1"])
                merged = True
                break
        if not merged:
            kept.append(b)

    # Strip the aux keys and convert back to (x,y,w,h) normalized form.
    out: list[dict] = []
    for k in kept:
        x0 = k["_x0"]
        y0 = k["_y0"]
        x1 = k["_x1"]
        y1 = k["_y1"]
        out.append(
            {
                "label": k["label"],
                "score": float(k["score"]),
                "x": x0,
                "y": y0,
                "w": x1 - x0,
                "h": y1 - y0,
            }
        )
    return out


def _rows_for_image(
    preds_by_file: dict[str, list[dict]], path: str
) -> list[dict]:
    matches = [
        rows
        for k, rows in preds_by_file.items()
        if k.endswith(Path(path).name)
    ]
    return matches[0] if matches else []


def results_to_dicts(results: list[ImageResult]) -> list[dict]:
    return [
        {**asdict(r), "top_predictions": [asdict(p) for p in r.top_predictions]}
        for r in results
    ]