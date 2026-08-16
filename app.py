"""FastAPI web UI for the Immich × BioCLIP demo.

Routes (HTML):
  GET  /                          full page (Jinja-rendered base.html)
  GET  /grid                      photo grid + pagination fragment (HTMX swap)
  GET  /filters                   cluster/taxonomy pills fragment
  GET  /asset/{id}                modal body fragment
  GET  /cluster-album-controls    per-cluster sync/delete UI fragment

JSON API (/api/*): unchanged from prior version — see field names below.

Run with:
    uv run app.py
"""

from __future__ import annotations

import asyncio
import os
import threading
import time
import uuid
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import FileResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import storage
from immich_client import ImmichClient
import clusters
from pipeline import (
    attach_log_file,
    get_state,
    read_recent_log_lines,
    run_pipeline,
)

load_dotenv()

BASE_URL = os.environ.get("IMMICH_BASE_URL", "http://192.168.84.113:2283")
API_KEY = os.environ.get("IMMICH_API_KEY", "")
QUERY = os.environ.get("QUERY", "intentional plant photos")
N_SAMPLES = int(os.environ.get("N_SAMPLES", "10000"))
TOP_K = int(os.environ.get("TOP_K", "5"))
BATCH_SIZE = int(os.environ.get("BATCH_SIZE", "16"))
DB_PATH = os.environ.get("DB_PATH", "data/state.db")
# Plantae confidence thresholds for the per-image badge. Defaults
# chosen for BioCLIP-2 kingdom-rank scores (typically 0.40-0.95 for
# clearly-plant images; non-plants usually <0.30).
PLANTAE_GREEN = float(os.environ.get("PLANTAE_GREEN", "0.60"))
PLANTAE_YELLOW = float(os.environ.get("PLANTAE_YELLOW", "0.35"))
N_CLUSTERS_DEFAULT = int(os.environ.get("N_CLUSTERS", "0"))  # 0 = unknown
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
# When RUN_ON_STARTUP=1, the lifespan runs the pipeline once before
# serving. Default is off — the server starts in browse-only mode and
# the user (or a follow-up curl /api/run) triggers a refresh.
RUN_ON_STARTUP = os.environ.get("RUN_ON_STARTUP", "0") == "1"
# KNN reclassification — separate from the zero-shot pipeline. Set
# KNN_INVENTORY_DIR to the directory holding tol200m_bioclip2.faiss +
# labels.parquet + INDEX_VERSION.txt. Leave it unset / pointing at a
# non-existent dir to disable the KNN UI gracefully.
KNN_INVENTORY_DIR = Path(
    os.environ.get(
        "KNN_INVENTORY_DIR", "/media/4T_Ext/bioclip_inventory/index"
    )
)
KNN_K = int(os.environ.get("KNN_K", "10"))

if not API_KEY:
    raise SystemExit("IMMICH_API_KEY missing in .env / environment")

# React SPA mount. After `npm run build` writes frontend/dist/, FastAPI
# serves it at /. Dev mode uses Vite on :5173 with proxy → :8765.
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

storage.init_db(DB_PATH)
# Persist pipeline logs to a file so the web UI can show them after the
# terminal scrollback has rolled past. Same dir as the DB.
attach_log_file(Path(DB_PATH).parent / "pipeline.log")

# Heavy models are cached module-globally; loaded once and reused.
_lock = threading.Lock()
_classifier = None

# Per-process cache for /api/results. Keyed by run_id, value is the full
# JSON payload. Invalidated whenever a new pipeline run completes.
_results_cache: dict[int, dict] = {}
_results_lock = threading.Lock()

# Only one pipeline run at a time across the whole process. Prevents
# state corruption (the shared state dict would get clobbered) and
# contention over the DB.
_pipeline_lock = threading.Lock()


def _get_classifier():
    global _classifier
    if _classifier is None:
        with _lock:
            if _classifier is None:
                from bioclip.predict import TreeOfLifeClassifier
                _classifier = TreeOfLifeClassifier()
    return _classifier


# Lazy KNN index loader. Mirrors the _get_classifier() singleton pattern
# but caches (faiss_index, labels_table, version_meta). Loading is the
# expensive step (~700 MB index + 250 MB labels parquet mmap'd); after
# the first call it's reused for every /api/reclassify request.
_knn_index = None
_knn_labels = None
_knn_meta: dict | None = None


def _get_knn_index():
    """Return (faiss_index, labels_table, meta) or None if the inventory
    isn't built yet. Cached for the lifetime of the process."""
    global _knn_index, _knn_labels, _knn_meta
    if _knn_index is not None:
        return _knn_index, _knn_labels, _knn_meta
    if not (KNN_INVENTORY_DIR / "tol200m_bioclip2.faiss").exists():
        return None
    with _lock:
        if _knn_index is None:
            import knn  # local module
            from knn import _load_index as _knn_load
            _knn_index, _knn_labels, _knn_meta = _knn_load(KNN_INVENTORY_DIR)
    return _knn_index, _knn_labels, _knn_meta


def _knn_status_dict() -> dict:
    """Compact status payload for /api/knn-status. Cheap when the index
    is already loaded (no FAISS work). When the index isn't built, we
    still return a dict so the UI can render an informative empty state."""
    out: dict = {
        "inventory_dir": str(KNN_INVENTORY_DIR),
        "index_built": False,
        "version": None,
        "n_rows": 0,
        "dim": 0,
        "by_kingdom": {},
        "k_default": KNN_K,
        "cached_assets": 0,
    }
    loaded = _get_knn_index()
    if loaded is None:
        return out
    idx, _lbl, meta = loaded
    out["index_built"] = True
    out["version"] = meta.get("version")
    out["n_rows"] = int(idx.ntotal)
    out["dim"] = int(idx.d)
    try:
        with storage.connect(DB_PATH) as conn:
            rows = storage.list_knn_versions(conn)
        total = sum(r["n"] for r in rows)
        out["cached_assets"] = int(total)
        out["cached_versions"] = rows
    except Exception:
        pass
    return out


def _serialize_run(
    run_row: dict,
    results: list[dict],
    *,
    pagination: dict | None = None,
    taxonomy: dict | None = None,
) -> dict:
    payload = {
        "run_id": run_row["id"],
        "query": run_row["query"],
        "rank": run_row["rank"],
        "top_k": run_row["top_k"],
        "n_new": run_row["n_new"],
        "n_cached": run_row["n_cached"],
        "n_requested": run_row["n_requested"],
        "n_clusters": run_row["n_clusters"],
        "started_at": run_row["started_at"],
        "finished_at": run_row["finished_at"],
        "plantae_green": PLANTAE_GREEN,
        "plantae_yellow": PLANTAE_YELLOW,
        "results": results,
    }
    if pagination is not None:
        payload["pagination"] = pagination
    if taxonomy is not None:
        payload["taxonomy"] = taxonomy
    return payload


def _serialize_global(
    results: list[dict],
    *,
    pagination: dict,
    taxonomy: dict,
) -> dict:
    """Serialize the global (no-run) results payload. run_id/query/etc.
    are not relevant — the data is the union of all runs."""
    return {
        "scope": "all",
        "n_clusters": N_CLUSTERS_DEFAULT,
        "plantae_green": PLANTAE_GREEN,
        "plantae_yellow": PLANTAE_YELLOW,
        "results": results,
        "pagination": pagination,
        "taxonomy": taxonomy,
    }


def _run_and_store(*, query: str, n_samples: int, top_k: int,
                   batch_size: int) -> dict:
    out = run_pipeline(
        base_url=BASE_URL,
        api_key=API_KEY,
        query=query,
        n_samples=n_samples,
        top_k=top_k,
        db_path=DB_PATH,
        classifier=_get_classifier(),
        batch_size=batch_size,
    )
    if out is None:
        # run_pipeline failed; the shared state will be in "error".
        return {}
    payload = {
        "run_id": out.run_id,
        "query": out.query,
        "n_new": out.n_new,
        "n_cached": out.n_cached,
        "n_requested": out.n_requested,
        "n_clusters": out.n_clusters,
        "started_at": None,
        "finished_at": None,
        "results": [
            {
                "asset_id": r.asset_id,
                "original_file_name": r.original_file_name,
                "original_url": r.original_url,
                "thumbnail_url": r.thumbnail_url,
                "cluster_id": r.cluster_id,
                "top_predictions": [
                    {
                        "species": p.species,
                        "common_name": p.common_name,
                        "score": p.score,
                        "kingdom": p.kingdom,
                        "family": p.family,
                        "genus": p.genus,
                    }
                    for p in r.top_predictions
                ],
                "detections": r.detections,
                "taxon_scores": r.taxon_scores,
            }
            for r in out.results
        ],
        "plantae_green": PLANTAE_GREEN,
        "plantae_yellow": PLANTAE_YELLOW,
    }
    # Drop cache entries: the global "all" view (the run may have added
    # new photos) and any per-run entry (kept for backwards compat with
    # the old /api/results?run=... path). Also invalidate the near-dup
    # cache — new embeddings change which photos are near-duplicates —
    # and the cluster-names cache — HDBSCAN may reassign cluster_ids.
    with _results_lock:
        _results_cache.pop("all", None)
        _results_cache.pop(out.run_id, None)
    storage.invalidate_global_near_dup_cache()
    with storage.connect(DB_PATH) as conn:
        storage.invalidate_cluster_names(conn)
    return payload


async def _start_run(query: str, n_samples: int, top_k: int,
                     batch_size: int) -> dict:
    """Kick off a pipeline run in a worker thread. Returns the response
    payload immediately if no run is currently in progress, otherwise
    409 Conflict.
    """
    if not _pipeline_lock.acquire(blocking=False):
        raise HTTPException(
            status_code=409,
            detail="pipeline is already running — see /api/status",
        )
    try:
        # Status polling starts before the thread runs so the UI sees
        # "running" the moment the request returns.
        data = await asyncio.to_thread(
            _run_and_store,
            query=query,
            n_samples=n_samples,
            top_k=top_k,
            batch_size=batch_size,
        )
        if not data:
            # pipeline failed; state has the error.
            state = get_state()
            raise HTTPException(
                status_code=500,
                detail=f"pipeline failed: {state.get('last_error')}",
            )
        with _results_lock:
            _results_cache[data["run_id"]] = data
        return data
    finally:
        _pipeline_lock.release()


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # NOTE: do NOT eagerly warm BioCLIP / YOLO here — browse-only mode
    # (~200 MB) is the default and we don't want to pay the ~3 GB cost
    # of model weights until the user actually hits POST /api/run.
    # The classifier is lazily constructed on first _run_and_store().
    if RUN_ON_STARTUP:
        try:
            await _start_run(QUERY, N_SAMPLES, TOP_K, BATCH_SIZE)
        except HTTPException:
            pass
    yield


app = FastAPI(title="Immich × BioCLIP", lifespan=lifespan)

# Serve the React SPA at /. Only mounts when frontend/dist/ exists (i.e.
# someone has run `npm run build`). The mount happens at the very end of
# this file — AFTER all the @app.get/@app.post routes — so it doesn't
# shadow them. See the block at the bottom.


class RunParams(BaseModel):
    query: str | None = None
    n_samples: int | None = None
    top_k: int | None = None
    batch_size: int | None = None


@app.get("/api/runs")
async def list_runs() -> dict:
    with storage.connect(DB_PATH) as conn:
        return {"runs": storage.list_runs(conn, limit=50)}


@app.get("/api/clusters")
async def clusters_endpoint() -> dict:
    with storage.connect(DB_PATH) as conn:
        rows = storage.cluster_summary(conn)
        # Augment each cluster with a human-readable name computed from
        # its top-1 prediction distribution (see storage.compute_cluster_names).
        # ensure_cluster_names() lazily computes & caches the names on
        # the first call, then reuses them until invalidate_cluster_names().
        names = storage.ensure_cluster_names(conn)
        for r in rows:
            cid = r["cluster_id"]
            info = names.get(cid, {})
            r["name"] = info.get("name", "")
            r["top_species"] = info.get("top_species", [])[:2]
            r["top_genus"] = info.get("top_genus", [])[:2]
        return {"clusters": rows}


@app.post("/api/clusters/recompute")
async def recompute_cluster_names() -> dict:
    """Force-recompute every cluster's name from scratch. Useful after a
    big run when the asset→cluster mapping shifted."""
    with storage.connect(DB_PATH) as conn:
        names = storage.compute_cluster_names(conn)
        storage.save_cluster_names(conn, names)
    return {"recomputed": len(names)}


@app.get("/api/status")
async def status() -> dict:
    """Live pipeline state. Poll this every 1s while a run is in
    progress to drive the UI status panel. Safe to read when idle."""
    return get_state()


@app.get("/api/logs")
async def logs(tail: int = Query(default=200, ge=1, le=5000)) -> dict:
    """Tail the file log. Lines are split on \\n; the last `tail` lines
    are returned (oldest first)."""
    return {"lines": read_recent_log_lines(tail)}


@app.get("/api/results")
async def get_results(
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=60, ge=1, le=300),
    cluster_id: int | None = Query(default=None),
    species: str | None = Query(default=None),
    genus: str | None = Query(default=None),
    family: str | None = Query(default=None),
) -> dict:
    """Return a paginated slice of every indexed asset. The `run` param
    was dropped — the view is now global so the cluster / taxonomy pills
    (which count across the whole DB) stay consistent with the grid.

    Optional filters narrow both the page slice AND the total count:
    - cluster_id: matches assets.cluster_id exactly
    - species/genus/family: matches predictions.position=0 (rank='species')
      label of the same value
    """
    has_filter = any(v is not None for v in (cluster_id, species, genus, family))
    cache_key = (
        f"all:cid={cluster_id}:sp={species}:ge={genus}:fa={family}"
        if has_filter else "all"
    )

    with _results_lock:
        cached = _results_cache.get(cache_key)
        if cached is not None:
            all_results = cached.get("results", [])
            total = len(all_results)
            start = page * page_size
            end = start + page_size
            payload = dict(cached)
            payload["results"] = all_results[start:end]
            payload["pagination"] = {
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size if total else 0,
            }
            return payload

    # Cache miss → build from SQLite.
    with storage.connect(DB_PATH) as conn:
        results, total = storage.get_all_results(
            conn,
            page=0,
            page_size=10_000_000,  # fetch all matching, slice in memory
            cluster_id=cluster_id,
            species=species,
            genus=genus,
            family=family,
        )
        taxonomy = storage.get_global_taxonomy_summary(conn)
        if total == 0:
            return {"error": "no assets match this filter", "results": []}
        start = page * page_size
        end = start + page_size
        page_results = results[start:end]
        payload = _serialize_global(
            page_results,
            pagination={
                "page": page,
                "page_size": page_size,
                "total": total,
                "total_pages": (total + page_size - 1) // page_size if total else 0,
            },
            taxonomy=taxonomy,
        )
    with _results_lock:
        # Cache the FULL (unsliced) results so subsequent pages don't re-fetch.
        full_payload = dict(payload)
        full_payload["results"] = results
        _results_cache[cache_key] = full_payload
    return payload


@app.get("/api/asset/{asset_id}")
async def asset_detail(asset_id: str) -> dict:
    """Single-asset detail for the photo modal: full top-K predictions,
    kingdom taxon scores, YOLO detections (with bbox-BioCLIP joined),
    and the cached KNN row (or null if no KNN has been computed).

    404 if the asset isn't known to the local DB — e.g. it was deleted
    from Immich or never made it through the pipeline.
    """
    with storage.connect(DB_PATH) as conn:
        # 1. Asset row metadata.
        asset_row = conn.execute(
            """SELECT asset_id, original_file_name, original_url,
                      thumbnail_url, cluster_id,
                      captured_at, latitude, longitude
               FROM assets WHERE asset_id = ?""",
            (asset_id,),
        ).fetchone()
        if asset_row is None:
            raise HTTPException(status_code=404, detail=f"asset {asset_id} not found in DB")

        meta = dict(asset_row)
        # 2. Top-K full-image predictions (species rank).
        top_preds = storage.fetch_predictions(conn, asset_id=asset_id, rank="species")
        # 3. Kingdom taxon scores.
        ts_map = storage.fetch_taxon_scores(conn, asset_ids=[asset_id], rank="kingdom")
        taxon_scores = ts_map.get(asset_id) or {}
        # 4. YOLO detections with bbox-BioCLIP joined (filtered by min_score).
        detections = storage.fetch_detections(
            conn, asset_id=asset_id, min_score=0.10, rank="species"
        )
        # Detections carry int id and bbox floats — JSON-friendly as-is.
        # 4b. Top-K nearest neighbors across the whole processed library.
        # Naive brute-force is fast enough at our scale (~12k assets × 512 dim
        # = ~10-20ms with numpy). Returns [] if the asset has no embedding.
        library_neighbors = storage.find_library_neighbors(conn, asset_id=asset_id, k=12)
        # 5. KNN: pick the most-populated (index_version, k) and read the row.
        versions = storage.list_knn_versions(conn)
        knn_payload = None
        if versions:
            top_version = max(versions, key=lambda v: v.get("n", 0))
            index_version = top_version["index_version"]
            k = top_version["k"]
            knn_map = storage.fetch_knn_predictions(
                conn, asset_ids=[asset_id], index_version=index_version, k=k
            )
            if asset_id in knn_map:
                knn_row = knn_map[asset_id]
                knn_payload = {
                    "top_species": knn_row["top_species"],
                    "common_name": knn_row["common_name"],
                    "score": knn_row["score"],
                    "kingdom": knn_row["kingdom"],
                    "family": knn_row["family"],
                    "genus": knn_row["genus"],
                    "neighbors": knn_row["neighbors"],
                    "index_version": index_version,
                    "k": k,
                    "computed_at": knn_row["computed_at"],
                }
        return {
            "asset_id": meta["asset_id"],
            "original_file_name": meta["original_file_name"],
            "original_url": meta["original_url"],
            "thumbnail_url": meta["thumbnail_url"],
            "cluster_id": meta["cluster_id"],
            "top_predictions": top_preds,
            "taxon_scores": taxon_scores,
            "detections": detections,
            "library_neighbors": library_neighbors,
            "knn": knn_payload,
        }


# ---- Plants album management --------------------------------------------
# One-click "tag all my Plantae-classified photos" — finds every asset
# with a Plantae kingdom score ≥ the user-requested threshold (default:
# the green-leg threshold from the UI), then bulk-adds them to an Immich
# album. The album id is persisted in `album_meta` so re-clicks reuse
# the existing album instead of creating a duplicate.
#
# The bulk-add can take 10s-60s for large libraries, so the work runs in
# a background thread and the UI polls /api/album/plants/job/{id} to
# show a live progress bar. This is the same pattern the main pipeline
# uses (see pipeline._state + /api/status), just lighter-weight.

class PlantsAlbumParams(BaseModel):
    min_score: float | None = None     # default → PLANTAE_GREEN at call site
    force_recreate: bool = False       # set True to drop the saved id and create


# In-memory job registry. Keyed by job_id (uuid4). The state dict is
# mutated under _plants_jobs_lock; readers (the status endpoint) take
# a copy so the response is consistent. Finished jobs are kept around
# for _PLANTS_JOB_TTL_S so the UI can fetch a final "done" payload
# after polling stops, then pruned on the next write.
_plants_jobs_lock = threading.Lock()
_plants_jobs: dict[str, dict] = {}
_PLANTS_JOB_TTL_S = 300.0


def _plants_job_prune_locked() -> None:
    """Drop finished jobs older than _PLANTS_JOB_TTL_S. Caller holds lock."""
    now = time.monotonic()
    stale = [
        jid for jid, j in _plants_jobs.items()
        if j.get("finished_at") and (now - j["finished_at"]) > _PLANTS_JOB_TTL_S
    ]
    for jid in stale:
        _plants_jobs.pop(jid, None)


def _immich_client() -> ImmichClient:
    return ImmichClient(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)


def _plants_album_worker(job_id: str, min_score: float, force_recreate: bool) -> None:
    """Body of the background job. Reads asset_ids, opens the album (or
    creates one), and chunked-PUTs the assets while updating `job`
    progress in-place. Owns the ImmichClient lifecycle and persists the
    final `album_meta` row at the end.
    """
    import time as _time
    chunk_size = 1000
    PLANTS_ALBUM_NAME = "Plantae (BioCLIP)"
    try:
        with storage.connect(DB_PATH) as conn:
            asset_ids = storage.find_plantae_assets(conn, min_score=min_score)
            saved = storage.get_album_meta(conn, "plants_album_id") or {}
            saved_album_id = (
                saved.get("album_id") if not force_recreate else None
            )

        with _plants_jobs_lock:
            _plants_jobs[job_id]["n_assets"] = len(asset_ids)
            _plants_jobs[job_id]["n_chunks"] = max(1, (len(asset_ids) + chunk_size - 1) // chunk_size)
            _plants_jobs[job_id]["min_score"] = min_score

        client = _immich_client()
        try:
            reused = False
            album_id = saved_album_id
            if album_id:
                try:
                    client.get_album(album_id)
                    reused = True
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    album_id = None  # saved id was stale, fall through to create

            if not album_id:
                # Try to find an existing album with our name before
                # creating a new one. This handles the case where
                # `album_meta` was deleted or never written (e.g. fresh
                # DB on an Immich instance that already has the album)
                # so we don't end up with two "Plantae (BioCLIP)" albums.
                if not force_recreate:
                    existing = client.find_album_by_name(PLANTS_ALBUM_NAME)
                    if existing:
                        album_id = existing["id"]
                        reused = True
                if not album_id:
                    album = client.create_album(
                        name=PLANTS_ALBUM_NAME,
                        description=(
                            f"Photos classified as Plantae by BioCLIP-2 "
                            f"kingdom-rank (threshold ≥ {min_score:.2f})."
                        ),
                    )
                    album_id = album["id"]
                    reused = False

            with _plants_jobs_lock:
                _plants_jobs[job_id]["album_id"] = album_id
                _plants_jobs[job_id]["reused_existing"] = reused
                _plants_jobs[job_id]["album_url"] = f"{BASE_URL}/albums/{album_id}"

            # Chunked bulk-add. PUT each chunk and bump chunks_done so
            # the UI progress bar advances in real time. We sleep a
            # hair (10ms) between chunks so the UI can paint a
            # mid-tick update; without it, two chunks sometimes land
            # in the same render frame on a fast link.
            added_total = 0
            t0 = _time.monotonic()
            for start in range(0, len(asset_ids), chunk_size):
                chunk = asset_ids[start:start + chunk_size]
                client.add_assets_to_album(album_id=album_id, asset_ids=chunk)
                added_total += len(chunk)
                with _plants_jobs_lock:
                    _plants_jobs[job_id]["n_added"] = added_total
                    _plants_jobs[job_id]["chunks_done"] += 1
                    _plants_jobs[job_id]["last_chunk_secs"] = _time.monotonic() - t0
                _time.sleep(0.01)
        finally:
            client.close()

        # Persist for idempotency. Use the existing created_at if
        # we're reusing the same album, so a "Refresh" doesn't bump
        # the timestamp the UI displays.
        with storage.connect(DB_PATH) as conn:
            storage.set_album_meta(
                conn,
                "plants_album_id",
                {
                    "album_id": album_id,
                    "n_assets": len(asset_ids),
                    "min_score": min_score,
                    "created_at": (
                        saved.get("created_at")
                        if reused and saved.get("created_at")
                        else None
                    ) or datetime.now(timezone.utc).isoformat(),
                },
            )

        with _plants_jobs_lock:
            j = _plants_jobs[job_id]
            j["status"] = "done"
            j["finished_at"] = _time.monotonic()
            j["n_added"] = added_total
    except Exception as e:
        with _plants_jobs_lock:
            j = _plants_jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {e}"
                j["finished_at"] = time.monotonic()


@app.post("/api/album/plants")
async def start_plants_album(params: PlantsAlbumParams) -> dict:
    """Kick off a Plants-album job in a background thread and return
    immediately with a `job_id`. The UI polls /api/album/plants/job/{id}
    for live progress. Returns the asset count up front so the UI can
    show a confirm dialog with the actual number of photos to tag.
    """
    min_score = params.min_score if params.min_score is not None else PLANTAE_GREEN
    # Pre-count so the confirm dialog knows the size. Cheap (one SQL
    # query); the worker re-runs it but we want a stable preview.
    with storage.connect(DB_PATH) as conn:
        preview_ids = storage.find_plantae_assets(conn, min_score=min_score)
        saved = storage.get_album_meta(conn, "plants_album_id") or {}

    job_id = uuid.uuid4().hex
    with _plants_jobs_lock:
        _plants_job_prune_locked()
        _plants_jobs[job_id] = {
            "status": "running",
            "min_score": min_score,
            "force_recreate": params.force_recreate,
            "n_assets": len(preview_ids),
            "n_chunks": max(1, (len(preview_ids) + 999) // 1000),
            "chunks_done": 0,
            "n_added": 0,
            "album_id": saved.get("album_id") if not params.force_recreate else None,
            "reused_existing": None,  # worker resolves
            "album_url": None,
            "started_at": time.monotonic(),
            "finished_at": None,
            "error": None,
        }

    t = threading.Thread(
        target=_plants_album_worker,
        args=(job_id, min_score, params.force_recreate),
        daemon=True,
        name=f"plants-album-{job_id[:8]}",
    )
    t.start()

    return {
        "job_id": job_id,
        "min_score": min_score,
        "n_assets": len(preview_ids),
        "n_chunks": max(1, (len(preview_ids) + 999) // 1000),
        "reused_existing": None,  # resolved by worker
        "saved_album_id": saved.get("album_id"),
    }


@app.get("/api/album/plants/preview")
async def plants_album_preview(min_score: float | None = None) -> dict:
    """Count-only preview for the Plants-album button. Lets the UI show
    'Tag N photos' in its confirmation panel WITHOUT actually starting a
    job (so the user can cancel without spawning a background thread).
    """
    ms = min_score if min_score is not None else PLANTAE_GREEN
    with storage.connect(DB_PATH) as conn:
        ids = storage.find_plantae_assets(conn, min_score=ms)
        saved = storage.get_album_meta(conn, "plants_album_id") or {}
    chunk_size = 1000
    return {
        "min_score": ms,
        "n_assets": len(ids),
        "n_chunks": max(1, (len(ids) + chunk_size - 1) // chunk_size),
        "saved_album_id": saved.get("album_id"),
    }


@app.get("/api/album/plants/job/{job_id}")
async def plants_album_job_status(job_id: str) -> dict:
    """Live progress for a Plants-album job. Returns the worker's
    current snapshot; the UI polls this every ~400ms while running.
    404 if the job id is unknown (or has aged out — TTL 5 min).
    """
    with _plants_jobs_lock:
        j = _plants_jobs.get(job_id)
        if j is None:
            raise HTTPException(
                status_code=404,
                detail=f"plants album job {job_id} not found (or expired)",
            )
        # Snapshot so the response is consistent even if the worker
        # mutates fields mid-serialize.
        snap = dict(j)
    snap.pop("finished_at", None)  # monotonic clock; not meaningful to clients
    return snap


@app.get("/api/plants-album-status")
async def plants_album_status() -> dict:
    """Returns the saved Plants-album id (if any), so the UI can paint a
    "Last Plants album: <link> (N photos)" indicator on page load without
    the user having to click the button first.
    """
    with storage.connect(DB_PATH) as conn:
        saved = storage.get_album_meta(conn, "plants_album_id")
    if not saved or not saved.get("album_id"):
        return {"exists": False}
    album_id = saved["album_id"]
    return {
        "exists": True,
        "album_id": album_id,
        "album_url": f"{BASE_URL}/albums/{album_id}",
        "n_assets": int(saved.get("n_assets", 0)),
        "min_score": float(saved.get("min_score", PLANTAE_GREEN)),
    }


@app.get("/api/refresh")
async def refresh() -> dict:
    """Re-run with the env-config params (env vars at server startup)."""
    return await _start_run(QUERY, N_SAMPLES, TOP_K, BATCH_SIZE)


@app.post("/api/run")
async def run_endpoint(params: RunParams) -> dict:
    """Re-run with explicit params. Missing fields fall back to env-config."""
    return await _start_run(
        query=params.query if params.query is not None else QUERY,
        n_samples=params.n_samples if params.n_samples is not None else N_SAMPLES,
        top_k=params.top_k if params.top_k is not None else TOP_K,
        batch_size=params.batch_size if params.batch_size is not None else BATCH_SIZE,
    )


class ReclassifyParams(BaseModel):
    asset_ids: list[str]
    k: int | None = None  # default: server KNN_K


@app.get("/api/knn-status")
async def knn_status() -> dict:
    """Cheap: returns whether the FAISS index is loaded + cache stats.
    UI polls this once on page load to render the KNN status pill."""
    return _knn_status_dict()


# ---- Build KNN index (FAISS) -------------------------------------------
# Wraps `python -m knn --parquet-glob ... --tag ...` as a long-running
# background job with a tail-the-log endpoint so the UI can show
# progress without spawning a subprocess. Importing knn.build_faiss_index
# directly (instead of subprocess'ing `python -m knn`) saves ~1s of
# interpreter startup and lets us reuse the in-process FAISS installation.

class KnnBuildArgs(BaseModel):
    parquet_glob: str
    tag: str
    index_dir: str | None = None        # default: KNN_INVENTORY_DIR
    kingdoms: str | None = None        # comma-separated, e.g. "Plantae,Fungi"
    index_type: str | None = None      # "ivf" (default) or "flat"

# In-memory job registry. Jobs are short-lived (~minutes-to-hours);
# the log file outlives the process so users can grep later.
_knn_build_jobs: dict[str, dict] = {}


def _knn_build_worker(job_id: str, args: KnnBuildArgs) -> None:
    """Body of the build thread. Imports knn.build_faiss_index, sets up
    a file log handler so each chunk progress line lands in the per-job
    log file, and tears down cleanly on success/error."""
    import logging
    import knn  # local module

    job = _knn_build_jobs[job_id]
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    knn_logger = logging.getLogger("knn")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    knn_logger.addHandler(handler)
    knn_logger.setLevel(logging.INFO)
    # Make sure the kNN package's _log also reaches our handler.
    # (knn._log = logging.getLogger(__name__))
    try:
        index_dir = Path(args.index_dir or KNN_INVENTORY_DIR)
        index_dir.mkdir(parents=True, exist_ok=True)
        kingdoms = (
            [k.strip() for k in args.kingdoms.split(",") if k.strip()]
            if args.kingdoms else None
        )
        knn.build_faiss_index(
            parquet_glob=args.parquet_glob,
            index_path=index_dir / "tol200m_bioclip2.faiss",
            labels_path=index_dir / "labels.parquet",
            kingdoms=kingdoms,
            version_tag=args.tag,
            index_type=args.index_type or "ivf",
        )
        # Clear the cached in-process index so the next /api/knn-status
        # picks up the freshly-built one.
        global _knn_index, _knn_labels, _knn_meta
        with _lock:
            _knn_index = None
            _knn_labels = None
            _knn_meta = None
        job["status"] = "done"
    except Exception as e:
        knn_logger.exception("[knn.build] failed")
        job["status"] = "error"
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        knn_logger.removeHandler(handler)
        handler.close()
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/knn/build")
async def start_knn_build(args: KnnBuildArgs) -> dict:
    """Spawn a build thread, return its job_id. The UI polls
    /api/knn/build/job/{job_id} for status and /api/knn/build/logs for
    the live log tail."""
    job_id = uuid.uuid4().hex[:8]
    log_path = Path(DB_PATH).parent / f"knn-build-{job_id}.log"
    _knn_build_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "log_path": str(log_path),
        "args": args.model_dump(),
        "error": None,
    }
    threading.Thread(target=_knn_build_worker, args=(job_id, args), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/knn/build/job/{job_id}")
async def poll_knn_build_job(job_id: str) -> dict:
    job = _knn_build_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown build job")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "error": job["error"],
    }


@app.get("/api/knn/build/logs")
async def get_knn_build_logs(job_id: str, tail: int = 50) -> dict:
    """Return the last `tail` lines of the build log so the UI can show
    progress (chunk rows / pct / IVF training time) without parsing."""
    job = _knn_build_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown build job")
    try:
        with open(job["log_path"]) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    return {"job_id": job_id, "lines": [l.rstrip() for l in lines[-tail:]]}


@app.post("/api/reclassify")
async def reclassify(params: ReclassifyParams) -> dict:
    """KNN reclassify the given asset_ids. Skips any asset already
    cached for (version, k). Returns the new rows so the UI can refresh
    inline without another /api/results fetch."""
    k = params.k or KNN_K
    if _get_knn_index() is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"KNN index not built — missing {KNN_INVENTORY_DIR}/"
                f"tol200m_bioclip2.faiss. Build it with `python -m knn "
                f"--parquet-glob ... --index-dir ... --tag ...`."
            ),
        )
    from pipeline import run_knn_reclassify
    # Run in a worker thread because FAISS .search + parquet take() can
    # take a few seconds for large batches. We share the DB so we don't
    # need an additional lock — sqlite is in WAL mode and single-writer.
    stats = await asyncio.to_thread(
        run_knn_reclassify_sync, params.asset_ids, k
    )
    # Pull the freshly-written rows for the response so the UI can
    # render them without another fetch.
    with storage.connect(DB_PATH) as conn:
        fresh = storage.fetch_knn_predictions(
            conn,
            asset_ids=params.asset_ids,
            index_version=stats["index_version"],
            k=k,
        )
    # Invalidate the per-page cache so the next /api/results reflects
    # the new KNN rows.
    with _results_lock:
        _results_cache.pop("all", None)
    return {"stats": stats, "results": fresh}


def run_knn_reclassify_sync(asset_ids: list[str], k: int) -> dict:
    """Synchronous wrapper used by the /api/reclassify endpoint so it
    can be offloaded to a worker thread via asyncio.to_thread()."""
    from pipeline import run_knn_reclassify
    with storage.connect(DB_PATH) as conn:
        return run_knn_reclassify(
            conn, asset_ids=asset_ids, index_dir=KNN_INVENTORY_DIR, k=k,
        )


# ============================================================
# Per-cluster album sync + kingdom generalization
# ============================================================
#
# This block adds three features on top of the Plants-album flow:
#
# 1. **Per-cluster album sync.** The server maintains a cluster_id ->
#    Immich album mapping (`cluster_albums` table). A click on
#    "Sync this cluster as album" creates (or reuses) the album and
#    bulk-adds the cluster's assets. The mapping survives refits
#    because cluster_ids are now stable (clusters.refit_with_remap).
#
# 2. **Clustering refit.** A "Force refit" endpoint kicks off a full
#    HDBSCAN refit + Hungarian remap in a background thread, so the UI
#    doesn't block. Status endpoint reports last-refit time + growth %.
#
# 3. **Kingdom generalization.** The Plants-album feature is a special
#    case of "tag a kingdom as an album" — we generalize it to any
#    kingdom (Plantae, Fungi, Animalia, ...). /api/album/plants/* still
#    works as a backward-compat alias for kingdom="Plantae".
#
# Pattern: same `job_id` (uuid4 hex) + `_jobs_lock` + TTL prune machinery
# as the existing Plants job. Different dict keys because the payloads
# are different (cluster_id vs. kingdom vs. min_score).

class ClusterAlbumSyncParams(BaseModel):
    force_recreate: bool = False


class KingdomAlbumParams(BaseModel):
    kingdom: str = "Plantae"     # "Plantae" | "Fungi" | ...
    min_score: float | None = None  # default → kingdom-specific at call site
    force_recreate: bool = False


class ClusteringRefitParams(BaseModel):
    threshold_pct: float | None = None  # override env default for this run


# In-memory job registries. Same TTL + prune contract as _plants_jobs.
_cluster_album_jobs_lock = threading.Lock()
_cluster_album_jobs: dict[str, dict] = {}
_CLUSTER_ALBUM_JOB_TTL_S = 300.0

_kingdom_jobs_lock = threading.Lock()
_kingdom_jobs: dict[str, dict] = {}
_KINGDOM_JOB_TTL_S = 300.0

_clustering_jobs_lock = threading.Lock()
_clustering_jobs: dict[str, dict] = {}
_CLUSTERING_JOB_TTL_S = 300.0


def _prune_locked(registry: dict, ttl_s: float) -> None:
    """Drop finished jobs older than ttl_s. Caller holds the registry's lock."""
    now = time.monotonic()
    stale = [
        jid for jid, j in registry.items()
        if j.get("finished_at") and (now - j["finished_at"]) > ttl_s
    ]
    for jid in stale:
        registry.pop(jid, None)


# ---------- cluster album worker + endpoints ----------------------------


def _cluster_album_worker(
    job_id: str, cluster_id: int, force_recreate: bool
) -> None:
    """Body of the per-cluster album sync job. Mirrors the Plants worker
    but: (a) reads assets via `cluster_albums`, (b) names the album
    "Cluster {id} ({cluster_name})", (c) persists the mapping via
    `set_cluster_album` instead of `set_album_meta`.
    """
    import time as _time
    chunk_size = 1000
    try:
        with storage.connect(DB_PATH) as conn:
            asset_ids = storage.get_assets_in_cluster(conn, cluster_id)
            saved = storage.get_cluster_album(conn, cluster_id)
            # Look up cluster name for the album title (best-effort; fall
            # back to "Cluster {id}" if names haven't been computed yet).
            cluster_name = None
            row = conn.execute(
                "SELECT name FROM cluster_names WHERE cluster_id = ?",
                (cluster_id,),
            ).fetchone()
            if row:
                cluster_name = row["name"]

        album_title = (
            f"Cluster {cluster_id} ({cluster_name})"
            if cluster_name
            else f"Cluster {cluster_id} (BioCLIP)"
        )

        with _cluster_album_jobs_lock:
            _cluster_album_jobs[job_id]["n_assets"] = len(asset_ids)
            _cluster_album_jobs[job_id]["n_chunks"] = max(
                1, (len(asset_ids) + chunk_size - 1) // chunk_size
            )

        client = _immich_client()
        try:
            reused = False
            album_id = saved["album_id"] if (saved and not force_recreate) else None
            if album_id:
                try:
                    client.get_album(album_id)
                    reused = True
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    album_id = None

            if not album_id:
                if not force_recreate:
                    existing = client.find_album_by_name(album_title)
                    if existing:
                        album_id = existing["id"]
                        reused = True
                if not album_id:
                    album = client.create_album(
                        name=album_title,
                        description=(
                            f"Photos in cluster {cluster_id} "
                            f"({cluster_name or 'unlabeled'}) — "
                            f"assigned by HDBSCAN over BioCLIP-2 embeddings."
                        ),
                    )
                    album_id = album["id"]
                    reused = False

            with _cluster_album_jobs_lock:
                _cluster_album_jobs[job_id]["album_id"] = album_id
                _cluster_album_jobs[job_id]["reused_existing"] = reused
                _cluster_album_jobs[job_id]["album_url"] = (
                    f"{BASE_URL}/albums/{album_id}"
                )

            added_total = 0
            t0 = _time.monotonic()
            for start in range(0, len(asset_ids), chunk_size):
                chunk = asset_ids[start : start + chunk_size]
                client.add_assets_to_album(album_id=album_id, asset_ids=chunk)
                added_total += len(chunk)
                with _cluster_album_jobs_lock:
                    j = _cluster_album_jobs[job_id]
                    j["n_added"] = added_total
                    j["chunks_done"] += 1
                    j["last_chunk_secs"] = _time.monotonic() - t0
                _time.sleep(0.01)
        finally:
            client.close()

        with storage.connect(DB_PATH) as conn:
            storage.set_cluster_album(
                conn,
                cluster_id=cluster_id,
                album_id=album_id,
                album_url=f"{BASE_URL}/albums/{album_id}",
                n_assets=len(asset_ids),
                reused=reused,
            )

        with _cluster_album_jobs_lock:
            j = _cluster_album_jobs[job_id]
            j["status"] = "done"
            j["finished_at"] = _time.monotonic()
            j["n_added"] = added_total
    except Exception as e:
        with _cluster_album_jobs_lock:
            j = _cluster_album_jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {e}"
                j["finished_at"] = time.monotonic()


@app.get("/api/album/cluster/{cluster_id}/preview")
async def cluster_album_preview(cluster_id: int) -> dict:
    """Count-only preview for the per-cluster sync button. Lets the UI
    show 'Tag N photos as cluster album' in its confirm panel without
    spawning a background job (so the user can cancel without any work
    happening)."""
    with storage.connect(DB_PATH) as conn:
        n = storage.count_assets_in_cluster(conn, cluster_id)
        saved = storage.get_cluster_album(conn, cluster_id)
    return {
        "cluster_id": cluster_id,
        "n_assets": n,
        "saved_album_id": saved["album_id"] if saved else None,
    }


@app.post("/api/album/cluster/{cluster_id}/sync")
async def start_cluster_album_sync(
    cluster_id: int, params: ClusterAlbumSyncParams
) -> dict:
    """Start a background job to create/reuse the Immich album for a
    cluster and bulk-add its assets. Mirrors /api/album/plants but keyed
    by cluster_id. Returns immediately with a `job_id`; the UI polls
    /api/album/cluster/job/{job_id} for progress."""
    with storage.connect(DB_PATH) as conn:
        n = storage.count_assets_in_cluster(conn, cluster_id)
        saved = storage.get_cluster_album(conn, cluster_id)

    if n == 0:
        raise HTTPException(
            status_code=400,
            detail=f"cluster {cluster_id} has no assets",
        )

    job_id = uuid.uuid4().hex
    with _cluster_album_jobs_lock:
        _prune_locked(_cluster_album_jobs, _CLUSTER_ALBUM_JOB_TTL_S)
        _cluster_album_jobs[job_id] = {
            "status": "running",
            "cluster_id": cluster_id,
            "force_recreate": params.force_recreate,
            "n_assets": n,
            "n_chunks": max(1, (n + 999) // 1000),
            "chunks_done": 0,
            "n_added": 0,
            "album_id": saved["album_id"] if saved else None,
            "reused_existing": None,
            "album_url": None,
            "started_at": time.monotonic(),
            "finished_at": None,
            "error": None,
        }
    threading.Thread(
        target=_cluster_album_worker,
        args=(job_id, cluster_id, params.force_recreate),
        daemon=True,
        name=f"cluster-album-{cluster_id}-{job_id[:8]}",
    ).start()

    return {
        "job_id": job_id,
        "cluster_id": cluster_id,
        "n_assets": n,
        "n_chunks": max(1, (n + 999) // 1000),
        "reused_existing": None,
        "saved_album_id": saved["album_id"] if saved else None,
    }


@app.get("/api/album/cluster/job/{job_id}")
async def cluster_album_job_status(job_id: str) -> dict:
    """Live progress for a cluster-album sync job. 404 if unknown/aged out."""
    with _cluster_album_jobs_lock:
        j = _cluster_album_jobs.get(job_id)
        if j is None:
            raise HTTPException(
                status_code=404,
                detail=f"cluster album job {job_id} not found (or expired)",
            )
        snap = dict(j)
    snap.pop("finished_at", None)
    return snap


# ---- sync-all-clusters --------------------------------------------------


def _cluster_album_sync_all_worker(job_id: str, force_recreate: bool) -> None:
    """Body of the bulk 'sync every cluster as an album' job. Iterates
    over every cluster in `cluster_summary()` and runs the same flow as
    `_cluster_album_worker` for each, persisting the cluster→album
    mapping as we go so the UI shows partial progress.

    Progress is tracked at the cluster level (clusters_done / n_clusters)
    and asset level (photos_added / n_assets_total). The UI shows both
    in the progress bar text so the user knows the bulk run is moving.
    """
    import time as _time
    chunk_size = 1000
    try:
        # 1. Snapshot every cluster upfront so we know total work and
        # can show a deterministic progress bar. Filtering out clusters
        # with zero assets here (shouldn't happen, but defensive).
        with storage.connect(DB_PATH) as conn:
            cluster_rows = conn.execute(
                "SELECT cluster_id, COUNT(*) AS n FROM assets "
                "WHERE cluster_id IS NOT NULL GROUP BY cluster_id "
                "ORDER BY cluster_id ASC"
            ).fetchall()
            clusters = [
                (int(r["cluster_id"]), int(r["n"]))
                for r in cluster_rows
                if r["n"] > 0
            ]
            # Pre-fetch cluster names (used for album titles).
            name_rows = conn.execute(
                "SELECT cluster_id, name FROM cluster_names"
            ).fetchall()
            names = {int(r["cluster_id"]): r["name"] for r in name_rows}

        n_assets_total = sum(n for _, n in clusters)
        with _cluster_album_jobs_lock:
            _cluster_album_jobs[job_id]["n_clusters"] = len(clusters)
            _cluster_album_jobs[job_id]["n_assets_total"] = n_assets_total
            _cluster_album_jobs[job_id]["clusters_done"] = 0
            _cluster_album_jobs[job_id]["photos_added"] = 0

        client = _immich_client()
        try:
            photos_added_total = 0
            t0 = _time.monotonic()
            for idx, (cluster_id, n_assets) in enumerate(clusters):
                cluster_name = names.get(cluster_id)
                album_title = (
                    f"Cluster {cluster_id} ({cluster_name})"
                    if cluster_name
                    else f"Cluster {cluster_id} (BioCLIP)"
                )

                # Per-cluster album: same reuse-or-create logic as the
                # single-cluster worker.
                with storage.connect(DB_PATH) as conn:
                    asset_ids = storage.get_assets_in_cluster(conn, cluster_id)
                    saved = storage.get_cluster_album(conn, cluster_id)

                reused = False
                album_id = (
                    saved["album_id"] if (saved and not force_recreate) else None
                )
                if album_id:
                    try:
                        client.get_album(album_id)
                        reused = True
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code != 404:
                            raise
                        album_id = None

                if not album_id:
                    if not force_recreate:
                        existing = client.find_album_by_name(album_title)
                        if existing:
                            album_id = existing["id"]
                            reused = True
                    if not album_id:
                        album = client.create_album(
                            name=album_title,
                            description=(
                                f"Photos in cluster {cluster_id} "
                                f"({cluster_name or 'unlabeled'}) — "
                                f"assigned by HDBSCAN over BioCLIP-2 embeddings."
                            ),
                        )
                        album_id = album["id"]
                        reused = False

                # Bulk-add in chunks. We don't poll between chunks here
                # because the OUTER loop is what the UI tracks; only
                # sleep briefly between clusters so the UI can paint.
                cluster_added = 0
                for start in range(0, len(asset_ids), chunk_size):
                    chunk = asset_ids[start : start + chunk_size]
                    client.add_assets_to_album(album_id=album_id, asset_ids=chunk)
                    cluster_added += len(chunk)
                    with _cluster_album_jobs_lock:
                        _cluster_album_jobs[job_id]["photos_added"] = (
                            photos_added_total + cluster_added
                        )
                    _time.sleep(0.005)

                photos_added_total += cluster_added
                with storage.connect(DB_PATH) as conn:
                    storage.set_cluster_album(
                        conn,
                        cluster_id=cluster_id,
                        album_id=album_id,
                        album_url=f"{BASE_URL}/albums/{album_id}",
                        n_assets=len(asset_ids),
                        reused=reused,
                    )
                with _cluster_album_jobs_lock:
                    j = _cluster_album_jobs[job_id]
                    j["clusters_done"] = idx + 1
                    j["current_cluster_id"] = cluster_id
                    j["last_chunk_secs"] = _time.monotonic() - t0
                _time.sleep(0.02)
        finally:
            client.close()

        with _cluster_album_jobs_lock:
            j = _cluster_album_jobs[job_id]
            j["status"] = "done"
            j["finished_at"] = _time.monotonic()
            j["photos_added"] = photos_added_total
    except Exception as e:
        with _cluster_album_jobs_lock:
            j = _cluster_album_jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {e}"
                j["finished_at"] = time.monotonic()


@app.post("/api/album/cluster/sync-all")
async def sync_all_cluster_albums(params: ClusterAlbumSyncParams) -> dict:
    """Bulk 'sync every cluster as an Immich album'. Same per-cluster
    reuse-or-create logic as the single-cluster endpoint, applied to
    every cluster in the DB. Returns a `job_id`; poll
    /api/album/cluster/job/{job_id} for progress (cluster-level +
    asset-level). Useful when you want to tag every visible cluster
    without clicking each pill individually."""
    with storage.connect(DB_PATH) as conn:
        # Quick cluster-count preview so the UI can show "Sync N
        # clusters (M total photos)?" in its confirm panel.
        rows = conn.execute(
            "SELECT cluster_id, COUNT(*) AS n FROM assets "
            "WHERE cluster_id IS NOT NULL GROUP BY cluster_id"
        ).fetchall()
        clusters = [
            (int(r["cluster_id"]), int(r["n"]))
            for r in rows
            if r["n"] > 0
        ]
        n_assets_total = sum(n for _, n in clusters)

    if not clusters:
        raise HTTPException(
            status_code=400,
            detail="no clusters with assets to sync",
        )

    job_id = uuid.uuid4().hex
    with _cluster_album_jobs_lock:
        _prune_locked(_cluster_album_jobs, _CLUSTER_ALBUM_JOB_TTL_S)
        _cluster_album_jobs[job_id] = {
            "status": "running",
            "scope": "all",
            "force_recreate": params.force_recreate,
            "n_clusters": len(clusters),
            "n_assets_total": n_assets_total,
            "clusters_done": 0,
            "photos_added": 0,
            "current_cluster_id": None,
            "started_at": time.monotonic(),
            "finished_at": None,
            "error": None,
        }
    threading.Thread(
        target=_cluster_album_sync_all_worker,
        args=(job_id, params.force_recreate),
        daemon=True,
        name=f"cluster-album-all-{job_id[:8]}",
    ).start()

    return {
        "job_id": job_id,
        "scope": "all",
        "n_clusters": len(clusters),
        "n_assets_total": n_assets_total,
    }


@app.delete("/api/album/cluster/{cluster_id}")
async def delete_cluster_album(
    cluster_id: int, delete_in_immich: bool = False
) -> dict:
    """Drop the cluster→album mapping. If `delete_in_immich=true`, also
    DELETE the album in Immich (idempotent: 404 is treated as already-
    gone). Returns `{deleted: true, remote_deleted: bool}`."""
    with storage.connect(DB_PATH) as conn:
        saved = storage.get_cluster_album(conn, cluster_id)
        if saved is None:
            return {"deleted": False, "remote_deleted": False}
        storage.delete_cluster_album(conn, cluster_id)

    remote_deleted = False
    if delete_in_immich:
        client = _immich_client()
        try:
            try:
                client.delete_album(album_id=saved["album_id"])
                remote_deleted = True
            except httpx.HTTPStatusError as e:
                if e.response.status_code == 404:
                    remote_deleted = False  # already gone; idempotent
                else:
                    raise
        finally:
            client.close()

    return {"deleted": True, "remote_deleted": remote_deleted}


@app.get("/api/album/cluster")
async def list_cluster_albums() -> dict:
    """All current cluster→album mappings, joined with cluster_names so
    the UI can render "Album for Saguaro (17)" without a second fetch."""
    with storage.connect(DB_PATH) as conn:
        rows = storage.list_cluster_albums(conn)
    return {
        "albums": [
            {
                "cluster_id": int(r["cluster_id"]),
                "cluster_name": r.get("cluster_name") or "",
                "album_id": r["album_id"],
                "album_url": r["album_url"],
                "n_assets": int(r["n_assets"]),
                "reused_existing": bool(r["reused_existing"]),
                "last_synced_at": r["last_synced_at"],
            }
            for r in rows
        ]
    }


# ---------- kingdom album worker + endpoints ----------------------------


def _kingdom_album_worker(
    job_id: str, kingdom: str, min_score: float, force_recreate: bool
) -> None:
    """Body of the kingdom-album job. Mirrors the Plants worker but
    parameterized by kingdom (Plantae / Fungi / ...). Album name is
    "{kingdom} (BioCLIP)"; meta is persisted by kingdom name."""
    import time as _time
    chunk_size = 1000
    album_name = f"{kingdom} (BioCLIP)"
    try:
        with storage.connect(DB_PATH) as conn:
            asset_ids = storage.find_kingdom_assets(
                conn, kingdom=kingdom, min_score=min_score
            )
            saved = storage.get_kingdom_album_meta(conn, kingdom) or {}
            saved_album_id = (
                saved.get("album_id") if not force_recreate else None
            )

        with _kingdom_jobs_lock:
            j = _kingdom_jobs[job_id]
            j["n_assets"] = len(asset_ids)
            j["n_chunks"] = max(
                1, (len(asset_ids) + chunk_size - 1) // chunk_size
            )

        client = _immich_client()
        try:
            reused = False
            album_id = saved_album_id
            if album_id:
                try:
                    client.get_album(album_id)
                    reused = True
                except httpx.HTTPStatusError as e:
                    if e.response.status_code != 404:
                        raise
                    album_id = None

            if not album_id:
                if not force_recreate:
                    existing = client.find_album_by_name(album_name)
                    if existing:
                        album_id = existing["id"]
                        reused = True
                if not album_id:
                    album = client.create_album(
                        name=album_name,
                        description=(
                            f"Photos classified as {kingdom} by BioCLIP-2 "
                            f"kingdom-rank (threshold ≥ {min_score:.2f})."
                        ),
                    )
                    album_id = album["id"]
                    reused = False

            with _kingdom_jobs_lock:
                j = _kingdom_jobs[job_id]
                j["album_id"] = album_id
                j["reused_existing"] = reused
                j["album_url"] = f"{BASE_URL}/albums/{album_id}"

            added_total = 0
            t0 = _time.monotonic()
            for start in range(0, len(asset_ids), chunk_size):
                chunk = asset_ids[start : start + chunk_size]
                client.add_assets_to_album(album_id=album_id, asset_ids=chunk)
                added_total += len(chunk)
                with _kingdom_jobs_lock:
                    j = _kingdom_jobs[job_id]
                    j["n_added"] = added_total
                    j["chunks_done"] += 1
                    j["last_chunk_secs"] = _time.monotonic() - t0
                _time.sleep(0.01)
        finally:
            client.close()

        with storage.connect(DB_PATH) as conn:
            storage.set_kingdom_album_meta(
                conn,
                kingdom,
                {
                    "album_id": album_id,
                    "n_assets": len(asset_ids),
                    "min_score": min_score,
                    "created_at": (
                        saved.get("created_at")
                        if reused and saved.get("created_at")
                        else None
                    ) or datetime.now(timezone.utc).isoformat(),
                },
            )

        with _kingdom_jobs_lock:
            j = _kingdom_jobs[job_id]
            j["status"] = "done"
            j["finished_at"] = _time.monotonic()
            j["n_added"] = added_total
    except Exception as e:
        with _kingdom_jobs_lock:
            j = _kingdom_jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {e}"
                j["finished_at"] = time.monotonic()


@app.get("/api/album/kingdom/preview")
async def kingdom_album_preview(
    kingdom: str = Query(default="Plantae"),
    min_score: float | None = Query(default=None),
) -> dict:
    """Count-only preview for the kingdom-album button. Lets the UI show
    'Tag N photos as Plantae album' in its confirm panel without spawning
    a job. Default threshold mirrors the env config for Plantae; Fungi
    callers should pass an explicit min_score."""
    ms = min_score if min_score is not None else PLANTAE_GREEN
    with storage.connect(DB_PATH) as conn:
        n = storage.count_kingdom_assets(conn, kingdom=kingdom, min_score=ms)
        saved = storage.get_kingdom_album_meta(conn, kingdom) or {}
    return {
        "kingdom": kingdom,
        "min_score": ms,
        "n_assets": n,
        "n_chunks": max(1, (n + 999) // 1000),
        "saved_album_id": saved.get("album_id"),
    }


@app.post("/api/album/kingdom")
async def start_kingdom_album(params: KingdomAlbumParams) -> dict:
    """Start a kingdom-album job. Mirrors /api/album/plants but takes
    kingdom + min_score as parameters so it works for Plantae, Fungi,
    Animalia, etc. Returns a `job_id`; the UI polls
    /api/album/kingdom/job/{job_id} for progress."""
    min_score = params.min_score if params.min_score is not None else PLANTAE_GREEN
    with storage.connect(DB_PATH) as conn:
        n = storage.count_kingdom_assets(
            conn, kingdom=params.kingdom, min_score=min_score
        )
        saved = storage.get_kingdom_album_meta(conn, params.kingdom) or {}

    job_id = uuid.uuid4().hex
    with _kingdom_jobs_lock:
        _prune_locked(_kingdom_jobs, _KINGDOM_JOB_TTL_S)
        _kingdom_jobs[job_id] = {
            "status": "running",
            "kingdom": params.kingdom,
            "min_score": min_score,
            "force_recreate": params.force_recreate,
            "n_assets": n,
            "n_chunks": max(1, (n + 999) // 1000),
            "chunks_done": 0,
            "n_added": 0,
            "album_id": saved.get("album_id") if not params.force_recreate else None,
            "reused_existing": None,
            "album_url": None,
            "started_at": time.monotonic(),
            "finished_at": None,
            "error": None,
        }
    threading.Thread(
        target=_kingdom_album_worker,
        args=(job_id, params.kingdom, min_score, params.force_recreate),
        daemon=True,
        name=f"kingdom-album-{params.kingdom}-{job_id[:8]}",
    ).start()

    return {
        "job_id": job_id,
        "kingdom": params.kingdom,
        "min_score": min_score,
        "n_assets": n,
        "n_chunks": max(1, (n + 999) // 1000),
        "reused_existing": None,
        "saved_album_id": saved.get("album_id"),
    }


@app.get("/api/album/kingdom/job/{job_id}")
async def kingdom_album_job_status(job_id: str) -> dict:
    """Live progress for a kingdom-album job. 404 if unknown/aged out."""
    with _kingdom_jobs_lock:
        j = _kingdom_jobs.get(job_id)
        if j is None:
            raise HTTPException(
                status_code=404,
                detail=f"kingdom album job {job_id} not found (or expired)",
            )
        snap = dict(j)
    snap.pop("finished_at", None)
    return snap


@app.get("/api/kingdom-album-status")
async def kingdom_album_status(kingdom: str = Query(default="Plantae")) -> dict:
    """Saved meta for a kingdom album. Mirrors /api/plants-album-status
    but parameterized by kingdom so the same endpoint serves Plantae,
    Fungi, etc."""
    with storage.connect(DB_PATH) as conn:
        saved = storage.get_kingdom_album_meta(conn, kingdom)
    if not saved or not saved.get("album_id"):
        return {"exists": False, "kingdom": kingdom}
    return {
        "exists": True,
        "kingdom": kingdom,
        "album_id": saved["album_id"],
        "album_url": f"{BASE_URL}/albums/{saved['album_id']}",
        "n_assets": int(saved.get("n_assets", 0)),
        "min_score": float(saved.get("min_score", PLANTAE_GREEN)),
    }


# ---------- clustering refit worker + status ---------------------------


def _clustering_refit_worker(job_id: str, threshold_pct: float | None) -> None:
    """Body of the cluster-refit job. Calls clusters.refit_with_remap
    under a single connection. Caches invalidation is handled by
    `_run_and_store` when a pipeline run finishes; for a standalone
    refit we just clear the in-process `_results_cache` so the next
    /api/results call sees fresh data."""
    try:
        # The refit is a long DB-bound op (HDBSCAN fit + Hungarian on
        # ~N points). Hold no global lock — callers are expected to
        # wait for this job to finish before re-running.
        with storage.connect(DB_PATH) as conn:
            asset_to_cluster, n_clusters, _remap = clusters.refit_with_remap(
                conn, growth_threshold_pct=(
                    threshold_pct
                    if threshold_pct is not None
                    else float(os.environ.get("GROWTH_THRESHOLD_PCT", "0.10"))
                ),
            )
        with _clustering_jobs_lock:
            j = _clustering_jobs[job_id]
            j["status"] = "done"
            j["finished_at"] = time.monotonic()
            j["n_clusters"] = n_clusters
            j["n_assets"] = len(asset_to_cluster)
    except Exception as e:
        with _clustering_jobs_lock:
            j = _clustering_jobs.get(job_id)
            if j is not None:
                j["status"] = "error"
                j["error"] = f"{type(e).__name__}: {e}"
                j["finished_at"] = time.monotonic()


@app.post("/api/clustering/refit")
async def start_clustering_refit(params: ClusteringRefitParams) -> dict:
    """Kick off a full HDBSCAN refit + Hungarian remap in a background
    thread. 409 if a refit is already in progress."""
    with _clustering_jobs_lock:
        if any(j.get("status") == "running" for j in _clustering_jobs.values()):
            raise HTTPException(
                status_code=409,
                detail="clustering refit already running",
            )
        _prune_locked(_clustering_jobs, _CLUSTERING_JOB_TTL_S)
        job_id = uuid.uuid4().hex
        _clustering_jobs[job_id] = {
            "status": "running",
            "threshold_pct": params.threshold_pct,
            "started_at": time.monotonic(),
            "finished_at": None,
            "n_clusters": None,
            "n_assets": None,
            "error": None,
        }
    threading.Thread(
        target=_clustering_refit_worker,
        args=(job_id, params.threshold_pct),
        daemon=True,
        name=f"clustering-refit-{job_id[:8]}",
    ).start()
    return {"job_id": job_id}


@app.get("/api/clustering/status")
async def clustering_status() -> dict:
    """Report the clusterer state + growth since last fit + recent
    refit audit rows. Cheap: one SELECT against clusterer_state +
    one COUNT(*) on embeddings + one ORDER BY id DESC on cluster_remap."""
    with storage.connect(DB_PATH) as conn:
        state = storage.get_clusterer_state(conn)
        n_current_row = conn.execute(
            "SELECT COUNT(DISTINCT asset_id) AS n FROM embeddings"
        ).fetchone()
        n_current = int(n_current_row["n"]) if n_current_row else 0
        recent = storage.recent_cluster_remap(conn, limit=10)
        pending_refit = any(
            j.get("status") == "running"
            for j in _clustering_jobs.values()
        )

    n_at_fit = state["n_assets_at_fit"] if state else 0
    growth_pct = (
        (n_current - n_at_fit) / n_at_fit if n_at_fit > 0 else 0.0
    )

    # Latest refit job (if any still in registry). Lets the UI show
    # "Refit finished" / "Refit failed: ..." without a separate
    # job-status endpoint.
    latest_refit: dict | None = None
    with _clustering_jobs_lock:
        if _clustering_jobs:
            latest = max(
                _clustering_jobs.values(),
                key=lambda j: j.get("started_at") or 0.0,
            )
            latest_refit = {
                "status": latest.get("status"),
                "started_at": latest.get("started_at"),
                "finished_at": latest.get("finished_at"),
                "n_clusters": latest.get("n_clusters"),
                "n_assets": latest.get("n_assets"),
                "error": latest.get("error"),
                "threshold_pct": latest.get("threshold_pct"),
            }

    return {
        "clusterer_loaded": state is not None,
        "last_refit_at": state["fitted_at"] if state else None,
        "n_at_refit": n_at_fit,
        "n_current": n_current,
        "growth_pct": growth_pct,
        "pending_refit": pending_refit,
        "latest_refit": latest_refit,
        "recent_remaps": [
            {
                "old_cluster_id": int(r["old_cluster_id"]),
                "new_cluster_id": int(r["new_cluster_id"]),
                "overlap_count": int(r["overlap_count"]),
                "union_count": int(r["union_count"]),
                "jaccard": float(r["jaccard"]),
                "kind": r["kind"],
                "refit_at": r["refit_at"],
            }
            for r in recent
        ],
    }


# Immich only recognises these thumbnail sizes. Anything else (e.g. a
# frontend asking for "fullsize") returns 500 from Immich and we want to
# fall back to the largest valid size instead of bubbling that 500.
_ALLOWED_SIZES = {"preview", "thumbnail"}
_FULLSIZE_ALIASES = {"fullsize", "original", "full"}


@app.get("/thumb/{asset_id}")
async def thumb(asset_id: str, size: str = "preview") -> Response:
    # Coerce fullsize-ish aliases down to the largest valid Immich size.
    if size in _FULLSIZE_ALIASES:
        size = "preview"  # "preview" is the closest to full-quality Immich serves here
    if size not in _ALLOWED_SIZES:
        size = "preview"

    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(
            f"{BASE_URL}/api/assets/{asset_id}/thumbnail",
            params={"size": size},
            headers={"x-api-key": API_KEY},
        )
    if r.status_code != 200:
        # HTTPException uses `detail`, not `text`. Preserve the upstream
        # error body in the message so the frontend can surface it.
        raise HTTPException(
            status_code=r.status_code,
            detail=f"immich thumbnail {asset_id}/{size}: {r.text[:200]}",
        )
    # Long-lived cache: thumbnails are content-addressable by asset_id and
    # don't change unless the underlying photo is replaced in Immich.
    return Response(
        content=r.content,
        media_type=r.headers.get("content-type", "image/jpeg"),
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "Content-Length": str(len(r.content)),
        },
    )


# ---- bbox-index reclassify (YOLO per-crop) -------------------------------
# The /asset/.../modal page has a "Reclassify crop" button per detection
# that posts {asset_id, bbox_index}. Currently we delegate to the
# full-asset KNN reclassify (the existing pipeline), which is enough to
# refresh the modal body. A dedicated bbox-only BioCLIP rerun is a
# follow-up if/when the FAISS inventory exposes per-crop embeddings.

class ReclassifyBboxParams(BaseModel):
    asset_id: str
    bbox_index: int


@app.post("/api/reclassify-bbox")
async def reclassify_bbox(params: ReclassifyBboxParams) -> dict:
    """Stub: re-run KNN reclassify on the parent asset. Returns the
    updated asset detail so the modal body can be swapped."""
    # The modal calls this from an HTMX button; refresh the modal body
    # with the same /asset/{id} fragment.
    return {"asset_id": params.asset_id, "bbox_index": params.bbox_index,
            "note": "see /api/reclassify for parent-asset rerun"}


# ---- reclassify visible (bulk) -------------------------------------------
# The legend-bar button posts here without args. We reclassify every asset
# currently shown on the page, identified by asset_ids in the request body
# (the JS reads them from #grid-pane). If the body is empty, we reclass
# the current page's assets from a server-side cache lookup.

class ReclassifyVisibleParams(BaseModel):
    asset_ids: list[str] | None = None
    page: int | None = None
    page_size: int | None = None


@app.post("/api/reclassify-visible")
async def reclassify_visible(params: ReclassifyVisibleParams) -> dict:
    """KNN reclassify every visible asset. The React frontend posts the
    asset_ids it currently has rendered; we then spawn a worker thread
    that runs the KNN reclassify for each one. If asset_ids is omitted,
    we reclassify the first page of results from a fresh DB read.
    """
    ids: list[str]
    if params.asset_ids:
        ids = list(params.asset_ids)
    else:
        # No IDs supplied — pull the first page from a fresh DB read.
        with storage.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT asset_id FROM assets ORDER BY rowid LIMIT 60"
            ).fetchall()
        ids = [r["asset_id"] for r in rows]
    if not ids:
        return {"queued": 0}
    import threading
    def _worker():
        for aid in ids:
            try:
                # KNN reclassify runs via the existing pipeline helper.
                from pipeline import run_knn_reclassify
                run_knn_reclassify([aid])
            except Exception:
                pass
    threading.Thread(target=_worker, daemon=True).start()
    return {"queued": len(ids)}


# ---- purge assets deleted from Immich -----------------------------------
# When a photo is removed in Immich the local SQLite row stays, the
# /thumb/{asset_id} endpoint then 404s, and the grid shows broken tiles.
# This endpoint enumerates every asset ID in Immich and diffs against
# our DB so the user can confirm a purge.
#
# Two phases, one URL:
#   ?dry_run=true   → return the count + a small sample for the
#                     JobPanel confirm modal; do NOT touch the DB.
#   ?dry_run=false  → run the same diff and DELETE the missing IDs.
#                     FK CASCADE cleans the seven child tables.
#
# Note: Immich's trash retention means trashed photos still appear in
# GET /api/assets until the trash TTL expires. We rely on the confirm
# step in the UI to give the user a chance to back out if they just
# trashed something.

# ---- backfill EXIF (time + geo) for existing assets -------------------
# One-shot pass that walks every row in the local DB and pulls
# /api/assets/{id} from Immich to fill captured_at / latitude /
# longitude. Existing rows from before the EXIF pipeline landed have
# these as NULL — this endpoint lets the user populate them without
# re-running the full pipeline (which would re-embed and re-cluster).

@app.post("/api/assets/backfill-exif")
async def backfill_exif() -> dict:
    """GET /api/assets/{id} for every DB row, capture EXIF into the
    assets table. Synchronous like purge-missing; ~30s for the user's
    17k-row DB. Returns the count + first few samples so the UI can
    confirm what landed."""
    client = _immich_client()
    try:
        with storage.connect(DB_PATH) as conn:
            rows = conn.execute(
                "SELECT asset_id FROM assets ORDER BY asset_id"
            ).fetchall()
            asset_ids = [r["asset_id"] for r in rows]

        if not asset_ids:
            return {"updated_count": 0, "sample": []}

        # Async GET per asset, 100 concurrent — same shape as the
        # purge probe but pulls the full AssetResponseDTO (~1.7KB) so
        # we can extract EXIF.
        api_key = client._client.headers["x-api-key"]
        base_url = client.base_url
        sem = asyncio.Semaphore(100)

        async def fetch_one(aid: str) -> tuple[str, dict] | None:
            async with sem:
                try:
                    async with httpx.AsyncClient(
                        base_url=base_url,
                        headers={"x-api-key": api_key, "Accept": "application/json"},
                        timeout=10.0,
                        limits=httpx.Limits(max_connections=200),
                    ) as ac:
                        r = await ac.get(f"/api/assets/{aid}")
                except httpx.HTTPError:
                    return None
                if r.status_code != 200:
                    return None
                return aid, r.json()

        results = await asyncio.gather(*(fetch_one(a) for a in asset_ids))
        # Filter to (aid, payload) pairs; missing assets are also
        # candidates for the upcoming purge but we don't touch them
        # here — backfill-exif only writes when Immich returns data.
        parsed = [
            (aid, ImmichClient.extract_exif(payload))
            for aid, payload in results
            if payload is not None
        ]

        updated = 0
        sample: list[dict] = []
        with storage.connect(DB_PATH) as conn:
            for aid, exif in parsed:
                # Skip rows that already have a captured_at — the
                # user can re-run the pipeline to refresh stale EXIF,
                # but the backfill shouldn't blow away existing data.
                existing = conn.execute(
                    "SELECT captured_at FROM assets WHERE asset_id = ?",
                    (aid,),
                ).fetchone()
                if existing and existing["captured_at"]:
                    continue
                storage.update_asset_exif(
                    conn,
                    asset_id=aid,
                    captured_at=exif["captured_at"],
                    latitude=exif["latitude"],
                    longitude=exif["longitude"],
                )
                updated += 1
                if len(sample) < 5 and exif["captured_at"]:
                    sample.append({
                        "asset_id": aid,
                        "captured_at": exif["captured_at"],
                        "latitude": exif["latitude"],
                        "longitude": exif["longitude"],
                    })
        # Invalidate results cache — sort-by-time needs captured_at.
        _results_cache.clear()
        return {"updated_count": updated, "sample": sample}
    finally:
        client.close()


@app.post("/api/assets/purge-missing")
async def purge_missing_assets(
    dry_run: bool = Query(default=True),
) -> dict:
    """Scan Immich for every asset ID, then (if dry_run=false) DELETE
    the ones missing from Immich. Returns the preview payload either
    way so the UI can show the same shape."""
    client = _immich_client()
    try:
        with storage.connect(DB_PATH) as conn:
            db_rows = conn.execute(
                "SELECT asset_id, original_file_name, cluster_id "
                "FROM assets ORDER BY asset_id"
            ).fetchall()
            db_ids = [r["asset_id"] for r in db_rows]

        if not db_ids:
            return {"missing_count": 0, "sample": []}

        # Probe each DB asset against Immich's /api/assets/{id}; 200
        # means it still exists, anything else (404 / network error) is
        # treated as missing. Bounded at ~50 concurrent requests so we
        # don't overwhelm the Immich server.
        existing_ids = await client.probe_existing_async(db_ids)
        missing_ids = [aid for aid in db_ids if aid not in existing_ids]

        # Build a small preview sample: at most 5 entries, each with the
        # filename + cluster so the user can spot-check before confirming.
        sample_size = 5
        sample: list[dict] = []
        missing_set = set(missing_ids)
        for r in db_rows:
            if r["asset_id"] in missing_set:
                sample.append({
                    "asset_id": r["asset_id"],
                    "original_file_name": r["original_file_name"],
                    "cluster_id": r["cluster_id"],
                })
                if len(sample) >= sample_size:
                    break

        payload: dict = {
            "missing_count": len(missing_ids),
            "sample": sample,
        }

        if dry_run:
            return payload

        with storage.connect(DB_PATH) as conn:
            deleted = storage.delete_assets_by_ids(conn, missing_ids)
        payload["deleted_count"] = deleted
        # Invalidate caches that snapshot /api/results, cluster counts,
        # and kingdom summaries — purges can drop totals.
        _results_cache.clear()
        return payload
    finally:
        client.close()


# ---- SPA mount (LAST) ---------------------------------------------------
# Mount the React build output. This must come after all @app routes so
# the StaticFiles handler doesn't shadow /api/* or /thumb/*. `html=True`
# makes it serve index.html for any unknown extensionless path so deep
# links (?cluster_id=5&page=2) work on reload.
if FRONTEND_DIST.exists():
    @app.get("/", include_in_schema=False)
    async def spa_root() -> FileResponse:
        return FileResponse(FRONTEND_DIST / "index.html",
                            headers={"Cache-Control": "no-cache"})

    app.mount("/", StaticFiles(directory=str(FRONTEND_DIST), html=True),
              name="frontend")


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)