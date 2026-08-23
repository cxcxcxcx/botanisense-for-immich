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
import json
import logging
import os
import threading
import time
import traceback
import tracemalloc
import uuid
from collections import deque, OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import storage
from immich_client import ImmichClient
import clusters
import pipeline  # full module — the SSE generator reads pipeline._log_file_path
from pipeline import (
    attach_log_file,
    get_state,
    read_recent_log_lines,
    run_person_backfill,
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
# Pickled near-dup map (~1 MB for 19k assets). Loaded at startup so
# the first hit after a fresh uvicorn is fast. Saved after the warm
# at pipeline end. Deleted when the embeddings change (manual
# invalidation scripts should `rm data/near_dup_cache.pkl`).
NEAR_DUP_CACHE_PATH = Path(DB_PATH).parent / "near_dup_cache.pkl"
# Plantae confidence thresholds for the per-image badge. Defaults
# chosen for BioCLIP-2 kingdom-rank scores (typically 0.40-0.95 for
# clearly-plant images; non-plants usually <0.30).
PLANTAE_GREEN = float(os.environ.get("PLANTAE_GREEN", "0.60"))
PLANTAE_YELLOW = float(os.environ.get("PLANTAE_YELLOW", "0.35"))
# Fungi kingdom scores are noisier than Plantae's (BioCLIP-2 was
# trained primarily on plants); default to the same 0.60 floor but
# let the user override via env. Used by the per-kingdom defaults
# surfaced by /api/settings/sync-defaults.
FUNGI_MIN_SCORE_DEFAULT = float(os.environ.get("FUNGI_MIN_SCORE_DEFAULT", "0.60"))
# People-filter default for the Run page "Sync to IMMICH" section.
# None = "doesn't matter" (apply to every photo). 0 = "without
# people". 1 = "with people". Parsed from env because there's no
# natural float for "no filter" — we accept "any"/"0"/"1" or "".
_sync_people_default_raw = os.environ.get("SYNC_PEOPLE_FILTER_DEFAULT", "").strip().lower()
if _sync_people_default_raw in ("1", "with", "with_people"):
    SYNC_PEOPLE_FILTER_DEFAULT: int | None = 1
elif _sync_people_default_raw in ("0", "without", "without_people", "no_people"):
    SYNC_PEOPLE_FILTER_DEFAULT = 0
else:
    SYNC_PEOPLE_FILTER_DEFAULT = None
N_CLUSTERS_DEFAULT = int(os.environ.get("N_CLUSTERS", "0"))  # 0 = unknown
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
# When RUN_ON_STARTUP=1, the lifespan runs the pipeline once before
# serving. Default is off — the server starts in browse-only mode and
# the user (or a follow-up curl /api/run) triggers a refresh.
RUN_ON_STARTUP = os.environ.get("RUN_ON_STARTUP", "0") == "1"

_log = logging.getLogger("app")
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

# Per-process cache for /api/results. Capped at _RESULTS_CACHE_MAX
# entries via OrderedDict LRU semantics — the unbounded dict that used
# to live here would grow to several GB after a few hours of UI
# browsing (one entry per (filter, sort) combo, each ~2 MB). Reads
# promote to the back; insertions push the oldest out the front.
# Invalidated whenever a new pipeline run completes (and on demand).
_RESULTS_CACHE_MAX = 8
_results_cache: OrderedDict[str, dict] = OrderedDict()
_results_lock = threading.Lock()


# Same idea for /api/clusters: cache the full payload per filter combo.
# The result depends only on (plantae, plantae_green, plantae_yellow)
# and the underlying cluster assignment, so a single entry per
# (plantae, threshold) tuple covers the UI's typical interaction
# pattern. Invalidated when a pipeline run completes.
_clusters_cache: dict[tuple, dict] = {}
_clusters_cache_lock = threading.Lock()


def _results_cache_get(key: str) -> dict | None:
    """Return the cached payload for `key`, or None. Promotes the key
    to the back of the LRU on hit so the most-recently-used entries
    survive eviction."""
    with _results_lock:
        if key in _results_cache:
            _results_cache.move_to_end(key)
            return _results_cache[key]
    return None


def _results_cache_put(key: str, value: dict) -> None:
    """Insert a payload, evicting the oldest entry if we'd exceed
    _RESULTS_CACHE_MAX. Combined with the move-to-end in GET, this
    makes the cache a strict LRU."""
    with _results_lock:
        _results_cache[key] = value
        while len(_results_cache) > _RESULTS_CACHE_MAX:
            _results_cache.popitem(last=False)


# ---- Profiling infrastructure --------------------------------------------
# /api/mem uses tracemalloc (stdlib) to attribute Python heap usage to
# specific source lines. It is lazy-started on the first call so the
# steady-state cost is zero, and it can be hard-disabled with
# MEM_TRACE=0 if the ~10-30% per-allocation overhead is unwelcome.
#
# /api/perf reads a per-request duration ring buffer (sampled by the
# HTTP middleware below; ~1µs per request). The two endpoints together
# answer "what is eating memory" and "what is slow" without re-running
# the workload under memray.
MEM_TRACE = os.environ.get("MEM_TRACE", "1") == "1"
_TRACEMALLOC_ACTIVE = False
_TRACEMALLOC_BASELINE = None  # type: ignore[var-annotated]  # tracemalloc.Snapshot
_MEM_LOCK = threading.Lock()

_TIMING: deque[dict] = deque(maxlen=500)
_TIMING_LOCK = threading.Lock()


def _invalidate_results_cache(prefix: str = "all:") -> None:
    """Pop every cache entry whose key starts with `prefix` (or equals
    `prefix` with the trailing colon stripped). The /api/results cache
    uses keys like "all:sort=rank" / "all:sort=time" / "all:sort=added"
    plus per-filter variants like "all:cid=3:sp=…:sort=rank". Popping the
    bare literal "all" — which the previous code did — only ever matched
    a single key, so the cache silently went stale across runs.
    """
    bare = prefix.rstrip(":")
    with _results_lock:
        for k in list(_results_cache):
            if k.startswith(prefix) or k == bare:
                _results_cache.pop(k, None)

# Pipeline runs execute in a child process so BioCLIP's ~5.5 GB resident
# memory never lives in uvicorn. The child writes status to a JSON file
# the parent polls for /api/status and /api/events. The parent keeps
# a tiny registry of the active job; concurrent runs are prevented by
# checking the registry under a short-lived mutex.
#
# There is NO long-lived lock and NO reaper thread. Each POST /api/run
# checks `proc.is_alive()` while holding the registry mutex for a few
# microseconds. If the previous child died without cleanup, the
# next POST cleans up the stale entry before proceeding. There is no
# way for uvicorn to wedge on this state — the only failure mode is
# "two clicks within microseconds of each other both start a run",
# which is acceptable (and even then, the second one will see the
# first's proc alive and return 409).
_pipeline_jobs_lock = threading.Lock()
_pipeline_active_job: dict | None = None  # {"proc", "job_id", "status_path", "stop_event", "log_path"}


def _get_active_job() -> dict | None:
    """Return the active job if its child is still alive; otherwise
    clear the registry entry and return None. Lazy cleanup — the next
    reader is the cleanup point. Cheap (`is_alive` is a single os.kill
    probe) so we call this from every endpoint that reads the
    registry."""
    global _pipeline_active_job
    with _pipeline_jobs_lock:
        active = _pipeline_active_job
        if active is None:
            return None
        if not active["proc"].is_alive():
            _pipeline_active_job = None
            return None
        return active


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
    # Let DB errors bubble — a misleading `cached_assets: 0` is worse
    # than a 500 here, since the UI uses this to decide whether the
    # index has usable data.
    with storage.connect(DB_PATH) as conn:
        rows = storage.list_knn_versions(conn)
    total = sum(r["n"] for r in rows)
    out["cached_assets"] = int(total)
    out["cached_versions"] = rows
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
    # Pass `classifier=None` so the BioCLIP model isn't loaded unless a
    # run actually has BioCLIP work to do. The pipeline resolves the
    # singleton lazily on first predict() — for fully-cached runs (n_new
    # == 0 across kingdom, species, and bbox-bioclip) the model never
    # loads, saving ~5.5 GB of resident memory.
    out = run_pipeline(
        base_url=BASE_URL,
        api_key=API_KEY,
        query=query,
        n_samples=n_samples,
        top_k=top_k,
        db_path=DB_PATH,
        classifier=None,
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
                # Slim attribution for the run payload — same shape as
                # the grid row, uses the whole-image top-K plus
                # taxon_scores. The YOLO bbox-bioclip join is too
                # expensive to ship with every run result; the asset
                # detail endpoint already carries the full version.
                "plant_attribution": storage.compute_plant_attribution(
                    taxon_scores=r.taxon_scores,
                    top_predictions=[
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
                    detections=r.detections,
                ) if r.detections else storage.compute_plant_attribution(
                    taxon_scores=r.taxon_scores,
                    top_predictions=[
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
                ),
            }
            for r in out.results
        ],
        "plantae_green": PLANTAE_GREEN,
        "plantae_yellow": PLANTAE_YELLOW,
    }
    # Drop cache entries: the global "all" view (the run may have added
    # new photos) and any per-run entry (kept for backwards compat with
    # the old /api/results?run=... path). Also invalidate the cluster
    # names cache — HDBSCAN may reassign cluster_ids.
    with _results_lock:
        _invalidate_results_cache("all:")
        _results_cache.pop(str(out.run_id), None)
    _invalidate_clusters_cache()
    # Precompute the near-dup map so the user's first /api/results after
    # the pipeline finishes hits the cache. compute_near_dup_groups is
    # ~3.6s with 19k assets (chunked matmul); we run it here, in the
    # pipeline thread, instead of making the first page load pay it.
    # The on-disk cache is invalidated first so the next save reflects
    # the post-pipeline embeddings. Wrapped in try/except so a warm
    # failure doesn't break the pipeline response — the cache just
    # stays cold.
    t0 = time.perf_counter()
    try:
        storage.invalidate_global_near_dup_cache()
        storage.delete_near_dup_cache(NEAR_DUP_CACHE_PATH)
        with storage.connect(DB_PATH) as conn:
            storage.get_global_near_dup_groups(conn)
        storage.save_near_dup_cache(NEAR_DUP_CACHE_PATH)
        _log.info(
            "near-dup cache rewarmed after pipeline run in %.1fs",
            time.perf_counter() - t0,
        )
    except Exception:
        _log.exception("near-dup cache warm failed; cache stays cold")
    with storage.connect(DB_PATH) as conn:
        storage.invalidate_cluster_names(conn)
    return payload


async def _start_run(query: str, n_samples: int, top_k: int,
                     batch_size: int) -> dict:
    """Kick off a pipeline run in a child process. Returns immediately
    with the new job_id once the child is forked.

    Concurrency: only one run at a time. We check the active-job
    registry under a short-lived mutex; if a previous child died but
    its entry is still there, we clean it up here (no reaper thread).

    409 Conflict if a run is already in progress.
    """
    import multiprocessing as _mp

    job_id = uuid.uuid4().hex[:8]
    status_path = Path(DB_PATH).parent / f"pipeline-{job_id}.status.json"
    log_path = Path(DB_PATH).parent / f"pipeline-{job_id}.log"
    stop_event = _mp.Event()

    # If a previous run died without cleaning up its registry entry,
    # _get_active_job drops it and returns None here. The single-run
    # invariant is checked via the helper instead of holding the
    # mutex across the spawn.
    if _get_active_job() is not None:
        raise HTTPException(
            status_code=409,
            detail="pipeline is already running — see /api/status",
        )

    # Fork outside any lock so concurrent /api/run/stop requests are
    # never blocked by spawn latency.
    proc = _mp.Process(
        target=_pipeline_child_entry,
        args=(query, n_samples, top_k, batch_size,
              str(status_path), str(log_path), stop_event),
        daemon=True,
        name=f"pipeline-{job_id}",
    )
    proc.start()

    with _pipeline_jobs_lock:
        global _pipeline_active_job
        _pipeline_active_job = {
            "proc": proc,
            "job_id": job_id,
            "status_path": status_path,
            "log_path": log_path,
            "stop_event": stop_event,
            "started_at": time.time(),
        }
    _log.info("[pipeline] child started pid=%d job_id=%s", proc.pid, job_id)

    return {
        "run_id": None,  # populated by the child; UI polls /api/status
        "job_id": job_id,
        "pid": proc.pid,
        "status": "running",
    }


def _pipeline_child_entry(query: str, n_samples: int, top_k: int,
                          batch_size: int, status_path: str,
                          log_path: str, stop_event) -> None:
    """Body of the child process. Mirrors the pipeline state to a JSON
    file the parent polls, runs `run_pipeline` (which can hit ~5.85 GB
    RSS at species-rank predict), invalidates cluster names in SQLite
    so the next reader sees fresh data, and exits. The kernel
    reclaims all child pages when we return.

    Imports BioCLIP / open_clip / torch here in the child only — the
    parent (uvicorn) never sees them, so its RSS stays at ~640 MB.
    """
    import logging as _logging
    log = _logging.getLogger(__name__)
    handler = None
    try:
        # Mirror pipeline state to status_path so the parent's
        # /api/status and /api/events can render live progress
        # without sharing memory.
        import pipeline as _pipeline
        _pipeline.set_status_file(Path(status_path))

        # Per-job file log.
        handler = _logging.FileHandler(Path(log_path), mode="w")
        handler.setFormatter(_logging.Formatter("%(asctime)s %(message)s"))
        log.addHandler(handler)
        log.setLevel(_logging.INFO)

        _pipeline.run_pipeline(
            base_url=BASE_URL,
            api_key=API_KEY,
            query=query,
            n_samples=n_samples,
            top_k=top_k,
            db_path=DB_PATH,
            classifier=None,
            batch_size=batch_size,
            stop_event=stop_event,
        )
        # Cluster names may have changed (HDBSCAN re-assigns cluster_ids
        # on incremental new points). Drop the cached names in SQLite
        # so the next /api/clusters call rebuilds them. The child has
        # the DB connection already; doing it here keeps uvicorn out
        # of the post-run path entirely.
        try:
            with storage.connect(DB_PATH) as conn:
                storage.invalidate_cluster_names(conn)
        except Exception:
            log.exception("[pipeline] cluster-name invalidation failed")
        log.info("[pipeline] child finished cleanly")
    except Exception as e:
        log.exception("[pipeline] child crashed: %s", e)
        try:
            import pipeline as _pipeline
            _pipeline.error_state(f"{type(e).__name__}: {e}")
        except Exception:
            pass
    finally:
        if handler is not None:
            try:
                handler.close()
                log.removeHandler(handler)
            except Exception:
                pass


@asynccontextmanager
async def lifespan(_app: FastAPI):
    # NOTE: do NOT eagerly warm BioCLIP / YOLO here — browse-only mode
    # (~200 MB) is the default and we don't want to pay the ~3 GB cost
    # of model weights until the user actually hits POST /api/run.
    # The classifier is lazily constructed on first _run_and_store().
    #
    # The near-dup map (compute_near_dup_groups) is persisted to
    # `data/near_dup_cache.pkl` (1 MB for 19k assets) at the end of
    # every pipeline run. At startup we try to load it first — that
    # covers the common case where the DB hasn't changed since the
    # last warm. If the file is missing or corrupt, fall back to a
    # fresh compute (3.6s with 19k assets).
    if storage._GLOBAL_NEAR_DUP_CACHE is None:
        try:
            def _load_or_warm():
                # Try disk first.
                loaded = storage.load_near_dup_cache(NEAR_DUP_CACHE_PATH)
                if loaded is not None:
                    storage._GLOBAL_NEAR_DUP_CACHE = loaded
                    return ("loaded", len(loaded))
                # Nothing on disk — compute from SQLite.
                with storage.connect(DB_PATH) as conn:
                    n = conn.execute(
                        "SELECT COUNT(*) AS n FROM embeddings"
                    ).fetchone()["n"]
                if n == 0:
                    return ("empty", 0)
                storage.invalidate_global_near_dup_cache()
                with storage.connect(DB_PATH) as conn:
                    storage.get_global_near_dup_groups(conn)
                # Persist for next start.
                storage.save_near_dup_cache(NEAR_DUP_CACHE_PATH)
                return ("computed", n)
            t0 = time.perf_counter()
            how, n = await asyncio.to_thread(_load_or_warm)
            dt = time.perf_counter() - t0
            if how == "loaded":
                _log.info(
                    "near-dup cache loaded from %s in %.2fs (%d entries)",
                    NEAR_DUP_CACHE_PATH, dt, n,
                )
            elif how == "computed":
                _log.info(
                    "near-dup cache computed at startup: %d embeddings in %.1fs",
                    n, dt,
                )
        except Exception:
            _log.exception("startup near-dup load/warm failed; cache stays cold")
    if RUN_ON_STARTUP:
        try:
            await _start_run(QUERY, N_SAMPLES, TOP_K, BATCH_SIZE)
        except HTTPException as e:
            # Don't crash startup, but log it — otherwise the user
            # wonders why the auto-run never happened.
            _log.warning(
                "startup auto-run failed: %s — %s",
                e.status_code, e.detail,
            )
    yield


app = FastAPI(title="Immich × BioCLIP", lifespan=lifespan)


@app.middleware("http")
async def _timing_middleware(request: Request, call_next):
    """Record per-request duration into the /api/perf ring buffer.

    Excludes the profiling endpoints themselves (so polling /api/perf
    doesn't drown out real work) and the long-lived SSE stream (one
    entry per connection lifetime, not useful for slow-load diagnosis).
    """
    path = request.url.path
    if path in ("/api/perf", "/api/mem", "/api/events"):
        return await call_next(request)
    t0 = time.perf_counter()
    status = 500
    try:
        response = await call_next(request)
        status = response.status_code
        return response
    finally:
        _TIMING.append({
            "t": t0,
            "method": request.method,
            "path": path,
            "status": status,
            "duration_ms": round((time.perf_counter() - t0) * 1000.0, 3),
        })


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
async def clusters_endpoint(
    plantae: str | None = Query(default=None, pattern="^(plant|non_plant|uncertain)$"),
) -> dict:
    # cluster_summary walks every (cluster_id, top-1 prediction) pair
    # with 3 nested sub-queries per cluster — ~450 ms for ~700 clusters.
    # The result is cacheable across requests because it depends only
    # on (plantae, plantae_green, plantae_yellow), which are exactly
    # the inputs the UI sends. The cache is invalidated when a pipeline
    # run completes (HDBSCAN may reassign cluster_ids).
    cache_key = (plantae, PLANTAE_GREEN, PLANTAE_YELLOW)
    with _clusters_cache_lock:
        cached = _clusters_cache.get(cache_key)
    if cached is not None:
        return cached

    # cluster_summary + ensure_cluster_names can take ~100-500ms once
    # the name cache is cold (walks every prediction). Offload so the
    # event loop stays free for the SSE generator.
    def _build():
        with storage.connect(DB_PATH) as conn:
            rows = storage.cluster_summary(
                conn,
                plantae=plantae,
                plantae_green=PLANTAE_GREEN,
                plantae_yellow=PLANTAE_YELLOW,
            )
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

    payload = await asyncio.to_thread(_build)
    with _clusters_cache_lock:
        _clusters_cache[cache_key] = payload
    return payload


def _invalidate_clusters_cache() -> None:
    """Drop the /api/clusters cache. Called after a pipeline run
    (HDBSCAN may have reassigned cluster_ids) and on demand."""
    with _clusters_cache_lock:
        _clusters_cache.clear()


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
    progress to drive the UI status panel. Safe to read when idle.

    Kept for backwards-compat / scripts / debugging. The React UI now
    reads the same payload via /api/events (SSE) instead of polling
    this endpoint.

    When a child-process run is active, the parent's `pipeline.get_state()`
    returns its own empty state (state lives in the child). We read the
    child's status file instead — the child mirrors its state there on
    every set_state / _stage_* call."""
    active = _get_active_job()
    if active is not None:
        sp = active["status_path"]
        if sp.exists():
            try:
                snap = json.loads(sp.read_text())
                # Stamp the job_id so the client can correlate with
                # /api/run's response payload.
                snap["job_id"] = active["job_id"]
                return snap
            except (json.JSONDecodeError, OSError):
                pass  # fall through to local state
    return get_state()


@app.get("/api/logs")
async def logs(tail: int = Query(default=200, ge=1, le=5000)) -> dict:
    """Tail the file log. Lines are split on \\n; the last `tail` lines
    are returned (oldest first)."""
    return {"lines": read_recent_log_lines(tail)}


# ---- Real-time event stream (SSE) ---------------------------------------
# One connection replaces the five independent polling loops the UI used
# to drive (StatusPanel 1Hz, LogTail 2Hz, JobPanel 400ms, KnnBuildModal
# 1Hz, BackfillExifModal 1Hz). The server snapshots all live state under
# their respective locks and emits only when the snapshot changes —
# latency is bounded by the 200ms tick, well below what 1Hz REST polling
# could deliver.
#
# Payload shape (one JSON object per `data:` line):
#   {
#     "pipeline":           {status, stage, chunk_idx, ...},   # pipeline.get_state()
#     "plants_jobs":        {job_id: {...}, ...},
#     "cluster_album_jobs": {job_id: {...}, ...},
#     "kingdom_jobs":       {job_id: {...}, ...},
#     "clustering_jobs":    {job_id: {...}, ...},
#     "knn_build_jobs":     {job_id: {...}, ...},
#     "backfill_exif_jobs": {job_id: {...}, ...},
#     "logs":               {"lines": [...], "mtime": float},
#   }
# The per-registry REST endpoints (/api/album/.../job/{id},
# /api/knn/build/job/{id}, /api/assets/backfill-exif/job/{id}) stay in
# place so scripts and one-off curl calls still work.


def _snapshot_jobs(
    registry: dict, lock: threading.Lock
) -> dict[str, dict]:
    """Snapshot the {job_id: state} pairs of an in-memory job registry.
    Caller-safe: takes the registry's lock and copies each entry so
    the response is internally consistent even if the worker is mid-
    update. Drops `finished_at` (monotonic clock; not meaningful over
    the wire)."""
    with lock:
        out: dict[str, dict] = {}
        for jid, j in registry.items():
            s = dict(j)
            s.pop("finished_at", None)
            out[jid] = s
        return out


def _build_event_snapshot() -> dict:
    """Assemble the full event payload. Each source takes its own short
    lock; the worst-case time is bounded by the sum of the lock hold
    times (microseconds). Safe to call from the SSE generator loop.

    `pipeline` is sourced from the child-process status file when a
    child run is active (see _pipeline_child_entry); otherwise it
    falls back to the in-process `pipeline.get_state()` (the parent's
    local copy — used for idle state)."""
    active = _get_active_job()
    pipeline_state: dict
    if active is not None:
        sp = active["status_path"]
        if sp.exists():
            try:
                pipeline_state = json.loads(sp.read_text())
                pipeline_state["job_id"] = active["job_id"]
            except (json.JSONDecodeError, OSError):
                pipeline_state = get_state()
        else:
            pipeline_state = get_state()
    else:
        pipeline_state = get_state()
    return {
        "pipeline": pipeline_state,
        "plants_jobs": _snapshot_jobs(_plants_jobs, _plants_jobs_lock),
        "cluster_album_jobs": _snapshot_jobs(
            _cluster_album_jobs, _cluster_album_jobs_lock
        ),
        "kingdom_jobs": _snapshot_jobs(_kingdom_jobs, _kingdom_jobs_lock),
        "clustering_jobs": _snapshot_jobs(
            _clustering_jobs, _clustering_jobs_lock
        ),
        "knn_build_jobs": _snapshot_jobs(
            _knn_build_jobs, _knn_build_jobs_lock
        ),
        "backfill_exif_jobs": _snapshot_jobs(
            _backfill_exif_jobs, _backfill_exif_jobs_lock
        ),
    }


# Cached tail of the pipeline log. The SSE generator refreshes this only
# when the log file's mtime advances — saves re-reading + re-parsing 200
# lines every 200ms when the pipeline is idle. (`read_recent_log_lines`
# itself is cheap, but reading + JSON-encoding still costs when nothing
# changed.)
_log_tail_cache: list[str] = []
_log_tail_mtime: float = 0.0


def _snapshot_logs() -> dict:
    """Tail the pipeline log file if its mtime has advanced. Returns the
    same shape as /api/logs plus a `mtime` field so the client can
    dedupe / know when the tail was last refreshed."""
    global _log_tail_cache, _log_tail_mtime
    # Attribute lookup (not bare `_log_file_path`) so we read the
    # post-attach_log_file value rather than the import-time None.
    path = pipeline._log_file_path
    if path is not None and path.exists():
        mt = path.stat().st_mtime
        if mt != _log_tail_mtime:
            _log_tail_cache = read_recent_log_lines(200)
            _log_tail_mtime = mt
    return {"lines": _log_tail_cache, "mtime": _log_tail_mtime}


@app.get("/api/events")
async def events_stream(request: Request) -> StreamingResponse:
    """Server-Sent Events stream. Emits a JSON snapshot whenever any of:
    pipeline state, any job registry, or the log file tail changes.
    Sends an SSE comment (`: keepalive`) every 15s so HTTP proxies
    don't drop an otherwise-quiet connection while idle.

    `request.is_disconnected()` lets the generator loop exit cleanly
    when the client closes the tab — no dangling coroutine."""

    async def gen():
        # Wrap the snapshot build so a single bad source (e.g. a typo
        # in a registry lock name — happened once during the SSE
        # rollout) doesn't hang the entire stream. We log + emit an
        # error event so the client at least sees something instead
        # of an empty body that closes after timeout.
        import logging as _logging
        _log = _logging.getLogger(__name__)

        def build_payload() -> str:
            snap = _build_event_snapshot()
            snap["logs"] = _snapshot_logs()
            return json.dumps(snap, default=str)

        try:
            last_payload = build_payload()
        except Exception as e:
            _log.exception("[events] initial snapshot failed")
            err = json.dumps({"error": f"{type(e).__name__}: {e}"})
            yield f"data: {err}\n\n"
            return
        yield f"data: {last_payload}\n\n"
        last_keepalive = time.monotonic()

        while True:
            if await request.is_disconnected():
                break
            try:
                payload = build_payload()
            except Exception as e:
                # Don't kill the stream on a transient snapshot error
                # — log + skip this tick so the next change still
                # reaches the client.
                _log.exception("[events] snapshot build failed")
                await asyncio.sleep(0.5)
                continue
            if payload != last_payload:
                yield f"data: {payload}\n\n"
                last_payload = payload
                last_keepalive = time.monotonic()
            else:
                now = time.monotonic()
                if now - last_keepalive > 15.0:
                    yield ": keepalive\n\n"
                    last_keepalive = now
            await asyncio.sleep(0.2)

    return StreamingResponse(
        gen(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            # Tell nginx (and friends) not to buffer the stream — otherwise
            # the client sees no events until the buffer fills or the
            # proxy decides to flush.
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/cache/invalidate")
async def cache_invalidate(prefix: str = Query(default="all:")) -> dict:
    """Drop entries from the in-process /api/results cache whose key starts
    with `prefix`. External scripts that write to the DB directly
    (e.g. scripts/person_apply_flips.py, scripts/person_classifier_backfill.py)
    should POST here to bust the cache so the next /api/results call
    re-reads from the DB. Without this, the running uvicorn worker
    keeps serving the pre-update payload until it is restarted.
    """
    _invalidate_results_cache(prefix)
    _invalidate_clusters_cache()
    return {"invalidated_prefix": prefix}


@app.get("/api/results")
async def get_results(
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=60, ge=1, le=300),
    cluster_id: int | None = Query(default=None),
    species: str | None = Query(default=None),
    genus: str | None = Query(default=None),
    family: str | None = Query(default=None),
    has_person: int | None = Query(default=None, ge=0, le=1),
    plantae: str | None = Query(default=None, pattern="^(plant|non_plant|uncertain)$"),
    sort: str = Query(default="rank", pattern="^(rank|time|added)$"),
) -> dict:
    """Return a paginated slice of every indexed asset. The `run` param
    was dropped — the view is now global so the cluster / taxonomy pills
    (which count across the whole DB) stay consistent with the grid.

    Optional filters narrow both the page slice AND the total count:
    - cluster_id: matches assets.cluster_id exactly
    - species/genus/family: matches predictions.position=0 (rank='species')
      label of the same value
    - has_person: 0 (no person) or 1 (has person). Plain WHERE on
      assets.has_person; NULL rows (unscanned) are excluded by both.
    - plantae: "plant" (≥PLANTAE_GREEN), "uncertain" (≥PLANTAE_YELLOW
      and <PLANTAE_GREEN), or "non_plant" (<PLANTAE_YELLOW). Uses the
      kingdom score stored in taxon_scores (already on every asset).
    - sort: 'rank' (default, server-defined order — Immich rank then
      cluster_id tiebreak), 'time' (newest captured_at first, NULLs
      last), or 'added' (newest first_seen_at first, NULLs last).
      Time/added sort happens server-side so the page slice is the
      actual most-recent N photos, not the top-N by rank re-sorted.

    `pagination.total` is the raw file count (every photo, pre-dedup)
    so the breadcrumb and "all" tile match. `pagination.total_pages`
    is computed from representative-card count so page buttons reflect
    what's actually browsable.
    """
    # Cache key includes every filter + sort param so different filter
    # combinations can't poison each other. has_person and plantae are
    # included so each value gets its own cached slice.
    has_filter = any(v is not None for v in (cluster_id, species, genus, family, has_person, plantae))
    cache_key = (
        f"all:cid={cluster_id}:sp={species}:ge={genus}:fa={family}:hp={has_person}:pl={plantae}:sort={sort}"
        if has_filter else f"all:hp={has_person}:pl={plantae}:sort={sort}"
    )

    cached = _results_cache_get(cache_key)
    if cached is not None:
        all_results = cached.get("results", [])
        total_assets = cached["pagination"]["total"]
        total_reps = len(all_results)
        start = page * page_size
        end = start + page_size
        payload = dict(cached)
        payload["results"] = all_results[start:end]
        payload["pagination"] = {
            "page": page,
            "page_size": page_size,
            "total": total_assets,
            "total_pages": (total_reps + page_size - 1) // page_size if total_reps else 0,
        }
        return payload

    # Cache miss → build from SQLite. The query asks for ALL matching
    # rows (page_size=10M) so we can sort/filter in Python once, then
    # slice per-page. For a 19k-row DB this is ~500ms of sync work —
    # offload so the event loop can keep ticking the SSE generator and
    # serve other requests in parallel.
    def _build():
        with storage.connect(DB_PATH) as conn:
            results, total_reps, total_assets = storage.get_all_results(
                conn,
                page=0,
                page_size=10_000_000,  # fetch all matching, slice in memory
                cluster_id=cluster_id,
                species=species,
                genus=genus,
                family=family,
                has_person=has_person,
                plantae=plantae,
                plantae_green=PLANTAE_GREEN,
                plantae_yellow=PLANTAE_YELLOW,
            )
            taxonomy = storage.get_global_taxonomy_summary(
                conn,
                plantae=plantae,
                plantae_green=PLANTAE_GREEN,
                plantae_yellow=PLANTAE_YELLOW,
            )
            if total_assets == 0:
                return {"error": "no assets match this filter", "results": []}, results
            # Server-side time sort. Without this, the client-side sortRows()
            # would only re-order the page slice the server already picked by
            # rank — recent photos with low Immich rank never appear. Split +
            # sort-with-reverse + concat puts NULLs last without a custom
            # comparator; ISO 8601 strings sort chronologically
            # lexicographically.
            if sort == "time":
                with_ts = [r for r in results if r.get("captured_at")]
                without_ts = [r for r in results if not r.get("captured_at")]
                with_ts.sort(key=lambda r: r["captured_at"], reverse=True)
                results = with_ts + without_ts
            elif sort == "added":
                # "added ↓" — newest first_seen_at first. first_seen_at is
                # set by `DEFAULT (datetime('now'))` on the initial INSERT
                # and is never overwritten by upsert, so it's the most
                # recently *ingested* asset. NULLs sort last (shouldn't
                # happen given the schema, but defensive).
                with_ts = [r for r in results if r.get("first_seen_at")]
                without_ts = [r for r in results if not r.get("first_seen_at")]
                with_ts.sort(key=lambda r: r["first_seen_at"], reverse=True)
                results = with_ts + without_ts
            start = page * page_size
            end = start + page_size
            page_results = results[start:end]
            payload = _serialize_global(
                page_results,
                pagination={
                    "page": page,
                    "page_size": page_size,
                    "total": total_assets,
                    "total_pages": (total_reps + page_size - 1) // page_size if total_reps else 0,
                },
                taxonomy=taxonomy,
            )
        return payload, results

    payload, results = await asyncio.to_thread(_build)
    if "results" in payload and not payload.get("error"):
        # Cache the FULL (unsliced) results so subsequent pages don't re-fetch.
        # The LRU cache (max _RESULTS_CACHE_MAX entries) bounds total memory.
        full_payload = dict(payload)
        full_payload["results"] = results
        _results_cache_put(cache_key, full_payload)
    return payload


@app.get("/api/asset/{asset_id}")
async def asset_detail(asset_id: str) -> dict:
    """Single-asset detail for the photo modal: full top-K predictions,
    kingdom taxon scores, YOLO detections (with bbox-BioCLIP joined),
    and the cached KNN row (or null if no KNN has been computed).

    404 if the asset isn't known to the local DB — e.g. it was deleted
    from Immich or never made it through the pipeline.
    """
    # The body does several SQLite queries + a numpy brute-force nearest-
    # neighbor search. On a large DB this can take 100-300ms — offload
    # so the event loop stays free for the SSE generator + other tabs.
    # HTTPException raised inside the thread doesn't propagate cleanly,
    # so we return a (status, body) tuple and translate after.
    def _build():
        with storage.connect(DB_PATH) as conn:
            # 1. Asset row metadata.
            asset_row = conn.execute(
                """SELECT asset_id, original_file_name, original_url,
                          thumbnail_url, cluster_id,
                          captured_at, latitude, longitude,
                          has_person
                   FROM assets WHERE asset_id = ?""",
                (asset_id,),
            ).fetchone()
            if asset_row is None:
                return 404, {"detail": f"asset {asset_id} not found in DB"}

            meta = dict(asset_row)
            # 2. Top-K full-image predictions (species rank).
            top_preds = storage.fetch_predictions(conn, asset_id=asset_id, rank="species")
            # 3. Kingdom taxon scores.
            ts_map = storage.fetch_taxon_scores(conn, asset_ids=[asset_id], rank="kingdom")
            taxon_scores = ts_map.get(asset_id) or {}
            # 4. YOLO detections with bbox-BioCLIP joined (filtered by min_score).
            # Bumped from 0.10 to 0.20: the new "plants in this image"
            # summary (compute_plant_attribution) absorbs the noise and
            # only surfaces boxes that justify drawing on the canvas.
            detections = storage.fetch_detections(
                conn, asset_id=asset_id, min_score=0.20, rank="species"
            )
            # 4a. Plant attribution summary — combines whole-image top-K
            # with YOLO bbox crops into a single "plants in this image"
            # list. The asset-detail modal renders this block first; the
            # top-K and YOLO sections below it move to a "debug" panel.
            plant_attribution = storage.compute_plant_attribution(
                taxon_scores=taxon_scores,
                top_predictions=top_preds,
                detections=detections,
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
            return 200, {
                "asset_id": meta["asset_id"],
                "original_file_name": meta["original_file_name"],
                "original_url": meta["original_url"],
                "thumbnail_url": meta["thumbnail_url"],
                "cluster_id": meta["cluster_id"],
                "captured_at": meta.get("captured_at"),
                "latitude": meta.get("latitude"),
                "longitude": meta.get("longitude"),
                # has_person: True / False / None (None = unscanned).
                # Drives the asset-detail "👤" chip in the frontend; pairs
                # with the grid's /api/results has_person filter.
                "has_person": (
                    None if meta.get("has_person") is None
                    else bool(meta["has_person"])
                ),
                "top_predictions": top_preds,
                "taxon_scores": taxon_scores,
                "detections": detections,
                "plant_attribution": plant_attribution,
                "library_neighbors": library_neighbors,
                "knn": knn_payload,
            }

    status, body = await asyncio.to_thread(_build)
    if status != 200:
        raise HTTPException(status_code=status, detail=body["detail"])
    return body


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
    # Membership filter for the Sync-to-IMMICH UX. None = any (default),
    # 0 = without people (also matches unscanned), 1 = with people.
    people_filter: int | None = None


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


def _plants_album_worker(
    job_id: str,
    min_score: float,
    force_recreate: bool,
    people_filter: int | None = None,
) -> None:
    """Body of the background job. Reads asset_ids, opens the album (or
    creates one), and chunked-PUTs the assets while updating `job`
    progress in-place. Owns the ImmichClient lifecycle and persists the
    final `album_meta` row at the end.

    `people_filter` narrows the asset set BEFORE bulk-add (any /
    without people / with people). Same semantics as the kingdom picker;
    threaded through so the Sync-to-IMMICH section can scope Plantae
    albums by membership without duplicating the album-create flow.
    """
    import time as _time
    chunk_size = 1000
    PLANTS_ALBUM_NAME = "Plantae (BioCLIP)"
    try:
        with storage.connect(DB_PATH) as conn:
            asset_ids = storage.find_plantae_assets(
                conn, min_score=min_score, people_filter=people_filter
            )
            saved = storage.get_album_meta(conn, "plants_album_id") or {}
            saved_album_id = (
                saved.get("album_id") if not force_recreate else None
            )

        with _plants_jobs_lock:
            _plants_jobs[job_id]["n_assets"] = len(asset_ids)
            _plants_jobs[job_id]["n_chunks"] = max(1, (len(asset_ids) + chunk_size - 1) // chunk_size)
            _plants_jobs[job_id]["min_score"] = min_score
            _plants_jobs[job_id]["people_filter"] = people_filter

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
    people_filter = params.people_filter
    # Pre-count so the confirm dialog knows the size. Cheap (one SQL
    # query); the worker re-runs it but we want a stable preview.
    with storage.connect(DB_PATH) as conn:
        preview_ids = storage.find_plantae_assets(
            conn, min_score=min_score, people_filter=people_filter
        )
        saved = storage.get_album_meta(conn, "plants_album_id") or {}

    job_id = uuid.uuid4().hex
    with _plants_jobs_lock:
        _plants_job_prune_locked()
        _plants_jobs[job_id] = {
            "status": "running",
            "min_score": min_score,
            "force_recreate": params.force_recreate,
            "people_filter": people_filter,
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
        args=(job_id, min_score, params.force_recreate, people_filter),
        daemon=True,
        name=f"plants-album-{job_id[:8]}",
    )
    t.start()

    return {
        "job_id": job_id,
        "min_score": min_score,
        "people_filter": people_filter,
        "n_assets": len(preview_ids),
        "n_chunks": max(1, (len(preview_ids) + 999) // 1000),
        "reused_existing": None,  # resolved by worker
        "saved_album_id": saved.get("album_id"),
    }


@app.get("/api/album/plants/preview")
async def plants_album_preview(
    min_score: float | None = None,
    people_filter: int | None = None,
) -> dict:
    """Count-only preview for the Plants-album button. Lets the UI show
    'Tag N photos' in its confirmation panel WITHOUT actually starting a
    job (so the user can cancel without spawning a background thread).

    `people_filter` (None / 0 / 1) narrows the count to the chosen
    membership scope so the per-action confirm modal shows the real
    number that the click will produce.
    """
    ms = min_score if min_score is not None else PLANTAE_GREEN
    with storage.connect(DB_PATH) as conn:
        ids = storage.find_plantae_assets(
            conn, min_score=ms, people_filter=people_filter
        )
        saved = storage.get_album_meta(conn, "plants_album_id") or {}
    chunk_size = 1000
    return {
        "min_score": ms,
        "people_filter": people_filter,
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


@app.post("/api/run/stop")
async def stop_run() -> dict:
    """Ask the running pipeline to stop at the next batch boundary.

    The worker checks `_stop_event` at every chunk boundary (plantae
    decision, species, kingdom, YOLO, bbox-bioclip) and raises
    `_PipelineStop` to unwind cleanly. Whatever the previous batch
    already wrote is committed — no work is lost.

    Returns:
      - 200 with `{stopped: true}` if a run was actually running and
        the stop flag was set
      - 200 with `{stopped: false, reason: "no run in progress"}` if
        nothing was running (no-op)

    With the child-process refactor, the worker lives in a separate
    process — its threading.Event is invisible from here. We set the
    multiprocessing.Event shared with the child instead. The pipeline
    module also sets the threading.Event on its own when it sees the
    multiprocessing one set, so any in-process callers (tests) also
    see the stop.
    """
    import logging
    _log = logging.getLogger(__name__)
    active = _get_active_job()
    if active is None:
        return {"stopped": False, "reason": "no run in progress"}
    active["stop_event"].set()
    # Also set the threading.Event in the parent so any in-process
    # callers (rare; only tests really) observe the stop too.
    import pipeline as _pipeline
    _pipeline.request_stop()
    _log.info(
        "[pipeline] stop requested by user; child pid=%s will unwind at next batch boundary",
        active["proc"].pid,
    )
    return {"stopped": True}


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
# Lock for /api/events to read the registry safely. The worker thread
# writes here without holding a lock (single writer) but the SSE
# snapshot in app.py reads it concurrently, so we need to gate that.
_knn_build_jobs_lock = threading.Lock()


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
        _invalidate_results_cache("all:")
    _invalidate_clusters_cache()
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
    # Membership filter — None = any, 0 = without people, 1 = with
    # people. Applied via storage.get_assets_in_cluster(...,
    # people_filter=...) so an unscanned asset counts as "without
    # people" until the person stage runs.
    people_filter: int | None = None


class KingdomAlbumParams(BaseModel):
    kingdom: str = "Plantae"     # "Plantae" | "Fungi" | ...
    min_score: float | None = None  # default → kingdom-specific at call site
    force_recreate: bool = False
    people_filter: int | None = None


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
    job_id: str,
    cluster_id: int,
    force_recreate: bool,
    people_filter: int | None = None,
) -> None:
    """Body of the per-cluster album sync job. Mirrors the Plants worker
    but: (a) reads assets via `cluster_albums`, (b) names the album
    "Cluster {id} ({cluster_name})", (c) persists the mapping via
    `set_cluster_album` instead of `set_album_meta`.

    `people_filter` narrows the asset set BEFORE bulk-add (any /
    without people / with people). Same NULL-as-without-people semantics
    as the kingdom picker so unscanned assets don't drop silently.
    """
    import time as _time
    chunk_size = 1000
    try:
        with storage.connect(DB_PATH) as conn:
            asset_ids = storage.get_assets_in_cluster(
                conn, cluster_id, people_filter=people_filter
            )
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
            _cluster_album_jobs[job_id]["people_filter"] = people_filter

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
async def cluster_album_preview(
    cluster_id: int,
    people_filter: int | None = None,
) -> dict:
    """Count-only preview for the per-cluster sync button. Lets the UI
    show 'Tag N photos as cluster album' in its confirm panel without
    spawning a background job (so the user can cancel without any work
    happening).

    `people_filter` (None / 0 / 1) narrows the count to the chosen
    membership scope so the per-action confirm modal shows the real
    number that the click will produce.
    """
    with storage.connect(DB_PATH) as conn:
        n = storage.count_assets_in_cluster(
            conn, cluster_id, people_filter=people_filter
        )
        saved = storage.get_cluster_album(conn, cluster_id)
    return {
        "cluster_id": cluster_id,
        "people_filter": people_filter,
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
        n = storage.count_assets_in_cluster(
            conn, cluster_id, people_filter=params.people_filter
        )
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
            "people_filter": params.people_filter,
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
        args=(job_id, cluster_id, params.force_recreate, params.people_filter),
        daemon=True,
        name=f"cluster-album-{cluster_id}-{job_id[:8]}",
    ).start()

    return {
        "job_id": job_id,
        "cluster_id": cluster_id,
        "people_filter": params.people_filter,
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


def _cluster_album_sync_all_worker(
    job_id: str,
    force_recreate: bool,
    people_filter: int | None = None,
) -> None:
    """Body of the bulk 'sync every cluster as an album' job. Iterates
    over every cluster in `cluster_summary()` and runs the same flow as
    `_cluster_album_worker` for each, persisting the cluster→album
    mapping as we go so the UI shows partial progress.

    Progress is tracked at the cluster level (clusters_done / n_clusters)
    and asset level (photos_added / n_assets_total). The UI shows both
    in the progress bar text so the user knows the bulk run is moving.

    `people_filter` narrows each cluster's asset list before bulk-add.
    A cluster that becomes empty after the filter is SKIPPED (not
    synced as an empty album) so the Run page "Sync clusters (all)"
    button doesn't produce phantom albums for filtered-out clusters.
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
            _cluster_album_jobs[job_id]["people_filter"] = people_filter

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
                # single-cluster worker. The picker applies the same
                # people_filter for every cluster so the user gets a
                # uniform membership scope across the bulk run.
                with storage.connect(DB_PATH) as conn:
                    asset_ids = storage.get_assets_in_cluster(
                        conn, cluster_id, people_filter=people_filter
                    )
                    # Skip clusters that became empty after the filter
                    # — creating an empty Immich album adds noise
                    # without contributing to the requested scope.
                    if not asset_ids:
                        with _cluster_album_jobs_lock:
                            _cluster_album_jobs[job_id]["clusters_done"] = idx + 1
                            _cluster_album_jobs[job_id]["current_cluster_id"] = cluster_id
                        continue
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
    without clicking each pill individually.

    `people_filter` (optional, from the Sync-to-IMMICH defaults) is
    applied per-cluster inside the worker — clusters that have no assets
    matching the filter are skipped to avoid empty albums. The preview
    numbers below reflect the unfiltered counts so the confirm modal
    can show the "before" picture; the worker reports the actual
    filtered counts as it goes.
    """
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
            "people_filter": params.people_filter,
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
        args=(job_id, params.force_recreate, params.people_filter),
        daemon=True,
        name=f"cluster-album-all-{job_id[:8]}",
    ).start()

    return {
        "job_id": job_id,
        "scope": "all",
        "people_filter": params.people_filter,
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


class ClusterAlbumsRemoveParams(BaseModel):
    """Body for POST /api/album/cluster/remove. `cluster_ids=None`
    means "all currently synced clusters" (the default Run-page
    action). Pass an explicit list to scope the removal; the endpoint
    validates each id against the existing cluster_albums rows so a
    typo'd id is silently ignored rather than producing a confusing
    error.
    """
    cluster_ids: list[int] | None = None
    delete_in_immich: bool = False


@app.post("/api/album/cluster/remove")
async def bulk_remove_cluster_albums(
    params: ClusterAlbumsRemoveParams,
) -> dict:
    """Bulk-remove cluster→album mappings. One round-trip replaces N
    DELETE /api/album/cluster/{id} calls for the Sync-to-IMMICH
    "Remove synced clusters" button.

    Behaviour:
      - `cluster_ids=None` → every currently synced cluster
      - `delete_in_immich=True` → also DELETE the album in Immich
        (404 is treated as already-gone, same as the single DELETE)
      - Per-cluster errors are collected in `errors` so one failure
        (e.g. transient network) doesn't abort the whole bulk

    Returns `{removed, remote_deleted, errors: [{cluster_id, error}]}`.
    """
    with storage.connect(DB_PATH) as conn:
        if params.cluster_ids is None:
            all_rows = storage.list_cluster_albums(conn)
            target_ids = [int(r["cluster_id"]) for r in all_rows]
        else:
            target_ids = [int(c) for c in params.cluster_ids]

    if not target_ids:
        return {"removed": 0, "remote_deleted": 0, "errors": []}

    removed = 0
    remote_deleted = 0
    errors: list[dict] = []

    # Single ImmichClient lifecycle for the whole batch — opening /
    # closing per cluster would burn ~30ms × N on the LAN.
    client = _immich_client() if params.delete_in_immich else None
    try:
        for cluster_id in target_ids:
            try:
                with storage.connect(DB_PATH) as conn:
                    saved = storage.get_cluster_album(conn, cluster_id)
                    if saved is None:
                        # No mapping — quietly skip. Tells the caller
                        # this id was already gone; the UI's "sync
                        # status" list will refresh on the next read.
                        continue
                    storage.delete_cluster_album(conn, cluster_id)
                removed += 1
                if client is not None:
                    try:
                        client.delete_album(album_id=saved["album_id"])
                        remote_deleted += 1
                    except httpx.HTTPStatusError as e:
                        if e.response.status_code == 404:
                            # already gone in Immich — local row was
                            # the source of truth; count it as success
                            remote_deleted += 1
                        else:
                            errors.append({
                                "cluster_id": cluster_id,
                                "error": (
                                    f"DELETE /api/albums/{saved['album_id']} → "
                                    f"HTTP {e.response.status_code}"
                                ),
                            })
            except Exception as e:  # noqa: BLE001 — surface anything else
                errors.append({
                    "cluster_id": cluster_id,
                    "error": f"{type(e).__name__}: {e}",
                })
    finally:
        if client is not None:
            client.close()

    return {
        "removed": removed,
        "remote_deleted": remote_deleted,
        "errors": errors,
    }


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
    job_id: str,
    kingdom: str,
    min_score: float,
    force_recreate: bool,
    people_filter: int | None = None,
) -> None:
    """Body of the kingdom-album job. Mirrors the Plants worker but
    parameterized by kingdom (Plantae / Fungi / ...). Album name is
    "{kingdom} (BioCLIP)"; meta is persisted by kingdom name.

    `people_filter` narrows the asset set BEFORE bulk-add (any /
    without people / with people). Threaded through so the Fungi and
    Plantae buttons can independently scope by membership.
    """
    import time as _time
    chunk_size = 1000
    album_name = f"{kingdom} (BioCLIP)"
    try:
        with storage.connect(DB_PATH) as conn:
            asset_ids = storage.find_kingdom_assets(
                conn,
                kingdom=kingdom,
                min_score=min_score,
                people_filter=people_filter,
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
            j["people_filter"] = people_filter

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
    people_filter: int | None = Query(default=None),
) -> dict:
    """Count-only preview for the kingdom-album button. Lets the UI show
    'Tag N photos as Plantae album' in its confirm panel without spawning
    a job. Default threshold mirrors the env config for Plantae; Fungi
    callers should pass an explicit min_score.

    `people_filter` narrows the count to the chosen membership scope so
    the per-action confirm modal shows the real number the click will
    produce.
    """
    # Kingdom-specific default threshold when caller leaves it blank.
    # Plantae uses the legacy PLANTAE_GREEN; every other kingdom
    # (Fungi / Animalia / ...) falls back to FUNGI_MIN_SCORE_DEFAULT
    # which is also the right ballpark for BioCLIP-2 kingdom scores.
    if min_score is not None:
        ms = min_score
    elif kingdom == "Plantae":
        ms = PLANTAE_GREEN
    else:
        ms = FUNGI_MIN_SCORE_DEFAULT
    with storage.connect(DB_PATH) as conn:
        n = storage.count_kingdom_assets(
            conn, kingdom=kingdom, min_score=ms, people_filter=people_filter
        )
        saved = storage.get_kingdom_album_meta(conn, kingdom) or {}
    return {
        "kingdom": kingdom,
        "min_score": ms,
        "people_filter": people_filter,
        "n_assets": n,
        "n_chunks": max(1, (n + 999) // 1000),
        "saved_album_id": saved.get("album_id"),
    }


@app.post("/api/album/kingdom")
async def start_kingdom_album(params: KingdomAlbumParams) -> dict:
    """Start a kingdom-album job. Mirrors /api/album/plants but takes
    kingdom + min_score as parameters so it works for Plantae, Fungi,
    Animalia, etc. Returns a `job_id`; the UI polls
    /api/album/kingdom/job/{job_id} for progress.

    Kingdom-specific default threshold mirrors the preview endpoint so
    the worker doesn't re-derive the threshold after the preview was
    already computed (and to keep the album_meta value consistent).
    """
    if params.min_score is not None:
        min_score = params.min_score
    elif params.kingdom == "Plantae":
        min_score = PLANTAE_GREEN
    else:
        min_score = FUNGI_MIN_SCORE_DEFAULT
    with storage.connect(DB_PATH) as conn:
        n = storage.count_kingdom_assets(
            conn,
            kingdom=params.kingdom,
            min_score=min_score,
            people_filter=params.people_filter,
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
            "people_filter": params.people_filter,
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
        args=(
            job_id,
            params.kingdom,
            min_score,
            params.force_recreate,
            params.people_filter,
        ),
        daemon=True,
        name=f"kingdom-album-{params.kingdom}-{job_id[:8]}",
    ).start()

    return {
        "job_id": job_id,
        "kingdom": params.kingdom,
        "min_score": min_score,
        "people_filter": params.people_filter,
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


# ============================================================
# Sync-to-IMMICH user defaults
# ============================================================
# Persisted on the server (album_meta key="sync_defaults") so they
# survive browser changes and are visible to non-browser callers
# (curl / scripts). Seeded from env on first read so the user gets a
# sensible starting point without clicking Save.
#
# Shape:
#   {
#     "plants":   {"people_filter": None|0|1, "min_score": float},
#     "fungi":    {"people_filter": None|0|1, "min_score": float},
#     "clusters": {"people_filter": None|0|1},
#     "sync_all": {"people_filter": None|0|1},
#   }
# `clusters` is the per-cluster card button; `sync_all` is the
# bulk "Sync every cluster" button. They could diverge in the future
# (e.g., sync_all might want a different default scope), so we keep
# them separate slots.
_SYNC_DEFAULTS_KEY = "sync_defaults"


class SyncDefaultsPayload(BaseModel):
    """Body for PUT /api/settings/sync-defaults. Each slot is optional
    so a partial PUT (e.g. only update the Fungi score) doesn't reset
    the others; we merge into the existing row on the server."""
    plants: dict | None = None
    fungi: dict | None = None
    clusters: dict | None = None
    sync_all: dict | None = None


def _sync_defaults_seeded() -> dict:
    """Build the seed defaults from env vars. Used on first read when
    the album_meta row doesn't exist yet."""
    return {
        "plants": {
            "people_filter": SYNC_PEOPLE_FILTER_DEFAULT,
            "min_score": PLANTAE_GREEN,
        },
        "fungi": {
            "people_filter": SYNC_PEOPLE_FILTER_DEFAULT,
            "min_score": FUNGI_MIN_SCORE_DEFAULT,
        },
        "clusters": {
            "people_filter": SYNC_PEOPLE_FILTER_DEFAULT,
        },
        "sync_all": {
            "people_filter": SYNC_PEOPLE_FILTER_DEFAULT,
        },
    }


def _sync_defaults_normalize(stored: dict) -> dict:
    """Merge stored defaults with the current seed so newly-added
    fields (e.g. a future `force_recreate_default`) appear in the
    payload without forcing a Save round-trip. Also clamps invalid
    `people_filter` values to None so a stale row from an older
    client can't crash the UI.
    """
    seed = _sync_defaults_seeded()
    out: dict = {}
    for slot, defaults in seed.items():
        merged = dict(defaults)
        stored_slot = stored.get(slot) if isinstance(stored, dict) else None
        if isinstance(stored_slot, dict):
            for k, v in stored_slot.items():
                if k in merged:
                    merged[k] = v
        # Clamp people_filter to None | 0 | 1
        if "people_filter" in merged and merged["people_filter"] not in (None, 0, 1):
            merged["people_filter"] = None
        # Clamp min_score to a sane float
        if "min_score" in merged:
            try:
                merged["min_score"] = float(merged["min_score"])
            except (TypeError, ValueError):
                merged["min_score"] = defaults["min_score"]
        out[slot] = merged
    return out


@app.get("/api/settings/sync-defaults")
async def get_sync_defaults() -> dict:
    """Return the persisted Sync-to-IMMICH defaults, seeded from env
    on first read so the UI always has a complete shape to render.

    `seeded: bool` is True when the server just wrote the seed row,
    so the frontend can show "Defaults initialised from environment"
    on the very first page load after upgrade.
    """
    with storage.connect(DB_PATH) as conn:
        stored = storage.get_album_meta(conn, _SYNC_DEFAULTS_KEY)
    if stored is None:
        seed = _sync_defaults_seeded()
        with storage.connect(DB_PATH) as conn:
            storage.set_album_meta(conn, _SYNC_DEFAULTS_KEY, seed)
        return {"defaults": seed, "seeded": True}
    return {"defaults": _sync_defaults_normalize(stored), "seeded": False}


@app.put("/api/settings/sync-defaults")
async def put_sync_defaults(payload: SyncDefaultsPayload) -> dict:
    """Upsert a partial payload over the existing defaults. Missing
    slots are preserved, so a client can update only the Fungi score
    without resetting the other slots.

    Returns the merged defaults so the client can re-render with the
    authoritative values (clamps applied server-side).
    """
    incoming = payload.model_dump(exclude_none=True)
    with storage.connect(DB_PATH) as conn:
        stored = storage.get_album_meta(conn, _SYNC_DEFAULTS_KEY) or _sync_defaults_seeded()
    merged = dict(stored)
    for slot, updates in incoming.items():
        if not isinstance(updates, dict):
            continue
        merged.setdefault(slot, {})
        for k, v in updates.items():
            merged[slot][k] = v
    normalised = _sync_defaults_normalize(merged)
    with storage.connect(DB_PATH) as conn:
        storage.set_album_meta(conn, _SYNC_DEFAULTS_KEY, normalised)
    return {"defaults": normalised}


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
    # COUNT(DISTINCT asset_id) on embeddings can be slow on a 19k-row
    # table; offload so it doesn't stall the SSE generator's tick.
    def _build():
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

    return await asyncio.to_thread(_build)


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
    import logging
    _worker_log = logging.getLogger("reclassify_sample")
    def _worker():
        for aid in ids:
            try:
                # KNN reclassify runs via the existing pipeline helper.
                from pipeline import run_knn_reclassify
                run_knn_reclassify([aid])
            except Exception:
                # The HTTP response has already returned; we cannot
                # bubble this to the caller. Log it so a per-asset
                # failure (e.g. a missing embedding row) is visible
                # instead of vanishing into a daemon thread.
                _worker_log.warning(
                    "reclassify-sample failed for %s", aid, exc_info=True
                )
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
#
# Spawned as a background job (like KNN build) so the UI can show
# live progress + log tail during the ~30s run. The worker writes
# per-chunk progress lines to data/backfill-exif-{job_id}.log and the
# UI polls /api/assets/backfill-exif/job/{id} for status + /logs
# for the tail.

_backfill_exif_jobs: dict[str, dict] = {}
# Lock for /api/events — same reason as _knn_build_jobs_lock above.
_backfill_exif_jobs_lock = threading.Lock()


def _backfill_exif_worker(job_id: str) -> None:
    """Body of the backfill thread. Walks every DB row, fetches the
    AssetResponseDTO from Immich, extracts EXIF, writes it back. Per-
    chunk progress + skip counts go to a per-job log file the UI
    tails."""
    import logging
    asyncio.run(_backfill_exif_async(job_id))


async def _backfill_exif_async(job_id: str) -> None:
    import logging
    job = _backfill_exif_jobs[job_id]
    log_path = Path(job["log_path"])
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger("backfill_exif")
    handler = logging.FileHandler(log_path, mode="w")
    handler.setFormatter(logging.Formatter("%(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    try:
        client = _immich_client()
        try:
            with storage.connect(DB_PATH) as conn:
                rows = conn.execute(
                    "SELECT asset_id FROM assets ORDER BY asset_id"
                ).fetchall()
                asset_ids = [r["asset_id"] for r in rows]

            if not asset_ids:
                logger.info("[backfill-exif] no assets in DB — nothing to do")
                job["updated_count"] = 0
                return

            logger.info(f"[backfill-exif] starting  total={len(asset_ids):,}")
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
                    except httpx.RequestError as e:
                        # Transport error — log and treat as missing so a
                        # 17k-asset batch keeps running; the missing
                        # count in the summary surfaces the magnitude.
                        logger.warning("[backfill-exif] %s fetch failed: %s", aid, e)
                        return None
                    return (aid, r.json()) if r.status_code == 200 else None

            # Stream progress instead of gather()-ing silently. The
            # old code waited ~30s on a 17k DB before logging anything;
            # as_completed surfaces results as they finish so the UI
            # sees steady scroll. Order doesn't matter — fetch_one
            # returns (aid, payload), and the DB write phase iterates
            # in arrival order without depending on the original list.
            fetch_log_every = max(1, min(200, len(asset_ids) // 25))
            results: list[tuple[str, dict] | None] = []
            tasks = [asyncio.create_task(fetch_one(aid)) for aid in asset_ids]
            done = 0
            for coro in asyncio.as_completed(tasks):
                try:
                    result = await coro
                except Exception as e:
                    # Backstop: fetch_one swallows httpx.RequestError,
                    # so anything reaching here is genuinely unexpected
                    # (e.g. JSON decode failure). Log it loudly so we
                    # don't silently count a programming error as a
                    # "missing" asset.
                    logger.warning(
                        "[backfill-exif] fetch_one raised: %s", e, exc_info=True
                    )
                    result = None
                results.append(result)
                done += 1
                if done % fetch_log_every == 0 or done == len(asset_ids):
                    logger.info(
                        f"[backfill-exif] fetch progress  "
                        f"{done:,}/{len(asset_ids):,}"
                    )
            parsed = [
                (aid, ImmichClient.extract_exif(payload))
                for aid, payload in results
                if payload is not None
            ]
            # Asset Immich returned nothing for (404 or network error) —
            # surface in the log so the user can see how many were
            # missing without opening a second tool.
            missing_count = sum(1 for r in results if r is None)
            if missing_count:
                logger.info(f"[backfill-exif] {missing_count:,} assets returned no data (deleted in Immich?)")
            logger.info(f"[backfill-exif] fetched  got={len(parsed):,}  missing={missing_count:,}")

            updated = 0
            skipped = 0
            no_exif = 0
            # Log every ~100 assets (or every asset for tiny batches)
            # so the UI sees steady scroll during the ~30s run. Tighter
            # than the fetch cadence — the DB phase is sequential, so
            # smaller chunks keep the log visibly moving.
            log_every = max(1, min(100, len(parsed) // 20))
            with storage.connect(DB_PATH) as conn:
                for i, (aid, exif) in enumerate(parsed, 1):
                    if not exif["captured_at"]:
                        no_exif += 1
                        continue
                    # Skip rows that already have a captured_at — the
                    # user can clear them via SQL or re-run the
                    # pipeline (which has its own refresh semantics).
                    existing = conn.execute(
                        "SELECT captured_at FROM assets WHERE asset_id = ?",
                        (aid,),
                    ).fetchone()
                    if existing and existing["captured_at"]:
                        skipped += 1
                    else:
                        storage.update_asset_exif(
                            conn,
                            asset_id=aid,
                            captured_at=exif["captured_at"],
                            latitude=exif["latitude"],
                            longitude=exif["longitude"],
                        )
                        updated += 1
                    # Log every N rows regardless of skip/update so the
                    # user sees steady scroll even when 90% of rows are
                    # already-set.
                    if i % log_every == 0:
                        logger.info(
                            f"[backfill-exif] progress  "
                            f"{i:,}/{len(parsed):,}  updated={updated:,}  "
                            f"skipped(already-set)={skipped:,}"
                        )
                # Commit before the `with` block closes — otherwise
                # every UPDATE rolls back on conn.close(). Python's
                # sqlite3 default isolation is deferred; the context
                # manager only closes, never commits.
                conn.commit()
            logger.info(
                f"[backfill-exif] done  "
                f"updated={updated:,}  skipped(already-set)={skipped:,}  "
                f"no-exif={no_exif:,}  missing-from-immich={missing_count:,}"
            )
            job["updated_count"] = updated
            job["skipped_count"] = skipped
            job["no_exif_count"] = no_exif
            job["missing_count"] = missing_count
        finally:
            client.close()
    except Exception as e:
        logger.exception("[backfill-exif] failed")
        job["error"] = f"{type(e).__name__}: {e}"
    finally:
        logger.removeHandler(handler)
        handler.close()
        if job["status"] != "error":
            job["status"] = "done"
        _results_cache.clear()  # sort-by-time needs fresh captured_at
        job["finished_at"] = datetime.now(timezone.utc).isoformat()


@app.post("/api/assets/backfill-exif")
async def start_backfill_exif() -> dict:
    """Spawn a backfill thread, return its job_id. The UI polls
    /api/assets/backfill-exif/job/{job_id} for status and /logs for
    the live tail."""
    job_id = uuid.uuid4().hex[:8]
    log_path = Path(DB_PATH).parent / f"backfill-exif-{job_id}.log"
    _backfill_exif_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "log_path": str(log_path),
        "updated_count": 0,
        "skipped_count": 0,
        "no_exif_count": 0,
        "missing_count": 0,
        "error": None,
    }
    threading.Thread(target=_backfill_exif_worker, args=(job_id,), daemon=True).start()
    return {"job_id": job_id}


@app.get("/api/assets/backfill-exif/job/{job_id}")
async def poll_backfill_exif_job(job_id: str) -> dict:
    job = _backfill_exif_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown backfill job")
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "updated_count": job["updated_count"],
        "skipped_count": job["skipped_count"],
        "no_exif_count": job["no_exif_count"],
        "missing_count": job["missing_count"],
        "error": job["error"],
    }


@app.get("/api/assets/backfill-exif/logs")
async def get_backfill_exif_logs(job_id: str, tail: int = 50) -> dict:
    """Tail the per-job log file. The worker writes progress lines as
    it processes each chunk of assets."""
    job = _backfill_exif_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown backfill job")
    try:
        with open(job["log_path"]) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    return {"job_id": job_id, "lines": [l.rstrip() for l in lines[-tail:]]}


# ---- Person-detection backfill ---------------------------------------
# Standalone sweep that runs ONLY the person stage (no BioCLIP). Used
# to fill in `assets.has_person` for the ~19k existing assets without
# rerunning the slow BioCLIP pipeline. Spawned as a background job
# (same pattern as backfill-exif): POST returns {job_id, n_to_scan};
# UI polls /api/person/backfill/job/{id} for status + scanned counts
# and /logs for the live tail.

_person_backfill_jobs: dict[str, dict] = {}
# Stop signals per job_id, used to ask a running child process to
# unwind at the next batch boundary. multiprocessing.Event is shared
# with the child by reference (it's a sync primitive that crosses the
# fork/spawn boundary cleanly).
_person_backfill_stops: dict[str, "multiprocessing.Event"] = {}


def _person_backfill_child_entry(
    base_url: str,
    api_key: str,
    db_path: str,
    log_path: str,
    status_path: str,
    stop_event,  # multiprocessing.Event — passed by parent
) -> None:
    """Entry point for the multiprocessing child. Runs the standalone
    pipeline.run_person_backfill() and writes:
    - log lines to `log_path`
    - status JSON (scanned/with/without counts) to `status_path`
    - the final "done"/"error" status to `status_path` and the log

    The parent (uvicorn worker) reads `status_path` on each /job poll
    — JSON file beats multiprocessing.Manager().dict() for our access
    pattern (parent reads every 400ms, child writes every ~30s). No
    IPC machinery, no extra threads in the parent.
    """
    import logging as _logging
    log_path = Path(log_path)
    status_path = Path(status_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    logger = _logging.getLogger("person_backfill")
    for h in list(logger.handlers):
        if isinstance(h, _logging.FileHandler):
            logger.removeHandler(h)
            h.close()
    handler = _logging.FileHandler(log_path, mode="w")
    handler.setFormatter(_logging.Formatter("%(asctime)s %(message)s",
                                            datefmt="%H:%M:%S"))
    logger.addHandler(handler)
    logger.setLevel(_logging.INFO)
    try:
        # Count up front so the log line is meaningful.
        from storage import count_assets_without_person_scan
        with storage.connect(db_path) as conn:
            n_to_scan = count_assets_without_person_scan(conn)
        logger.info(f"[person-backfill] starting  n_to_scan={n_to_scan:,}")

        def progress_cb(scanned: int, with_p: int, without_p: int) -> None:
            logger.info(
                f"[person-backfill] progress  scanned={scanned:,}  "
                f"has_person={with_p:,}  no_person={without_p:,}"
            )

        result = run_person_backfill(
            base_url=base_url,
            api_key=api_key,
            db_path=db_path,
            progress_cb=progress_cb,
            status_path=status_path,
            stop_event=stop_event,
        )

        # Stop requested vs. natural completion — distinguish so the
        # UI can show "Stopped" vs "Done" cleanly.
        if stop_event.is_set():
            logger.info(
                f"[person-backfill] stopped  scanned={result['scanned_count']:,}"
            )
        else:
            logger.info(
                f"[person-backfill] done  scanned={result['scanned_count']:,}  "
                f"has_person={result['with_person_count']:,}  "
                f"no_person={result['without_person_count']:,}"
            )
    except Exception as e:  # noqa: BLE001 — surface anything to the UI
        logger.exception("[person-backfill] failed")
        try:
            import json as _json
            status_path.write_text(_json.dumps({
                "stage": "error",
                "error": f"{type(e).__name__}: {e}",
            }))
        except Exception:
            # The status JSON is what the UI polls to see what
            # happened. If we can't write it, log loudly so the user
            # can see the failure mode (full disk, bad path, ...) —
            # otherwise the job just looks stuck.
            logger.warning(
                "[person-backfill] could not write error status to %s",
                status_path, exc_info=True,
            )
    finally:
        handler.close()
        logger.removeHandler(handler)


@app.post("/api/person/backfill")
async def start_person_backfill() -> dict:
    """Spawn the standalone person-backfill process. Returns
    {job_id, n_to_scan} — the count tells the UI how much work is
    queued (and is what the confirm message should show).

    Runs as a separate **multiprocessing.Process** (not a thread) so
    YOLO's CPU-bound work doesn't hold the GIL and starve the
    FastAPI event loop. The child writes progress to a JSON status
    file; the parent reads it on each /job poll — no shared memory.
    """
    import multiprocessing as _mp
    # Concurrent-job guard. Even though each job is now in its own
    # process, two concurrent jobs would still contend on the SQLite
    # write lock. Refuse the second until the first finishes.
    running = [
        j for j in _person_backfill_jobs.values()
        if j["status"] == "running"
    ]
    if running:
        existing = running[0]
        return {
            "job_id": existing["job_id"],
            "n_to_scan": existing["n_to_scan"],
            "noop": False,
            "already_running": True,
        }
    # Count up front so the UI can show the work size in the confirm
    # modal without an extra round-trip.
    with storage.connect(DB_PATH) as conn:
        n_to_scan = storage.count_assets_without_person_scan(conn)
    if n_to_scan == 0:
        # Nothing to do — surface as a 200 with a structured payload
        # so the UI can show "all done" without spawning a process.
        return {"job_id": None, "n_to_scan": 0, "noop": True}
    job_id = uuid.uuid4().hex[:8]
    log_path = Path(DB_PATH).parent / f"person-backfill-{job_id}.log"
    status_path = Path(DB_PATH).parent / f"person-backfill-{job_id}.status.json"
    stop_event = _mp.Event()
    _person_backfill_stops[job_id] = stop_event
    _person_backfill_jobs[job_id] = {
        "job_id": job_id,
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "finished_at": None,
        "log_path": str(log_path),
        "status_path": str(status_path),
        "n_to_scan": n_to_scan,
        "scanned_count": 0,
        "with_person_count": 0,
        "without_person_count": 0,
        "error": None,
    }
    # Resolve Immich creds in the parent (env vars, not from the
    # singleton which is in this process anyway).
    ic = _immich_client()
    base_url = ic.base_url
    api_key = ic._client.headers["x-api-key"]
    proc = _mp.Process(
        target=_person_backfill_child_entry,
        args=(base_url, api_key, DB_PATH, str(log_path),
              str(status_path), stop_event),
        daemon=True,
        name=f"person-backfill-{job_id}",
    )
    proc.start()
    _person_backfill_jobs[job_id]["_proc"] = proc
    return {"job_id": job_id, "n_to_scan": n_to_scan, "noop": False, "already_running": False}


@app.post("/api/person/backfill/stop")
async def stop_person_backfill(job_id: str) -> dict:
    """Ask a running backfill to unwind at the next batch boundary.
    The child process checks the stop Event we share with it; the
    loop exits and marks the job as 'stopped' in its JSON status
    file. The poll endpoint then surfaces that."""
    job = _person_backfill_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown person-backfill job")
    if job["status"] != "running":
        return {"stopped": False, "reason": "job not running"}
    stop_event = _person_backfill_stops.get(job_id)
    if stop_event is not None:
        stop_event.set()
    return {"stopped": True, "job_id": job_id}


@app.get("/api/person/backfill/job/{job_id}")
async def poll_person_backfill_job(job_id: str) -> dict:
    job = _person_backfill_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown person-backfill job")
    # The child writes JSON status to `status_path`. Read it here so
    # progress is visible across processes. Falls back to the
    # in-memory dict (init values) before the child has had time to
    # write its first snapshot.
    status_path = job.get("status_path")
    if status_path:
        try:
            import json as _json
            snap = _json.loads(Path(status_path).read_text())
            # Update in-memory job from the JSON snapshot so /job
            # reflects the child's latest counts. The child is the
            # source of truth; the in-memory dict is the bookkeeping.
            if snap.get("stage") in ("running", "stopped", "error", "finished"):
                job["scanned_count"] = snap.get("scanned_count", job["scanned_count"])
                job["with_person_count"] = snap.get(
                    "with_person_count", job["with_person_count"]
                )
                job["without_person_count"] = snap.get(
                    "without_person_count", job["without_person_count"]
                )
                if snap.get("stage") == "running":
                    pass  # still running
                elif snap.get("stage") == "stopped":
                    job["status"] = "stopped"
                    if job.get("finished_at") is None:
                        job["finished_at"] = datetime.now(timezone.utc).isoformat()
                elif snap.get("stage") == "error":
                    job["status"] = "error"
                    job["error"] = snap.get("error", "unknown")
                    if job.get("finished_at") is None:
                        job["finished_at"] = datetime.now(timezone.utc).isoformat()
                elif snap.get("stage") == "finished":
                    job["status"] = "done"
                    if job.get("finished_at") is None:
                        job["finished_at"] = datetime.now(timezone.utc).isoformat()
                    # Once the child reports done, invalidate the
                    # results cache so the grid reflects new has_person.
                    if not job.get("_cache_invalidated"):
                        with _results_lock:
                            _invalidate_results_cache("all:")
                        _invalidate_clusters_cache()
                        job["_cache_invalidated"] = True
        except FileNotFoundError:
            pass  # child hasn't written its first snapshot yet
    return {
        "job_id": job["job_id"],
        "status": job["status"],
        "started_at": job["started_at"],
        "finished_at": job["finished_at"],
        "n_to_scan": job["n_to_scan"],
        "scanned_count": job["scanned_count"],
        "with_person_count": job["with_person_count"],
        "without_person_count": job["without_person_count"],
        "error": job["error"],
    }


@app.get("/api/person/backfill/logs")
async def get_person_backfill_logs(job_id: str, tail: int = 80) -> dict:
    """Tail the per-job log file. The worker writes progress lines
    after each outer page (500 assets) so the UI gets steady scroll."""
    job = _person_backfill_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown person-backfill job")
    try:
        with open(job["log_path"]) as f:
            lines = f.readlines()
    except FileNotFoundError:
        lines = []
    return {"job_id": job_id, "lines": [l.rstrip() for l in lines[-tail:]]}


@app.get("/api/person/backfill/preview")
async def preview_person_backfill() -> dict:
    """Cheap count-only call: how many assets would be scanned if the
    user kicked off the backfill right now. Drives the confirm-modal
    message in the UI. Mirrors the shape of `/api/album/.../preview`
    so the JobPanel config can reuse the same hasPreview flow."""
    with storage.connect(DB_PATH) as conn:
        n = storage.count_assets_without_person_scan(conn)
    return {"n_assets": n}


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


# ---- Profiling endpoints -----------------------------------------------
# Used during development to attribute memory and time without
# re-running the workload under memray + py-spy. Both are safe to
# poll, both skip themselves in the timing middleware.


@app.get("/api/mem")
async def _mem_endpoint(
    top: int = Query(default=25, ge=1, le=200),
    reset: bool = Query(default=False),
) -> dict:
    """Live memory stats.

    Returns:
      - `rss_mb / vms_mb / peak_mb / data_mb`: process memory from
        /proc/self/status — no Python overhead, available even when
        tracemalloc is disabled.
      - `top_by_line`: the top-N allocation sites (file + line), traced
        by `tracemalloc`. Lazy-started on the first call.
      - `top_by_module`: the same data, grouped by top-level package.
      - `growth_vs_baseline`: net allocation since the first call (or
        the last `?reset=1`). Reads new minus old per file:line.

    `tracemalloc` adds ~10-30% overhead per allocation. Disable with
    `MEM_TRACE=0` in the env to get only the /proc stats.

    Typical workflow for "what did this request just allocate?"::
        curl 'localhost:8765/api/mem?reset=1'   # set baseline
        curl -X POST 'localhost:8765/api/run'  # trigger the heavy call
        curl 'localhost:8765/api/mem'          # see growth_vs_baseline
    """
    # /proc/self/status gives process-level memory without touching
    # any Python allocator. Works even with tracemalloc off.
    rss_mb = vms_mb = peak_mb = data_mb = 0.0
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(("VmRSS:", "VmSize:", "VmPeak:", "VmData:")):
                    label = line.split(":")[0]
                    val = float(line.split()[1]) / 1024.0  # KB → MB
                    if label == "VmRSS": rss_mb = val
                    elif label == "VmSize": vms_mb = val
                    elif label == "VmPeak": peak_mb = val
                    elif label == "VmData": data_mb = val
    except OSError:
        pass

    payload: dict = {
        "rss_mb": round(rss_mb, 1),
        "vms_mb": round(vms_mb, 1),
        "peak_mb": round(peak_mb, 1),
        "data_mb": round(data_mb, 1),
        "tracemalloc_active": False,
        "top_by_line": [],
        "top_by_module": [],
        "growth_vs_baseline": [],
    }

    if not MEM_TRACE:
        return payload

    # Tracemalloc start is one-shot lazy. Frame depth 1 (just the
    # leaf frame) keeps the per-allocation overhead small (~5%);
    # the previous depth of 25 was the reason /api/mem stalled over
    # hours of running — every tracemalloc bookkeeping struct was
    # 25× larger. Set `MEM_TRACE=0` in the env to skip entirely.
    global _TRACEMALLOC_ACTIVE, _TRACEMALLOC_BASELINE
    with _MEM_LOCK:
        if not _TRACEMALLOC_ACTIVE:
            tracemalloc.start(1)
            _TRACEMALLOC_ACTIVE = True
            _TRACEMALLOC_BASELINE = tracemalloc.take_snapshot()
        if reset:
            _TRACEMALLOC_BASELINE = tracemalloc.take_snapshot()

    snap = tracemalloc.take_snapshot()

    by_line = [
        {
            "file": str(s.traceback[0].filename),
            "line": s.traceback[0].lineno,
            "size_mb": round(s.size / 1024 / 1024, 3),
            "count": s.count,
        }
        for s in snap.statistics("lineno")[:top]
    ]
    payload["top_by_line"] = by_line

    # Group by top-level package so the response stays scannable when
    # the line-level list is long. `site-packages/<pkg>/...` →
    # `<pkg>`. Local code under `/immich_bioclip/<module>.py` → module.
    by_module: dict[str, dict] = {}
    for entry in by_line:
        fname = entry["file"]
        if "site-packages/" in fname:
            mod = fname.split("site-packages/")[1].split("/")[0]
        elif "/immich_bioclip/" in fname:
            mod = fname.split("/immich_bioclip/")[1].split("/")[0].split(".")[0]
        elif "/.venv/lib/" in fname or "/lib/python" in fname:
            mod = "stdlib"
        else:
            mod = "other"
        agg = by_module.setdefault(mod, {"size_mb": 0.0, "count": 0})
        agg["size_mb"] += entry["size_mb"]
        agg["count"] += entry["count"]
    payload["top_by_module"] = sorted(
        [{"module": m, **v} for m, v in by_module.items()],
        key=lambda x: x["size_mb"], reverse=True,
    )[:top]

    # Diff against the baseline. `size_diff` is bytes added by an
    # allocation site since the baseline; negative values mean code
    # freed more than it allocated at that site.
    if _TRACEMALLOC_BASELINE is not None:
        diff = snap.compare_to(_TRACEMALLOC_BASELINE, "lineno")
        payload["growth_vs_baseline"] = [
            {
                "file": str(s.traceback[0].filename),
                "line": s.traceback[0].lineno,
                "size_diff_mb": round(s.size_diff / 1024 / 1024, 3),
                "count_diff": s.count_diff,
            }
            for s in diff[:top]
        ]

    payload["tracemalloc_active"] = _TRACEMALLOC_ACTIVE
    return payload


@app.get("/api/perf")
async def _perf_endpoint(
    window: int = Query(default=500, ge=1, le=5000),
    route: str | None = Query(
        default=None,
        description="Filter to a single route, e.g. 'POST /api/run'",
    ),
) -> dict:
    """Per-request timing for the ring buffer.

    `by_route` aggregates the last `window` requests by (method, path):
      - `first_ms`: duration of the first hit — the "first load" cost
        (model load, FAISS mmap, near-dup matmul, etc.)
      - `mean_ms` / `max_ms` / `last_ms`: steady-state latency
      - `first_at` / `last_at`: monotonic timestamps so you can see
        whether the slow first hit just happened or was 10 minutes ago.

    `slowest_recent` is the top 20 individual requests across the
    window — useful when a single weird request skews the aggregate.

    Filter to a single route with `?route=POST%20%2Fapi%2Frun`.
    """
    with _TIMING_LOCK:
        records = [
            r for r in _TIMING
            if route is None or f"{r['method']} {r['path']}" == route
        ][-window:]

    by_route: dict[str, dict] = {}
    for r in records:
        key = f"{r['method']} {r['path']}"
        agg = by_route.setdefault(key, {
            "method": r["method"],
            "path": r["path"],
            "count": 0,
            "first_ms": 0.0,
            "first_status": 0,
            "first_at": None,
            "total_ms": 0.0,
            "max_ms": 0.0,
            "last_ms": 0.0,
            "last_status": 0,
            "last_at": None,
        })
        agg["count"] += 1
        agg["total_ms"] += r["duration_ms"]
        if r["duration_ms"] > agg["max_ms"]:
            agg["max_ms"] = r["duration_ms"]
        agg["last_ms"] = r["duration_ms"]
        agg["last_status"] = r["status"]
        agg["last_at"] = r["t"]
        if agg["first_at"] is None or r["t"] < agg["first_at"]:
            agg["first_at"] = r["t"]
            agg["first_ms"] = r["duration_ms"]
            agg["first_status"] = r["status"]

    by_route_list = []
    for v in by_route.values():
        v["mean_ms"] = round(v["total_ms"] / v["count"], 3) if v["count"] else 0.0
        v["total_ms"] = round(v["total_ms"], 3)
        v["max_ms"] = round(v["max_ms"], 3)
        v["last_ms"] = round(v["last_ms"], 3)
        v["first_ms"] = round(v["first_ms"], 3)
        by_route_list.append(v)
    by_route_list.sort(key=lambda x: x["total_ms"], reverse=True)

    slowest = sorted(records, key=lambda r: r["duration_ms"], reverse=True)[:20]

    return {
        "window_size": len(records),
        "buffer_capacity": _TIMING.maxlen,
        "by_route": by_route_list,
        "slowest_recent": [
            {
                "method": r["method"],
                "path": r["path"],
                "status": r["status"],
                "duration_ms": r["duration_ms"],
                "ts": r["t"],
            }
            for r in slowest
        ],
    }


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