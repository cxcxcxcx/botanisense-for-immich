"""FastAPI web UI for BotaniSense for Immich — an Immich companion that
classifies plant photos via the BioCLIP-2 model and pushes the results
back to Immich as albums. BioCLIP-2 (the underlying ML engine) and the
app name "BotaniSense for Immich" are distinct: keep technical
references to BioCLIP/-2 alone, this module/docstring deals with the
user-facing product name.

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
import sys
import threading
import time
import traceback
import tracemalloc
import uuid
from collections import deque, OrderedDict
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import TYPE_CHECKING

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, Response, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import storage
from immich_client import ImmichClient, ImmichProbeError
import clusters
if TYPE_CHECKING:
    import numpy as np
import pipeline  # full module — the SSE generator reads pipeline._log_file_path
from pipeline import (
    attach_log_file,
    get_state,
    read_recent_log_lines,
    run_person_backfill,
    run_pipeline,
)

load_dotenv()

# --- Runtime configuration ---------------------------------------------------
#
# Two surfaces only (config.toml support was removed; see
# _migrate_legacy_config for the one-time bridge from an old file):
#
#   1. Environment variables — IMMICH_BASE_URL / IMMICH_API_KEY / QUERY /
#      … — how docker-compose and .env configure a deployment.
#   2. The Immich connection row in SQLite (album_meta key
#      "immich_connection"), written by POST /api/settings/immich from
#      the web UI (Welcome screen, or HealthBanner → "Fix connection").
#
# The Immich connection resolves per call with precedence env > DB, so a
# UI save takes effect on the next request without a restart. See
# immich_base_url() / immich_api_key() below.
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
# In-app scheduler state (UX-19b). _scheduler_loop sets this to the
# ISO timestamp of the next scheduled run; the StatusPanel surfaces
# it to the user. None when PIPELINE_CRON is unset.
_next_run_at_iso: str | None = None
# KNN reclassification — separate from the zero-shot pipeline. Set
# KNN_INVENTORY_DIR to the directory holding tol200m_bioclip2.faiss +
# labels.parquet + INDEX_VERSION.txt (or the official imageomics
# layout). Leave it unset / pointing at a non-existent dir to disable
# the KNN UI gracefully. Resolved per call so a changed env takes
# effect without a restart.
def KNN_INVENTORY_DIR() -> Path:
    """Resolve the KNN inventory dir fresh on every call (env only)."""
    return Path(os.environ.get("KNN_INVENTORY_DIR", ""))
KNN_K = int(os.environ.get("KNN_K", "10"))


# --- Immich connection (URL + API key) ---------------------------------------
#
# The API key is stored in the DB in plaintext — the same posture as a
# .env file on disk — and is never returned by any endpoint (only a
# boolean api_key_set). These are plain functions rather than captured
# constants so a connection saved from the web UI applies to the next
# request without a restart, including in pipeline child processes
# (which fork this module and call the same accessors).

def _connection_settings() -> dict:
    """Read the UI-saved connection row, tolerating a missing/unready DB."""
    try:
        with storage.connect(DB_PATH) as conn:
            return storage.get_connection_settings(conn) or {}
    except Exception:
        # Fresh file before init_db, or a schema-less DB (tests).
        return {}


def immich_base_url() -> str:
    """Resolved Immich base URL: env var, else the UI-saved row."""
    return (
        os.environ.get("IMMICH_BASE_URL", "")
        or str(_connection_settings().get("base_url") or "")
    )


def immich_api_key() -> str:
    """Resolved Immich API key: env var, else the UI-saved row."""
    return (
        os.environ.get("IMMICH_API_KEY", "")
        or str(_connection_settings().get("api_key") or "")
    )


def immich_connection_source() -> str:
    """Where the effective connection comes from: "env" | "db" | "none"."""
    if os.environ.get("IMMICH_BASE_URL", ""):
        return "env"
    if _connection_settings().get("base_url"):
        return "db"
    return "none"


def _require_immich_connection() -> tuple[str, str]:
    """Guard for endpoints that need a working Immich connection."""
    base_url, api_key = immich_base_url(), immich_api_key()
    if not base_url or not api_key:
        raise HTTPException(
            status_code=409,
            detail="Immich is not configured — enter the Immich URL and "
                   "API key in the app (Welcome screen, or 'Fix "
                   "connection' in the health banner).",
        )
    return base_url, api_key


def _migrate_legacy_config() -> None:
    """One-time bridge from the removed config.toml.

    config.toml is no longer a supported configuration surface. While a
    legacy file exists (checked on every boot, so this is idempotent):

      - [immich].base_url is copied into the DB connection row when the
        row has no URL yet — the file is going away; the value must not
        be lost. (The API key is NOT copied — it stays env-only unless
        the user enters it in the web UI.)
      - [knn].inventory_dir / [cors].origins are injected into the
        environment when the corresponding env var is unset — those are
        deployment knobs whose permanent home is the environment.

    Deleting config.toml completes the migration.
    """
    if sys.version_info >= (3, 11):
        import tomllib
    else:
        try:
            import tomli as tomllib
        except ImportError:
            _log.warning(
                "legacy config.toml present but no TOML reader available "
                "(python<3.11 without tomli) — ignoring it"
            )
            return
    path = Path(os.environ.get("BOTANISENSE_CONFIG") or "config.toml")
    if not path.exists():
        return
    try:
        with path.open("rb") as f:
            cfg = tomllib.load(f)
    except Exception as e:
        _log.warning("legacy config.toml unreadable (%s) — ignoring", e)
        return

    knn_dir = cfg.get("knn", {}).get("inventory_dir")
    if knn_dir and not os.environ.get("KNN_INVENTORY_DIR"):
        os.environ["KNN_INVENTORY_DIR"] = str(knn_dir)
        _log.info(
            "migrated [knn].inventory_dir from config.toml into "
            "KNN_INVENTORY_DIR for this process — set the env var "
            "(or delete config.toml) to make it permanent"
        )
    cors = cfg.get("cors", {}).get("origins")
    if cors and not os.environ.get("CORS_ORIGINS"):
        os.environ["CORS_ORIGINS"] = ",".join(str(o) for o in cors)
        _log.info(
            "migrated [cors].origins from config.toml into CORS_ORIGINS "
            "for this process — set the env var (or delete config.toml) "
            "to make it permanent"
        )
    base_url = cfg.get("immich", {}).get("base_url")
    if base_url:
        with storage.connect(DB_PATH) as conn:
            saved = storage.get_connection_settings(conn) or {}
            if not saved.get("base_url"):
                storage.set_connection_settings(
                    conn,
                    base_url=str(base_url),
                    api_key=str(saved.get("api_key") or ""),
                )
                _log.info(
                    "migrated [immich].base_url from config.toml into the "
                    "app database — config.toml is no longer read"
                )

# React SPA mount. After `npm run build` writes frontend/dist/, FastAPI
# serves it at /. Dev mode uses Vite on :5173 with proxy → :8765.
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"

storage.init_db(DB_PATH)
_migrate_legacy_config()
if not immich_base_url() or not immich_api_key():
    # Not an error: the app boots so the web UI can collect the
    # connection (Welcome screen). Every Immich-touching endpoint
    # answers 409 until it's saved.
    _log.warning(
        "Immich is not configured yet — enter the Immich URL and API "
        "key in the web UI, or set IMMICH_BASE_URL / IMMICH_API_KEY."
    )
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
# While a run is ACTIVE, entries also expire after this many seconds —
# the pipeline commits results per batch, and without the TTL the
# first mid-run request pinned its (usually empty) payload for the
# entire multi-hour run. Values are (payload, monotonic_ts) tuples.
_RESULTS_CACHE_MAX = 8
_RESULTS_CACHE_RUNNING_TTL = 10.0
_results_cache: "OrderedDict[str, tuple[dict, float]]" = OrderedDict()
_results_lock = threading.Lock()


# Same idea for /api/clusters: cache the full payload per filter combo.
# The result depends only on (plantae, plantae_green, plantae_yellow)
# and the underlying cluster assignment, so a single entry per
# (plantae, threshold) tuple covers the UI's typical interaction
# pattern. Invalidated when a pipeline run completes.
_clusters_cache: dict[tuple, dict] = {}
_clusters_cache_lock = threading.Lock()

# Centroid vectors per cluster (see clusters.cluster_centroids) backing
# /api/clusters/related. One global entry — like the payload cache above
# it depends only on the cluster assignment + embeddings, so it's
# invalidated at exactly the same moments (run completes, refit).
_cluster_sim_cache: dict[str, object] = {}
_CLUSTER_SIM_UNSET = object()


# ---- /thumb pipeline ----------------------------------------------------
# Two-layer cache that fixes the slow-grid paint:
#
# 1. ONE httpx.AsyncClient for the lifetime of the process. The previous
#    code did `async with httpx.AsyncClient(...) as c: c.get(...)` for
#    EVERY thumbnail request — no connection pooling, no HTTP keepalive,
#    so a 60-card grid triggered 60 fresh TCP+TLS handshakes to Immich.
#    Reusing one client lets httpx keep connections open between calls
#    (max_keepalive_connections below) and bounds concurrency so a
#    runaway grid can't exhaust Immich's worker pool.
#
# 2. Small in-memory LRU of thumbnail bytes keyed by (asset_id, size).
#    Thumbnails are content-addressable — they only change when the
#    underlying photo is replaced in Immich — so even a 500-entry cache
#    (~15 MB for ~250px thumbs) absorbs a full 60-card grid hit on
#    repeat visits. The Cache-Control header is still 24h so clients
#    (and any reverse proxy) also cache downstream.
_thumb_client: httpx.AsyncClient | None = None
_thumb_client_lock = threading.Lock()  # guards lazy init / shutdown / swap


def _new_thumb_client() -> httpx.AsyncClient:
    """Build the shared /thumb/* client from the CURRENT connection.

    Called at startup and again whenever the connection is re-saved via
    POST /api/settings/immich (the client captures base_url + api key
    in its headers, so it must be rebuilt when those change).
    """
    return httpx.AsyncClient(
        base_url=immich_base_url(),
        headers={"x-api-key": immich_api_key(), "Accept": "image/*"},
        timeout=30.0,
        limits=httpx.Limits(
            max_connections=50,
            max_keepalive_connections=20,
            keepalive_expiry=30.0,
        ),
    )


def _rebuild_thumb_client() -> None:
    """Swap in a thumb client for the new connection, closing the old.

    In-flight requests holding the old client may error once it closes;
    the browser retry covers that. No-op before lifespan startup has
    created the client.
    """
    global _thumb_client
    with _thumb_client_lock:
        old = _thumb_client
        _thumb_client = _new_thumb_client()
    if old is None:
        return
    try:
        asyncio.get_running_loop().create_task(old.aclose())
    except RuntimeError:
        old.close()  # no running loop in this thread
_THUMB_CACHE_MAX = 500
_thumb_cache: OrderedDict[tuple[str, str], tuple[bytes, str]] = OrderedDict()
_thumb_cache_lock = asyncio.Lock()  # all cache mutations

# Reference-photo cache for /api/knn/ref-thumb. Separate LRU from the
# Immich thumb cache: these are external gbif/eol images fetched over
# the public internet, so they're slower to miss and worth keeping, but
# they shouldn't evict Immich thumbnails (which the grid hammers).
_REF_THUMB_CACHE_MAX = 400
_ref_thumb_cache: OrderedDict[str, tuple[bytes, str]] = OrderedDict()
_ref_thumb_cache_lock = asyncio.Lock()
# Cap on a single reference image. gbif/eol media are typically
# 50-500 KB; anything past this is a misconfigured source (or a hostile
# one trying to fill our RAM) and gets rejected rather than cached.
_REF_THUMB_MAX_BYTES = 8 * 1024 * 1024
# Reference images are rendered as ~100px squares in a 3-col grid, but
# the sources serve originals (a typical iNaturalist photo is 1536×2048
# / ~1 MB). Downscale before caching: 6 neighbours × 1 MB is 6 MB over
# the wire for thumbnails, and the LRU would hold ~400 MB at its cap.
_REF_THUMB_EDGE = 320


def _results_cache_get(key: str) -> dict | None:
    """Return the cached payload for `key`, or None. Promotes the key
    to the back of the LRU on hit so the most-recently-used entries
    survive eviction.

    While a pipeline job is active, entries older than
    _RESULTS_CACHE_RUNNING_TTL are treated as misses: the search and
    species stages commit new rows per batch, and this cache used to
    pin the pre-run (often empty) grid for the whole multi-hour run —
    the classic "ran for hours and every page is still empty". Idle
    entries keep the previous never-expire behavior; run-end
    invalidation happens in the reaper (_invalidate_post_run_caches)."""
    running = _get_active_job() is not None
    with _results_lock:
        if key in _results_cache:
            value, ts = _results_cache[key]
            if running and (time.monotonic() - ts) > _RESULTS_CACHE_RUNNING_TTL:
                del _results_cache[key]
                return None
            _results_cache.move_to_end(key)
            return value
    return None


def _results_cache_put(key: str, value: dict) -> None:
    """Insert a payload, evicting the oldest entry if we'd exceed
    _RESULTS_CACHE_MAX. Combined with the move-to-end in GET, this
    makes the cache a strict LRU. Entries are stamped with a
    monotonic time for the run-active TTL in _results_cache_get."""
    with _results_lock:
        _results_cache[key] = (value, time.monotonic())
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
# There is NO long-lived lock and NO reaper thread. Each reader that
# touches the registry checks `proc.poll()` under the registry mutex;
# if the previous child died without cleanup, the reader reaps the
# stale entry (and invalidates run-derived caches) before proceeding.
# POST /api/run checks AND spawns inside the same critical section, so
# a concurrent click sees the first child alive and gets a 409 — the
# spawn never leaks an untracked worker.
_pipeline_jobs_lock = threading.Lock()
_pipeline_active_job: dict | None = None  # {"proc", "job_id", "status_path", "log_path"}
# Holds the last worker's terminal state when it ended in anything
# other than a clean "done" — either "error" (crash/OOM, or the worker's
# own error_state) or "stopped" (user clicked Stop and the run unwound
# cleanly at a batch boundary). Preferred over the "last completed run"
# fallback by the status snapshots until the next run starts, so a
# killed or stopped run never masquerades as a successful one.
_last_worker_outcome: dict | None = None


def _invalidate_post_run_caches() -> None:
    """Drop every in-process cache derived from the pipeline's DB
    writes. The worker is a separate process, so it cannot clear the
    parent uvicorn's caches itself — this runs exactly once, when the
    reaper notices the child has exited."""
    _invalidate_results_cache("all:")
    _invalidate_clusters_cache()
    # The near-dup sibling map is computed from all embeddings at
    # startup/warm; new photos from the run would otherwise show as
    # singletons (and stale sibling links persist) for the whole
    # process lifetime. Invalidated here; recomputed lazily on the
    # next /api/results call.
    storage.invalidate_global_near_dup_cache()


def _get_active_job_locked() -> dict | None:
    """Same as _get_active_job, but called WITH _pipeline_jobs_lock
    held (so _start_run can check-and-register atomically)."""
    global _pipeline_active_job
    active = _pipeline_active_job
    if active is None:
        return None
    if active["proc"].poll() is not None:
        # The worker exited. Trust whatever terminal state it
        # managed to write to its status file:
        #   "stopped" — user clicked Stop; the worker unwound at a
        #     batch boundary and called stopped_state(). NOT an
        #     error: the UI shows a neutral "stopped" badge.
        #   "error"   — the worker caught an exception and wrote
        #     error_state() itself.
        # Only when the file STILL says "running" did the worker
        # die without writing anything (SIGKILL / OOM hard-kill) —
        # that is the genuine "died unexpectedly" case.
        global _last_worker_outcome
        try:
            import json as _json

            status_file = Path(active["status_path"])
            snap = _json.loads(status_file.read_text())
            term = snap.get("status")
            if term == "running":
                snap["status"] = "error"
                snap["last_error"] = (
                    "worker process died unexpectedly "
                    "(crash or OOM) — see the job log"
                )
                status_file.write_text(_json.dumps(snap))
                term = "error"
            if term in ("error", "stopped"):
                _last_worker_outcome = {
                    "status": term,
                    "stage": snap.get("stage", ""),
                    "query": snap.get("query", ""),
                    "n_requested": snap.get("n_requested"),
                    "started_at": snap.get("started_at"),
                    "finished_at": snap.get("finished_at"),
                    "last_error": snap.get(
                        "last_error",
                        "worker process died unexpectedly"
                        if term == "error" else "stopped by user",
                    ),
                }
        except Exception:
            pass
        _pipeline_active_job = None
        # The child committed new rows per batch but, being a separate
        # process, cannot invalidate THIS process's caches. Do it here,
        # once, at the live→dead transition (done/error/stopped alike).
        _invalidate_post_run_caches()
        return None
    return active


def _get_active_job() -> dict | None:
    """Return the active job if its child is still alive; otherwise
    clear the registry entry and return None. Lazy cleanup — the next
    reader is the cleanup point. Cheap (`poll` is a waitpid) so we call
    this from every endpoint that reads the registry."""
    with _pipeline_jobs_lock:
        return _get_active_job_locked()


def _last_completed_run() -> dict | None:
    """Return a snapshot of the most recent finished run, read from
    SQLite. Used by /api/status and /api/events when there's no live
    child — gives the UI a "last completed" view instead of an empty
    idle state. Cheap (one indexed SELECT on the runs table)."""
    from datetime import datetime
    with storage.connect(DB_PATH) as conn:
        row = conn.execute(
            "SELECT id, query, n_new, n_cached, n_requested, "
            "       started_at, finished_at "
            "  FROM runs "
            " WHERE finished_at IS NOT NULL "
            " ORDER BY id DESC LIMIT 1"
        ).fetchone()
        if row is None:
            return None
        # Shape mirrors pipeline._state so the frontend doesn't need
        # to special-case this. stage="" with status="done" is what
        # finish_state writes after a successful run.
        started = row["started_at"]
        finished = row["finished_at"]
        elapsed = 0.0
        try:
            if started and finished:
                t0 = datetime.fromisoformat(started)
                t1 = datetime.fromisoformat(finished)
                elapsed = (t1 - t0).total_seconds()
        except (ValueError, TypeError):
            pass
        return {
            "status": "done",
            "stage": "",
            "chunk_idx": 0,
            "chunk_total": 0,
            "rate": 0.0,
            "eta_seconds": 0.0,
            "current_run_id": row["id"],
            "query": row["query"],
            "n_new": row["n_new"],
            "n_cached": row["n_cached"],
            "n_requested": row["n_requested"],
            "started_at": started,
            "finished_at": finished,
            "last_error": None,
            # Render the elapsed time as a single "stage_times" entry so
            # the UI's stage-times bar shows total runtime.
            "stage_times": {"total": elapsed},
            "stage_t0": None,
        }


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
# Path the cached index was loaded FROM. When KNN_INVENTORY_DIR() resolves
# to a different path, we invalidate _knn_index/_knn_labels/_knn_meta so a
# changed KNN_INVENTORY_DIR picks up a fresh index without a restart.
_knn_loaded_dir: Path | None = None


def _get_knn_index():
    """Return (faiss_index, labels_table, meta) or None if the inventory
    isn't built yet. Cached for the lifetime of the process; invalidated
    if KNN_INVENTORY_DIR() resolves to a different path."""
    global _knn_index, _knn_labels, _knn_meta, _knn_loaded_dir
    # Re-resolve every call so an env change takes effect on the next
    # request, no restart needed. Also: if the inventory_dir changed to
    # a different path, invalidate the cached index — the file might not
    # even exist there.
    inventory = KNN_INVENTORY_DIR()
    # Detect which layout is on disk (knn_io._load_index does the same).
    custom_index = inventory / "tol200m_bioclip2.faiss"
    official_index = inventory / "faiss" / "index.index"
    if not (custom_index.exists() or official_index.exists()):
        return None
    if _knn_index is not None and _knn_loaded_dir == inventory:
        return _knn_index, _knn_labels, _knn_meta
    with _lock:
        if _knn_index is None or _knn_loaded_dir != inventory:
            import knn_io as _knn_io
            _knn_index, _knn_labels, _knn_meta = _knn_io._load_index(inventory)
            _knn_loaded_dir = inventory
            _log.info(
                f"[knn.load] loaded {_knn_index.ntotal:,} vectors from {inventory}"
            )
    return _knn_index, _knn_labels, _knn_meta


def _knn_status_dict() -> dict:
    """Compact status payload for /api/knn-status. Cheap when the index
    is already loaded (no FAISS work). When the index isn't built, we
    still return a dict so the UI can render an informative empty state.

    The `building` flag is True iff a /api/knn/build child process is
    still running — the UI uses this to suppress the "reference index
    missing" alert while a build is in progress (the JobPanel already
    shows progress, and showing "missing" alongside an active build is
    confusing — looks like the build did nothing)."""
    out: dict = {
        # "" when unset — str(Path("")) is "." and the UI would print
        # that verbatim ("Looks for an index at: .").
        "inventory_dir": os.environ.get("KNN_INVENTORY_DIR", ""),
        "index_built": False,
        "building": any(
            j.get("status") == "running" for j in _knn_build_jobs.values()
        ),
        "version": None,
        "n_rows": 0,
        "dim": 0,
        "by_kingdom": {},
        "k_default": KNN_K,
        "cached_assets": 0,
        # Index health, from knn.validate_index() at load time. None when
        # the index isn't loaded. `healthy: False` means the stored
        # vectors aren't L2-normalized and the neighbours are unreliable
        # — the UI warns rather than silently presenting bad species.
        "healthy": None,
        "health_reason": "",
    }
    loaded = _get_knn_index()
    if loaded is None:
        return out
    idx, _lbl, meta = loaded
    out["index_built"] = True
    out["version"] = meta.get("version")
    out["n_rows"] = int(idx.ntotal)
    out["dim"] = int(idx.d)
    health = (meta or {}).get("health") or {}
    out["healthy"] = health.get("ok")
    out["health_reason"] = health.get("reason", "")
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

async def _start_run(query: str, n_samples: int, top_k: int,
                     batch_size: int,
                     stop_criteria_pct: float | None = None) -> dict:
    """Kick off a pipeline run in a child process. Returns immediately
    with the new job_id once the child is forked.

    Concurrency: only one run at a time. We check the active-job
    registry under a short-lived mutex; if a previous child died but
    its entry is still there, we clean it up here (no reaper thread).

    409 Conflict if a run is already in progress.
    """
    import subprocess

    # Fail fast with a clear message before spawning — the child would
    # only discover the missing connection at its first Immich call.
    _require_immich_connection()

    # The run child is a plain exec'd subprocess (`python -m
    # pipeline_worker`), NOT multiprocessing. Children born through
    # multiprocessing — fork or spawn — run the BioCLIP encode pinned
    # to ~1 core (~2.0 s/img, OpenMP pool never engaging); the
    # identical encode in a freshly exec'd interpreter gets full torch
    # parallelism (~0.33 s/img on 8 threads). pipeline_worker.py has
    # the diagnosis trail. Stops arrive as SIGTERM (see /api/run/stop).
    job_id = uuid.uuid4().hex[:8]
    status_path = Path(DB_PATH).parent / f"pipeline-{job_id}.status.json"
    log_path = Path(DB_PATH).parent / f"pipeline-{job_id}.log"
    worker_args = json.dumps({
        "base_url": immich_base_url(),
        "api_key": immich_api_key(),
        "query": query,
        "n_samples": n_samples,
        "top_k": top_k,
        "db_path": DB_PATH,
        "batch_size": batch_size,
        "stop_criteria_pct": stop_criteria_pct,
        "status_path": str(status_path),
        "log_path": str(log_path),
    })
    # Check-and-spawn atomically under the registry mutex. Spawning
    # first and checking afterwards orphaned the losing child on a 409
    # (a full BioCLIP worker running untracked, writing to the same
    # SQLite DB) and let two near-simultaneous clicks both register.
    # Popen is a few-ms fork/exec, so holding the lock across it never
    # noticeably blocks /api/run/stop.
    with _pipeline_jobs_lock:
        # Reap a dead previous run in the same critical section so a
        # stale entry can't race the new registration.
        if _get_active_job_locked() is not None:
            raise HTTPException(
                status_code=409,
                detail="pipeline is already running — see /api/status",
            )
        proc = subprocess.Popen(
            [sys.executable, "-m", "pipeline_worker"],
            cwd=Path(__file__).parent,
            env={
                **os.environ,
                # Worker args (incl. the Immich API key) ride in the
                # environment, not argv — argv is world-readable via
                # /proc/<pid>/cmdline.
                "PIPELINE_WORKER_ARGS": worker_args,
                # MKL_DYNAMIC=true (its default) lets MKL downgrade to its
                # sequential `def` kernel path per call; pin thread counts
                # explicitly so BLAS never loses its pool. (The stage that
                # actually dominated search was full-resolution PIL decode
                # — see pipeline._open_rgb_small.)
                "MKL_DYNAMIC": "FALSE",
                "MKL_NUM_THREADS": str(min(8, os.cpu_count() or 8)),
                "OMP_NUM_THREADS": str(min(8, os.cpu_count() or 8)),
            },
        )
        global _pipeline_active_job, _last_worker_outcome
        _pipeline_active_job = {
            "proc": proc,
            "job_id": job_id,
            "status_path": status_path,
            "log_path": log_path,
            "started_at": time.time(),
        }
        _last_worker_outcome = None
    _log.info("[pipeline] child started pid=%d job_id=%s", proc.pid, job_id)

    return {
        "run_id": None,  # populated by the child; UI polls /api/status
        "job_id": job_id,
        "pid": proc.pid,
        "status": "running",
    }


@asynccontextmanager
def _terminate_worker_children() -> None:
    """SIGTERM any running worker subprocess (pipeline run, KNN build,
    person backfill) at shutdown. Without this, a host/dev uvicorn
    restart leaves the child alive invisibly — doing ML work and
    per-batch DB writes alongside the new server, which will happily
    spawn a second concurrent run. (Docker stop kills the whole cgroup
    so containers don't need this; host restarts do.) We don't wait or
    escalate — the process is exiting, and a worker that ignores
    SIGTERM unwinds at its next batch boundary regardless."""
    procs = []
    with _pipeline_jobs_lock:
        active = _pipeline_active_job
        if active is not None and active["proc"].poll() is None:
            procs.append(active["proc"])
    for job in _knn_build_jobs.values():
        proc = job.get("proc")
        if proc is not None and proc.poll() is None:
            procs.append(proc)
    for job in _person_backfill_jobs.values():
        proc = job.get("_proc")
        if proc is not None and proc.poll() is None:
            procs.append(proc)
    for proc in procs:
        try:
            proc.terminate()
        except Exception:
            pass
    if procs:
        _log.warning(
            "[shutdown] sent SIGTERM to %d worker child(ren)", len(procs)
        )


async def lifespan(_app: FastAPI):
    # One shared httpx.AsyncClient for /thumb/* across the whole process
    # — see _thumb_client docstring. Keeping keepalive_connections at 20
    # matches Immich's per-IP limit on most installs; 50 max-connections
    # lets a single grid request fan out without head-of-line blocking.
    global _thumb_client
    _thumb_client = _new_thumb_client()
    try:
        await _lifespan_inner()
        yield
    finally:
        _terminate_worker_children()
        client, _thumb_client = _thumb_client, None
        if client is not None:
            await client.aclose()


async def _scheduler_loop(cron: str) -> None:
    """In-app pipeline scheduler (UX-19b).

    Format for `cron`: "HH:MM" 24-hour, daily. We deliberately avoid
    pulling in croniter/apscheduler for a single feature — the
    format is intentionally minimal. Users who need a full cron
    expression can run an external `crontab` line (documented in
    README.md → Known limits) until a richer scheduler ships.

    Loop: compute seconds-until-next HH:MM, sleep, kick off a run via
    _start_run (the subprocess worker — never in-process), repeat. The
    /api/status endpoint exposes the next scheduled time via
    _next_run_at_iso so the UI can show "Next run at: …".
    """
    global _next_run_at_iso
    try:
        hh, mm = cron.split(":")
        hh_i, mm_i = int(hh), int(mm)
        if not (0 <= hh_i <= 23 and 0 <= mm_i <= 59):
            raise ValueError("hour or minute out of range")
    except (ValueError, AttributeError) as e:
        _log.warning("scheduler: bad PIPELINE_CRON=%r (%s) — disabled", cron, e)
        return

    while True:
        # Seconds from now until the next HH:MM, in the container's
        # local zone (TZ env / mounted /etc/localtime — README documents
        # PIPELINE_CRON as a local HH:MM). If that time is past today,
        # roll forward to tomorrow.
        from datetime import timedelta
        now = datetime.now().astimezone()
        target = now.replace(hour=hh_i, minute=mm_i, second=0, microsecond=0)
        if target <= now:
            target = target + timedelta(days=1)
        sleep_seconds = max(1.0, (target - now).total_seconds())
        _next_run_at_iso = target.isoformat()
        _log.info("scheduler: next run at %s (in %.0fs)", _next_run_at_iso, sleep_seconds)
        try:
            await asyncio.sleep(sleep_seconds)
        except asyncio.CancelledError:
            _next_run_at_iso = None
            raise

        # Kick off the same subprocess worker as POST /api/run — never
        # run_pipeline in this process (BioCLIP in uvicorn = ~6 GB and no
        # job registry). A run already in progress makes _start_run raise
        # 409; that's the expected "skip today" case.
        try:
            if not immich_base_url() or not immich_api_key():
                _log.warning(
                    "scheduler: skipping run — Immich is not configured yet"
                )
                continue
            await _start_run(
                query=QUERY,
                n_samples=N_SAMPLES,
                top_k=TOP_K,
                batch_size=BATCH_SIZE,
            )
            _log.info("scheduler: scheduled run started")
        except HTTPException as e:
            if e.status_code == 409:
                _log.info("scheduler: a run is already active — skipping")
            else:
                _log.warning("scheduler: scheduled run failed: %s", e.detail)
        except Exception as e:  # noqa: BLE001
            _log.warning("scheduler: scheduled run failed: %s", e)


def _sweep_old_job_files(max_age_days: float = 7.0) -> None:
    """Per-job logs/status files accumulate in the data dir forever —
    pipeline-<job>.log/.status.json, knn-build-*.log, backfill-exif-*.log,
    person-backfill-*.log/.status.json — one pair per run/job for the
    process's lifetime. Sweep files older than the retention window at
    boot (the UI only tails the active/recent job). Also removes
    atomic-write *.tmp leftovers from processes that died mid-rename.
    /tmp scratch dirs are swept separately by the pipeline worker
    (pipeline._sweep_stale_scratch)."""
    data_dir = Path(DB_PATH).parent
    now = time.time()
    cutoff = now - max_age_days * 86400.0
    tmp_cutoff = now - 86400.0
    patterns = (
        "pipeline-*.log", "pipeline-*.status.json",
        "knn-build-*.log",
        "backfill-exif-*.log",
        "person-backfill-*.log", "person-backfill-*.status.json",
    )
    removed = 0
    try:
        for pat in patterns:
            for f in data_dir.glob(pat):
                try:
                    if f.is_file() and f.stat().st_mtime < cutoff:
                        f.unlink()
                        removed += 1
                except OSError:
                    continue
        # Atomic-write leftovers (status JSON / near-dup pickle). A live
        # write holds its .tmp for milliseconds, so the 1-day gate only
        # ever matches crashed writes.
        for f in data_dir.glob("*.tmp"):
            try:
                if f.is_file() and f.stat().st_mtime < tmp_cutoff:
                    f.unlink()
                    removed += 1
            except OSError:
                continue
        if removed:
            _log.info(
                "[cleanup] removed %d old job/log file(s) from %s",
                removed, data_dir,
            )
    except OSError:
        pass


async def _lifespan_inner() -> None:
    """Existing startup work (near-dup warm + optional auto-run +
    optional in-app scheduler), pulled out so lifespan() stays focused
    on the shared HTTP client."""
    _sweep_old_job_files()
    # Spawn the in-app scheduler task if PIPELINE_CRON is set. We
    # use a simple HH:MM daily schedule (e.g. "02:00") rather than a
    # full cron parser to avoid pulling in croniter/apscheduler for
    # a single feature. Format: "HH:MM" 24-hour. Empty / unset =
    # scheduler disabled (default).
    cron = os.environ.get("PIPELINE_CRON", "").strip()
    if cron:
        sched_task = asyncio.create_task(
            _scheduler_loop(cron), name="pipeline-scheduler",
        )
        _log.info("scheduler: PIPELINE_CRON=%r enabled", cron)
    else:
        sched_task = None

    # NOTE: do NOT eagerly warm BioCLIP / YOLO here — browse-only mode
    # (~200 MB) is the default and we don't want to pay the ~6 GB cost
    # of model weights until the user actually hits POST /api/run (the
    # classifier lives in the worker process, never here).
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


app = FastAPI(title="BotaniSense for Immich", lifespan=lifespan)

# CORS: BotaniSense is a self-hosted single-user app — the typical
# deployment is uvicorn serving both the SPA and the API (same-origin),
# so CORS doesn't apply. But in dev the SPA is served by Vite on
# :5173 and the API by uvicorn on :8765, and client.ts deliberately
# bypasses Vite's /api proxy for /api/* (the SSE /api/events stream
# would otherwise get buffered by Vite). That makes the dev case a
# cross-origin request and CORS becomes load-bearing.
#
# Default: allow the Vite dev origin only (see the block below).
# To widen (e.g. phone on LAN hitting uvicorn directly),
# set CORS_ORIGINS (comma-separated, e.g. "https://a.example.com,
# https://b.example.com"). An empty value means same-origin only —
# stricter, and distinct from leaving it unset. Allow_credentials
# stays False.
#
# Default: the Vite dev-server origin only. The old default ("*" = any
# origin) let any web page open in the user's browser read the API
# cross-origin (photo library, cluster data) and get JSON POSTs
# preflighted through. Production is same-origin (uvicorn serves the
# SPA), so only dev needs an entry.
_cors_env = os.environ.get("CORS_ORIGINS")
_cors_origins = (
    [o.strip() for o in _cors_env.split(",") if o.strip()]
    if _cors_env is not None
    else ["http://localhost:5173", "http://127.0.0.1:5173"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
    # SSE: the browser sends a preflight OPTIONS for EventSource.
    # Returning the right Access-Control-Allow-Origin is enough; no
    # special expose_headers needed.
)


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
    # Stop-criteria threshold expressed as a percent (0-100) of a
    # batch that must score above the Plantae/Fungi confidence floor
    # for the search to keep going. Lower = the pipeline stops sooner
    # (stricter — only batches with mostly plants continue). None
    # falls back to the module default in pipeline.py (30%).
    stop_criteria_pct: float | None = None


@app.get("/api/runs")
async def list_runs() -> dict:
    with storage.connect(DB_PATH) as conn:
        return {"runs": storage.list_runs(conn, limit=50)}


@app.get("/api/clusters")
async def clusters_endpoint(
    plantae: str | None = Query(default=None, pattern="^(plant|non_plant|uncertain)$"),
) -> dict:
    return await _clusters_payload(plantae)


async def _clusters_payload(plantae: str | None) -> dict:
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


async def _cluster_similarity_index() -> tuple[list[int], "np.ndarray"]:
    """(cluster_ids, centroid matrix) for /api/clusters/related, cached.

    Loading every embedding costs the same O(N) read as a refit, so the
    result is memoized in _cluster_sim_cache and dropped by
    _invalidate_clusters_cache (a refit can renumber clusters AND shift
    membership, and new embeddings move the centroids).
    """
    with _clusters_cache_lock:
        cached = _cluster_sim_cache.get("index", _CLUSTER_SIM_UNSET)
    if cached is not _CLUSTER_SIM_UNSET:
        return cached

    def _build():
        with storage.connect(DB_PATH) as conn:
            return clusters.cluster_centroids(conn)

    index = await asyncio.to_thread(_build)
    with _clusters_cache_lock:
        _cluster_sim_cache["index"] = index
    return index


@app.get("/api/clusters/related")
async def related_clusters_endpoint(
    cluster_id: int = Query(...),
    limit: int = Query(default=8, ge=1, le=24),
) -> dict:
    """Clusters closest in embedding space to `cluster_id`.

    Similarity = cosine between cluster centroids (mean of member
    embeddings, L2-normalized — see clusters.cluster_centroids). The
    queried cluster itself is excluded. Display fields ride along from
    the same cached payload /api/clusters serves (unfiltered tier, so
    counts match the cluster cards' default view), which also means a
    name-less/Mixed cluster shows up with exactly the label the cards
    grid uses.
    """
    cluster_ids, centroids = await _cluster_similarity_index()
    neighbors = clusters.nearest_clusters(
        cluster_ids, centroids, cluster_id=cluster_id, k=limit
    )
    if not neighbors:
        return {"related": [], "cluster_id": cluster_id}

    payload = await _clusters_payload(None)
    by_id = {r["cluster_id"]: r for r in payload.get("clusters", [])}
    related = []
    for nb in neighbors:
        meta = by_id.get(nb["cluster_id"], {})
        related.append({
            "cluster_id": nb["cluster_id"],
            "similarity": nb["similarity"],
            "name": meta.get("name", ""),
            "n": meta.get("n", 0),
            "sample_asset_id": meta.get("sample_asset_id"),
            "sample_thumbnail_url": meta.get("sample_thumbnail_url"),
            "top_genus": meta.get("top_genus", []),
        })
    return {"related": related, "cluster_id": cluster_id}


# cluster super-groups (the hierarchy level ABOVE clusters). group_id 0
# is the "Ungrouped" bucket (clusters with no close siblings); real
# families are numbered from 1 in the families list's (stable) order.
UNGROUPED_GROUP_ID = 0


async def _cluster_group_index() -> tuple[list[list[int]], list[int]]:
    """(member-cluster lists per group, ungrouped list), cached.

    The member lists ARE the group-id contract: group_id N (1-based) is
    `index[N - 1]`, group_id 0 is the ungrouped list. Families are
    sorted by photo count desc BEFORE numbering so both /api/cluster-groups
    cards and /api/results?cluster_group=N resolve through this exact
    list — one structure, no id/sort drift. Photo counts come from the
    same cached /api/clusters payload the cards use; all caches drop
    together on refit.
    """
    with _clusters_cache_lock:
        cached = _cluster_sim_cache.get("groups", _CLUSTER_SIM_UNSET)
    if cached is not _CLUSTER_SIM_UNSET:
        return cached

    cluster_ids, centroids = await _cluster_similarity_index()

    def _build():
        return clusters.build_cluster_groups(cluster_ids, centroids)

    families, solo = await asyncio.to_thread(_build)
    meta_by_id = {r["cluster_id"]: r for r in (await _clusters_payload(None)).get("clusters", [])}
    n_of = lambda member_ids: sum(
        int((meta_by_id.get(cid) or {}).get("n") or 0) for cid in member_ids
    )
    families_sorted = sorted(families, key=lambda m: (-n_of(m), m[0]))
    index = (families_sorted, solo)
    with _clusters_cache_lock:
        _cluster_sim_cache["groups"] = index
    return index


def _resolve_group_clusters(
    group_id: int, group_members: list[list[int]], ungrouped: list[int]
) -> list[int]:
    """Cluster-id list for a super-group id (0 = Ungrouped bucket)."""
    if group_id == UNGROUPED_GROUP_ID:
        return ungrouped
    if 1 <= group_id <= len(group_members):
        return group_members[group_id - 1]
    return []


def _group_card(group_id: int, member_ids: list[int], meta_by_id: dict) -> dict | None:
    """Card payload for one super-group: name from a genus vote across
    member clusters (weighted by their top-genus counts, same ≥40% noise
    rule as storage.compute_cluster_names), counts/thumbnail from the
    largest member's row. None when no member carries metadata.

    `members` carries per-cluster display rows (id/name/n, biggest
    first) for the GroupMembersStrip — capped at 12: for the Ungrouped
    bucket the full list would be hundreds of chips, and the strip is a
    browse affordance, not an exhaustive index.
    """
    n = 0
    votes: dict[str, int] = {}
    vote_total = 0
    sample_meta: dict | None = None
    member_rows: list[dict] = []
    for cid in member_ids:
        meta = meta_by_id.get(cid)
        if meta is None:
            continue
        n += int(meta.get("n") or 0)
        member_rows.append({
            "cluster_id": cid,
            "name": meta.get("name", "") or f"cluster #{cid}",
            "n": int(meta.get("n") or 0),
        })
        for g in (meta.get("top_genus") or [])[:2]:
            votes[g["genus"]] = votes.get(g["genus"], 0) + int(g.get("count") or 0)
            vote_total += int(g.get("count") or 0)
        if sample_meta is None or int(meta.get("n") or 0) > int(sample_meta.get("n") or 0):
            sample_meta = meta
    if sample_meta is None:
        return None
    member_rows.sort(key=lambda m: (-m["n"], m["cluster_id"]))
    # Naming, two regimes:
    #   - One member dominates the family (≥60% of photos): the family
    #     IS that cluster's identity — use its name. Without this, the
    #     giant "Mixed" cluster (8.6k photos on prod) joined a small
    #     hortensia family and its dominant genus voted the whole family
    #     "Prunus" — technically the vote winner, practically a lie.
    #   - Otherwise (a genuine peer group): genus vote, same ≥40% noise
    #     rule as storage.compute_cluster_names.
    largest = member_rows[0]
    if n and largest["n"] / n >= 0.60:
        name = largest["name"] or "Mixed"
    else:
        top_genus, top_count = (max(votes.items(), key=lambda kv: kv[1]) if votes else ("", 0))
        share = (top_count / vote_total) if vote_total else 0.0
        name = top_genus if share >= 0.40 else "Mixed"
    return {
        "group_id": group_id,
        "name": name,
        "n": n,
        "n_clusters": len(member_ids),
        "sample_asset_id": sample_meta.get("sample_asset_id"),
        "sample_thumbnail_url": sample_meta.get("sample_thumbnail_url"),
        "cluster_ids": member_ids,
        "members": member_rows[:12],
    }


@app.get("/api/cluster-groups")
async def cluster_groups_endpoint() -> dict:
    """Super-groups of clusters (the hierarchy level above clusters).

    Built by complete-linkage agglomerative clustering over the cluster
    centroids, cut at a cosine-distance threshold
    (clusters.CLUSTER_GROUP_DISTANCE, default 0.15) — a threshold, not a
    fixed K, so only genuinely-close clusters merge and the long tail
    lands in the Ungrouped bucket (group_id 0) instead of a meaningless
    mega-group. Families are sorted by photo count desc; group ids are
    their 1-based position in that list (stable within a cache
    generation; the cache drops on refit).

    `?cluster_group=N` on /api/results filters the photo grid to a
    group's member clusters, so these ids are the URL-stable contract
    between the two endpoints.
    """
    families, solo = await _cluster_group_index()
    payload = await _clusters_payload(None)
    meta_by_id = {r["cluster_id"]: r for r in payload.get("clusters", [])}
    groups = []
    for idx, member_ids in enumerate(families, start=1):
        card = _group_card(idx, member_ids, meta_by_id)
        if card is not None:
            groups.append(card)
    ungrouped = _group_card(UNGROUPED_GROUP_ID, solo, meta_by_id) if solo else None
    if ungrouped is not None:
        # The bucket's genus vote is meaningless by construction (its
        # members failed the family cut) — label it for what it is.
        ungrouped["name"] = "Ungrouped"
    return {"groups": groups, "ungrouped": ungrouped}


def _invalidate_clusters_cache() -> None:
    """Drop the /api/clusters cache. Called after a pipeline run
    (HDBSCAN may have reassigned cluster_ids) and on demand."""
    with _clusters_cache_lock:
        _clusters_cache.clear()
        _cluster_sim_cache.clear()


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
    """Pipeline state for the UI status panel.

    - During a run: child's live status file (mirrored per state change).
    - After a run completes: last-completed-run summary from SQLite.
      Without this fallback, /api/status returns the parent's empty
      idle state once the child exits, and the UI keeps showing the
      "Pipeline started" message with no final state.
    - Never run yet: parent's empty idle state.

    The returned dict also picks up `next_run_at` from the in-app
    scheduler (UX-19b) — null when PIPELINE_CRON is unset.
    """
    active = _get_active_job()
    if active is not None:
        sp = active["status_path"]
        if sp.exists():
            try:
                snap = json.loads(sp.read_text())
                snap["job_id"] = active["job_id"]
                snap["next_run_at"] = _next_run_at_iso
                return snap
            except (json.JSONDecodeError, OSError):
                pass
    last = await asyncio.to_thread(_last_completed_run)
    if last is not None:
        if isinstance(last, dict):
            last["next_run_at"] = _next_run_at_iso
        return last
    base = get_state()
    base["next_run_at"] = _next_run_at_iso
    return base


@app.get("/api/health")
async def health() -> dict:
    """Pre-flight check for the SPA HealthBanner.

    Best-effort: every field returns ok=True/False plus context, never
    raises. The UI decides what to surface (red for blockers, yellow
    for warnings) and the user decides what to do — this endpoint
    shouldn't refuse to respond just because, e.g., Immich is down.
    """
    # --- Immich reachability + API key ---
    immich_ok = False
    immich_version: str | None = None
    immich_err: str | None = None
    api_key_ok = False
    api_key_scope = "none"
    api_key_err: str | None = None
    _base_url = immich_base_url()
    _api_key = immich_api_key()
    if not _base_url or not _api_key:
        # Unconfigured: the web UI collects the connection. Don't hit
        # the network — both checks report the same actionable state.
        immich_err = "not configured"
        api_key_err = "not configured"
    else:
        try:
            # Cheap call: server info, no admin scope required.
            async with httpx.AsyncClient(base_url=_base_url, timeout=4.0) as c:
                r = await c.get("/api/server/ping", headers={"x-api-key": _api_key})
            if r.status_code == 200:
                immich_ok = True
                immich_version = r.json().get("version", None) if r.headers.get("content-type", "").startswith("application/json") else None
            else:
                immich_err = f"HTTP {r.status_code} from /api/server/ping"
        except Exception as e:
            immich_err = f"{type(e).__name__}: {e}"

        # Distinct check for the API key's scope. /api/albums returns a
        # paginated list and 401s without a valid key. We don't actually
        # need the data, just the status.
        try:
            async with httpx.AsyncClient(base_url=_base_url, timeout=4.0) as c:
                r = await c.get("/api/albums", headers={"x-api-key": _api_key}, params={"take": 1})
            if r.status_code == 200:
                api_key_ok = True
                api_key_scope = "ok"
            elif r.status_code in (401, 403):
                api_key_err = "invalid or revoked key"
            else:
                api_key_err = f"HTTP {r.status_code}"
        except Exception as e:
            api_key_err = f"{type(e).__name__}: {e}"

    # --- KNN index ---
    knn_path = str(KNN_INVENTORY_DIR()) if KNN_INVENTORY_DIR() else ""
    knn_built = False
    knn_version: str | None = None
    if knn_path:
        # Two layouts supported: custom (BotaniSense-built) and official
        # (imageomics/bioclip-image-search-lite preloaded).
        custom_index = Path(knn_path) / "tol200m_bioclip2.faiss"
        custom_version = Path(knn_path) / "INDEX_VERSION.txt"
        official_index = Path(knn_path) / "faiss" / "index.index"
        if custom_index.exists() and custom_version.exists():
            knn_built = True
            try:
                knn_version = custom_version.read_text().strip()
            except OSError:
                knn_version = None
        elif official_index.exists():
            knn_built = True
            knn_version = "imageomics-official"
        else:
            knn_built = False

    # --- BioCLIP-2 weights ---
    # HuggingFace stores snapshots under
    #   $HF_HOME/hub/models--imageomics--bioclip-2/snapshots/*/.
    # Earlier this code read $HF_HOME/models--imageomics--bioclip-2 (no
    # /hub/) which never matched the actual layout — a false-negative
    # that made the UI say "weights not downloaded yet" even though
    # they had been on disk since July.
    from huggingface_hub.constants import HF_HUB_CACHE
    bioclip_weights = False
    bioclip_dir = Path(HF_HUB_CACHE) / "models--imageomics--bioclip-2"
    if bioclip_dir.exists():
        for snap in bioclip_dir.glob("snapshots/*/"):
            if any(snap.glob("*.safetensors")) or any(snap.glob("*.bin")):
                bioclip_weights = True
                break

    # --- DB ---
    db_ok = Path(DB_PATH).exists()
    db_size = Path(DB_PATH).stat().st_size if db_ok else 0

    return {
        "immich": {"ok": immich_ok, "version": immich_version, "error": immich_err},
        "api_key": {"ok": api_key_ok, "scope": api_key_scope, "error": api_key_err},
        "knn": {
            "built": knn_built,
            "version": knn_version,
            "path": knn_path,
            "configured": bool(knn_path),
        },
        "bioclip": {
            "weights_present": bioclip_weights,
            "cache_path": str(bioclip_dir),
        },
        "db": {"ok": db_ok, "path": DB_PATH, "size_bytes": db_size},
    }


@app.get("/api/config")
async def public_config() -> dict:
    """Non-secret runtime config surfaced to the SPA.

    `immich_public_url` drives AssetDetail's "Open in Immich" button:
    IMMICH_PUBLIC_URL when set, else the effective base URL — the
    override matters when the server reaches Immich over a hostname
    the browser can't resolve (e.g. both in one docker-compose
    network: the server says http://immich:2283, the browser needs
    http://<lan-host>:2283).

    `base_url` + `api_key_set` + `source` drive the connection form
    (Welcome screen / "Fix connection"): the URL is not a secret and
    is returned so the form can pre-fill it, but the API key is only
    ever surfaced as a boolean — never its value.
    """
    return {
        "immich_public_url": os.environ.get("IMMICH_PUBLIC_URL")
        or immich_base_url(),
        "base_url": immich_base_url(),
        "api_key_set": bool(immich_api_key()),
        "source": immich_connection_source(),
        "env_pinned": bool(
            os.environ.get("IMMICH_BASE_URL")
            or os.environ.get("IMMICH_API_KEY")
        ),
    }


class ImmichConnectionParams(BaseModel):
    base_url: str
    # Blank = keep the already-saved key (the form pre-fills nothing
    # for the key; it can't show what it doesn't have).
    api_key: str = ""


@app.post("/api/settings/immich")
async def set_immich_connection(params: ImmichConnectionParams) -> dict:
    """Save the Immich connection from the web UI.

    Stores the URL + API key in the SQLite DB (album_meta key
    "immich_connection"); both resolve via env > DB, so an env-pinned
    deployment keeps working while the save waits in the wings —
    reported back as env_pinned so the form can say so.

    Returns the fresh /api/health payload so the form can show
    "connected" or the concrete error immediately, without a second
    round trip.
    """
    from urllib.parse import urlsplit

    url = params.base_url.strip().rstrip("/")
    if not url:
        raise HTTPException(status_code=422, detail="Immich URL is required")
    parsed = urlsplit(url)
    if (
        parsed.scheme not in ("http", "https")
        or not parsed.hostname
        or parsed.username
        or parsed.password
    ):
        raise HTTPException(
            status_code=422,
            detail="Enter a plain http(s) URL, e.g. http://192.168.1.10:2283",
        )

    env_key = os.environ.get("IMMICH_API_KEY", "")
    # A blank key means "keep the existing key" — but ONLY when the URL
    # is unchanged. Blank key + new URL would point the thumb client (and
    # the /api/health poll below) at the new host while attaching the
    # current Immich key to every request: any page that can reach this
    # unauthenticated endpoint could exfiltrate the stored key to a host
    # it controls. Require a fresh key whenever the host changes.
    current_url = (immich_base_url() or "").rstrip("/")
    key_blank = not params.api_key.strip()
    if key_blank and url != current_url and not env_key:
        raise HTTPException(
            status_code=422,
            detail="API key is required when changing the Immich URL",
        )
    api_key = params.api_key.strip() or env_key or immich_api_key()
    if not api_key:
        raise HTTPException(
            status_code=422,
            detail="Immich API key is required — create one in Immich "
                   "under Administration → Settings → API Keys",
        )

    with storage.connect(DB_PATH) as conn:
        storage.set_connection_settings(conn, base_url=url, api_key=api_key)
    _log.info(
        "Immich connection saved from the web UI (%s, source env_pinned=%s)",
        url,
        bool(os.environ.get("IMMICH_BASE_URL")),
    )
    # Rebuild the shared thumbnail client so /thumb/* follows the new
    # connection immediately.
    _rebuild_thumb_client()

    env_pinned = bool(
        os.environ.get("IMMICH_BASE_URL") or os.environ.get("IMMICH_API_KEY")
    )
    # Reuse the /api/health logic for the immediate connected/broken
    # signal (aliased: a local named `health` would shadow the function).
    fresh_health = await health()
    return {"saved": True, "env_pinned": env_pinned, "health": fresh_health}


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

    `pipeline` is sourced from:
    - The child-process status file when a run is active (live).
    - The last completed run from the runs table after a run ends.
    - The parent's empty idle state when no run has ever happened.
    """
    active = _get_active_job()
    pipeline_state: dict
    if active is not None:
        sp = active["status_path"]
        file_state: dict | None = None
        if sp.exists():
            try:
                file_state = json.loads(sp.read_text())
            except (json.JSONDecodeError, OSError):
                file_state = None
        if file_state is not None:
            file_state["job_id"] = active["job_id"]
            pipeline_state = file_state
        else:
            # Child spawned but hasn't written its first status yet —
            # interpreter boot + torch import take a few seconds. Report
            # an explicit "starting" state instead of get_state(): the
            # parent's module state still holds the PREVIOUS run's
            # terminal snapshot here, which made the SSE tell clients
            # "done" seconds after they had just started a run (and the
            # Run button re-enabled itself mid-boot). Field shape mirrors
            # pipeline._state's initial dict so the UI can render it
            # unchanged; started_at is the child's spawn time (same
            # wall-clock float the status file carries).
            pipeline_state = {
                "status": "running",
                "stage": "starting",
                "chunk_idx": 0,
                "chunk_total": 0,
                "rate": 0.0,
                "eta_seconds": 0.0,
                "current_run_id": None,
                "query": "",
                "n_new": 0,
                "n_cached": 0,
                "n_requested": 0,
                "started_at": active["started_at"],
                "finished_at": None,
                "last_error": None,
                "stage_times": {},
                "stage_t0": None,
                "job_id": active["job_id"],
            }
    else:
        # No live child — show the last completed run, or the
        # empty idle state if nothing has ever run. The UI needs
        # this to show the user that the previous run finished
        # and what it produced.
        if _last_worker_outcome is not None:
            # The last worker ended in error OR was stopped by the
            # user — both outrank the last successful run. A red
            # "error" or neutral "stopped" must not masquerade as a
            # green "done" from yesterday.
            pipeline_state = _last_worker_outcome
        else:
            last = _last_completed_run()
            pipeline_state = last if last is not None else get_state()
    return {
        "pipeline": pipeline_state,
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
    # Super-group filter (/api/cluster-groups id). Resolved server-side
    # to the member cluster list, so group membership stays consistent
    # with the cards and the URL stays short.
    cluster_group: int | None = Query(default=None, ge=0),
    species: str | None = Query(default=None),
    genus: str | None = Query(default=None),
    family: str | None = Query(default=None),
    has_person: int | None = Query(default=None, ge=0, le=1),
    plantae: str | None = Query(default=None, pattern="^(plant|non_plant|uncertain)$"),
    include_non_plants: bool = Query(default=False),
    # Default is 'time' (capture date DESC) — matches the frontend
    # DEFAULT_FILTERS.sort and the "Newest first" pill in Grid.tsx.
    # Clients passing plantae/score instead send sort='rank' (no-op
    # server order) and resort the page slice client-side via
    # Grid.sortRows. 'rank' remains in the pattern for users who
    # explicitly ask for Immich's search rank.
    sort: str = Query(default="time", pattern="^(rank|time|added)$"),
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
      An explicit tier always wins regardless of `include_non_plants`.
    - include_non_plants: opt-in toggle (default False). When False,
      the unfiltered "all" view hides non-plant photos entirely — grid
      and `pagination.total` only count assets with Plantae score ≥
      PLANTAE_YELLOW. This is the project-wide "no non-plant counts"
      policy applied to results. Set True to restore legacy behavior
      and include every photo in the all-view. Has no effect when an
      explicit `plantae` tier is given.
    - sort: 'time' (default — newest captured_at first, NULLs last),
      'added' (newest first_seen_at first, NULLs last), or 'rank'
      (Immich search-rank from the most recent run, then cluster_id
      tiebreak). Time/added sort happens server-side so the page slice
      is the actual most-recent N photos, not the top-N by rank
      re-sorted.

    `pagination.total` is the raw file count (every photo, pre-dedup)
    so the breadcrumb and "all" tile match. `pagination.total_pages`
    is computed from representative-card count so page buttons reflect
    what's actually browsable.
    """
    # Cache key includes every filter + sort param so different filter
    # combinations can't poison each other. has_person, plantae, and
    # include_non_plants are included so each value gets its own cached
    # slice — flipping the toggle on the same filter combination MUST
    # bust the cache, otherwise stale results leak across toggles.
    has_filter = any(v is not None for v in (cluster_id, cluster_group, species, genus, family, has_person, plantae))
    cache_key = (
        f"all:cid={cluster_id}:cg={cluster_group}:sp={species}:ge={genus}:fa={family}:hp={has_person}:pl={plantae}:inp={int(include_non_plants)}:sort={sort}"
        if has_filter else f"all:hp={has_person}:pl={plantae}:inp={int(include_non_plants)}:sort={sort}"
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

    # Resolve the super-group to its member cluster ids (cache-miss path
    # only — the cache key already distinguishes groups). An unknown or
    # empty group id means "nothing matches", never "no filter".
    cluster_ids_filter: list[int] | None = None
    if cluster_group is not None:
        group_members, ungrouped = await _cluster_group_index()
        cluster_ids_filter = _resolve_group_clusters(cluster_group, group_members, ungrouped)
        if not cluster_ids_filter:
            return {
                "error": "no assets match this filter",
                "results": [],
                "taxonomy": {},
                "pagination": {
                    "page": page,
                    "page_size": page_size,
                    "total": 0,
                    "total_pages": 0,
                },
            }

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
                cluster_ids=cluster_ids_filter,
                species=species,
                genus=genus,
                family=family,
                has_person=has_person,
                plantae=plantae,
                plantae_green=PLANTAE_GREEN,
                plantae_yellow=PLANTAE_YELLOW,
                include_non_plants=include_non_plants,
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
            # 5. KNN: prefer (index_version, k) that has a row for THIS asset
            # (a freshly-computed Re-run KNN row may live in a less-populated
            # version than the global max — e.g. just-cached imageomics-official
            # while v2 is still the dominant global key). Fall back to the
            # most-populated pair if no version has a row for this asset.
            versions = storage.list_knn_versions(conn)
            knn_payload = None
            if versions:
                # Asset-first: any version with a row for this asset wins.
                # Order: list_knn_versions is already last_at DESC, so the
                # most-recently-used matching version comes first.
                for v in versions:
                    if storage.knn_predictions_have_asset(
                        conn,
                        asset_id=asset_id,
                        index_version=v["index_version"],
                        k=v["k"],
                    ):
                        index_version, k = v["index_version"], v["k"]
                        break
                else:
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


# NOTE: the old "/api/album/plants" endpoints were folded into the
# kingdom surface below ("/api/album/kingdom*" with kingdom="Plantae")
# — the plants worker was a strict subset of the kingdom worker. The
# one-click Plantae button in the UI now calls the kingdom endpoints.
def _immich_client() -> ImmichClient:
    base_url, api_key = _require_immich_connection()
    return ImmichClient(base_url=base_url, api_key=api_key, timeout=60.0)



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
        stop_criteria_pct=params.stop_criteria_pct,
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

    With the worker being a plain subprocess (see _start_run), there is
    no shared Event — we SIGTERM it. The worker's handler arms
    pipeline.request_stop(), so the run still unwinds at the next
    batch boundary with per-batch commits intact. We also set the
    threading.Event in the parent so any in-process callers (rare;
    only tests really) observe the stop.

    Escalation: SIGTERM only works if the worker reaches a batch
    boundary. If it is wedged somewhere with no stop check (a model
    load thrashing in swap, a hung download), it will never exit. A
    second click — or the 45s watchdog armed by the first — escalates
    to SIGKILL. The parent writes the terminal "stopped" state itself
    in that case (a SIGKILLed worker cannot), so the dead-process
    reaper reports a neutral stop rather than "died unexpectedly".
    """
    import json as _json
    import logging
    _log = logging.getLogger(__name__)
    active = _get_active_job()
    if active is None:
        return {"stopped": False, "reason": "no run in progress"}
    proc = active["proc"]

    def _force_kill(reason: str) -> None:
        # A SIGKILLed process cannot write its own status file — write
        # the terminal "stopped" state for it, otherwise the reaper
        # would brand a forced stop as an unexpected crash.
        try:
            status_file = Path(active["status_path"])
            snap = (
                _json.loads(status_file.read_text())
                if status_file.exists() else {}
            )
            snap["status"] = "stopped"
            snap["last_error"] = reason
            snap["finished_at"] = time.time()
            status_file.write_text(_json.dumps(snap))
        except Exception:
            pass
        try:
            proc.kill()
        except Exception:
            pass
        _log.warning("[pipeline] child pid=%s force-killed (%s)", proc.pid, reason)

    if active.get("terminated_at"):
        # Second click while the first SIGTERM is still pending —
        # escalate immediately instead of waiting for the watchdog.
        _force_kill("stopped by user (forced — worker did not unwind)")
        return {"stopped": True, "escalated": True}

    proc.terminate()
    active["terminated_at"] = time.time()
    # Also set the threading.Event in the parent so any in-process
    # callers (rare; only tests really) observe the stop too.
    import pipeline as _pipeline
    _pipeline.request_stop()
    _log.info(
        "[pipeline] stop requested by user; child pid=%s will unwind at next batch boundary",
        proc.pid,
    )

    def _stop_watchdog() -> None:
        # Grace period: one batch is ~10-15s of encode plus downloads;
        # a healthy worker unwinds well within this. Still alive after
        # 45s means it is wedged where no stop check runs.
        time.sleep(45)
        if proc.poll() is None:
            _force_kill("stopped by user (forced — worker did not unwind in 45s)")

    threading.Thread(target=_stop_watchdog, daemon=True).start()
    return {"stopped": True}


class ReclassifyParams(BaseModel):
    asset_ids: list[str]
    k: int | None = None  # default: server KNN_K
    # Recompute even when a row already exists for (index_version, k).
    # The per-asset "Re-run KNN" button sets this — without it a second
    # click is a silent no-op.
    force: bool = False


@app.get("/api/knn-status")
async def knn_status() -> dict:
    """Cheap: returns whether the FAISS index is loaded + cache stats.
    UI polls this once on page load to render the KNN status pill.

    Runs in a worker thread: on first call this loads the index
    (mmap ~1 GB + labels metadata), which must not block the event
    loop — it would freeze every concurrent request including the SSE
    ticks for seconds. Subsequent calls hit the in-memory cache and
    are fast either way."""
    return await asyncio.to_thread(_knn_status_dict)


# ---- Build KNN index (FAISS) -------------------------------------------
# Wraps `python -m knn --parquet-glob ... --tag ...` as a long-running
# background job with a tail-the-log endpoint so the UI can show
# progress. The build runs in a real subprocess (see start_knn_build):
# a runaway build must not take uvicorn down with it, and the same CLI
# entry path is what the terminal workflow has always exercised.

class KnnBuildArgs(BaseModel):
    parquet_glob: str
    tag: str
    index_dir: str | None = None        # default: KNN_INVENTORY_DIR() (re-read each call)
    kingdoms: str | None = None        # comma-separated, e.g. "Plantae,Fungi"
    ivf_nlist: int | None = None       # override IVF cluster count (default: sqrt(train_size))
    ivf_train_size: int | None = None  # #vectors held in RAM for k-means training (default: 200K, lower=less RAM)

# In-memory job registry. Jobs are short-lived (~minutes-to-hours);
# the log file outlives the process so users can grep later.
_knn_build_jobs: dict[str, dict] = {}
# Lock for /api/events to read the registry safely. The worker thread
# writes here without holding a lock (single writer) but the SSE
# snapshot in app.py reads it concurrently, so we need to gate that.
_knn_build_jobs_lock = threading.Lock()


# Soft memory cap on the KNN build child process. We tried a hard
# RLIMIT_DATA cap at 12 GB but it tripped false positives — DuckDB
# mmap's every parquet shard (~100 MB × 666 = 66 GB of virtual
# address space), so the kernel's `mmap: VmData exceed data ulimit`
# check fired before physical RAM filled up. cgroup v2 memory.max
# would be the right tool here, but it's not writable to non-root
# users on this box.
#
# Approach: set a "soft warning" cap based on RSS in the watcher
# thread (NOT a kernel rlimit). If the build process's RSS exceeds
# this value the parent SIGKILLs it. The OS OOM killer is the
# backstop. Sized to leave headroom on a 29 GB host for the rest
# of the system: ~29 total − ~6 baseline (immich_ml, postgresql,
# memray) − 0.5 uvicorn = ~22 GB safe for the build. The build's
# actual steady-state RSS at 130M rows is ~2 GB (in-RAM IVFPQ +
# DuckDB + Python), so 22 GB gives us 10x headroom — the only way
# it would hit the cap is a leak we haven't found yet, in which
# case SIGKILL surfaces as a clean error instead of crashing uvicorn.
_KNN_BUILD_MEM_LIMIT_BYTES = 10 * 1024**3  # parent SIGKILLs build if RSS > 10 GB


def _tail_error_from_log(log_path: str) -> str | None:
    """Pull the most recent `[knn.build] error: ...` line from the log
    file. Worker writes this on caught exceptions; it's the cleanest
    single-line summary for the UI."""
    try:
        with open(log_path) as f:
            lines = f.readlines()
    except (FileNotFoundError, OSError):
        return None
    for line in reversed(lines[-50:]):
        line = line.strip()
        if line.startswith("[knn.build] error:"):
            return line.split("[knn.build] error:", 1)[1].strip()
    return None


@app.post("/api/knn/build")
async def start_knn_build(args: KnnBuildArgs) -> dict:
    """Spawn a build child process, return its job_id. The UI polls
    /api/knn/build/job/{job_id} for status and /api/knn/build/logs for
    the live log tail."""
    # Pre-flight: create the index dir NOW so a permission / mount /
    # typo error surfaces here as a clean 422 instead of an opaque
    # FileNotFoundError traceback in the child (which would also flood
    # the log file with stack traces). The child re-runs mkdir with
    # exist_ok=True; this is the canary.
    target_dir = Path(args.index_dir or KNN_INVENTORY_DIR())
    try:
        target_dir.mkdir(parents=True, exist_ok=True)
    except (PermissionError, OSError) as e:
        raise HTTPException(
            status_code=422,
            detail=(
                f"Cannot create index directory {target_dir}: "
                f"{type(e).__name__}: {e}. "
                f"Pick a path the server process can write to "
                f"(e.g. data/knn-index next to the SQLite DB)."
            ),
        )
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
    # Spawn the build in a real subprocess (not a forked mp worker), so
    # a runaway build can't take down the uvicorn worker. Physical
    # memory is bounded by the parent-side RSS watchdog below
    # (_watch_subprocess_build SIGKILLs if RSS exceeds the cap). We do
    # NOT set RLIMIT_DATA in the child: on this kernel it counts
    # anonymous mmap as well as brk, and faiss reserves ~2.7 GB of
    # virtual address space at import and grows ~174 B/vector — so an
    # RLIMIT_DATA cap trips on VIRTUAL reservation while RSS is only
    # ~2-3 GB, killing healthy builds around 35-60M vectors. RSS is the
    # real pressure; watch that instead.
    import subprocess
    # Use `python -m knn` rather than importing the module — same as
    # the CLI entry path that has been smoke-tested many times. Avoids
    # the import-after-fork gotchas that multiprocessing.Process hit.
    # sys.executable (not a hardcoded ./.venv path) so this works from
    # any install layout, including the Docker image.
    cmd = [
        sys.executable, "-u", "-m", "knn",
        "--parquet-glob", args.parquet_glob,
        "--index-dir", str(Path(args.index_dir or KNN_INVENTORY_DIR())),
        "--tag", args.tag,
    ]
    if args.kingdoms:
        cmd += ["--kingdoms", args.kingdoms]
    if args.ivf_nlist is not None:
        cmd += ["--ivf-nlist", str(args.ivf_nlist)]
    if args.ivf_train_size is not None:
        cmd += ["--ivf-train-size", str(args.ivf_train_size)]
    proc = subprocess.Popen(
        cmd,
        stdout=open(log_path, "w"),
        stderr=subprocess.STDOUT,
        start_new_session=True,  # child becomes process group leader
    )
    _knn_build_jobs[job_id]["proc"] = proc
    # Watch in a daemon thread (don't block the request). The thread
    # exits when the child exits; it doesn't keep uvicorn alive.
    threading.Thread(
        target=_watch_subprocess_build, args=(job_id, proc), daemon=True
    ).start()
    return {"job_id": job_id}


def _watch_subprocess_build(job_id: str, proc: subprocess.Popen) -> None:
    """Daemon thread that waits for the KNN build subprocess to exit and
    updates the job dict based on the exit code. Mirrors the
    multiprocessing watcher but for subprocess.Popen — exitcode semantics
    match: 0 = success, -9 = SIGKILL (parent's RSS cap exceeded),
    other = error.

    Also polls the child's RSS every 5s and SIGKILLs it if it crosses
    the parent's soft cap (22 GB default). This is the backstop in
    place of RLIMIT_DATA — we couldn't use a kernel rlimit because
    DuckDB's mmap'd parquet files exceeded RLIMIT_DATA well before
    physical RAM did, and cgroup v2 isn't writable by non-root users.
    """
    import os as _os
    import time as _time
    # Poll the child for two things: (1) did it exit on its own, (2)
    # is its RSS over the soft cap. `proc.poll()` returns None while
    # the child is alive. We sleep 5s between polls so a runaway
    # build gets ~5s of grace after the first overshoot — enough for
    # a chunk boundary, not enough to take down uvicorn.
    last_kill_at = 0.0
    while proc.poll() is None:
        _time.sleep(5.0)
        try:
            with open(f"/proc/{proc.pid}/status") as _f:
                _status = _f.read()
        except (FileNotFoundError, ProcessLookupError):
            break
        _rss_kb = 0
        for _line in _status.splitlines():
            if _line.startswith("VmRSS:"):
                _rss_kb = int(_line.split()[1])
                break
        _rss_mb = _rss_kb / 1024
        if (
            _KNN_BUILD_MEM_LIMIT_BYTES is not None
            and _rss_kb > _KNN_BUILD_MEM_LIMIT_BYTES / 1024
        ):
            # Throttle: don't kill more than once per 30s so a stuck
            # proc gets one chance to release memory before we
            # escalate to SIGKILL.
            if _time.time() - last_kill_at > 30:
                last_kill_at = _time.time()
                _log.warning(
                    f"[knn.build] job {job_id} RSS={_rss_mb:.0f} MB > "
                    f"soft cap {_KNN_BUILD_MEM_LIMIT_BYTES // (1024**3)} GB; "
                    f"sending SIGKILL"
                )
                try:
                    _os.kill(proc.pid, 9)
                except ProcessLookupError:
                    pass
    proc.wait()
    job = _knn_build_jobs.get(job_id)
    if job is None:
        return
    job["finished_at"] = datetime.now(timezone.utc).isoformat()
    if proc.returncode == 0:
        job["status"] = "done"
        job["error"] = None
    elif proc.returncode == -9 or proc.returncode == 137:
        # SIGKILL — the parent killed us because RSS exceeded the
        # soft cap. Surface a clear message instead of an empty log.
        job["status"] = "error"
        msg = (
            f"KNN build worker exceeded its {_KNN_BUILD_MEM_LIMIT_BYTES // (1024**3)} GB "
            f"RSS soft cap and was killed. Try a smaller "
            f"ivf_train_size (e.g. 100000)."
        )
        job["error"] = f"MemoryError: {msg}"
        try:
            with open(job["log_path"], "a") as f:
                f.write(f"\n[knn.build] KILLED by parent watchdog (exitcode {proc.returncode} / SIGKILL).\n")
                f.write(f"[knn.build] Cause: RSS exceeded the {_KNN_BUILD_MEM_LIMIT_BYTES // (1024**3)} GB soft cap (physical memory).\n")
                f.write(f"[knn.build] Hint: re-run with ivf_train_size <= 100000.\n")
        except Exception:
            pass
    else:
        job["status"] = "error"
        err = _tail_error_from_log(job["log_path"])
        job["error"] = err or f"KNN build worker exited with code {proc.returncode}"


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
    # to_thread: first call mmaps the ~1 GB index and must not block
    # the event loop.
    if await asyncio.to_thread(_get_knn_index) is None:
        raise HTTPException(
            status_code=503,
            detail=(
                f"KNN index not built — missing {KNN_INVENTORY_DIR()}/"
                f"tol200m_bioclip2.faiss. Build it with `python -m knn "
                f"--parquet-glob ... --index-dir ... --tag ...`."
            ),
        )
    from pipeline import run_knn_reclassify
    # Run in a worker thread because FAISS .search + parquet take() can
    # take a few seconds for large batches. We share the DB so we don't
    # need an additional lock — sqlite is in WAL mode and single-writer.
    stats = await asyncio.to_thread(
        run_knn_reclassify_sync, params.asset_ids, k, params.force
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
    # the new KNN rows. _invalidate_results_cache acquires
    # _results_lock internally — calling it under the lock deadlocks.
    _invalidate_results_cache("all:")
    _invalidate_clusters_cache()
    return {"stats": stats, "results": fresh}


def run_knn_reclassify_sync(asset_ids: list[str], k: int, force: bool = False) -> dict:
    """Synchronous wrapper used by the /api/reclassify endpoint so it
    can be offloaded to a worker thread via asyncio.to_thread()."""
    from pipeline import run_knn_reclassify
    with storage.connect(DB_PATH) as conn:
        return run_knn_reclassify(
            conn, asset_ids=asset_ids, index_dir=KNN_INVENTORY_DIR(), k=k,
            force=force,
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
# 3. **Kingdom generalization.** "Tag a kingdom as an album" works for
#    any kingdom (Plantae, Fungi, Animalia, ...). This used to be a
#    separate /api/album/plants endpoint pair; the plants worker was a
#    strict subset of the kingdom worker, so the two were merged and
#    the Plantae button now calls the kingdom endpoints.
#
# Pattern: same `job_id` (uuid4 hex) + `_jobs_lock` + TTL prune machinery
# across the job registries in this file. Different dict keys because
# the payloads are different (cluster_id vs. kingdom vs. min_score).

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


# In-memory job registries. Same TTL + prune contract throughout.
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
            f"Cluster {cluster_id} ({cluster_name}) #BotaniSense"
            if cluster_name
            else f"Cluster {cluster_id} (BioCLIP) #BotaniSense"
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
                    f"{immich_base_url()}/albums/{album_id}"
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
                album_url=f"{immich_base_url()}/albums/{album_id}",
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
    cluster and bulk-add its assets. Same album reuse + chunked-add
    flow as the kingdom worker, keyed by cluster_id. Returns immediately with a `job_id`; the UI polls
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

    "Junk drawer" clusters — those with no species-rank taxon_scores,
    i.e. HDBSCAN's cluster-of-last-resort that captured scattered
    kingdom-only points — are filtered out before the totals are
    computed. Without this filter a single cluster with 50%+ of all
    photos would dominate the sync and produce one giant album of
    unlabeled photos, exactly what `n_assets_total=19784 / n_clusters=1`
    was showing before this change. Skipped clusters count toward
    neither the denominator nor `clusters_done`.
    """
    import time as _time
    chunk_size = 1000
    try:
        # 1. Snapshot every cluster upfront so we know total work and
        # can show a deterministic progress bar. Filtering out clusters
        # with zero assets (shouldn't happen, but defensive) and clusters
        # without species-rank predictions (junk-drawer clusters that
        # would otherwise dominate the sync — see docstring above).
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
                and storage.cluster_has_species_predictions(
                    conn, int(r["cluster_id"])
                )
            ]
            skipped = len(cluster_rows) - len(clusters)
            # Pre-fetch cluster names (used for album titles).
            name_rows = conn.execute(
                "SELECT cluster_id, name FROM cluster_names"
            ).fetchall()
            names = {int(r["cluster_id"]): r["name"] for r in name_rows}

        n_assets_total = sum(n for _, n in clusters)
        with _cluster_album_jobs_lock:
            _cluster_album_jobs[job_id]["n_clusters"] = len(clusters)
            _cluster_album_jobs[job_id]["n_clusters_skipped"] = skipped
            _cluster_album_jobs[job_id]["n_assets_total"] = n_assets_total
            _cluster_album_jobs[job_id]["clusters_done"] = 0
            _cluster_album_jobs[job_id]["photos_added"] = 0
            _cluster_album_jobs[job_id]["people_filter"] = people_filter
        if skipped:
            _log.info(
                f"[cluster-sync-all] skipping {skipped} junk-drawer "
                f"cluster(s) with no species-rank scores; "
                f"{len(clusters)} cluster(s) will be synced"
            )

        client = _immich_client()
        try:
            photos_added_total = 0
            t0 = _time.monotonic()
            for idx, (cluster_id, n_assets) in enumerate(clusters):
                cluster_name = names.get(cluster_id)
                album_title = (
                    f"Cluster {cluster_id} ({cluster_name}) #BotaniSense"
                    if cluster_name
                    else f"Cluster {cluster_id} (BioCLIP) #BotaniSense"
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
                        album_url=f"{immich_base_url()}/albums/{album_id}",
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
        # clusters (M total photos)?" in its confirm panel. Junk-drawer
        # clusters (no species-rank scores) are filtered out so the
        # preview matches what the worker will actually do.
        rows = conn.execute(
            "SELECT cluster_id, COUNT(*) AS n FROM assets "
            "WHERE cluster_id IS NOT NULL GROUP BY cluster_id"
        ).fetchall()
        clusters = [
            (int(r["cluster_id"]), int(r["n"]))
            for r in rows
            if r["n"] > 0
            and storage.cluster_has_species_predictions(
                conn, int(r["cluster_id"])
            )
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
async def delete_cluster_album(cluster_id: int) -> dict:
    """Drop the cluster→album mapping AND DELETE the album in Immich
    (idempotent: 404 is treated as already-gone). The user-facing
    'cluster sync' feature is marked beta, and the policy is "never
    leave orphans": when you untrack a cluster album locally, the
    matching Immich album goes with it. Returns `{deleted, remote_deleted}`.
    """
    with storage.connect(DB_PATH) as conn:
        saved = storage.get_cluster_album(conn, cluster_id)
        if saved is None:
            return {"deleted": False, "remote_deleted": False}
        storage.delete_cluster_album(conn, cluster_id)

    remote_deleted = False
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

    Note: Immich albums are ALWAYS deleted alongside the local
    mapping (the `delete_in_immich` toggle was removed when the
    cluster-sync feature went beta — policy is "never leave orphans").
    """
    cluster_ids: list[int] | None = None


class ClusterAlbumsUndoRemoveParams(BaseModel):
    """Body for POST /api/album/cluster/undo-remove. The frontend
    captures the full ClusterAlbumMapping rows before bulk-remove and
    POSTs them back here within the undo TTL window (default 30s).
    The endpoint re-inserts the cluster_albums rows but does NOT
    recreate the Immich albums (those were deleted server-side) —
    the user has to re-sync to get a fresh album with the same name.
    """
    rows: list[dict]


@app.post("/api/album/cluster/remove")
async def bulk_remove_cluster_albums(
    params: ClusterAlbumsRemoveParams,
) -> dict:
    """Bulk-remove cluster→album mappings AND DELETE the matching
    Immich albums. One round-trip replaces N DELETE
    /api/album/cluster/{id} calls for the Sync-to-IMMICH "Remove
    synced clusters" button.

    Behaviour:
      - `cluster_ids=None` → every currently synced cluster
      - Every mapping's matching Immich album is DELETEd (404 is
        idempotent)
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
    client = _immich_client()
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


@app.post("/api/album/cluster/undo-remove")
async def undo_bulk_remove_cluster_albums(
    params: ClusterAlbumsUndoRemoveParams,
) -> dict:
    """Recreate the cluster_albums rows from a snapshot captured before
    a bulk-remove. The Immich albums themselves are NOT restored —
    they were DELETEd on the Immich side during the bulk-remove and
    only the local mapping comes back. Re-sync to recreate the actual
    albums.

    Idempotent: re-inserting an existing cluster_id via INSERT OR
    REPLACE overwrites it (so the user can Undo twice without
    duplicating rows).
    """
    inserted = 0
    skipped = 0
    with storage.connect(DB_PATH) as conn:
        for row in params.rows:
            try:
                cid = int(row["cluster_id"])
                album_id = row.get("album_id") or ""
                album_url = row.get("album_url") or ""
                if not album_id or not album_url:
                    skipped += 1
                    continue
                storage.insert_cluster_album(conn, {
                    "cluster_id": cid,
                    "album_id": album_id,
                    "album_url": album_url,
                    "n_assets": int(row.get("n_assets", 0)),
                    "reused_existing": int(row.get("reused_existing", 0)),
                    "created_at": row.get("created_at"),
                    "last_synced_at": row.get("last_synced_at"),
                })
                inserted += 1
            except Exception as e:  # noqa: BLE001 — per-row failures don't abort
                _log.warning("undo-remove row %s failed: %s", row, e)
                skipped += 1
    return {"restored": inserted, "skipped": skipped}


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


def _kingdom_album_meta(conn, kingdom: str) -> dict:
    """Saved album meta for a kingdom, adopting the pre-merge legacy row.

    Albums synced before the plants/kingdom merge were persisted under
    album_meta key "plants_album_id"; the merged code stores under the
    kingdom name ("Plantae"). On first read after upgrading, fall back
    to the legacy row so an existing Plantae album is reused rather
    than duplicated. Writes always go to the new key.
    """
    saved = storage.get_kingdom_album_meta(conn, kingdom)
    if saved is None and kingdom == "Plantae":
        saved = storage.get_album_meta(conn, "plants_album_id")
    return saved or {}


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
            saved = _kingdom_album_meta(conn, kingdom)
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
                j["album_url"] = f"{immich_base_url()}/albums/{album_id}"

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
        saved = _kingdom_album_meta(conn, kingdom)
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
    """Start a kingdom-album job. Takes kingdom + min_score as
    parameters so it works for Plantae, Fungi, Animalia, etc. Returns
    a `job_id`; the UI polls
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
        saved = _kingdom_album_meta(conn, params.kingdom)

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
        saved = _kingdom_album_meta(conn, kingdom)
    if not saved or not saved.get("album_id"):
        return {"exists": False, "kingdom": kingdom}
    return {
        "exists": True,
        "kingdom": kingdom,
        "album_id": saved["album_id"],
        "album_url": f"{immich_base_url()}/albums/{saved['album_id']}",
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
    under a single connection. This runs in-process (HDBSCAN refit is
    CPU-bound but not BioCLIP), so it invalidates the in-process
    /api/results and /api/clusters caches itself when it finishes."""
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
        # Refit reassigns cluster_ids — cached results/clusters payloads
        # and the DB-side cluster-name cache are all stale now.
        _invalidate_results_cache("all:")
        _invalidate_clusters_cache()
        with storage.connect(DB_PATH) as conn:
            storage.invalidate_cluster_names(conn)
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
    # Thumbnails are proxied from Immich — nothing to serve until the
    # connection exists. (Keeps an unconfigured first boot clean: the
    # grid can't render anyway.)
    if not immich_base_url() or not immich_api_key():
        raise HTTPException(
            status_code=409,
            detail="Immich is not configured — enter the Immich URL and "
                   "API key in the app first.",
        )
    # Coerce fullsize-ish aliases down to the largest valid Immich size.
    if size in _FULLSIZE_ALIASES:
        size = "preview"  # "preview" is the closest to full-quality Immich serves here
    if size not in _ALLOWED_SIZES:
        size = "preview"

    key = (asset_id, size)
    # Cache HIT — bypass the network entirely. Lock is held only long
    # enough to read+move; the bytes are returned without acquiring
    # anything.
    async with _thumb_cache_lock:
        hit = _thumb_cache.get(key)
        if hit is not None:
            _thumb_cache.move_to_end(key)
            body, media_type = hit
            return Response(
                content=body,
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=86400, immutable",
                    "Content-Length": str(len(body)),
                    "X-Thumb-Cache": "HIT",
                },
            )

    # Cache MISS — fetch from Immich through the shared client (kept
    # alive for the lifetime of the process via lifespan). We release
    # the cache lock during the network round-trip so concurrent misses
    # for DIFFERENT asset_ids can fan out in parallel; only the
    # final put+evict re-acquires the lock.
    client = _thumb_client
    if client is None:
        # Lifespan didn't run (e.g. ASGI server misconfigured) — fall
        # back to a one-shot client so the endpoint still works, just
        # without connection reuse.
        async with httpx.AsyncClient(timeout=30.0) as fallback:
            r = await fallback.get(
                f"/api/assets/{asset_id}/thumbnail",
                params={"size": size},
            )
    else:
        r = await client.get(
            f"/api/assets/{asset_id}/thumbnail",
            params={"size": size},
        )
    if r.status_code != 200:
        # HTTPException uses `detail`, not `text`. Preserve the upstream
        # error body in the message so the frontend can surface it.
        raise HTTPException(
            status_code=r.status_code,
            detail=f"immich thumbnail {asset_id}/{size}: {r.text[:200]}",
        )
    body = r.content
    media_type = r.headers.get("content-type", "image/jpeg")
    # Server-side LRU (process-local, ~15 MB at the 500-entry cap). The
    # Cache-Control header below is for downstream caches and the
    # browser — they all benefit from the same content-addressable
    # invariant: a thumbnail for asset_id=X never changes unless the
    # underlying photo is replaced in Immich.
    async with _thumb_cache_lock:
        _thumb_cache[key] = (body, media_type)
        while len(_thumb_cache) > _THUMB_CACHE_MAX:
            _thumb_cache.popitem(last=False)
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=86400, immutable",
            "Content-Length": str(len(body)),
            "X-Thumb-Cache": "MISS",
        },
    )


# ---- KNN reference-photo proxy ------------------------------------------
# The KNN neighbor list carries an `identifier` — the source image URL
# of the reference photo the match came from (gbif / eol media). The UI
# renders those as thumbnails so a user can eyeball "does my photo
# actually look like this?" instead of trusting a species string.
#
# We proxy rather than hotlink for three reasons: the browser would
# otherwise leak the user's IP to a few hundred herbarium hosts; many
# of those hosts set no CORS headers and some block hotlinking; and a
# proxy lets us cache (the same reference photo recurs across assets).
#
# The proxy is CLOSED, not open: `url` must already appear as a
# neighbor identifier in one of the KNN caches. See
# storage.knn_reference_url_known for why.


def _shrink_ref_image(body: bytes, media_type: str) -> tuple[bytes, str]:
    """Downscale a reference photo to _REF_THUMB_EDGE on its long side
    and re-encode as JPEG. Returns (body, media_type) unchanged if the
    image is already small or can't be decoded."""
    import io
    from PIL import Image

    try:
        img = Image.open(io.BytesIO(body))
        if max(img.size) <= _REF_THUMB_EDGE:
            return body, media_type
        img = img.convert("RGB")
        img.thumbnail((_REF_THUMB_EDGE, _REF_THUMB_EDGE), Image.LANCZOS)
        buf = io.BytesIO()
        img.save(buf, format="JPEG", quality=82, optimize=True)
        return buf.getvalue(), "image/jpeg"
    except Exception as e:
        _log.warning("ref-thumb: could not shrink image (%s); serving as-is", e)
        return body, media_type


@app.get("/api/knn/ref-thumb")
async def knn_ref_thumb(url: str = Query(..., description="reference image URL")) -> Response:
    if not url.startswith("https://"):
        # http:// would send the request in the clear and is not used by
        # any source dataset in the index.
        raise HTTPException(status_code=400, detail="url must be https://")

    async with _ref_thumb_cache_lock:
        hit = _ref_thumb_cache.get(url)
        if hit is not None:
            _ref_thumb_cache.move_to_end(url)
            body, media_type = hit
            return Response(
                content=body,
                media_type=media_type,
                headers={
                    "Cache-Control": "public, max-age=604800",
                    "Content-Length": str(len(body)),
                    "X-Ref-Thumb-Cache": "HIT",
                },
            )

    def _known() -> bool:
        with storage.connect(DB_PATH) as conn:
            return storage.knn_reference_url_known(conn, url)

    if not await asyncio.to_thread(_known):
        raise HTTPException(
            status_code=403,
            detail="url is not a known KNN reference image",
        )

    # follow_redirects: gbif identifiers frequently 302 to a CDN. The
    # allowlist above means we only ever start from a URL that came out
    # of the reference index, so the redirect chain is as trustworthy as
    # the index itself.
    try:
        async with httpx.AsyncClient(
            timeout=20.0, follow_redirects=True, max_redirects=5,
        ) as client:
            r = await client.get(url, headers={"User-Agent": "BotaniSense/1.0"})
    except httpx.HTTPError as e:
        raise HTTPException(status_code=502, detail=f"reference fetch failed: {e}")

    if r.status_code != 200:
        raise HTTPException(
            status_code=r.status_code,
            detail=f"reference image {url}: HTTP {r.status_code}",
        )
    body = r.content
    if len(body) > _REF_THUMB_MAX_BYTES:
        raise HTTPException(status_code=502, detail="reference image too large")
    media_type = r.headers.get("content-type", "image/jpeg")
    if not media_type.startswith("image/"):
        raise HTTPException(
            status_code=502, detail=f"reference is not an image ({media_type})",
        )

    # Downscale to thumbnail size before caching. Pillow decode + encode
    # is CPU-bound, so it goes to a worker thread rather than blocking
    # the event loop. A source we can't decode is served through
    # unchanged — better a big image than none.
    body, media_type = await asyncio.to_thread(
        _shrink_ref_image, body, media_type
    )

    async with _ref_thumb_cache_lock:
        _ref_thumb_cache[url] = (body, media_type)
        while len(_ref_thumb_cache) > _REF_THUMB_CACHE_MAX:
            _ref_thumb_cache.popitem(last=False)
    return Response(
        content=body,
        media_type=media_type,
        headers={
            "Cache-Control": "public, max-age=604800",
            "Content-Length": str(len(body)),
            "X-Ref-Thumb-Cache": "MISS",
        },
    )


# ---- bbox-index reclassify (YOLO per-crop) -------------------------------
# The AssetDetail "Reclassify crop" button (one per YOLO detection,
# inside the debug disclosure) posts {asset_id, detection_id}.
# We crop the parent asset to that detection's bbox xywh, run BioCLIP
# on the crop, query the FAISS KNN index, and cache the result in
# `bbox_knn_predictions`. The endpoint is synchronous (one crop, one
# embedding pass, ~hundreds of ms). Requires KNN_INVENTORY_DIR to
# have INDEX_VERSION.txt — returns 503 otherwise so the frontend can
# surface a useful toast instead of a generic 500.

class ReclassifyBboxParams(BaseModel):
    asset_id: str
    detection_id: int


@app.post("/api/reclassify-bbox")
async def reclassify_bbox(params: ReclassifyBboxParams) -> dict:
    """KNN-reclassify a single cropped bbox. See
    pipeline.run_knn_reclassify_bbox for the full flow (download
    image, crop, BioCLIP embed, FAISS query, cache)."""
    if not str(KNN_INVENTORY_DIR()):
        raise HTTPException(
            status_code=503,
            detail="KNN not configured — set KNN_INVENTORY_DIR and build an index first",
        )
    inv = Path(KNN_INVENTORY_DIR())
    has_custom = (inv / "tol200m_bioclip2.faiss").exists() and (inv / "INDEX_VERSION.txt").exists()
    has_official = (inv / "faiss" / "index.index").exists() and (inv / "duckdb" / "metadata.duckdb").exists()
    if not (has_custom or has_official):
        raise HTTPException(
            status_code=503,
            detail=f"No KNN index found in {KNN_INVENTORY_DIR()} — set inventory_dir to a directory with either tol200m_bioclip2.faiss + INDEX_VERSION.txt (custom) or faiss/index.index + duckdb/metadata.duckdb (imageomics official)",
        )
    # This call transiently loads BioCLIP INSIDE uvicorn (~6 GB, freed
    # immediately after via _unload_classifier). Refuse while a pipeline
    # run is active: the worker's resident copy plus ours would coexist.
    if _get_active_job() is not None:
        raise HTTPException(
            status_code=409,
            detail="a pipeline run is active — retry after it finishes",
        )
    _require_immich_connection()
    try:
        return await asyncio.to_thread(
            pipeline.run_knn_reclassify_bbox,
            detection_id=params.detection_id,
            index_dir=Path(KNN_INVENTORY_DIR()),
            db_path=DB_PATH,
            base_url=immich_base_url(),
            api_key=immich_api_key(),
            k=KNN_K,
        )
    except FileNotFoundError as e:
        # Same as the upper check, but raised from the pipeline thread.
        raise HTTPException(status_code=503, detail=str(e))
    except ValueError as e:
        # detection_id not found.
        raise HTTPException(status_code=404, detail=str(e))


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
    if await asyncio.to_thread(_get_knn_index) is None:
        raise HTTPException(
            status_code=503,
            detail="KNN index not built — see the KNN panel to build one.",
        )
    import threading
    import logging
    _worker_log = logging.getLogger("reclassify_sample")
    ids_for_worker = list(ids)

    def _worker():
        # run_knn_reclassify_sync opens its own connection and passes
        # the correct (conn, asset_ids=, index_dir=) signature — the
        # previous code called run_knn_reclassify([aid]), which passed
        # the id list as the connection and TypeError'd on every asset.
        try:
            run_knn_reclassify_sync(ids_for_worker, KNN_K, force=True)
        except Exception:
            # The HTTP response has already returned; log so a failure
            # (e.g. missing embeddings) is visible instead of vanishing.
            _worker_log.warning(
                "reclassify-visible failed", exc_info=True
            )
            return
        # New KNN rows change the species shown on cards.
        _invalidate_results_cache("all:")
        _invalidate_clusters_cache()

    threading.Thread(target=_worker, daemon=True).start()
    return {"queued": len(ids_for_worker)}


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


class _SigtermFlag:
    """Stands in for the old shared mp.Event: SIGTERM flips it, and the
    backfill loop's existing stop_event.is_set() checks pick it up at
    the next batch boundary."""

    def __init__(self) -> None:
        self._flag = False

    def is_set(self) -> bool:
        return self._flag


def _person_backfill_worker_main(args: dict) -> None:
    """Body of the exec'd backfill subprocess — see the Popen call in
    the backfill start endpoint for why this is not multiprocessing."""
    import signal

    flag = _SigtermFlag()

    def _on_sigterm(signum, frame):  # noqa: ANN001
        flag._flag = True  # noqa: SLF001 — same module

    signal.signal(signal.SIGTERM, _on_sigterm)
    _person_backfill_child_entry(
        args["base_url"], args["api_key"], args["db_path"],
        args["log_path"], args["status_path"], flag,
    )


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


def _person_backfill_reap() -> None:
    """Lazy reaper: reconcile jobs whose child process has exited. A
    child that died without writing a terminal status (SIGKILL/OOM)
    would otherwise stay 'running' forever and the start guard below
    would refuse every future backfill until restart. Mirrors the
    pipeline-run reaper — the next reader (start guard / poll) is the
    cleanup point."""
    import json as _json

    term_map = {"stopped": "stopped", "error": "error", "finished": "done"}
    for job in _person_backfill_jobs.values():
        if job["status"] != "running":
            continue
        proc = job.get("_proc")
        if proc is None or proc.poll() is None:
            continue
        term = None
        try:
            snap = _json.loads(Path(job["status_path"]).read_text())
            term = term_map.get(snap.get("stage"))
        except Exception:
            term = None
        if term is None:
            # Exited with no terminal status: crash or hard kill.
            term = "error"
            job["error"] = (
                "backfill worker died unexpectedly (crash or OOM) "
                "— see the job log"
            )
        job["status"] = term
        if job.get("finished_at") is None:
            job["finished_at"] = datetime.now(timezone.utc).isoformat()
        if term == "done" and not job.get("_cache_invalidated"):
            _invalidate_results_cache("all:")
            _invalidate_clusters_cache()
            job["_cache_invalidated"] = True


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
    import subprocess
    # Concurrent-job guard. Even though each job is now in its own
    # process, two concurrent jobs would still contend on the SQLite
    # write lock. Refuse the second until the first finishes. The
    # reaper runs first so a crashed previous job doesn't block here
    # forever.
    _person_backfill_reap()
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
    # Plain exec'd subprocess, same rationale as the pipeline run child
    # in _start_run: multiprocessing children (any start method) run
    # torch inference pinned to ~1 core; a fresh interpreter doesn't.
    # Stops arrive as SIGTERM; _person_backfill_worker_main's handler
    # sets the flag the entry already polls.
    proc = subprocess.Popen(
        [sys.executable, "-c",
         "import json,os,app; "
         "app._person_backfill_worker_main(json.loads(os.environ['PERSON_BACKFILL_ARGS']))"],
        cwd=Path(__file__).parent,
        env={
            **os.environ,
            # Args (incl. the Immich API key) via env, not argv — see
            # _start_run.
            "PERSON_BACKFILL_ARGS": json.dumps({
                "base_url": base_url,
                "api_key": api_key,
                "db_path": DB_PATH,
                "log_path": str(log_path),
                "status_path": str(status_path),
            }),
        },
    )
    _person_backfill_jobs[job_id]["_proc"] = proc
    return {"job_id": job_id, "n_to_scan": n_to_scan, "noop": False, "already_running": False}


@app.post("/api/person/backfill/stop/{job_id}")
async def stop_person_backfill(job_id: str) -> dict:
    """Ask a running backfill to unwind at the next batch boundary.
    The child's SIGTERM handler flips the stop flag its loop polls;
    the loop exits and marks the job 'stopped' in its JSON status
    file, which the poll endpoint surfaces. A second click while the
    first SIGTERM is still pending escalates to SIGKILL immediately,
    and a watchdog escalates after 45 s (same contract as the pipeline
    run stop) so a wedged child can't stay 'running' forever."""
    job = _person_backfill_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown person-backfill job")
    if job["status"] != "running":
        return {"stopped": False, "reason": "job not running"}
    proc = job.get("_proc")
    if proc is not None and proc.poll() is None:
        if job.get("_terminated_at"):
            try:
                proc.kill()
            except Exception:
                pass
            return {"stopped": True, "escalated": True, "job_id": job_id}
        job["_terminated_at"] = time.monotonic()
        proc.terminate()

        def _kill_watchdog(proc=proc) -> None:
            time.sleep(45)
            if proc.poll() is None:
                try:
                    proc.kill()
                except Exception:
                    pass
                _log.warning(
                    "[person-backfill] child force-killed (did not unwind in 45s)"
                )

        threading.Thread(target=_kill_watchdog, daemon=True).start()
    return {"stopped": True, "job_id": job_id}


@app.get("/api/person/backfill/job/{job_id}")
async def poll_person_backfill_job(job_id: str) -> dict:
    job = _person_backfill_jobs.get(job_id)
    if not job:
        raise HTTPException(status_code=404, detail="unknown person-backfill job")
    # Reap a dead child first so a crash is surfaced even if the start
    # guard is never hit again.
    _person_backfill_reap()
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
                    # No _results_lock here — _invalidate_results_cache
                    # takes it itself and the lock is not reentrant.
                    if not job.get("_cache_invalidated"):
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


class AssetPlantaeOverridePayload(BaseModel):
    """Body for POST /api/assets/batch/plantae-override (UX-17).

    `value` semantics:
      - null → clear the user override (revert to model score)
      - 0     → force non-plant (excluded from Plantae albums even if BioCLIP says plant)
      - 1     → force plant (included even if BioCLIP says non-plant)
    """
    asset_ids: list[str]
    value: int | None


@app.post("/api/assets/batch/plantae-override")
async def batch_set_plantae_override(payload: AssetPlantaeOverridePayload) -> dict:
    """Bulk-set `plantae_override` on the given assets. The first batch
    action surfaced by the multi-select UI (UX-17); 'Mark selected as
    non-plant' writes 0.

    Asset ids that don't exist in the local DB are silently skipped
    (the user batched a page-level selection and a photo may have
    been re-synced between selection and action). Returns the number
    of rows actually changed.

    Invalidate /api/results and /api/clusters caches since downstream
    filter counts and the 'Plantae only' cluster listing both depend
    on this column.
    """
    updated = 0
    with storage.connect(DB_PATH) as conn:
        updated = storage.set_plantae_overrides(conn, payload.asset_ids, payload.value)
    _invalidate_clusters_cache()
    # Force /api/results to re-pull on next read; the cached payload
    # includes plantae_score + plantae tier that no longer match.
    with _results_lock:
        _results_cache.clear()
    return {"updated": updated}


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
        # means it still exists, 404 means missing. A 401/403/429/5xx
        # (or a network error) raises ImmichProbeError — we must NOT
        # treat those as missing, or every asset would look deleted and
        # the confirmed purge would wipe the whole local DB.
        try:
            existing_ids = await client.probe_existing_async(db_ids)
        except ImmichProbeError as e:
            raise HTTPException(status_code=502, detail=str(e)) from e
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
        # Invalidate every cache derived from the asset set: results,
        # cluster counts/names, and the near-dup sibling map.
        _invalidate_results_cache("all:")
        _invalidate_clusters_cache()
        storage.invalidate_global_near_dup_cache()
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