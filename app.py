"""FastAPI web UI for the Immich × BioCLIP demo.

Routes:
  GET  /                  HTML page with image grid + params form (no run picker)
  GET  /api/results       JSON of the latest run (or ?run=<id>)
  GET  /api/runs          list of recent runs (newest first)
  GET  /api/clusters      cluster summary (cluster_id → # of images)
  GET  /api/refresh       re-run with current env-config params; returns JSON
  POST /api/run           re-run with explicit params (query, n_samples, ...)
  GET  /api/status        live pipeline state (stage, chunk, eta, ...)
  GET  /api/logs          last N log lines from the file log
  GET  /thumb/{id}        proxies an Immich thumbnail

Run with:
    uv run app.py
"""

from __future__ import annotations

import asyncio
import os
import threading
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import storage
from immich_client import ImmichClient
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
N_SAMPLES = int(os.environ.get("N_SAMPLES", "120"))
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
app.mount("/static", StaticFiles(directory="static"), name="static")


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
                      thumbnail_url, cluster_id
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

class PlantsAlbumParams(BaseModel):
    min_score: float | None = None     # default → PLANTAE_GREEN at call site
    force_recreate: bool = False       # set True to drop the saved id and create


def _immich_client() -> ImmichClient:
    return ImmichClient(base_url=BASE_URL, api_key=API_KEY, timeout=60.0)


@app.post("/api/album/plants")
async def create_plants_album(params: PlantsAlbumParams) -> dict:
    min_score = params.min_score if params.min_score is not None else PLANTAE_GREEN
    asset_ids = []
    with storage.connect(DB_PATH) as conn:
        asset_ids = storage.find_plantae_assets(conn, min_score=min_score)
        saved = storage.get_album_meta(conn, "plants_album_id") or {}
        saved_album_id = saved.get("album_id") if not params.force_recreate else None
        saved_count = int(saved.get("n_assets", 0)) if saved_album_id else 0

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
            album = client.create_album(
                name="Plantae (BioCLIP)",
                description=(
                    f"Photos classified as Plantae by BioCLIP-2 kingdom-rank "
                    f"(threshold ≥ {min_score:.2f})."
                ),
            )
            album_id = album["id"]
            reused = False

        # Chunked bulk-add. Immich has no published per-call cap; we
        # mirror the project's _MAX_PAGE_SIZE for consistency with the
        # smart-search pagination. PUT each chunk.
        chunk_size = 1000
        added_total = 0
        for start in range(0, len(asset_ids), chunk_size):
            chunk = asset_ids[start:start + chunk_size]
            client.add_assets_to_album(album_id=album_id, asset_ids=chunk)
            added_total += len(chunk)
    finally:
        client.close()

    # Persist for idempotency.
    with storage.connect(DB_PATH) as conn:
        storage.set_album_meta(
            conn,
            "plants_album_id",
            {
                "album_id": album_id,
                "n_assets": len(asset_ids),
                "min_score": min_score,
                "created_at": (saved.get("created_at") if reused and saved.get("created_at") else None)
                              or datetime.now(timezone.utc).isoformat(),
            },
        )

    return {
        "album_id": album_id,
        "album_url": f"{BASE_URL}/albums/{album_id}",
        "n_assets": len(asset_ids),
        "n_added_this_call": added_total,
        "reused_existing": reused,
        "min_score": min_score,
    }


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


@app.get("/thumb/{asset_id}")
async def thumb(asset_id: str, size: str = "preview") -> Response:
    async with httpx.AsyncClient(timeout=30.0) as c:
        r = await c.get(
            f"{BASE_URL}/api/assets/{asset_id}/thumbnail",
            params={"size": size},
            headers={"x-api-key": API_KEY},
        )
    if r.status_code != 200:
        raise HTTPException(status_code=r.status_code, text=r.text)
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


PAGE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Immich × BioCLIP</title>
<link rel="stylesheet" href="/static/css/app.css">
</head>
<body>
<header>
  <h1>Immich × BioCLIP</h1>
  <span class="grow"></span>
  <span class="knn-pill" id="knn-pill" title="KNN inventory status">KNN: ?</span>
  <span class="meta" id="run-info"></span>
  <button id="refresh">Refresh</button>
</header>
<div class="params">
  <label>query <input type="text" id="param-query" placeholder="intentional plant photos"></label>
  <label>n <input type="number" id="param-n" min="1" max="1000"></label>
  <label>top_k <input type="number" id="param-k" min="1" max="20"></label>
  <label>batch <input type="number" id="param-batch" min="1" max="64"></label>
  <button id="run-btn">Run</button>
  <span id="param-hint" style="color:#888; font-size: 12px;">empty fields = use server default</span>
</div>
<div class="status-panel" id="status-panel">
  <span class="idle" id="status-text">idle</span>
  <div class="progress"><div id="status-bar" style="width:0%"></div></div>
  <div id="stage-time-bar" class="stage-time-bar" style="display:none"></div>
  <div id="stage-time-legend" class="stage-time-legend"></div>
</div>
<div class="log-tail">
  <div class="log-toggle" id="log-toggle">▸ logs (click to expand)</div>
  <pre id="log-pre" style="display:none"></pre>
</div>
<div class="toolbar" id="clusters-bar">
  <span style="color:#888">group by:</span>
  <select id="group-by">
    <option value="species">species</option>
    <option value="genus">genus</option>
    <option value="family">family</option>
    <option value="cluster">cluster (HDBSCAN)</option>
  </select>
  <input id="cluster-search" class="cluster-search" type="search" placeholder="search species…" autocomplete="off" style="display:none">
  <div id="cluster-pills" class="cluster-pills"></div>
  <label class="sort-mode">sort:
    <select id="sort-mode">
      <option value="rank">search rank</option>
      <option value="plantae">plantae confidence</option>
      <option value="score">top-1 score</option>
    </select>
  </label>
</div>
<div class="toolbar" id="legend-bar" style="border-top:0; padding-top:0; font-size: 12px;">
  <span id="plantae-legend" style="color:#888"></span>
  <span class="grow"></span>
  <span id="plants-album-status" style="color:#888; margin-right:12px"></span>
  <button class="reclassify-btn" id="plants-album-btn" title="Create (or refresh) an Immich album containing every asset classified as Plantae" style="margin-right:8px">Tag as Plants album</button>
  <button class="reclassify-btn" id="reclassify-visible-btn" title="KNN-reclassify every asset visible on this page">Reclassify visible with KNN</button>
</div>
<div class="pagination-bar" id="pagination-bar"></div>
<main id="root">
  <div class="empty" id="loading-msg">Loading from <span id="server-url"></span>…</div>
</main>

<!-- Photo detail modal (native <dialog>). Body populated by modal.js -->
<dialog id="detail-dialog" class="detail-dialog">
  <div class="modal-shell">
    <div class="modal-header">
      <div class="modal-title" id="modal-title">Asset detail</div>
      <button class="modal-close" data-action="close-modal">✕</button>
    </div>
    <div class="modal-body" id="modal-body">
      <div class="modal-loading">Loading…</div>
    </div>
  </div>
</dialog>

<script type="module" src="/static/js/app.js"></script>
</body>
</html>
"""


@app.get("/", response_class=HTMLResponse)
async def index() -> Response:
    # Force the browser to revalidate so CSS/JS changes show up without
    # a hard refresh. The page is small and cheap to serve.
    return Response(
        content=PAGE,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-cache"},
    )


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=PORT)