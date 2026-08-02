"""FastAPI web UI for the Immich × BioCLIP demo.

Routes:
  GET  /                  HTML page with run picker + image grid + params form
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
from pathlib import Path

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse, Response
from pydantic import BaseModel

import storage
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
HOST = os.environ.get("HOST", "0.0.0.0")
PORT = int(os.environ.get("PORT", "8765"))
# When RUN_ON_STARTUP=1, the lifespan runs the pipeline once before
# serving. Default is off — the server starts in browse-only mode and
# the user (or a follow-up curl /api/run) triggers a refresh.
RUN_ON_STARTUP = os.environ.get("RUN_ON_STARTUP", "0") == "1"

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
    # Drop the cache entry for this run so the next GET rebuilds it.
    with _results_lock:
        _results_cache.pop(out.run_id, None)
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
        return {"clusters": storage.cluster_summary(conn)}


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
    run: int | None = Query(default=None),
    page: int = Query(default=0, ge=0),
    page_size: int = Query(default=60, ge=1, le=300),
) -> dict:
    with _results_lock:
        # Serve from the in-process cache if we have it.
        target_id: int | None = None
        if run is None:
            with storage.connect(DB_PATH) as conn:
                latest = storage.latest_run(conn)
            if latest is None:
                return {"error": "no runs yet — try Refresh"}
            target_id = latest["id"]
        else:
            target_id = run
        cached = _results_cache.get(target_id)
        if cached is not None:
            # Cached payload has the full results list — slice in memory.
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
        run_row = storage.get_run(conn, target_id)
        if run_row is None:
            raise HTTPException(
                status_code=404, detail=f"run {target_id} not found"
            )
        results = storage.get_run_results(conn, run_id=target_id)
        taxonomy = storage.get_run_taxonomy_summary(
            conn, run_id=target_id, rank_name=run_row["rank"]
        )
        total = len(results)
        start = page * page_size
        end = start + page_size
        page_results = results[start:end]
        payload = _serialize_run(
            run_row,
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
        _results_cache[target_id] = full_payload
    return payload


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
<style>
  :root { color-scheme: light dark; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0; background:#111; color:#eee; }
  header { padding: 14px 24px; border-bottom: 1px solid #333; display:flex; gap:16px; align-items:center; flex-wrap: wrap; }
  h1 { margin:0; font-size: 18px; }
  .meta { color:#9ad; font-family: ui-monospace, monospace; font-size: 13px; }
  .grow { flex:1; }
  select { background:#222; color:#eee; border:1px solid #444; padding:6px 10px; border-radius:6px; max-width: 560px; }
  button { background:#234; color:#eee; border:1px solid #456; padding:6px 12px; border-radius:6px; cursor:pointer; }
  button:hover { background:#345; }
  button:disabled { opacity: 0.5; cursor: progress; }
  .toolbar { padding: 8px 24px; border-bottom: 1px solid #222; display:flex; gap:12px; flex-wrap: wrap; align-items:center; font-size: 13px; }
  .cluster-pill { background:#1c2; color:#cfb; padding: 3px 10px; border-radius: 12px; cursor: pointer; user-select: none; }
  .cluster-pill.on { background:#2c4; color:#fff; }
  main { padding: 24px; display:grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 18px; align-items: start; }
  .card { background:#1a1a1a; border:1px solid #2a2a2a; border-radius:10px; overflow:hidden; display:flex; flex-direction:column; }
  .img-wrap { position: relative; background:#000; }
  .img-wrap img { width:100%; height:auto; display:block; }
  .img-wrap svg.bbox { position:absolute; inset:0; width:100%; height:100%; pointer-events:none; }
  .img-wrap .bbox rect { fill: none; stroke: #5f5; stroke-width: 2; vector-effect: non-scaling-stroke; }
  .img-wrap .bbox text { fill: #5f5; font: 11px/1 ui-monospace, monospace; paint-order: stroke; stroke: #000; stroke-width: 3; }
  .img-wrap .cluster-tag { position:absolute; top:6px; left:6px; background:rgba(0,0,0,0.6); color:#cfb; font:11px ui-monospace, monospace; padding:2px 6px; border-radius:4px; }
  .img-wrap .rank-tag { position:absolute; top:6px; right:6px; background:#1a1a1a; color:#fff; font:11px ui-monospace, monospace; padding:2px 6px; border-radius:4px; box-shadow:0 1px 3px rgba(0,0,0,.4); }
  .toolbar select { background:#1a1a1a; color:#eee; border:1px solid #333; padding:2px 6px; border-radius:4px; font-size:12px; }
  .toolbar label.sort-mode { color:#888; font-size:12px; margin-left:8px; }
  .meta-card { padding: 10px 12px; font-size: 13px; }
  .meta-card .name { font-weight: 600; margin-bottom: 4px; word-break: break-all; }
  .meta-card .urls a { color:#7ad; text-decoration:none; display:block; font-family: ui-monospace, monospace; font-size: 11px; margin-top:2px; word-break: break-all; }
  .meta-card .urls a:hover { text-decoration: underline; }
  .preds { padding: 0 12px 12px; }
  .pred { display:flex; justify-content:space-between; gap:10px; padding: 4px 0; border-top:1px solid #222; font-size: 13px; }
  .pred:first-child { border-top: 0; }
  .pred .sp { flex:1; }
  .pred .sc { color:#ad9; font-variant-numeric: tabular-nums; min-width: 56px; text-align:right; }
  .pred .cn { color:#888; font-size: 11px; }
  .badge { display:inline-block; background:#223; color:#9cf; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right:6px; }
  .plantae { display:inline-block; font-size: 11px; padding: 2px 6px; border-radius: 4px; margin-right:6px; font-family: ui-monospace, monospace; }
  .plantae.green  { background:#1a3a1a; color:#6f6; }
  .plantae.yellow { background:#3a3a1a; color:#ff6; }
  .plantae.red    { background:#3a1a1a; color:#f88; }
  .dets { padding: 0 12px 12px; border-top: 1px dashed #333; margin-top: 4px; }
  .det-block { margin-top: 8px; }
  .det-header { font-size: 11px; color:#ad9; margin-bottom: 2px; font-family: ui-monospace, monospace; }
  .empty { padding: 40px; text-align:center; color:#888; grid-column: 1 / -1; white-space: pre-wrap; }
  .error { color:#f88; }
  /* Params form, status panel, log tail */
  .params { display:flex; gap:8px; flex-wrap: wrap; align-items:center; padding: 10px 24px; border-bottom: 1px solid #222; font-size: 13px; }
  .params input { background:#222; color:#eee; border:1px solid #444; padding:5px 9px; border-radius:6px; font-family: ui-monospace, monospace; }
  .params input[type=text] { min-width: 240px; }
  .params input[type=number] { width: 90px; }
  .params label { color:#888; }
  .status-panel { padding: 8px 24px; border-bottom: 1px solid #2a2a2a; background:#1a1a1a; font: 12px/1.6 ui-monospace, monospace; }
  .status-panel .idle { color:#888; }
  .status-panel .running { color:#fc6; }
  .status-panel .done { color:#6f6; }
  .status-panel .error { color:#f88; }
  .status-panel .progress { background:#222; height:6px; border-radius:3px; margin-top:6px; overflow:hidden; }
  .status-panel .progress > div { background:#6ad; height:100%; transition: width 0.4s ease; }
  /* Time-per-stage stacked bar. Each slice width = (stage_sec / total_sec) * 100% */
  .stage-time-bar { display:flex; height:6px; border-radius:3px; margin-top:4px; overflow:hidden; background:#222; }
  .stage-time-bar .slice { height:100%; transition: width 0.4s ease; }
  .stage-time-legend { display:flex; flex-wrap:wrap; gap:8px 12px; margin-top:4px; font-size:10px; color:#888; }
  .stage-time-legend .swatch { display:inline-block; width:8px; height:8px; border-radius:2px; margin-right:4px; vertical-align:middle; }
  .stage-time-legend .stage-name { color:#aaa; }
  .stage-time-legend .stage-sec { color:#777; margin-left:3px; }
  .log-tail { padding: 0 0 12px; border-bottom: 1px solid #2a2a2a; }
  .log-tail pre { margin:0; padding: 8px 24px; background:#0a0a0a; color:#bcb; font: 11px/1.4 ui-monospace, monospace; max-height: 160px; overflow:auto; white-space: pre; }
  .log-tail .log-toggle { padding: 6px 24px; font-size: 12px; color:#888; cursor: pointer; user-select: none; }
  .log-tail .log-toggle:hover { color:#ad9; }
  /* Pagination bar */
  .pagination-bar { display:flex; align-items:center; justify-content:center; gap:4px; padding: 10px 24px; flex-wrap: wrap; background:#1a1a1a; border-bottom: 1px solid #2a2a2a; font: 12px ui-monospace, monospace; }
  .pagination-bar button { background:#222; color:#eee; border:1px solid #333; padding: 4px 10px; border-radius:4px; cursor:pointer; font: inherit; min-width: 32px; }
  .pagination-bar button:hover:not(:disabled) { background:#333; border-color:#555; }
  .pagination-bar button:disabled { opacity:0.4; cursor: not-allowed; }
  .pagination-bar button.current { background:#2c4; color:#fff; border-color:#4c6; }
  .pagination-bar .ellipsis { color:#888; padding: 0 4px; }
  .pagination-bar .summary { color:#888; margin-right: 12px; }
  /* Near-duplicate panel */
  .near-dup { padding: 0 12px 12px; border-top: 1px dashed #333; margin-top: 4px; }
  .near-dup-toggle { background:transparent; border:0; color:#9cf; padding: 8px 0 4px; cursor:pointer; font: 12px ui-monospace, monospace; text-align: left; }
  .near-dup-toggle:hover { color:#cef; }
  .near-dup-panel { display: flex; gap: 8px; flex-wrap: wrap; padding-top: 4px; }
  .near-dup-item { width: 110px; background:#0a0a0a; border:1px solid #222; border-radius: 6px; padding: 4px; display:flex; flex-direction:column; gap: 4px; }
  .near-dup-item img { width: 100%; height: 90px; object-fit: cover; border-radius: 4px; }
  .near-dup-name { font: 10px/1.3 ui-monospace, monospace; color:#aaa; max-height: 26px; overflow:hidden; word-break: break-all; }
  .near-dup-pick { background:#234; color:#eee; border:1px solid #456; padding: 3px 0; border-radius:4px; font-size: 11px; cursor:pointer; }
  .near-dup-pick:hover { background:#345; }
  .near-dup-pick.active { background:#2a4; border-color:#4c6; }
</style>
</head>
<body>
<header>
  <h1>Immich × BioCLIP</h1>
  <select id="run-picker" title="Run"></select>
  <span class="grow"></span>
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
  <span id="cluster-pills" style="margin-left:12px"></span>
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
</div>
<div class="pagination-bar" id="pagination-bar"></div>
<main id="root">
  <div class="empty" id="loading-msg">Loading from <span id="server-url"></span>…</div>
</main>

<script>
const root = document.getElementById('root');
const picker = document.getElementById('run-picker');
const info = document.getElementById('run-info');
const refreshBtn = document.getElementById('refresh');
const runBtn = document.getElementById('run-btn');
const clustersBar = document.getElementById('clusters-bar');
const statusPanel = document.getElementById('status-panel');
const statusText = document.getElementById('status-text');
const statusBar = document.getElementById('status-bar');
const stageTimeBar = document.getElementById('stage-time-bar');
const stageTimeLegend = document.getElementById('stage-time-legend');
const logToggle = document.getElementById('log-toggle');
const logPre = document.getElementById('log-pre');
const paramQuery = document.getElementById('param-query');
const paramN = document.getElementById('param-n');
const paramK = document.getElementById('param-k');
const paramBatch = document.getElementById('param-batch');

// Surface the server URL we're actually talking to, so connection issues
// are obvious instead of mysterious "Loading…" forever.
document.getElementById('server-url').textContent = location.origin;

let currentGroupBy = (typeof localStorage !== 'undefined' && localStorage.getItem('groupBy')) || 'species';
let currentGroupFilter = null;  // null = all; otherwise cluster_id (number) or taxon name (string)
let currentSortMode = (typeof localStorage !== 'undefined' && localStorage.getItem('sortMode')) || 'rank';
let lastLogCount = 0;             // for /api/logs polling delta

function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
}

async function safeFetch(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

function showError(msg, detail) {
  console.error(msg, detail);
  root.innerHTML = `<div class="empty error">${escapeHtml(msg)}${
    detail ? `<br><small>${escapeHtml(String(detail))}</small>` : ''
  }</div>`;
}

function fmtRunLabel(r) {
  const date = r.started_at || '';
  return `#${r.id} — ${r.query}  (new=${r.n_new}, cached=${r.n_cached})  ${date}`;
}

async function loadRuns(selectId) {
  try {
    const data = await safeFetch('/api/runs');
    picker.innerHTML = data.runs.map(rn =>
      `<option value="${rn.id}">${escapeHtml(fmtRunLabel(rn))}</option>`
    ).join('');
    if (selectId != null) {
      picker.value = String(selectId);
    } else if (data.runs.length) {
      picker.value = String(data.runs[0].id);
    }
    return data.runs;
  } catch (e) {
    showError('Could not reach the server.', `${location.origin}/api/runs — ${e.message}`);
    return [];
  }
}

// Compute taxonomy pill data from the current run's results, OR fall back
// to /api/clusters when grouping by cluster (HDBSCAN labels are global).
async function loadClusterBar() {
  const pillsContainer = document.getElementById('cluster-pills');
  if (currentGroupBy === 'cluster') {
    try {
      const data = await safeFetch('/api/clusters');
      const pills = [
        `<span class="cluster-pill${currentGroupFilter === null ? ' on' : ''}" data-cluster="">all</span>`
      ];
      for (const c of data.clusters) {
        const cid = Number(c.cluster_id);
        pills.push(
          `<span class="cluster-pill${currentGroupFilter === cid ? ' on' : ''}" data-cluster="${cid}">` +
          `cluster ${cid} (${c.n})</span>`
        );
      }
      pillsContainer.innerHTML = pills.join('');
      pillsContainer.querySelectorAll('.cluster-pill').forEach(el => {
        el.addEventListener('click', () => {
          const v = el.getAttribute('data-cluster');
          currentGroupFilter = v === '' ? null : Number(v);
          loadClusterBar();
          if (picker.value) loadResults(picker.value, {page: 0, refreshPills: false});
        });
      });
    } catch (e) {
      pillsContainer.innerHTML = '<span style="color:#888">cluster data unavailable</span>';
    }
    return;
  }
  // Taxonomy mode — counts come from the server-side summary, which
  // aggregates the entire run (not just the current pagination slice).
  const runId = picker.value;
  if (!runId) {
    pillsContainer.innerHTML = '<span style="color:#888">(no run loaded)</span>';
    return;
  }
  let data;
  try {
    // Use page=0 to fetch the first page; the taxonomy summary field
    // is full-run regardless of the page slice.
    data = await safeFetch(`/api/results?run_id=${runId}&page=0&page_size=1`);
  } catch (e) {
    pillsContainer.innerHTML = '<span style="color:#888">results unavailable</span>';
    return;
  }
  const entries = (data.taxonomy && data.taxonomy[currentGroupBy]) || [];
  const pills = [
    `<span class="cluster-pill${currentGroupFilter === null ? ' on' : ''}" data-cluster="">all</span>`
  ];
  for (const {name, n} of entries) {
    const esc = escapeHtml(name);
    pills.push(
      `<span class="cluster-pill${currentGroupFilter === name ? ' on' : ''}" data-cluster="${esc}">` +
      `${esc} (${n})</span>`
    );
  }
  pillsContainer.innerHTML = pills.join('');
  pillsContainer.querySelectorAll('.cluster-pill').forEach(el => {
    el.addEventListener('click', () => {
      const v = el.getAttribute('data-cluster');
      currentGroupFilter = v === '' ? null : v;
      loadClusterBar();
      if (picker.value) loadResults(picker.value, {page: 0, refreshPills: false});
    });
  });
}

function plantaeClass(score, green, yellow) {
  if (score == null) return 'red';
  if (score >= green) return 'green';
  if (score >= yellow) return 'yellow';
  return 'red';
}

function renderPlantaeLegend(data) {
  const legend = document.getElementById('plantae-legend');
  const g = (data.plantae_green ?? 0.60).toFixed(2);
  const y = (data.plantae_yellow ?? 0.35).toFixed(2);
  legend.innerHTML =
    `Plantae ` +
    `<span class="plantae green">≥ ${g}</span>` +
    `<span class="plantae yellow">≥ ${y}</span>` +
    `<span class="plantae red">&lt; ${y}</span>` +
    ` (BioCLIP kingdom-rank confidence)`;
}

// Numbered pagination bar: « first ‹ prev  1 2 ... 50  next › last »
// With ellipsis for runs that span many pages so the bar stays compact.
function renderPagination(p) {
  const bar = document.getElementById('pagination-bar');
  if (!bar) return;
  if (!p || !p.total) {
    bar.innerHTML = '';
    return;
  }
  const cur = p.page;
  const total = p.total_pages;
  const pageSize = p.page_size;
  const totalItems = p.total;
  if (total <= 1) {
    bar.innerHTML = `<span class="summary">${totalItems} photo${totalItems === 1 ? '' : 's'}</span>`;
    return;
  }

  // Choose which page numbers to render. Always show first, last, current
  // and one on each side of current. Collapse the rest into ellipsis.
  const want = new Set([0, total - 1, cur, cur - 1, cur + 1]);
  const pages = [];
  for (let i = 0; i < total; i++) {
    if (want.has(i)) pages.push(i);
    else if (i > 0 && i < total - 1 && (want.has(i - 1) || want.has(i + 1))) {
      pages.push(-1); // ellipsis marker
    }
  }
  // de-dup adjacent ellipsis
  const compact = [];
  for (const x of pages) {
    if (x === -1 && compact.length && compact[compact.length - 1] === -1) continue;
    compact.push(x);
  }

  const btn = (label, target, opts = {}) => {
    const cls = opts.current ? ' class="current"' : '';
    const disabled = opts.disabled ? ' disabled' : '';
    return `<button${cls}${disabled} data-page="${target}">${escapeHtml(label)}</button>`;
  };

  const parts = [];
  parts.push(`<span class="summary">${totalItems} photos, page ${cur + 1}/${total}</span>`);
  parts.push(btn('« first', 0, {disabled: cur === 0}));
  parts.push(btn('‹ prev', cur - 1, {disabled: cur === 0}));
  for (const x of compact) {
    if (x === -1) {
      parts.push(`<span class="ellipsis">…</span>`);
    } else {
      parts.push(btn(String(x + 1), x, {current: x === cur}));
    }
  }
  parts.push(btn('next ›', cur + 1, {disabled: cur >= total - 1}));
  parts.push(btn('last »', total - 1, {disabled: cur >= total - 1}));

  bar.innerHTML = parts.join('');
  bar.querySelectorAll('button[data-page]').forEach(el => {
    el.addEventListener('click', () => {
      const target = Number(el.getAttribute('data-page'));
      if (target === currentPage) return;
      loadResults(picker.value, {page: target, refreshPills: false});
    });
  });
}

function render(data) {
  if (data.error) {
    root.innerHTML = `<div class="empty">${escapeHtml(data.error)}</div>`;
    info.textContent = '';
    return;
  }
  info.textContent =
    `query=${data.query}  ·  rank=${data.rank || '?'}  ·  k=${data.top_k || '?'}  ·  ` +
    `new=${data.n_new}  cached=${data.n_cached}  ·  clusters=${data.n_clusters || 0}`;
  renderPlantaeLegend(data);
  const green = data.plantae_green ?? 0.60;
  const yellow = data.plantae_yellow ?? 0.35;

  let rows = data.results;
  if (currentGroupFilter !== null) {
    if (currentGroupBy === 'cluster') {
      rows = rows.filter(r => r.cluster_id === currentGroupFilter);
    } else {
      // taxonomy: match the chosen species/genus/family against top-1
      rows = rows.filter(r => {
        const top = (r.top_predictions || [])[0];
        return top && top[currentGroupBy] === currentGroupFilter;
      });
    }
  }

  // Sort by current mode (after cluster filter). Search-rank is the
  // identity transform — Immich already returned the assets in the
  // desired order; get_run_results preserves it through the SQL join.
  if (currentSortMode === 'plantae') {
    rows = rows.slice().sort((a, b) =>
      (b.taxon_scores && b.taxon_scores.Plantae != null ? b.taxon_scores.Plantae : -1) -
      (a.taxon_scores && a.taxon_scores.Plantae != null ? a.taxon_scores.Plantae : -1));
  } else if (currentSortMode === 'score') {
    rows = rows.slice().sort((a, b) =>
      ((b.top_predictions && b.top_predictions[0] && b.top_predictions[0].score) || -1) -
      ((a.top_predictions && a.top_predictions[0] && a.top_predictions[0].score) || -1));
  } // 'rank' = leave as-is

  if (!rows.length) {
    root.innerHTML = '<div class="empty">No results in this view.</div>';
    return;
  }

  root.innerHTML = rows.map(r => {
    const preds = r.top_predictions.map(p => `
      <div class="pred">
        <div class="sp">${escapeHtml(p.species)}<div class="cn">${escapeHtml(p.common_name || '')}</div></div>
        <div class="sc">${(p.score * 100).toFixed(2)}%</div>
      </div>
    `).join('');
    const k = r.top_predictions[0] || {};
    const fullTop1 = k.species || '';
    const fullTop1Score = k.score || 0;
    const dets = (r.detections || []).filter(d => d.score >= 0.10);
    const bboxSvg = dets.map(d => {
      const x = (d.x * 100).toFixed(2);
      const y = (d.y * 100).toFixed(2);
      const w = (d.w * 100).toFixed(2);
      const h = (d.h * 100).toFixed(2);
      const lbl = `${escapeHtml(d.label)} ${(d.score * 100).toFixed(0)}%`;
      return `<rect x="${x}%" y="${y}%" width="${w}%" height="${h}%"/>` +
             `<text x="${x}%" y="${(Math.max(0, parseFloat(y) - 0.5)).toFixed(2)}%">${lbl}</text>`;
    }).join('');
    // For each detection, show top-3 species BioCLIP predicted on the
    // cropped bbox. Highlight any that beat the full-image top-1 score.
    const detBlocks = dets.map((d, di) => {
      const bp = (d.bbox_predictions || []).slice(0, 3);
      if (!bp.length) return '';
      const rowsHtml = bp.map(p => {
        const beats = p.species === fullTop1 && p.score > fullTop1Score;
        return `
          <div class="pred">
            <div class="sp">${escapeHtml(p.species)}<div class="cn">${escapeHtml(p.common_name || '')}</div></div>
            <div class="sc">${(p.score * 100).toFixed(2)}%${beats ? ' ↑' : ''}</div>
          </div>
        `;
      }).join('');
      return `
        <div class="det-block">
          <div class="det-header">crop ${di + 1}: ${escapeHtml(d.label)} ${(d.score * 100).toFixed(0)}%</div>
          ${rowsHtml}
        </div>
      `;
    }).join('');
    const nearDups = r.near_dups || [];
    const nearDupPanel = nearDups.length ? `
      <div class="near-dup">
        <button class="near-dup-toggle" data-aid="${escapeHtml(r.asset_id)}">
          ▸ ${nearDups.length} similar image${nearDups.length > 1 ? 's' : ''} (cosine ≥ 0.99)
        </button>
        <div class="near-dup-panel" id="ndp-${escapeHtml(r.asset_id)}" style="display:none">
          ${nearDups.map(d => `
            <div class="near-dup-item" data-aid="${escapeHtml(r.asset_id)}" data-other="${escapeHtml(d.asset_id)}">
              <img src="/thumb/${encodeURIComponent(d.asset_id)}" alt="${escapeHtml(d.original_file_name || '')}" loading="lazy" decoding="async">
              <div class="near-dup-name" title="${escapeHtml(d.original_file_name || '')}">${escapeHtml(d.original_file_name || d.asset_id)}</div>
              <button class="near-dup-pick" title="Use this image as the primary on this card">use as primary</button>
            </div>
          `).join('')}
        </div>
      </div>
    ` : '';
    return `
      <div class="card">
        <div class="img-wrap">
          <img src="/thumb/${encodeURIComponent(r.asset_id)}" alt="${escapeHtml(r.original_file_name)}" loading="lazy" decoding="async" data-primary-asset="${escapeHtml(r.asset_id)}">
          ${r.cluster_id !== null && r.cluster_id !== undefined ? `<span class="cluster-tag">cluster ${r.cluster_id}</span>` : ''}
          ${r.rank !== undefined && r.rank !== null ? `<span class="rank-tag" title="Immich smart-search rank (0-based)">#${r.rank + 1}</span>` : ''}
          ${bboxSvg ? `<svg class="bbox" viewBox="0 0 100 100" preserveAspectRatio="none">${bboxSvg}</svg>` : ''}
        </div>
        <div class="meta-card">
          <div class="name">${escapeHtml(r.original_file_name)}</div>
          <div><span class="badge">${escapeHtml(k.kingdom || '?')}</span><span class="badge">${escapeHtml(k.family || '?')}</span><span class="badge">${escapeHtml(k.genus || '?')}</span>${(() => {
            const ts = r.taxon_scores || {};
            const plantae = ts.Plantae;
            if (plantae == null) return '';
            const cls = plantaeClass(plantae, green, yellow);
            const others = Object.entries(ts)
              .filter(([n]) => n !== 'Plantae')
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([n, s]) => `${escapeHtml(n)}=${(s*100).toFixed(0)}`)
              .join(' · ');
            return `<span class="plantae ${cls}" title="BioCLIP kingdom-rank score for Plantae${others ? ' — other kingdoms: '+others : ''}">Plantae ${(plantae*100).toFixed(1)}%</span>`;
          })()}</div>
          <div class="urls">
            <a href="${escapeHtml(r.original_url)}" target="_blank">original →</a>
            <a href="${escapeHtml(r.thumbnail_url)}" target="_blank">thumb →</a>
          </div>
        </div>
        <div class="preds">${preds}</div>
        ${detBlocks ? `<div class="dets">${detBlocks}</div>` : ''}
        ${nearDupPanel}
      </div>
    `;
  }).join('');
}

let currentPage = 0;
let currentPageSize = 60;

async function loadResults(runId, opts = {}) {
  const page = opts.page ?? currentPage;
  const pageSize = opts.pageSize ?? currentPageSize;
  currentPage = page;
  currentPageSize = pageSize;
  const runParam = runId != null ? `run=${runId}` : '';
  const url = `/api/results?${runParam}&page=${page}&page_size=${pageSize}`.replace(/\?$/, '');
  try {
    const data = await safeFetch(url);
    lastResults = data;
    render(data);
    renderPagination(data.pagination);
    // Pills only need to be re-built when the run itself changes.
    if (opts.refreshPills !== false) loadClusterBar();
  } catch (e) {
    showError('Could not load results.', `${url} — ${e.message}`);
  }
}

function fmtDuration(s) {
  if (!s || s <= 0) return '';
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  if (m < 60) return `${m}m${r.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h${(m % 60).toString().padStart(2, '0')}m`;
}

function renderStatus(s) {
  if (!s) return;
  const pct = s.chunk_total > 0
    ? Math.min(100, (s.chunk_idx / s.chunk_total) * 100)
    : (s.status === 'running' ? 5 : 100);
  statusBar.style.width = `${pct}%`;
  statusPanel.className = 'status-panel';  // reset
  if (s.status === 'running') {
    statusPanel.classList.add('running');
    const stage = s.stage || '?';
    const ci = s.chunk_idx || 0;
    const ct = s.chunk_total || 0;
    const eta = s.eta_seconds ? `  eta=${fmtDuration(s.eta_seconds)}` : '';
    const rate = s.rate ? `  ${s.rate.toFixed(1)}/s` : '';
    const q = s.query ? `  query=${s.query}` : '';
    statusText.textContent = `[${stage}] chunk ${ci}/${ct}${rate}${eta}${q}`;
    refreshBtn.disabled = true;
    runBtn.disabled = true;
    renderStageTimes(s);  // live: include the stage currently being timed
  } else if (s.status === 'done') {
    statusPanel.classList.add('done');
    statusText.textContent = `done  run #${s.current_run_id ?? '?'}  ` +
      `(${s.n_new} new, ${s.n_cached} cached)`;
    refreshBtn.disabled = false;
    runBtn.disabled = false;
    renderStageTimes(s);  // final: show all accumulated stages
  } else if (s.status === 'error') {
    statusPanel.classList.add('error');
    statusText.textContent = `error: ${s.last_error || 'unknown'}`;
    refreshBtn.disabled = false;
    runBtn.disabled = false;
    renderStageTimes(s);
  } else {
    statusPanel.classList.add('idle');
    statusText.textContent = 'idle';
    refreshBtn.disabled = false;
    runBtn.disabled = false;
    stageTimeBar.style.display = 'none';
    stageTimeLegend.innerHTML = '';
  }
}

// Per-stage timing bar — each stage = colored slice, width = pct of total.
const STAGE_COLORS = {
  search: '#4a9eff',       // blue
  species: '#50c878',      // green
  cluster: '#f5a623',      // orange
  kingdom: '#9b59b6',      // purple
  yolo: '#e74c3c',         // red
  'bbox-bioclip': '#1abc9c', // teal
};
function renderStageTimes(s) {
  const times = s.stage_times || {};
  if (!Object.keys(times).length) {
    stageTimeBar.style.display = 'none';
    stageTimeLegend.innerHTML = '';
    return;
  }
  // While running, project the current stage's elapsed-so-far into the
  // bar so the slice is visibly growing in real time. We compute it from
  // (now - stage_t0) but only if the server's stage_t0 matches stage_times.
  const live = Object.assign({}, times);
  if (s.status === 'running' && s.stage && s.stage_t0) {
    const elapsed = (Date.now() / 1000) - s.stage_t0;
    if (elapsed > 0) live[s.stage] = (live[s.stage] || 0) + elapsed;
  }
  const total = Object.values(live).reduce((a, b) => a + b, 0) || 1;
  stageTimeBar.innerHTML = '';
  stageTimeLegend.innerHTML = '';
  let first = true;
  for (const [name, sec] of Object.entries(live)) {
    const pct = (sec / total) * 100;
    const color = STAGE_COLORS[name] || '#777';
    const slice = document.createElement('div');
    slice.className = 'slice';
    slice.style.background = color;
    slice.style.width = `${pct}%`;
    slice.title = `${name}: ${fmtDuration(sec)}`;
    stageTimeBar.appendChild(slice);
    const item = document.createElement('span');
    item.innerHTML = `<span class="swatch" style="background:${color}"></span>` +
      `<span class="stage-name">${name}</span>` +
      `<span class="stage-sec">${fmtDuration(sec)}</span>`;
    if (!first) item.style.marginLeft = '0';
    first = false;
    stageTimeLegend.appendChild(item);
  }
  stageTimeBar.style.display = 'flex';
}

async function pollStatus() {
  try {
    const s = await safeFetch('/api/status');
    renderStatus(s);
  } catch (e) {
    // Transient — keep the last status visible.
    console.warn('status poll failed:', e);
  }
}

async function pollLogs() {
  if (logPre.style.display === 'none') return;  // collapsed
  try {
    const data = await safeFetch('/api/logs?tail=200');
    const lines = data.lines || [];
    logPre.textContent = lines.join('\\n');
    // Auto-scroll to bottom if user hasn't scrolled up.
    logPre.scrollTop = logPre.scrollHeight;
  } catch (e) {
    console.warn('logs poll failed:', e);
  }
}

logToggle.addEventListener('click', () => {
  const showing = logPre.style.display !== 'none';
  logPre.style.display = showing ? 'none' : 'block';
  logToggle.textContent = showing ? '▸ logs (click to expand)' : '▾ logs';
  if (!showing) pollLogs();
});

function collectParams() {
  const params = {};
  const q = paramQuery.value.trim();
  const n = parseInt(paramN.value, 10);
  const k = parseInt(paramK.value, 10);
  const b = parseInt(paramBatch.value, 10);
  if (q) params.query = q;
  if (Number.isFinite(n) && n > 0) params.n_samples = n;
  if (Number.isFinite(k) && k > 0) params.top_k = k;
  if (Number.isFinite(b) && b > 0) params.batch_size = b;
  return params;
}

async function runWithParams() {
  const params = collectParams();
  runBtn.disabled = true;
  refreshBtn.disabled = true;
  runBtn.textContent = 'Running…';
  root.innerHTML = '<div class="empty">Running pipeline…</div>';
  try {
    const data = await safeFetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    await loadRuns(data.run_id);
    await loadClusterBar();
    await loadResults(data.run_id);
  } catch (e) {
    showError('Run failed.', e.message);
  } finally {
    runBtn.disabled = false;
    refreshBtn.disabled = false;
    runBtn.textContent = 'Run';
  }
}

async function refresh() {
  refreshBtn.disabled = true;
  refreshBtn.textContent = 'Refreshing…';
  root.innerHTML = '<div class="empty">Running pipeline…</div>';
  try {
    const data = await safeFetch('/api/refresh');
    await loadRuns(data.run_id);
    await loadClusterBar();
    await loadResults(data.run_id);
  } catch (e) {
    showError('Refresh failed.', e.message);
  } finally {
    refreshBtn.disabled = false;
    refreshBtn.textContent = 'Refresh';
  }
}

picker.addEventListener('change', () => loadResults(picker.value));
refreshBtn.addEventListener('click', refresh);
runBtn.addEventListener('click', runWithParams);
const sortModeSel = document.getElementById('sort-mode');
if (sortModeSel) {
  sortModeSel.value = currentSortMode;
  sortModeSel.addEventListener('change', () => {
    currentSortMode = sortModeSel.value;
    try { localStorage.setItem('sortMode', currentSortMode); } catch (_) {}
    // Re-render the latest data with the new sort.
    loadResults(picker.value || null);
  });
}
const groupBySel = document.getElementById('group-by');
if (groupBySel) {
  groupBySel.value = currentGroupBy;
  groupBySel.addEventListener('change', () => {
    currentGroupBy = groupBySel.value;
    currentGroupFilter = null;  // reset filter when switching grouping
    try { localStorage.setItem('groupBy', currentGroupBy); } catch (_) {}
    loadClusterBar();
    if (picker.value) loadResults(picker.value, {page: 0, refreshPills: false});
  });
}

// Re-render the last-loaded results without re-fetching — used by group-by
// Last fetched /api/results payload, used by the filter pill handlers to
// re-render the current page without an extra round-trip when filter-only
// changes happen (e.g. clicking a cluster pill while staying on the same
// page).
let lastResults = null;
function renderCurrent() {
  if (lastResults) render(lastResults);
}

// Near-dup handlers: delegated on the grid so we don't have to rebind
// every render. Pick-as-primary swaps the primary card's thumbnail
// (cosmetic — we don't touch the DB or the run).
document.addEventListener('click', (e) => {
  const t = e.target.closest('.near-dup-toggle');
  if (t) {
    const aid = t.getAttribute('data-aid');
    const panel = document.getElementById('ndp-' + aid);
    if (!panel) return;
    const showing = panel.style.display !== 'none';
    panel.style.display = showing ? 'none' : 'flex';
    t.textContent = showing
      ? `▸ ${t.textContent.replace(/^▸ |^▾ /, '')}`
      : `▾ ${t.textContent.replace(/^▸ |^▾ /, '')}`;
    return;
  }
  const pick = e.target.closest('.near-dup-pick');
  if (pick) {
    const item = pick.closest('.near-dup-item');
    const primaryAid = item.getAttribute('data-aid');
    const otherAid = item.getAttribute('data-other');
    const card = pick.closest('.card');
    const img = card.querySelector('.img-wrap img');
    const altName = item.querySelector('.near-dup-name').textContent;
    if (img) img.src = '/thumb/' + encodeURIComponent(otherAid);
    const nameEl = card.querySelector('.meta-card .name');
    if (nameEl) nameEl.textContent = altName + '  (was ' + primaryAid + ')';
    // Mark active in the panel.
    card.querySelectorAll('.near-dup-pick').forEach(b => b.classList.remove('active'));
    pick.classList.add('active');
  }
});

(async () => {
  await loadRuns();
  await loadClusterBar();
  await pollStatus();
  if (picker.value) await loadResults(picker.value);
})();
setInterval(pollStatus, 1000);
setInterval(pollLogs, 2000);
setInterval(async () => {
  const prev = picker.value;
  await loadRuns();
  if (prev) picker.value = prev;
}, 30000);
</script>
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