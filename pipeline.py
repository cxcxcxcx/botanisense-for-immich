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
import json
import logging
import os
import sqlite3
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import threading

# Heavy imports (torch/bioclip/sklearn-via-clusters/ultralytics-via-yolo)
# are deferred to run_pipeline() so that `import pipeline` stays cheap and
# the FastAPI browse-only server doesn't pay ~3 GB of model weights at
# startup. See run_pipeline() for the deferred-import block.

import storage
from storage import PERSON_LABELS, PERSON_MIN_CONF, HAND_LABEL, HAND_MIN_CONF, has_real_person
from immich_client import ImmichAsset, ImmichClient

# PEP 562 module-level __getattr__ — lazy-loads the heavy bits
# (torch/bioclip/sklearn-via-clusters/ultralytics-via-yolo) the first
# time anything looks them up by name. Keeps `import pipeline` cheap
# for the browse-only FastAPI server, while still allowing
# `pipeline.TreeOfLifeClassifier`, `pipeline.clusters`, `pipeline.yolo`,
# `pipeline.Rank` to work for both test patches (mock.patch.object) and
# the deferred body of run_pipeline().
_LAZY_ATTRS = ("TreeOfLifeClassifier", "Rank", "clusters", "yolo", "knn")


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
    if name == "knn":
        import knn as _knn
        globals()["knn"] = _knn
        return _knn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


THUMBNAIL_SIZE = "preview"
DEFAULT_BATCH_SIZE = 16
# Fractional growth in the embedded-asset count that triggers a full
# HDBSCAN refit (vs. incremental assignment for new points only).
# 0.10 = "refit when N grows by 10% since the last fit". Override with
# the GROWTH_THRESHOLD_PCT env var; set to 0.0 to force a refit every
# run, or to a very large value to effectively disable refits.
GROWTH_THRESHOLD_PCT = float(os.environ.get("GROWTH_THRESHOLD_PCT", "0.10"))
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


def _close_image(im) -> None:
    """Best-effort PIL image close. Closes can fail on already-freed
    images or unusual file types; we log at debug instead of silently
    passing so teardown regressions are still traceable."""
    try:
        im.close()
    except Exception:
        _log.debug("image close failed", exc_info=True)


# Process-wide classifier cache. The BioCLIP model on CPU holds ~5.5 GB
# resident — loading it on every run is wasteful, but loading it on a
# fully-cached run (where it never gets called) is even worse. Resolving
# happens lazily inside run_pipeline so an entirely-cached run can
# finish without ever importing the model.
#
# Thread-safe: run_pipeline runs in a worker thread; the lazy-init below
# uses a lock so two concurrent runs can't double-construct.
_clf_lock = threading.Lock()
_clf_cache: "object | None" = None  # TreeOfLifeClassifier instance, lazily


def _resolve_classifier() -> "object | None":
    """Return the cached TreeOfLifeClassifier, instantiating on first
    call. Returns the same instance for the lifetime of the process so
    BioCLIP weights stay mmap'd and the second-onwards n_new>0 runs
    don't pay the load cost again."""
    global _clf_cache
    if _clf_cache is not None:
        return _clf_cache
    with _clf_lock:
        if _clf_cache is None:
            _TreeOfLifeClassifier = globals()["TreeOfLifeClassifier"]
            _clf_cache = _TreeOfLifeClassifier()
    return _clf_cache



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
    "stage_times": {},
    "stage_t0": None,
}

# Cooperative-stop signal. Set by POST /api/run/stop; checked at every
# batch/chunk boundary in the loops below so the worker unwinds cleanly,
# commits whatever it has, and exits with status='stopped'.
_stop_event = threading.Event()


class _PipelineStop(Exception):
    """Raised by `check_stop()` when the user requested a stop. The
    outer try/except in run_pipeline catches this and runs the same
    cleanup as success/error paths (commit, set state, log)."""


def request_stop() -> bool:
    """Signal the running pipeline to stop at the next batch boundary.
    Returns True if a stop was actually requested (i.e. the pipeline
    was running); False if nothing was running."""
    with _state_lock:
        was_running = _state["status"] == "running"
    if was_running:
        _stop_event.set()
    return was_running


def clear_stop() -> None:
    """Reset the stop flag. Called by reset_state() at the start of a
    new run so a previous stop doesn't leak into the next run."""
    _stop_event.clear()


def is_stopped() -> bool:
    """Cheap check used at batch boundaries. Returns True if a stop
    has been requested since the last clear_stop()."""
    return _stop_event.is_set()


def check_stop(stop_event=None) -> None:
    """Same as is_stopped() but raises _PipelineStop so callers can
    use try/except for control flow rather than checking on every line.

    `stop_event` is an optional multiprocessing.Event passed by the
    parent when `run_pipeline` is running in a child process. We check
    it in addition to the module-level `_stop_event` so the parent's
    /api/run/stop can reach the worker across the process boundary
    without sending signals."""
    if (stop_event is not None and stop_event.is_set()) or _stop_event.is_set():
        raise _PipelineStop("user requested stop")

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
    # Clear any leftover stop flag from a previous run that was
    # actually stopped (or killed) — otherwise the new run would
    # immediately bail at the first check_stop().
    clear_stop()
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
                # Per-stage elapsed seconds, accumulated across the run.
                # Each stage pushes its own value via _stage_end(name).
                "stage_times": {},
                "stage_t0": None,
            }
        )
    _maybe_mirror_state()


def _stage_begin(name: str, *, chunk_total: int | None = None) -> None:
    """Mark the start of `name` for the per-stage timing bar.

    `chunk_total` is the planned number of chunks for this stage; the
    caller can update `chunk_idx` via set_state() as work progresses so
    the UI progress bar advances in real time.
    """
    with _state_lock:
        _state["stage"] = name
        # wall-clock seconds so the JS UI can subtract from Date.now()/1000
        # to project the running slice's elapsed time live.
        _state["stage_t0"] = time.time()
        _state["chunk_idx"] = 0
        _state["chunk_total"] = chunk_total if chunk_total is not None else 0
    _maybe_mirror_state()


def _stage_end(name: str) -> None:
    """Accumulate elapsed seconds into _state['stage_times'][name]."""
    with _state_lock:
        t0 = _state.get("stage_t0")
        if t0 is not None:
            elapsed = time.time() - t0
            times = _state.get("stage_times")
            if times is None:
                times = {}
                _state["stage_times"] = times
            times[name] = times.get(name, 0.0) + elapsed
            _state["stage_t0"] = None
    _maybe_mirror_state()


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
    _maybe_mirror_state()


def error_state(message: str) -> None:
    with _state_lock:
        _state.update(
            status="error", last_error=message, finished_at=time.time()
        )
    _maybe_mirror_state()


def stopped_state(message: str = "stopped by user") -> None:
    """Mark the run as stopped — the user clicked Stop. Distinct from
    `error_state` so the UI can render a neutral "stopped" badge
    instead of a red error one. We clear the request_stop flag so the
    next run starts fresh."""
    with _state_lock:
        _state.update(
            status="stopped",
            last_error=message,
            finished_at=time.time(),
            eta_seconds=0.0,
        )
    _maybe_mirror_state()
    _stop_event.clear()


def set_state(**updates) -> None:
    with _state_lock:
        _state.update(updates)
    _maybe_mirror_state()


def set_chunk_progress(idx: int, total: int) -> None:
    """Update the in-stage chunk counter. Called per chunk from the
    main loops so the UI progress bar advances."""
    with _state_lock:
        _state["chunk_idx"] = idx
        _state["chunk_total"] = total
    _maybe_mirror_state()


# When running in a child process (see _pipeline_child_entry in app.py),
# the parent's `pipeline.get_state()` returns its own empty state because
# `_state` is module-level per process. We mirror the child's state to
# a JSON file on every set_state / _stage_* call so the parent's snapshot
# builder can read it for /api/status and /api/events. Cheap (one fsync'd
# write per state change, ~200 Hz max during a run) and survives the
# child crashing — the file persists after exit so the parent can show
# the final state for a moment before clearing it.
_status_file_path: Path | None = None


def set_status_file(path: Path | None) -> None:
    """Tell the pipeline module where to mirror state. Pass None to
    disable mirroring (the default, used when run_pipeline is in the
    same process as the FastAPI worker). The child-process wrapper
    _pipeline_child_entry calls this on entry."""
    global _status_file_path
    _status_file_path = path


def _maybe_mirror_state() -> None:
    if _status_file_path is None:
        return
    try:
        with _state_lock:
            snap = dict(_state)
        # Atomic-ish: write to .tmp then rename. The parent reads on
        # every SSE tick (~5 Hz) so it sees either the old or the new
        # payload, never a half-written one.
        tmp = _status_file_path.with_suffix(_status_file_path.suffix + ".tmp")
        tmp.write_text(json.dumps(snap, default=str))
        tmp.replace(_status_file_path)
    except Exception:
        # Don't let a status-mirror error kill the pipeline. Worst case
        # the parent sees stale state until the next successful write.
        pass


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


def _load_vectors(
    conn: sqlite3.Connection, asset_ids: list[str]
) -> np.ndarray:
    """Fetch the BioCLIP embeddings for the given asset_ids, returned as
    a (N, EMB_DIM) float32 contiguous array aligned with asset_ids.

    Order matches asset_ids exactly. We don't assume the DB rows come
    back in any particular order — we index by position so the caller
    can zip the result with the input list.

    Used by the cluster stage in `run_pipeline` to feed
    clusters.assign_new_points() after the species embeddings have been
    written for this run's new assets.
    """
    import storage as _storage

    if not asset_ids:
        return np.empty((0, 0), dtype="<f4")
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"""
        SELECT asset_id, vector FROM embeddings
        WHERE asset_id IN ({placeholders})
        """,
        asset_ids,
    ).fetchall()
    by_id = {
        r["asset_id"]: np.frombuffer(r["vector"], dtype="<f4").copy()
        for r in rows
    }
    # Re-order to match asset_ids. Missing (shouldn't happen) become zeros.
    dim = next(iter(by_id.values())).shape[0] if by_id else 0
    out = np.zeros((len(asset_ids), dim), dtype="<f4")
    for i, aid in enumerate(asset_ids):
        v = by_id.get(aid)
        if v is not None:
            out[i] = v
    return np.ascontiguousarray(out)


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
PLANT_MIN_PROB = 0.60            # Plantae or Fungi > this counts as a candidate
PLANT_BATCH_MAJORITY = 0.30      # 30% of the batch must be candidates to continue
# Tail batches (the leftover from page_size % batch_size) can be tiny —
# e.g. 500 % 32 = 20 items. With n=20 a 30% majority is ±6 candidates;
# with n=4 it's ±1. A single BioCLIP noise spike can flip the boundary
# decision and terminate the search prematurely. Skip the boundary check
# on batches smaller than this — accumulate candidates and continue to
# the next page, where the next (full-size) batch will get a reliable
# signal. Set to 0 to disable.
PLANT_MIN_BATCH_FOR_BOUNDARY = 24


def _score_batch_for_kingdoms(
    clf,
    paths: list[str],
) -> dict[str, dict[str, float]]:
    """Run BioCLIP kingdom-rank predict on `paths` and return
    {file_path: {kingdom: score}} for Plantae and Fungi. Other kingdoms
    (Animalia, etc.) are skipped — discovery only cares about plants and
    mushrooms. Cached scores should be checked before calling this; we
    don't cache here because the caller (discover_plantae_assets)
    persists the rows after we get them back."""
    if not paths:
        return {}
    # BioCLIP needs the Rank enum, not a string. Resolve lazily via the
    # PEP 562 __getattr__ on this module so the import cost stays
    # deferred until the first live run.
    import sys as _sys
    _Rank = _sys.modules[__name__].Rank
    raw = clf.predict(images=paths, rank=_Rank.KINGDOM, k=5)
    # Build {path: {Plantae: x, Fungi: y}} so the discovery loop can
    # decide on Plantae OR Fungi candidates in one pass.
    out: dict[str, dict[str, float]] = {}
    for row in raw:
        kingdom = row.get("kingdom")
        if kingdom not in ("Plantae", "Fungi"):
            continue
        f = row["file_name"]
        score = float(row["score"])
        d = out.setdefault(f, {})
        if score > d.get(kingdom, 0.0):
            d[kingdom] = score
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
    stop_event=None,
) -> list[ImmichAsset]:
    """Stream-search Immich and return assets up to the plant/fungi boundary.

    Algorithm:
      1. Fetch pages of PLANT_PAGE_SIZE from Immich (size=500, page=1, 2…).
      2. For each page, walk through it in decision batches of
         `decision_batch_size` (caller passes 2× the inference batch so each
         decision step is a whole number of GPU chunks).
      3. For each batch, ensure thumbs exist (download if needed) and score
         each asset's Plantae + Fungi confidence — cache hit reads the
         stored row, cache miss runs BioCLIP kingdom-rank predict on the
         batch and persists the result.
      4. If >= PLANT_BATCH_MAJORITY of the batch have
         max(Plantae, Fungi) > PLANT_MIN_PROB, include those assets and
         continue to the next batch.
      5. If < PLANT_BATCH_MAJORITY AND the batch has at least
         PLANT_MIN_BATCH_FOR_BOUNDARY items, we treat it as a real
         boundary: include EVERY asset from this batch (plants, mushrooms,
         and everything else) so the user can review the boundary with
         their kingdom scores visible, then return.
      6. If < PLANT_BATCH_MAJORITY but the batch is *smaller* than
         PLANT_MIN_BATCH_FOR_BOUNDARY (the tail of a page), the majority
         ratio is too noisy to trust — a single candidate can flip the
         decision and prematurely terminate the search. Skip the boundary
         termination: include plants-or-fungi from this batch and keep
         going. The next full-size batch will produce a reliable signal.

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
        # User-visible stop check between Immich pages. Stopping here
        # means we've already committed the previous page's batch into
        # the DB so no work is lost.
        check_stop(stop_event)
        n_batches = (len(items) + batch_size - 1) // batch_size
        _log.info(
            f"[search] page {page} from Immich: got {len(items)} photos to score "
            f"(target={n_samples}, run so far={len(accepted)})"
        )

        for batch_start in range(0, len(items), batch_size):
            batch = items[batch_start : batch_start + batch_size]
            batch_idx = batch_start // batch_size + 1

            # Per-batch stop check. The previous batch already committed
            # all its DB writes (we commit after each batch's writes
            # above), so stopping here is safe.
            check_stop(stop_event)

            # Gather cache hits and download preds in one pass.
            # plantae_by_path: {path: {"Plantae": p, "Fungi": f}} — max of
            # the two is used downstream to decide if a photo counts as a
            # "candidate" (plant OR mushroom).
            plantae_by_path: dict[str, dict[str, float]] = {}
            to_predict_paths: list[str] = []
            to_predict_assets: list[ImmichAsset] = []
            # Per-batch breakdown for the log so the user can see what
            # the batch actually did:
            #   n_cached      — taxon_scores read from DB (no BioCLIP)
            #   n_predict     — BioCLIP kingdom-rank predict() runs
            #   n_thumb_dl    — of those, how many needed a fresh thumb
            #   n_dedup       — already seen in an earlier batch on this run
            n_cached = 0
            n_predict = 0
            n_thumb_dl = 0
            n_dedup = 0
            for asset in batch:
                if asset.id in seen_ids:
                    n_dedup += 1
                    continue
                # Ensure the asset row exists before we write any
                # taxon_scores (FK on taxon_scores.asset_id). The full
                # URL fields are filled in later by run_pipeline's stitch.
                original_url = f"{base_url}/api/assets/{asset.id}/original"
                thumbnail_url = (
                    f"{base_url}/api/assets/{asset.id}/thumbnail"
                    f"?size={THUMBNAIL_SIZE}"
                )
                if storage.taxon_scores_have_asset(
                    conn, asset_id=asset.id, rank="kingdom"
                ):
                    # Cache hit: kingdom score is already stored. Skip
                    # the GET /api/assets/{id} round-trip (the per-asset
                    # capture-time + GPS live in the local `assets`
                    # row from the first run) and only refresh the
                    # `assets` row when one of the URL/filename fields
                    # has actually changed.
                    n_cached += 1
                    if not storage.asset_row_matches(
                        conn,
                        asset_id=asset.id,
                        original_file_name=asset.original_file_name,
                        original_url=original_url,
                        thumbnail_url=thumbnail_url,
                    ):
                        # Refresh name/URLs only; keep existing
                        # captured_at / lat / lng.
                        storage.upsert_asset(
                            conn,
                            asset_id=asset.id,
                            original_file_name=asset.original_file_name,
                            original_url=original_url,
                            thumbnail_url=thumbnail_url,
                        )
                    cached = storage.fetch_taxon_scores(
                        conn, asset_ids=[asset.id], rank="kingdom"
                    )
                    d = cached.get(asset.id, {})
                    plantae_by_path[str(thumb_dir / f"{asset.id}.jpg")] = {
                        "Plantae": float(d.get("Plantae") or 0.0),
                        "Fungi": float(d.get("Fungi") or 0.0),
                    }
                    continue
                # Cache miss — fetch EXIF so the freshly-inserted row
                # has the capture time + GPS (captured_at IS NOT NULL
                # check gates against re-fetching photos we already
                # know about from an earlier run).
                if not storage.asset_has_captured_at(
                    conn, asset_id=asset.id
                ):
                    exif = ImmichClient.extract_exif(
                        client.get_asset_metadata(asset.id)
                    )
                else:
                    exif = {"captured_at": None, "latitude": None, "longitude": None}
                storage.upsert_asset(
                    conn,
                    asset_id=asset.id,
                    original_file_name=asset.original_file_name,
                    original_url=original_url,
                    thumbnail_url=thumbnail_url,
                    captured_at=exif["captured_at"],
                    latitude=exif["latitude"],
                    longitude=exif["longitude"],
                )
                n_predict += 1
                p = thumb_dir / f"{asset.id}.jpg"
                if not p.exists():
                    n_thumb_dl += 1
                    thumb_bytes = client.download_bytes(
                        f"/api/assets/{asset.id}/thumbnail"
                        f"?size={THUMBNAIL_SIZE}"
                    )
                    p.write_bytes(thumb_bytes)
                to_predict_paths.append(str(p))
                to_predict_assets.append(asset)
            # Compact suffix shared by all three log branches below.
            # `n_added` falls out of the branch-local counter; the
            # breakdown is computed once up front.
            breakdown = (
                f"cached={n_cached} predict={n_predict} "
                f"(thumbs_dl={n_thumb_dl}) dedup={n_dedup}"
            )

            # Run BioCLIP kingdom-rank on the misses. Persist each row.
            if to_predict_paths:
                if clf is None:
                    # Lazy: only construct the model when we actually
                    # have BioCLIP work to do. The caller passes None
                    # so the singleton resolves on first use.
                    clf = _resolve_classifier()
                scores = _score_batch_for_kingdoms(clf, to_predict_paths)
                for asset, p in zip(to_predict_assets, to_predict_paths):
                    kingdom_scores = scores.get(p, {})
                    plantae = kingdom_scores.get("Plantae", 0.0)
                    fungi = kingdom_scores.get("Fungi", 0.0)
                    # Persist whichever kingdoms came back above 0; the full
                    # kingdom loop later will fill in the rest.
                    persist = {
                        k: v for k, v in (("Plantae", plantae), ("Fungi", fungi)) if v > 0
                    }
                    if persist:
                        storage.upsert_taxon_scores(
                            conn,
                            asset_id=asset.id,
                            rank="kingdom",
                            scores=persist,
                        )
                    plantae_by_path[p] = kingdom_scores

            # Build ordered list of (asset, max_kingdom_score) for the batch.
            # A photo counts as a candidate if EITHER kingdom exceeds the
            # threshold — so both plants AND mushrooms enter the run.
            scored: list[tuple[ImmichAsset, float]] = []
            for asset in batch:
                p = str(thumb_dir / f"{asset.id}.jpg")
                d = plantae_by_path.get(p, {})
                max_score = max(d.get("Plantae", 0.0), d.get("Fungi", 0.0))
                scored.append((asset, max_score))

            n_plant = sum(1 for _, s in scored if s > PLANT_MIN_PROB)
            majority = n_plant / len(scored) if scored else 0.0
            threshold_pct = int(PLANT_BATCH_MAJORITY * 100)

            if majority >= PLANT_BATCH_MAJORITY:
                # Continue: include plants-or-fungi in this batch.
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
                    f"{n_plant} of {len(scored)} are plants-or-fungi (>= {threshold_pct}%) "
                    f"[{breakdown}] "
                    f"-> keep going, added {n_added} (run={len(accepted)})"
                )
            elif len(scored) < PLANT_MIN_BATCH_FOR_BOUNDARY:
                # Small batch (the tail of a page or a short page) — the
                # majority ratio is too noisy to decide a boundary. A
                # single BioCLIP noise spike can flip the decision and
                # terminate the search prematurely. Skip the boundary
                # termination: include plants-or-fungi from this batch and
                # keep going. The next full-size batch will give a reliable
                # signal.
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
                    f"only {n_plant} of {len(scored)} are plants "
                    f"(< {threshold_pct}%) but small batch "
                    f"(< {PLANT_MIN_BATCH_FOR_BOUNDARY}) — too noisy, "
                    f"[{breakdown}] keep going, added {n_added} "
                    f"(run={len(accepted)})"
                )
            else:
                # Boundary batch: < PLANT_BATCH_MAJORITY candidates on a
                # full-size batch. Include EVERY asset from this batch —
                # plantae, fungi, and anything else alike — so the user
                # can review the entire batch in Immich search order,
                # with their Plantae/Fungi scores visible via the per-asset
                # badge. Then stop searching.
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
                    f"(< {threshold_pct}%) [{breakdown}] "
                    f"-> boundary, adding all {n_added} "
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


def run_knn_reclassify(
    conn,
    *,
    asset_ids: list[str],
    index_dir: Path,
    k: int = 10,
) -> dict:
    """KNN reclassify a list of assets using a pre-built FAISS index.

    Reuses the cached BioCLIP image embeddings stored in the
    `embeddings` table — does NOT re-embed photos. Skips any asset that
    already has a `knn_predictions` row for (version, k).

    Returns a stats dict {"requested", "cached", "scanned", "index_version"}.
    Logs to stderr and to the file log via the existing `_log` so the
    UI's tail picks it up.
    """
    import sys as _sys
    _self = _sys.modules[__name__]
    _knn = _self.knn

    index_dir = Path(index_dir)
    version_path = index_dir / "INDEX_VERSION.txt"
    if not version_path.exists():
        raise FileNotFoundError(
            f"INDEX_VERSION.txt missing in {index_dir} — did you build "
            f"the index with knn.build_faiss_index(...) first?"
        )
    # Read the version tag (first non-empty line).
    version_tag = next(
        (line.strip() for line in version_path.read_text().splitlines() if line.strip()),
        "unknown",
    )

    if not asset_ids:
        return {
            "requested": 0, "cached": 0, "scanned": 0,
            "index_version": version_tag,
        }

    # Filter: only assets that have an embedding AND don't already have
    # a KNN row for this (version, k). The DB-side check keeps the
    # "already classified" count accurate per call.
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"""
        SELECT e.asset_id, e.vector
        FROM embeddings e
        WHERE e.asset_id IN ({placeholders})
          AND NOT EXISTS (
            SELECT 1 FROM knn_predictions kp
            WHERE kp.asset_id = e.asset_id
              AND kp.index_version = ?
              AND kp.k = ?
          )
        """,
        [*asset_ids, version_tag, k],
    ).fetchall()

    if not rows:
        return {
            "requested": len(asset_ids), "cached": len(asset_ids),
            "scanned": 0, "index_version": version_tag,
        }

    ids = [r["asset_id"] for r in rows]
    mat = np.stack([np.frombuffer(r["vector"], dtype="<f4").copy() for r in rows])
    _log.info(
        f"[knn] classifying {len(ids)} assets  k={k}  "
        f"index={version_tag}"
    )

    results = _knn.classify_via_knn(
        query_embeddings=mat, index_dir=index_dir, k=k
    )

    for aid, r in zip(ids, results):
        storage.replace_knn_predictions(
            conn,
            asset_id=aid,
            index_version=version_tag,
            k=k,
            rows=[r],
        )
    conn.commit()

    return {
        "requested": len(asset_ids),
        "cached": len(asset_ids) - len(ids),
        "scanned": len(ids),
        "index_version": version_tag,
    }


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
    stop_event=None,
) -> RunResult:
    """Run a smart-search on Immich for `query`, embed + predict any assets
    not yet cached, recompute clusters, run YOLO-World on uncached images,
    persist everything, and return the ordered result list for this run.

    Heavy passes run in chunks of `batch_size` so peak memory stays small.

    Each cache check below is explicit: if a row exists for the (asset,
    rank) pair, we skip the work. Add new caches here so a freshly-cached
    asset still gets the new feature on the next run.

    `stop_event` is an optional `multiprocessing.Event` set by the parent
    process when running this in a child (see `_pipeline_child_entry` in
    app.py). The threading-module `_stop_event` is always checked too,
    so /api/run/stop works whether the worker is in-thread or
    in-subprocess.
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
    set_state(query=query, n_requested=n_samples)
    _stage_begin("search")

    client = ImmichClient(base_url=base_url, api_key=api_key)
    try:
        with storage.connect(db_path) as conn:
            # Lazy classifier: only construct the TreeOfLifeClassifier
            # (and load BioCLIP weights + open_clip helpers) the first
            # time the kingdom stage has real BioCLIP work to do. The
            # BioCLIP model on CPU is ~5.5 GB resident — loading it for
            # an entirely-cached run wastes that until the process dies.
            # Both downstream stages gate on `if needs_bioclip:` /
            # `if to_predict_paths:`, so the singleton is only created
            # when inference is genuinely about to run.
            clf = None

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
                stop_event=stop_event,
            )
            _stage_end("search")

            needs_bioclip: list[tuple[ImmichAsset, Path | None]] = []
            needs_yolo: list[tuple[ImmichAsset, Path | None]] = []
            ordered_ids: list[str] = []

            for asset in assets:
                original_url = f"{base_url}/api/assets/{asset.id}/original"
                thumbnail_url = (
                    f"{base_url}/api/assets/{asset.id}/thumbnail"
                    f"?size={THUMBNAIL_SIZE}"
                )
                # Pull EXIF in the same family as the chunk loop
                # above — same cache, same parser. Existing rows keep
                # their previous EXIF via the ON CONFLICT clause in
                # upsert_asset().
                exif = ImmichClient.extract_exif(
                    client.get_asset_metadata(asset.id)
                )
                storage.upsert_asset(
                    conn,
                    asset_id=asset.id,
                    original_file_name=asset.original_file_name,
                    original_url=original_url,
                    thumbnail_url=thumbnail_url,
                    captured_at=exif["captured_at"],
                    latitude=exif["latitude"],
                    longitude=exif["longitude"],
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
                if clf is None:
                    clf = classifier if classifier is not None else _resolve_classifier()
                _stage_begin("species", chunk_total=species_total_chunks)
                species_t0 = time.monotonic()
                _log.info(
                    f"[species] starting  {len(needs_bioclip)} assets, "
                    f"{species_total_chunks} chunks of {batch_size}"
                )
                i = 0
                while i < len(needs_bioclip):
                    # Stop check at chunk boundary. Each chunk commits
                    # its embeddings + predictions before the next, so
                    # stopping here loses nothing.
                    check_stop(stop_event)
                    if species_total_chunks > 0:
                        set_chunk_progress(i // batch_size + 1, species_total_chunks)
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
                        _close_image(im)
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
                _stage_end("species")

            # ---- Clustering: incremental assign + conditional refit ----
            # Strategy B: when a trained clusterer is persisted, do the
            # fast path (approximate_predict on newly-embedded vectors);
            # only when the corpus has grown by GROWTH_THRESHOLD_PCT since
            # the last fit do we pay the full refit + Hungarian remap.
            # cluster_names / _results_cache / near-dup caches are
            # invalidated by `_run_and_store` after this function returns,
            # so we don't bother here.
            _stage_begin("cluster")
            cluster_t0 = time.monotonic()
            newly_embedded_ids: list[str] = [a.id for a, _p in needs_bioclip]
            asset_to_cluster: dict[str, int] = {}
            n_clusters = 0
            refit_done = False

            clusterer_loaded = _clusters.load_clusterer(conn)
            if clusterer_loaded is not None and newly_embedded_ids:
                try:
                    new_vectors = _load_vectors(conn, newly_embedded_ids)
                    assign = _clusters.assign_new_points(
                        conn,
                        new_asset_ids=newly_embedded_ids,
                        new_vectors=new_vectors,
                    )
                    if assign is not None:
                        asset_to_cluster = assign
                except Exception as e:
                    _log.warning(
                        f"[cluster] incremental assign failed: {type(e).__name__}: {e}; "
                        "falling back to refit"
                    )
                    clusterer_loaded = None

            needs_refit = (
                clusterer_loaded is None
                or _clusters.should_refit(
                    conn, growth_threshold_pct=GROWTH_THRESHOLD_PCT
                )
            )

            if needs_refit:
                _log.info(
                    f"[cluster] full refit "
                    f"(growth > {GROWTH_THRESHOLD_PCT:.0%} since last fit "
                    f"or no prior clusterer)"
                )
                asset_to_cluster, n_clusters, _remap = _clusters.refit_with_remap(
                    conn, growth_threshold_pct=GROWTH_THRESHOLD_PCT
                )
                refit_done = True
            elif asset_to_cluster:
                # Incremental path produced new assignments only; the
                # full mapping (old + new) already lives in assets.cluster_id.
                # Compute n_clusters from the DB rather than scanning just
                # the new points.
                row = conn.execute(
                    "SELECT COUNT(DISTINCT cluster_id) AS n "
                    "FROM assets WHERE cluster_id IS NOT NULL AND cluster_id != -1"
                ).fetchone()
                n_clusters = int(row["n"]) if row else 0
                _log.info(
                    f"[cluster] incremental assign: {len(asset_to_cluster)} new points, "
                    f"no refit needed"
                )
            else:
                # Edge case: no new embeddings AND no refit. Pull n_clusters
                # from the DB so the runs.n_clusters column stays sane.
                row = conn.execute(
                    "SELECT COUNT(DISTINCT cluster_id) AS n "
                    "FROM assets WHERE cluster_id IS NOT NULL AND cluster_id != -1"
                ).fetchone()
                n_clusters = int(row["n"]) if row else 0
                _log.info(
                    f"[cluster] no-op: no new points, growth under threshold"
                )

            # set_cluster_ids is a no-op when asset_to_cluster is empty
            # (e.g. cache-hit re-run with no new embeddings). It also
            # wouldn't hurt on the refit path — refit_with_remap already
            # wrote the full mapping — but skipping it saves a write.
            if asset_to_cluster and not refit_done:
                storage.set_cluster_ids(conn, asset_to_cluster=asset_to_cluster)

            _log.info(
                f"[cluster] done  n_clusters={n_clusters}  "
                f"refit={refit_done}  "
                f"elapsed={_fmt_eta(time.monotonic() - cluster_t0)}"
            )
            _stage_end("cluster")

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
            _stage_begin("kingdom", chunk_total=kingdom_total_chunks)
            kingdom_t0 = time.monotonic()
            if kingdom_assets:
                if clf is None:
                    clf = classifier if classifier is not None else _resolve_classifier()
                _log.info(
                    f"[kingdom] starting  {len(kingdom_assets)} assets, "
                    f"{kingdom_total_chunks} chunks of {batch_size}"
                )
            else:
                _log.info("[kingdom] no work — all assets already cached")

            i = 0
            while i < len(kingdom_assets):
                check_stop(stop_event)
                j = min(i + batch_size, len(kingdom_assets))
                if kingdom_total_chunks > 0:
                    set_chunk_progress(i // batch_size + 1, kingdom_total_chunks)
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
                    _close_image(im)
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
            _stage_end("kingdom")
            del kingdom_assets, kingdom_paths

            # ---- YOLO-World: detect on uncached assets (chunked) ----
            # Detection rows are persisted as we go; the bbox BioCLIP
            # pass below queries the detections table directly so it
            # handles cache-hit assets without needing tuples handed
            # off from here.
            yolo_total_chunks = (len(needs_yolo) + batch_size - 1) // batch_size
            yolo_t0 = time.monotonic()
            _stage_begin("yolo", chunk_total=yolo_total_chunks)
            if needs_yolo:
                _log.info(
                    f"[yolo] starting  {len(needs_yolo)} assets, "
                    f"{yolo_total_chunks} chunks of {batch_size}"
                )
                yolo_model_loaded = False
                try:
                    i = 0
                    while i < len(needs_yolo):
                        if yolo_total_chunks > 0:
                            set_chunk_progress(i // batch_size + 1, yolo_total_chunks)
                        check_stop(stop_event)
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
                            # Strip person/face/human boxes — those are
                            # the person stage's responsibility (it
                            # uses append_detections). Letting them
                            # through here would (a) duplicate every
                            # person box when an asset flows through
                            # both stages in the same run and (b) route
                            # person crops through BioCLIP-reclassify
                            # (wrong — BioCLIP is plant-focused).
                            plant_dets = [
                                d for d in dets
                                if d["label"] not in PERSON_LABELS
                            ]
                            new_ids = storage.replace_detections(
                                conn, asset_id=asset.id, detections=plant_dets
                            )
                            n_dets += len(new_ids)
                        for im in pil_images:
                            _close_image(im)
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
            _stage_end("yolo")

            # ---- Person detection: scan all assets (one-time backfill) ----
            # Unlike the yolo stage above (which only iterates plantae
            # candidates from `discover_plantae_assets`), this stage
            # scans EVERY asset in the DB where `has_person IS NULL`.
            # On the first run after this change that's ~19,300 assets.
            # Subsequent runs only touch newly-ingested assets. Uses
            # `append_detections` (NOT `replace_detections`) so existing
            # plant detections and their `bbox_predictions` FK CASCADE
            # children stay cached — only person/face/human boxes are
            # added. Gate: `WHERE has_person IS NULL`.
            _stage_begin("person")
            try:
                n_total = 0
                page_num = 0
                # Outer page loop — 500 unscanned assets per page keeps
                # memory low and progress visible. Uses cursor pagination
                # (matching get_assets_without_person_scan's signature) so
                # a SIGKILL between pages can't cause the next run to
                # re-scan the same asset — see storage.py for why we
                # abandoned OFFSET for this query.
                cursor_ts: str | None = None
                cursor_id: str | None = None
                outer_page = 500
                while True:
                    check_stop(stop_event)
                    todo = storage.get_assets_without_person_scan(
                        conn, limit=outer_page,
                        after_first_seen_at=cursor_ts,
                        after_asset_id=cursor_id,
                    )
                    if not todo:
                        break
                    page_num += 1
                    # Inner batch loop — chunked for the YOLO forward
                    # pass (matches batch_size used elsewhere).
                    todo_paths: list[tuple[str, Path, str]] = []
                    for asset_id, _fn, _ts in todo:
                        p = thumb_dir / f"{asset_id}.jpg"
                        if not p.exists():
                            thumb_bytes = client.download_bytes(
                                f"/api/assets/{asset_id}/thumbnail"
                                f"?size={THUMBNAIL_SIZE}"
                            )
                            p.write_bytes(thumb_bytes)
                        todo_paths.append((asset_id, p, _ts))
                    # Process in chunks of `batch_size`. YOLO-World's
                    # first forward pass here rebakes the 17-label
                    # embedding table under the singleton's _LOCK; the
                    # previous yolo stage's finally already unloaded so
                    # we get the fresh 17-class init.
                    person_t0 = time.monotonic()
                    for i in range(0, len(todo_paths), batch_size):
                        check_stop(stop_event)
                        chunk = todo_paths[i : i + batch_size]
                        pil_images = [Image.open(p).convert("RGB") for _, p, _ in chunk]
                        detections = _yolo.detect(pil_images, conf=yolo_conf)
                        for (asset_id, _, _ts), dets in zip(chunk, detections):
                            # Class-agnostic IoU merge — same as yolo
                            # stage. A photo with overlapping person +
                            # face boxes collapses to one detection.
                            dets = _merge_overlapping_boxes(dets)
                            # Keep ONLY person/face/human/hand boxes —
                            # the yolo stage already persisted the plant
                            # boxes for any plantae candidate that
                            # flowed through it. If we appended plant
                            # boxes here too, plantae candidates would
                            # end up with duplicate plant rows in the
                            # detections table. Hand is kept because
                            # has_real_person() cross-checks it to
                            # suppress hand-silhouette false positives.
                            person_dets = [
                                d for d in dets
                                if d["label"] in PERSON_LABELS
                                and d["score"] >= PERSON_MIN_CONF
                            ]
                            hand_dets = [
                                d for d in dets
                                if d["label"] == HAND_LABEL
                                and d["score"] >= HAND_MIN_CONF
                            ]
                            if person_dets or hand_dets:
                                # Append (no delete) — preserves any
                                # existing plant detections and keeps
                                # their bbox_predictions cache intact
                                # via FK CASCADE.
                                storage.append_detections(
                                    conn, asset_id=asset_id,
                                    detections=person_dets + hand_dets,
                                )
                            # has_real_person applies the hand-suppression
                            # rule — see storage.has_real_person().
                            storage.mark_person_scanned(
                                conn,
                                asset_id=asset_id,
                                has_person=has_real_person(dets),
                            )
                        for im in pil_images:
                            _close_image(im)
                        del pil_images
                        gc.collect()
                    n_total += len(todo_paths)
                    elapsed = time.monotonic() - person_t0
                    _log.info(
                        f"[person] page {page_num} "
                        f"items={len(todo_paths)}  scanned_so_far={n_total}  "
                        f"elapsed={_fmt_eta(elapsed)}"
                    )
                    # Advance cursor to the last row of this page so a
                    # SIGKILL/OOM between pages resumes where we left off.
                    cursor_ts = todo_paths[-1][2]
                    cursor_id = todo_paths[-1][0]
                    # Periodic commit so a 70-min backfill flushes
                    # progress to disk and survives a SIGKILL/OOM
                    # between pages. The pipeline's outer commit at
                    # line 1688 only fires once per full run.
                    conn.commit()
                # Final summary log — only emit when we actually scanned.
                n_remaining = storage.count_assets_without_person_scan(conn)
                if n_total or n_remaining == 0:
                    _log.info(
                        f"[person] done  scanned {n_total} assets this run, "
                        f"{n_remaining} still unscanned total"
                    )
            finally:
                _stage_end("person")

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
                    # Skip person/face/human boxes — BioCLIP is
                    # plant-focused, so re-classifying a person crop
                    # would produce a meaningless Plantae score. Plant
                    # crops go through BioCLIP as before.
                    if d["label"] in PERSON_LABELS:
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
            _stage_begin("bbox-bioclip", chunk_total=bbox_total_chunks)
            if all_bbox_targets:
                if clf is None:
                    clf = classifier if classifier is not None else _resolve_classifier()
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
                    check_stop(stop_event)
                    if bbox_total_chunks > 0:
                        set_chunk_progress(i // batch_size + 1, bbox_total_chunks)
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
                            _close_image(im)

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
                        _close_image(im)
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
            _stage_end("bbox-bioclip")
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

            # Per-stage timing summary for the log + UI timing bar.
            times = get_state().get("stage_times", {})
            total = sum(times.values()) or 0.0
            parts = "  ".join(
                f"{name}={_fmt_eta(t)}" for name, t in times.items()
            )
            _log.info(
                f"[run] done  run={run_id}  total={_fmt_eta(total)}  "
                f"{parts}"
            )

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


def run_person_backfill(
    *,
    base_url: str,
    api_key: str,
    db_path: str | Path,
    yolo_conf: float = 0.05,
    batch_size: int = DEFAULT_BATCH_SIZE,
    progress_cb=None,
    status_path: Path | None = None,
    stop_event=None,
) -> dict:
    """Standalone person-detection backfill — runs ONLY the person
    stage, no BioCLIP, no search, no species classification, no
    clustering. Used by the 'Detect people' LegendBar button for the
    one-time sweep over the ~19k existing assets where `has_person IS
    NULL`.

    Same chunked pattern as the in-pipeline person stage but lifted
    out so it can be triggered without re-running the slow BioCLIP
    pipeline. Uses cursor pagination (`after_first_seen_at` +
    `after_asset_id`) so re-runs after a crash or stop always resume
    cleanly — no rows skipped due to offset drift. Runs YOLO with the
    17-label model, persists person-only boxes via `append_detections`,
    sets `has_person` accordingly.

    Args:
      progress_cb: optional callable invoked with `(scanned_count,
        has_person_count, no_person_count)` after each outer page so
        the UI can show live progress without re-querying the DB.
      status_path: optional Path. When set, the worker writes a small
        JSON snapshot after each outer page so the parent process can
        poll progress without sharing memory. Used by the
        multiprocessing-based app worker — JSON file beats
        multiprocessing.Manager().dict() for our access pattern
        (parent reads every 400ms, child writes every ~30s).
      stop_event: optional `multiprocessing.Event`. When set, the
        worker checks `stop_event.is_set()` between batches in
        addition to the module-level `_stop_event`. Lets the parent
        cancel a running child without sending signals.

    Returns a small dict the caller can serialize to the job snapshot.
    """
    from immich_client import ImmichClient  # local import (heavy? cheap; here for clarity)

    # yolo is a PEP 562 lazy-loaded attribute on this module — access
    # it via `sys.modules[__name__]` to trigger the lazy loader. Same
    # pattern as run_pipeline() at the top.
    import sys as _sys
    _self = _sys.modules[__name__]
    _yolo = _self.yolo

    db_path = Path(db_path)
    thumb_dir = db_path.parent / "thumbs"
    thumb_dir.mkdir(parents=True, exist_ok=True)

    client = ImmichClient(base_url=base_url, api_key=api_key)
    n_total_scanned = 0
    n_with_person = 0
    n_without_person = 0

    def _stop_requested() -> bool:
        """Stop check used by both threaded and multiprocess callers.
        `stop_event` is the multiprocessing.Event passed by the parent
        (None for the in-pipeline thread, which uses the module-level
        `_stop_event` via check_stop()).
        """
        if stop_event is not None and stop_event.is_set():
            return True
        try:
            check_stop()
            return False
        except _PipelineStop:
            return True

    def _write_status(stage: str) -> None:
        """Write a small JSON snapshot so the parent can poll progress
        without holding a Manager proxy. Atomic via tmp+rename."""
        if status_path is None:
            return
        import json as _json
        payload = {
            "stage": stage,
            "scanned_count": n_total_scanned,
            "with_person_count": n_with_person,
            "without_person_count": n_without_person,
        }
        tmp = status_path.with_suffix(status_path.suffix + ".tmp")
        tmp.write_text(_json.dumps(payload))
        tmp.replace(status_path)

    try:
        with storage.connect(db_path) as conn:
            # Cursor pagination: track the last (first_seen_at,
            # asset_id) we processed; the next query skips past it.
            # Resilient to interruption — re-runs always resume
            # without skipping or duplicating.
            cursor_ts: str | None = None
            cursor_id: str | None = None
            outer_page = 500
            while True:
                if _stop_requested():
                    break
                todo = storage.get_assets_without_person_scan(
                    conn, limit=outer_page,
                    after_first_seen_at=cursor_ts,
                    after_asset_id=cursor_id,
                )
                if not todo:
                    break
                # Resolve thumbs (download if missing).
                todo_paths: list[tuple[str, Path, str]] = []
                for asset_id, _fn, _ts in todo:
                    p = thumb_dir / f"{asset_id}.jpg"
                    if not p.exists():
                        thumb_bytes = client.download_bytes(
                            f"/api/assets/{asset_id}/thumbnail"
                            f"?size={THUMBNAIL_SIZE}"
                        )
                        p.write_bytes(thumb_bytes)
                    todo_paths.append((asset_id, p, _ts))
                # Chunked YOLO forward pass.
                for i in range(0, len(todo_paths), batch_size):
                    if _stop_requested():
                        break
                    chunk = todo_paths[i : i + batch_size]
                    pil_images = [Image.open(p).convert("RGB") for _, p, _ in chunk]
                    detections = _yolo.detect(pil_images, conf=yolo_conf)
                    for (asset_id, _, _ts), dets in zip(chunk, detections):
                        dets = _merge_overlapping_boxes(dets)
                        person_dets = [
                            d for d in dets
                            if d["label"] in PERSON_LABELS
                            and d["score"] >= PERSON_MIN_CONF
                        ]
                        hand_dets = [
                            d for d in dets
                            if d["label"] == HAND_LABEL
                            and d["score"] >= HAND_MIN_CONF
                        ]
                        if person_dets or hand_dets:
                            storage.append_detections(
                                conn, asset_id=asset_id,
                                detections=person_dets + hand_dets,
                            )
                        # has_real_person applies the hand-suppression
                        # rule — see storage.has_real_person().
                        has_p = has_real_person(dets)
                        storage.mark_person_scanned(
                            conn, asset_id=asset_id, has_person=has_p
                        )
                        n_total_scanned += 1
                        if has_p:
                            n_with_person += 1
                        else:
                            n_without_person += 1
                    for im in pil_images:
                        _close_image(im)
                    del pil_images
                    gc.collect()
                if progress_cb is not None:
                    progress_cb(
                        n_total_scanned, n_with_person, n_without_person
                    )
                # Periodic commit so a 70-min backfill flushes
                # progress to disk and survives an OOM kill / SIGKILL
                # between outer pages. Also makes has_person visible
                # to other SQLite connections (the API + UI) without
                # waiting for the full run to finish.
                conn.commit()
                # Advance cursor to the last row of this page.
                cursor_ts = todo_paths[-1][2]
                cursor_id = todo_paths[-1][0]
                _write_status("running")
    finally:
        try:
            _yolo.unload_model()
        except Exception:
            # Teardown in a finally block — must NOT re-raise, or it
            # would mask the primary exception. Log so a stuck teardown
            # doesn't disappear silently.
            _log.warning("yolo unload failed", exc_info=True)
        _write_status("finished")
    return {
        "scanned_count": n_total_scanned,
        "with_person_count": n_with_person,
        "without_person_count": n_without_person,
    }


def _run_pipeline_safe(**kwargs) -> RunResult | None:
    """Wrapper that converts pipeline exceptions into error_state() so the
    UI can surface them instead of just hanging the request."""
    try:
        return run_pipeline(**kwargs)
    except _PipelineStop:
        # User clicked Stop. The check_stop() raise was already at a
        # batch boundary, so the previous chunk's writes are committed.
        # We just need to mark the state, log it, and return so the
        # wrapping request releases the pipeline lock.
        _log.warning("pipeline stopped by user")
        stopped_state("stopped by user")
        return None
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
        except Exception as je:
            # Could be HTML, plain text, or truncated JSON — log so
            # the raw shape is recoverable when debugging "Immich said
            # something the user can't see" reports.
            _log.debug("immich smart-search body not JSON: %s", je)
            body = None
        msg = _flatten(body).strip()
        if not msg:
            try:
                msg = (e.response.text or "").strip()
            except Exception as te:
                _log.debug("immich smart-search body unreadable: %s", te)
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