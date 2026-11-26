"""Shared pipeline: Immich smart-search → BioCLIP embed+predict →
DB → k-means cluster → detector (RTMDet by default; see YOLO_BACKEND).

Memory note: BioCLIP-2 (~4.3 GB of artifacts: weights + species label
table) and the detector ONNX (~20 MB) are both resident during the first run
on uncached assets; peak RSS ~6 GB in the pipeline child — see
docs/BIOCLIP_MEMORY.md. To keep peak RAM
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
import shutil
import sqlite3
import concurrent.futures
import io as _io
import tempfile
import time
from collections import defaultdict
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image
import threading

# Heavy imports (torch/bioclip/sklearn-via-clusters/onnxruntime-via-yolo)
# are deferred to run_pipeline() so that `import pipeline` stays cheap and
# the FastAPI browse-only server doesn't pay ~3 GB of model weights at
# startup. See run_pipeline() for the deferred-import block.

import storage
from storage import PERSON_LABELS, PERSON_MIN_CONF, HAND_LABEL, HAND_MIN_CONF, has_real_person
from immich_client import ImmichAsset, ImmichClient

# PEP 562 module-level __getattr__ — lazy-loads the heavy bits
# (torch/bioclip/sklearn-via-clusters/onnxruntime-via-yolo) the first
# time anything looks them up by name. Keeps `import pipeline` cheap
# for the browse-only FastAPI server, while still allowing
# `pipeline.TreeOfLifeClassifier`, `pipeline.clusters`, `pipeline.yolo`,
# `pipeline.Rank` to work for both test patches (mock.patch.object) and
# the deferred body of run_pipeline().
_LAZY_ATTRS = ("TreeOfLifeClassifier", "Rank", "clusters", "yolo", "knn", "face")


def _yolo_backend_module() -> str:
    """Detection backend chosen once per process: 'rtmdet' (default) is
    the Apache-2.0 RTMDet export shipped in models/; 'obj6' is the
    opt-in Ultralytics YOLO26 export (AGPL-3.0 weights, bring your own
    ONNX). Both are served through onnxruntime and expose the same
    get_model()/unload_model()/detect() interface. 'gemini1' is the
    retired pre-rename id for 'obj6' and is still accepted so old
    deployments keep working. The YOLO-World backend ('world') was
    removed with the ultralytics dependency."""
    backend = (os.environ.get("YOLO_BACKEND") or "rtmdet").strip().lower()
    if backend in ("obj6", "gemini1"):
        return "yolo_obj6"
    if backend == "rtmdet":
        return "yolo_rtmdet"
    raise ValueError(
        f"unknown YOLO_BACKEND={backend!r} — use 'rtmdet' (default) or "
        f"'obj6'; 'world' (YOLO-World) was removed with the ultralytics "
        f"dependency (see docs/detector_backends.md)"
    )


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
        import importlib

        module = importlib.import_module(_yolo_backend_module())
        globals()["yolo"] = module
        return module
    if name == "face":
        import face as _face
        globals()["face"] = _face
        return _face
    if name == "knn":
        import knn_io as _knn
        globals()["knn"] = _knn
        return _knn
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


THUMBNAIL_SIZE = "preview"
DEFAULT_BATCH_SIZE = 16

# Person-flag classifier veto (2026-09-10). The box heuristics + YuNet
# rescue decide "is there person evidence", but YOLO-World reads
# hand-holding-X photos (ubiquitous in a plant library) as a big
# `person` box and never emits `hand`. The BioCLIP whole-image
# classifier separates exactly that class: in the review's disagreement
# sample, every hand-holding-X photo scored p < 0.05 (12/12) while every
# visually-confirmed real person scored p >= 0.055 (all 19 YuNet-rescued
# rows >= 0.11). So a confident low p vetoes the flag; a missing
# embedding or missing model file disables the veto (falls back to the
# box stack, which is right on the no-embedding population too).
PERSON_VETO_P = 0.05
_PERSON_CLF: dict | None = None  # cached {"coef", "intercept"}; {} = unavailable


def _load_person_clf() -> dict:
    """Lazy-load the BioCLIP person-flag logistic regression (trained by
    scripts/person_classifier_retrain.py, artifact
    data/person_classifier.joblib — gitignored; a fresh checkout without
    it simply runs veto-disabled)."""
    global _PERSON_CLF
    if _PERSON_CLF is None:
        path = Path(os.environ.get("PERSON_CLF_PATH",
                                   "data/person_classifier.joblib"))
        try:
            import joblib

            blob = joblib.load(path)
            _PERSON_CLF = {
                "coef": np.asarray(blob["coef"], dtype=np.float32).ravel(),
                "intercept": float(blob["intercept"]),
            }
            _log.info("[person] classifier veto loaded from %s", path)
        except Exception as e:
            _log.info("[person] classifier veto unavailable (%s: %s) — "
                      "disabled", type(e).__name__, e)
            _PERSON_CLF = {}
    return _PERSON_CLF


def _person_clf_p(conn: sqlite3.Connection, asset_id: str) -> float | None:
    """Classifier p for one asset, computed from its PERSISTED embedding
    (encode-once: the species stage stored it earlier in the run — no
    image decode or re-encode here). None when the asset has no
    embedding or the model is unavailable."""
    clf = _load_person_clf()
    if not clf:
        return None
    row = conn.execute(
        "SELECT vector FROM embeddings WHERE asset_id=?", (asset_id,)
    ).fetchone()
    if row is None or row[0] is None:
        return None
    vec = np.frombuffer(row[0], dtype="<f4")
    return float(1.0 / (1.0 + np.exp(-(vec @ clf["coef"] + clf["intercept"]))))


def apply_person_veto(has_person: bool, clf_p: float | None) -> bool:
    """The veto rule: a confident classifier negative beats person-box
    evidence. `clf_p is None` (no embedding / no model) never vetoes."""
    if has_person and clf_p is not None and clf_p < PERSON_VETO_P:
        return False
    return has_person
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
    """Return the cached classifier for the selected ENGINE, instantiating
    on first call. Returns the same instance for the lifetime of the
    process so model weights stay loaded and the second-onwards n_new>0
    runs don't pay the load cost again.

    ENGINE=teacher (default) → TreeOfLifeClassifier (BioCLIP-2, ~6 GB).
    ENGINE=mobile-* → MobileEngine (BioCLIP-Mobile ONNX package + label
    table, ~1.2 GB; see mobile_engine.py). ENGINE=coreml / auto → on
    macOS with the Core ML package, CoreMLEngine (the teacher with its
    image tower on the ANE/GPU; see coreml_engine.py), else the teacher.
    All satisfy the same duck-type surface, so every stage below is
    engine-agnostic."""
    global _clf_cache
    if _clf_cache is not None:
        return _clf_cache
    with _clf_lock:
        if _clf_cache is None:
            from coreml_engine import select_engine
            from mobile_engine import resolve_engine_name

            if select_engine(resolve_engine_name()) == "coreml":
                engine = _load_coreml_engine()
                if engine is not None:
                    _clf_cache = engine
                    return _clf_cache
            elif resolve_engine_name().startswith("mobile"):
                from mobile_engine import MobileEngine

                _t0 = time.monotonic()
                engine = MobileEngine()
                _log.info(
                    "[model] ENGINE=%s  package=%s  version=%s  "
                    "(loaded in %.1fs; session builds on first use)",
                    resolve_engine_name(),
                    engine.package_dir,
                    engine.version_key(),
                    time.monotonic() - _t0,
                )
                _clf_cache = engine
                return _clf_cache
            _TreeOfLifeClassifier = globals()["TreeOfLifeClassifier"]
            # The load (and on first use ever, the ~4.3 GB download)
            # takes minutes with no callback of our own — say so, or the
            # log tail looks hung mid-run. HF's own tqdm bar only goes
            # to the child's stderr (docker logs), never this file.
            _log.info(
                "[model] loading BioCLIP-2 (first use downloads ~4.3 GB "
                "of weights + label table; progress bar in docker logs)"
            )
            # Pin the intra-op pool explicitly. torch's default is
            # already the physical-core count, but the run child is
            # forked from uvicorn and a run once crawled at
            # ~2 s/img (~1.5 threads' worth) for no reason the
            # environment showed — an explicit set here makes the
            # intended parallelism independent of anything inherited.
            # Measured on the 16-thread dev box (encode 32 previews):
            # 1T 1.58 s/img · 2T 0.86 · 4T 0.60 · 8T 0.33 · 16T 0.36
            # — 8 (physical cores) is the knee; hyperthreads regress.
            import torch

            torch.set_num_threads(min(8, os.cpu_count() or 8))
            _t0 = time.monotonic()
            _clf_cache = _TreeOfLifeClassifier()
            _log.info(
                f"[model] BioCLIP-2 ready in {time.monotonic() - _t0:.0f}s "
                f"(torch threads={torch.get_num_threads()})"
            )
    return _clf_cache


def _load_coreml_engine() -> "object | None":
    """CoreMLEngine if it loads and passes its golden gate, else None
    (caller falls back to the torch teacher). Never raises."""
    try:
        from coreml_engine import CoreMLEngine

        _t0 = time.monotonic()
        engine = CoreMLEngine()
        ok, mean, low = engine.self_check()
    except Exception as e:  # noqa: BLE001 — any load failure → teacher
        _log.info("[model] Core ML engine failed to load (%s) — using BioCLIP-2 torch", e)
        return None
    if not ok:
        _log.info(
            "[model] Core ML golden gate FAILED (cos mean %.6f / min %.6f) "
            "— using BioCLIP-2 torch", mean, low,
        )
        return None
    _log.info(
        "[model] ENGINE=coreml  version=%s  golden cos %.6f/%.6f  (ready in %.1fs)",
        engine.version_key(), mean, low, time.monotonic() - _t0,
    )
    return engine


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
    Returns True if the pipeline looked like it was running.

    Deliberately lock-free: this runs inside the SIGTERM handler, on the
    main thread while it is paused mid-bytecode. If it took
    _state_lock (non-reentrant) and the main thread was inside any
    `with _state_lock:` block — every chunk boundary is — the handler
    would deadlock against the thread that can never release the lock,
    and only the parent's 45s SIGKILL watchdog would recover. Event.set
    is a plain flag flip, safe from a signal handler. An early stop
    (before the run marks itself "running") is honored by the worker,
    which checks stop_requested() right before run_pipeline.
    """
    was_set = _stop_event.is_set()
    _stop_event.set()
    return was_set or _state.get("status") == "running"


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
    # Visible pipeline log line so the run timeline reads top-to-bottom
    # (search → species → cluster → kingdom → yolo → person → bbox-bioclip)
    # even when a stage has no work to do. The caller still emits a
    # detailed "starting N assets" line below when there IS work.
    _log.info(f"[stage] {name}")


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
    # Visible summary line so the log timeline ends with what the
    # run actually accomplished, not just "done".
    _log.info(
        f"[run] done  run={run_id}  n_new={n_new}  n_cached={n_cached}  "
        f"n_requested={n_requested}"
    )


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
    items_done: int | None = None,
    note: str = "",
) -> None:
    """Log a chunk progress line with ETA. Safe to call on every chunk
    (called once per chunk). chunk_idx is 1-based for human readability.

    `items_done` is the cumulative number of items processed in the
    stage so far; the rate is items_done/elapsed. The old code divided
    the CURRENT chunk's n_items by the CUMULATIVE elapsed, which decayed
    to 0.0/s by mid-stage (16 items / 8000 s) even though throughput was
    steady. Callers pass the stage's running total (e.g. `j`, the chunk
    loop cursor); when omitted we fall back to chunk_idx * n_items."""
    if total_chunks <= 0:
        return
    done = chunk_idx
    per_chunk = elapsed / done if done > 0 else 0.0
    remaining = max(0, total_chunks - done) * per_chunk
    cumulative_items = (
        items_done if items_done is not None else done * max(1, n_items)
    )
    rate = cumulative_items / elapsed if elapsed > 0 else 0.0
    tail = f"  {note}" if note else ""
    _log.info(
        f"[{stage}] chunk {done}/{total_chunks}  items={n_items}  "
        f"rate={rate:.1f}/s  elapsed={_fmt_eta(elapsed)}  "
        f"eta={_fmt_eta(remaining)}{tail}"
    )
    # Mirror into the shared state so the UI can show the same numbers.
    set_state(
        stage=stage,
        chunk_idx=chunk_idx,
        chunk_total=total_chunks,
        rate=rate,
        eta_seconds=remaining,
    )
BBOX_BIOCLIP_MIN = 0.15
# The Linux pipeline can re-classify each crop, but it only writes
# bbox_predictions — it does NOT write the bbox_embeddings used by the
# Similar panel. On the teacher model that pass is also the heaviest
# stage in the run (tens of thousands of crops, ~11 GB peak); the Mac
# CoreML worker computes both predictions and embeddings far faster.
# Set BBOX_BIOCLIP_ENABLED=0 on teacher instances that use the Mac box
# backfill, so a normal run cannot OOM here.
BBOX_BIOCLIP_ENABLED = os.environ.get(
    "BBOX_BIOCLIP_ENABLED", "1"
).strip().lower() not in {"0", "false", "no", "off"}
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


def _classify_features(
    clf,
    feats,
    keys: list[str],
    *,
    rank_enum,
    k: int,
    min_prob: float = 1e-9,
) -> list[dict]:
    """Run BioCLIP's text classification head on PRECOMPUTED image
    features and return the same list-of-dicts shape clf.predict()
    returns (rows tagged with `keys` as file_name).

    This is numerically clf.predict()'s classification step — softmax
    over logit_scale * feats @ txt_embeddings, then top-k / rank-grouped
    formatting — minus the image encoding. clf.predict() re-opens and
    re-encodes every image internally (a full ViT-L/14 forward pass on
    CPU), so any caller that already encoded images — e.g. to persist
    the embedding, or because it loaded a stored embedding from the DB —
    must go through here instead of calling predict() again, or it pays
    for the vision tower twice per chunk.
    """
    probs = clf.create_probabilities(feats, clf.get_txt_embeddings()).detach().cpu()
    is_species = getattr(rank_enum, "name", str(rank_enum)).upper() == "SPECIES"
    out: list[dict] = []
    for i, key in enumerate(keys):
        if is_species:
            out.extend(clf.format_species_probs(key, probs[i], k))
        else:
            out.extend(clf.format_grouped_probs(key, probs[i], rank_enum, min_prob, k))
    return out


_PIL_DRAFT_SIZE = 448  # ~2x the model's 224px input — decode-cheap


def _yolo_decode_size() -> int:
    """Preview decode cap for the detection stages. The RTMDet / obj6
    exports letterbox to 960 and were trained/gated at that size —
    upsampling a 448 BioCLIP draft decode would blur small boxes away.
    Each backend module declares its own cap; the fallback is the
    BioCLIP draft size."""
    import importlib

    mod = importlib.import_module(_yolo_backend_module())
    return int(getattr(mod, "DECODE_PREVIEW_MAX_SIZE", _PIL_DRAFT_SIZE))


def _decode_preview(blob: bytes, max_size: int = _PIL_DRAFT_SIZE) -> Image.Image:
    """Decode a downloaded preview for BioCLIP — in memory only; raw
    photos are never written to disk.

    The model squashes everything to 224×224, but the naive
    `Image.open(...).convert("RGB")` decodes the FULL preview (1440px+ —
    occasionally a multi-MP original Immich serves in its place) and
    the transforms then ToTensor that whole frame before Resize squashes
    it. That was ~2 s of single-threaded decode/resize per photo — the
    search stage's real bottleneck, invisible in every thread/MKL
    profile. `draft` downscales during JPEG decode; `thumbnail` caps
    whatever survives (WebP ignores draft).

    `max_size` lets the detection stages request a larger decode (see
    _yolo_decode_size) — BioCLIP callers keep the cheap default.
    """
    img = Image.open(_io.BytesIO(blob))
    try:
        img.draft("RGB", (max_size, max_size))
    except Exception:
        pass
    img = img.convert("RGB")
    if max(img.size) > max_size:
        img.thumbnail((max_size, max_size), Image.BILINEAR)
    return img


def fetch_thumbs(client, jobs, tag: str, max_size: int | None = None) -> tuple[dict, set]:
    """Download + decode thumbnails in parallel (IO-bound threads).

    `jobs`: iterable of (key, asset_id). Returns ({key: PIL}, {dead keys}).
    Semantics match the old serial loops: per-asset containment (warn +
    skip), no rows for dead assets. Thumbnails are uncached everywhere,
    so every stage (kingdom/species/yolo) pays this — keep it shared.
    """
    jobs = list(jobs)
    out: dict = {}
    dead: set = set()

    def _one(job):
        key, aid = job
        try:
            blob = client.download_bytes(
                f"/api/assets/{aid}/thumbnail"
                f"?size={THUMBNAIL_SIZE}"
            )
            img = _decode_preview(blob) if max_size is None else _decode_preview(blob, max_size)
            return (key, img, None)
        except Exception as e:  # noqa: BLE001 — containment
            return (key, None, e)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=min(16, max(4, len(jobs)))) as ex:
        for key, pil, err in ex.map(_one, jobs):
            if err is not None:
                dead.add(key)
                _log.warning(
                    "[%s] %s: %s: %s — skipping asset",
                    tag, key, type(err).__name__, err,
                )
            else:
                out[key] = pil
    return out, dead


_KINGDOM_BIN_CACHE: dict[int, tuple] = {}


def _kingdom_bins(clf) -> tuple["torch.Tensor", list[str]]:
    """Index every label-table row into a kingdom bin, cached per
    classifier instance (the table is ~785k rows; building the index is
    a one-time ~1s walk). The cache keeps a strong reference to the
    classifier so its id() can't be recycled onto a different one."""
    cached = _KINGDOM_BIN_CACHE.get(id(clf))
    if cached is not None:
        return cached[1], cached[2]
    import torch

    kingdoms = [name_ary[0][0] for name_ary in clf.get_current_txt_names()]
    bins = sorted(set(kingdoms))
    bin_of = {name: i for i, name in enumerate(bins)}
    idx = torch.tensor([bin_of[k] for k in kingdoms], dtype=torch.long)
    _KINGDOM_BIN_CACHE[id(clf)] = (clf, idx, bins)
    return idx, bins


def _unload_classifier() -> None:
    """Drop the cached BioCLIP classifier and hand its memory back.

    BioCLIP's residency (~5-6 GB: fp32 weights, the 784k×768 text table
    ≈ 2.4 GB, the python label names, allocator-retained activations) is
    held for the classifier's lifetime by design — but it never needs to
    overlap with YOLO. Stages run grouped by model: BioCLIP finishes
    (search/species/kingdom), we unload, YOLO loads (yolo/person), and
    bbox-bioclip re-resolves the classifier when it needs it again (one
    ~10-20 s warm-cache load). Keeping both resident at once is what
    OOM-killed runs on memory-tight hosts."""
    global _clf_cache
    if _clf_cache is None:
        return
    _clf_cache = None
    # The kingdom-bin index holds a strong ref to the classifier —
    # dropping the cache without clearing this would keep the whole
    # model alive anyway.
    _KINGDOM_BIN_CACHE.clear()
    gc.collect()
    try:
        import ctypes

        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except Exception:
        pass


def _score_batch_for_kingdoms(
    clf,
    items: list[tuple[str, bytes]],
) -> tuple[dict[str, dict[str, float]], dict[str, np.ndarray]]:
    """Run BioCLIP kingdom-rank prediction on downloaded previews.

    `items` is [(asset_id, preview_bytes)] — raw photos stay in memory
    only, nothing is written to disk.

    Returns (scores, vec_by_id):
      scores     — {asset_id: {kingdom: score}} for Plantae and Fungi
                   only. Other kingdoms (Animalia, etc.) are skipped —
                   discovery only cares about plants and mushrooms.
      vec_by_id  — {asset_id: normalized 768-d float32 image feature
                   vector} so the caller can persist it as the asset's
                   embedding. The species stage would otherwise encode
                   these exact same images a second time (and the old
                   clf.predict() path encoded them once here AND again
                   inside predict — this encode-once path is the fix).

    Kingdom probabilities are the same numbers BioCLIP's grouped
    formatting produces for rank=KINGDOM — softmax over the full label
    table, then each row's probability summed into its kingdom bin —
    but computed as one index_add instead of BioCLIP's per-image Python
    loop over all ~785k label rows (~2 s/img in format_grouped_probs;
    triage only needs 7 kingdom numbers).

    Cached scores should be checked before calling this; we don't cache
    here because the caller (discover_plantae_assets) persists the rows
    after we get them back."""
    if not items:
        return {}, {}
    import torch

    # Encode once. clf.predict(images=paths) would do this same encode
    # internally — we do it here so the feature vector can be (a) reused
    # for kingdom classification in this call and (b) persisted to the
    # embeddings table for the species stage to reuse.
    pil_images = []
    kept_items = []
    for _aid, _blob in items:
        try:
            pil_images.append(_decode_preview(_blob))
            kept_items.append((_aid, _blob))
        except Exception as e:
            # One corrupt/truncated thumbnail must not abort the run —
            # skip the asset; the next run re-tries it (nothing was
            # cached for it).
            _log.warning(
                "[search] %s: undecodable thumbnail (%s: %s) — skipping",
                _aid, type(e).__name__, e,
            )
    items = kept_items
    if not items:
        return {}, {}
    try:
        feats = clf.create_image_features(pil_images, normalize=True)
        with torch.no_grad():
            probs = clf.create_probabilities(
                feats, clf.get_txt_embeddings()
            ).detach().cpu()
            bin_idx, bins = _kingdom_bins(clf)
            sums = torch.zeros(
                (probs.shape[0], len(bins)), dtype=probs.dtype
            ).index_add_(1, bin_idx, probs)
        feats_np = feats.detach().cpu().numpy().astype("<f4")
    finally:
        for im in pil_images:
            _close_image(im)
    vec_by_id = {key: feats_np[i] for i, (key, _) in enumerate(items)}
    # Build {asset_id: {Plantae: x, Fungi: y}} so the discovery loop can
    # decide on Plantae OR Fungi candidates in one pass.
    out: dict[str, dict[str, float]] = {}
    for i, (key, _) in enumerate(items):
        d = out.setdefault(key, {})
        for j, kingdom in enumerate(bins):
            if kingdom in ("Plantae", "Fungi"):
                d[kingdom] = float(sums[i][j])
    return out, vec_by_id


def discover_plantae_assets(
    conn,
    client,
    clf,
    *,
    query: str,
    n_samples: int,
    base_url: str,
    api_key: str,
    decision_batch_size: int = PLANT_BATCH_SIZE,
    stop_criterion: float | None = None,
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
    # Rate/ETA sources for the per-batch breakdown lines and the UI
    # panel. Without these the panel renders the stale default 0.0/s
    # for the whole (potentially hours-long) stage. scored_total drives
    # the rate; the ETA denominator is the ACCEPT target because Immich
    # may dry up before n_samples photos are ever streamed.
    search_t0 = time.monotonic()
    scored_total = 0

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
            # plantae_by_id: {asset_id: {"Plantae": p, "Fungi": f}} —
            # max of the two is used downstream to decide if a photo
            # counts as a "candidate" (plant OR mushroom).
            plantae_by_id: dict[str, dict[str, float]] = {}
            # (asset_id, preview bytes) — raw photos stay in memory only.
            to_predict_items: list[tuple[str, bytes]] = []
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
            # Per-batch phase timers (seconds), surfaced in the
            # breakdown so a slow batch names its slow phase instead of
            # hiding inside one wall-clock number.
            t_meta = t_dl = t_enc = t_db = 0.0
            # Bulk DB reads for this batch — replaces 3 per-asset
            # SELECTs that previously dominated the search loop
            # (~30s for 10k cached photos). One SELECT for the cache
            # check, one for the cached kingdom SCORES (from
            # taxon_scores — the assets table has no Plantae/Fungi
            # columns), one for the existing assets rows, one for the
            # missing-captured_at check (drives the EXIF fetch).
            batch_ids = [a.id for a in batch]
            cached_ids = storage.bulk_taxon_scores_have_assets(
                conn, asset_ids=batch_ids, rank="kingdom"
            )
            cached_scores = storage.bulk_fetch_taxon_scores(
                conn, asset_ids=list(cached_ids), rank="kingdom"
            )
            existing_assets = storage.bulk_asset_rows(
                conn, asset_ids=batch_ids
            )
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
                if asset.id in cached_ids:
                    # Cache hit: kingdom score is already stored. Skip
                    # the GET /api/assets/{id} round-trip (the per-asset
                    # capture-time + GPS live in the local `assets`
                    # row from the first run) and only refresh the
                    # `assets` row when one of the URL/filename fields
                    # has actually changed.
                    n_cached += 1
                    existing_asset = existing_assets.get(asset.id, {})
                    if (
                        existing_asset.get("original_file_name") != asset.original_file_name
                        or existing_asset.get("original_url") != original_url
                        or existing_asset.get("thumbnail_url") != thumbnail_url
                    ):
                        # Refresh name/URLs only — never overwrite
                        # captured_at / lat / lng. The old code used
                        # upsert_asset here, whose SQL always writes
                        # the EXIF columns and so clobbered them with
                        # NULL on every URL-drift run.
                        storage.refresh_asset_urls(
                            conn,
                            asset_id=asset.id,
                            original_file_name=asset.original_file_name,
                            original_url=original_url,
                            thumbnail_url=thumbnail_url,
                        )
                    # IMPORTANT: cached_scores comes from taxon_scores,
                    # not from the assets row. The assets table has no
                    # Plantae/Fungi columns — fetching those from
                    # existing_asset gave 0.0 for every asset, which
                    # made the search boundary think nothing was a
                    # plant. Don't reintroduce that bug.
                    d = cached_scores.get(asset.id, {})
                    plantae_by_id[asset.id] = {
                        "Plantae": float(d.get("Plantae") or 0.0),
                        "Fungi": float(d.get("Fungi") or 0.0),
                    }
                    continue
                # Cache miss — fetch EXIF so the freshly-inserted row
                # has the capture time + GPS (captured_at IS NOT NULL
                # check gates against re-fetching photos we already
                # know about from an earlier run). Per-asset errors are
                # contained: a 500/undecodable asset must not abort the
                # whole run — it stays un-cached, so the next run
                # retries it.
                try:
                    existing = existing_assets.get(asset.id, {})
                    if existing.get("captured_at") is None:
                        _t0 = time.monotonic()
                        exif = ImmichClient.extract_exif(
                            client.get_asset_metadata(asset.id)
                        )
                        t_meta += time.monotonic() - _t0
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
                    n_thumb_dl += 1
                    _t0 = time.monotonic()
                    thumb_bytes = client.download_bytes(
                        f"/api/assets/{asset.id}/thumbnail"
                        f"?size={THUMBNAIL_SIZE}"
                    )
                    t_dl += time.monotonic() - _t0
                    to_predict_items.append((asset.id, thumb_bytes))
                except Exception as e:
                    _log.warning(
                        "[search] %s: %s: %s — skipping asset",
                        asset.id, type(e).__name__, e,
                    )
                    continue
            # Compact suffix shared by all three log branches below.
            # `n_added` falls out of the branch-local counter; the
            # breakdown is computed once up front. The trailing rate is
            # cumulative photos-scored / elapsed for the whole stage —
            # per-batch counts (n_items/this-batch-elapsed) swing
            # wildly with cache hit rate, the cumulative number is the
            # one that predicts finish time.
            scored_total += n_predict + n_cached
            search_elapsed = time.monotonic() - search_t0
            search_rate = (
                scored_total / search_elapsed if search_elapsed > 0 else 0.0
            )

            # Run BioCLIP kingdom-rank on the misses. Persist each row.
            if to_predict_items:
                if clf is None:
                    # Lazy: only construct the model when we actually
                    # have BioCLIP work to do. The caller passes None
                    # so the singleton resolves on first use.
                    clf = _resolve_classifier()
                _t0 = time.monotonic()
                scores, vec_by_id = _score_batch_for_kingdoms(
                    clf, to_predict_items
                )
                t_enc += time.monotonic() - _t0
                _t0 = time.monotonic()
                for asset_id, _blob in to_predict_items:
                    kingdom_scores = scores.get(asset_id, {})
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
                            asset_id=asset_id,
                            rank="kingdom",
                            scores=persist,
                        )
                    # Persist the image feature vector computed above so
                    # the species stage classifies from it instead of
                    # encoding this image a second time. The vector is
                    # model-specific (dim checked on reuse), so no rank
                    # dependency here.
                    vec = vec_by_id.get(asset_id)
                    if vec is not None:
                        storage.upsert_embedding(
                            conn, asset_id=asset_id, vector=vec
                        )
                    plantae_by_id[asset_id] = kingdom_scores
                t_db += time.monotonic() - _t0

            # Built AFTER the encode/persist phases so their timers are
            # real — this string is consumed by the log branches below.
            breakdown = (
                f"cached={n_cached} predict={n_predict} "
                f"(thumbs_dl={n_thumb_dl}) dedup={n_dedup} "
                f"{search_rate:.1f}/s "
                f"[meta {t_meta:.1f}s dl {t_dl:.1f}s "
                f"enc {t_enc:.1f}s db {t_db:.1f}s]"
            )

            # Build ordered list of (asset, max_kingdom_score) for the batch.
            # A photo counts as a candidate if EITHER kingdom exceeds the
            # threshold — so both plants AND mushrooms enter the run.
            scored: list[tuple[ImmichAsset, float]] = []
            for asset in batch:
                d = plantae_by_id.get(asset.id, {})
                max_score = max(d.get("Plantae", 0.0), d.get("Fungi", 0.0))
                scored.append((asset, max_score))

            n_plant = sum(1 for _, s in scored if s > PLANT_MIN_PROB)
            majority = n_plant / len(scored) if scored else 0.0
            # Use the caller-provided stop_criterion override when given
            # (so a per-run UI value flows through); otherwise fall back
            # to the module default PLANT_BATCH_MAJORITY (30%).
            stop_crit = (
                stop_criterion if stop_criterion is not None
                else PLANT_BATCH_MAJORITY
            )
            threshold_pct = int(stop_crit * 100)

            if majority >= stop_crit:
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
                conn.commit()
                return accepted

            if len(accepted) >= n_samples:
                _log.info(
                    f"[search] reached target of {n_samples}, stopping"
                )
                # Final progress update at 100% before returning.
                set_chunk_progress(n_samples, n_samples)
                conn.commit()
                return accepted[:n_samples]
            # Live progress for the UI bar. Updates the SSE stream so
            # StatusPanel advances per-batch even when each batch's
            # commit takes tens of ms (10k photos / 16 per batch =
            # 625 batches = several seconds of "silent" work
            # otherwise).
            set_chunk_progress(len(accepted), n_samples)
            # Speed readout for the UI panel. ETA extrapolates the
            # ACCEPT rate so far over the remaining accept target —
            # rough by design (acceptance ratio drifts page to page and
            # Immich may run out of results first), but it beats the
            # stale 0.0/s the panel showed before.
            if search_elapsed > 0:
                accept_rate = len(accepted) / search_elapsed
                remaining = max(0, n_samples - len(accepted))
                set_state(
                    stage="search",
                    rate=search_rate,
                    eta_seconds=(
                        remaining / accept_rate if accept_rate > 0 else 0.0
                    ),
                )
            # Commit THIS batch's asset rows + kingdom scores now.
            # storage.connect() only commits when its whole with-block
            # exits — i.e. when the entire multi-hour run finishes — so
            # without this every batch sat in one open transaction:
            # /api/results saw a permanently empty DB mid-run, and any
            # crash/stop/crash-loop ROLLED BACK the entire run (the
            # "ran for hours, every page empty" report).
            conn.commit()

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


def run_knn_reclassify_bbox(
    *,
    detection_id: int,
    index_dir: Path,
    db_path: str | Path,
    base_url: str,
    api_key: str,
    k: int = 10,
) -> dict:
    """KNN reclassify a single cropped bbox (per-crop 'Reclassify crop'
    button in AssetDetail).

    Downloads the parent asset's image, crops to the bbox xywh,
    runs BioCLIP on the crop to get a 768-d embedding, queries the
    FAISS index, and writes the result to bbox_knn_predictions.

    Cost: ~1 BioCLIP forward pass + 1 FAISS query (~ms). The pipeline
    bbox-BioCLIP stage does the same per-crop work in batch — this is
    a one-off trigger for users who want to retry a single crop with
    a new KNN index without re-running the whole pipeline.

    Returns the KNN result dict (same shape as
    `run_knn_reclassify` minus the asset_id field).
    """
    import sys as _sys
    import io as _io
    _self = _sys.modules[__name__]
    _knn = _self.knn

    index_dir = Path(index_dir)
    version_path = index_dir / "INDEX_VERSION.txt"
    if version_path.exists():
        version_tag = next(
            (line.strip() for line in version_path.read_text().splitlines() if line.strip()),
            "unknown",
        )
    elif (index_dir / "faiss" / "index.index").exists():
        version_tag = "imageomics-official"
    else:
        raise FileNotFoundError(
            f"No KNN index found in {index_dir} — need either INDEX_VERSION.txt (custom) or faiss/index.index (imageomics official)"
        )

    # Look up the detection + the asset's image URL.
    with storage.connect(db_path) as conn_ro:
        det = storage.get_detection_with_asset_url(
            conn_ro, detection_id=detection_id,
        )
    if det is None:
        raise ValueError(f"detection {detection_id} not found")

    # Skip if cached for this (version, k).
    with storage.connect(db_path) as conn_ck:
        if storage.bbox_knn_predictions_have_detection(
            conn_ck,
            detection_id=detection_id,
            index_version=version_tag,
            k=k,
        ):
            cached = storage.fetch_bbox_knn_predictions(
                conn_ck,
                detection_ids=[detection_id],
                index_version=version_tag,
                k=k,
            ).get(detection_id)
            if cached is not None:
                _log.info(
                    f"[knn bbox {detection_id}] cache hit  "
                    f"v={version_tag} k={k} -> {cached.get('top_species')}"
                )
                return {
                    "detection_id": detection_id,
                    "asset_id": det["asset_id"],
                    "index_version": version_tag,
                    "k": k,
                    "result": cached,
                    "cached": True,
                }

    # Download the image. Prefer the asset's stored original_url
    # (whatever Immich returned at ingest time); fall back to the
    # /api/assets/{id}/original endpoint if the URL has since rotated.
    client = ImmichClient(base_url=base_url, api_key=api_key)
    url = det.get("original_url") or f"/api/assets/{det['asset_id']}/original"
    try:
        img_bytes = client.download_bytes(url)
    except Exception as e:
        raise RuntimeError(f"download failed for {url}: {e}") from e

    # Crop + BioCLIP embedding.
    from PIL import Image
    with Image.open(_io.BytesIO(img_bytes)) as im:
        w_img, h_img = im.size
        x = max(0.0, float(det["x"]))
        y = max(0.0, float(det["y"]))
        # detections stored as top-left xywh (floats 0..1 over image)
        bw = float(det["w"])
        bh = float(det["h"])
        x0 = int(round(x * w_img))
        y0 = int(round(y * h_img))
        x1 = int(round((x + bw) * w_img))
        y1 = int(round((y + bh) * h_img))
        # Clamp to image bounds (the bbox coords are floats; rounding
        # can spill past the edge by a pixel).
        x0 = max(0, min(w_img, x0))
        y0 = max(0, min(h_img, y0))
        x1 = max(x0 + 1, min(w_img, x1))
        y1 = max(y0 + 1, min(h_img, y1))
        crop = im.convert("RGB").crop((x0, y0, x1, y1))

    # BioCLIP embedding for the crop. classifier is heavy (5 GB) — only
    # loaded on first use. _resolve_classifier() is process-global; the
    # first call pays the load cost, subsequent calls share.
    clf = _resolve_classifier()
    if clf is None:
        raise RuntimeError("classifier unavailable (BioCLIP weights not loaded?)")
    with _clf_lock:
        feats = clf.create_image_features([crop], normalize=True)
    feats_np = feats.detach().cpu().numpy().astype("<f4")
    del feats
    # This entry point runs inside uvicorn (asyncio.to_thread), not in
    # the pipeline worker — free the ~6 GB classifier immediately so a
    # later pipeline run doesn't coexist with a resident copy here.
    _unload_classifier()

    # FAISS KNN.
    results = _knn.classify_via_knn(
        query_embeddings=feats_np, index_dir=index_dir, k=k,
    )
    if not results:
        raise RuntimeError("KNN classifier returned no rows (empty index?)")
    result = results[0]

    # Persist + return.
    with storage.connect(db_path) as conn_wr:
        storage.replace_bbox_knn_predictions(
            conn_wr,
            detection_id=detection_id,
            index_version=version_tag,
            k=k,
            row=result,
        )
    _log.info(
        f"[knn bbox {detection_id}] asset={det['asset_id']} "
        f"v={version_tag} k={k} -> {result.get('top_species')} "
        f"({result.get('score'):.3f})"
    )
    return {
        "detection_id": detection_id,
        "asset_id": det["asset_id"],
        "index_version": version_tag,
        "k": k,
        "result": result,
        "cached": False,
    }


def run_knn_reclassify(
    conn,
    *,
    asset_ids: list[str],
    index_dir: Path,
    k: int = 10,
    force: bool = False,
) -> dict:
    """KNN reclassify a list of assets using a pre-built FAISS index.

    Reuses the cached BioCLIP image embeddings stored in the
    `embeddings` table — does NOT re-embed photos. Skips any asset that
    already has a `knn_predictions` row for (version, k), unless
    `force` is set — the per-asset "Re-run KNN" button passes force so
    an explicit user click always recomputes (and so rows written by an
    older code version get refreshed with any new neighbor fields).

    Returns a stats dict {"requested", "cached", "scanned", "index_version"}.
    Logs to stderr and to the file log via the existing `_log` so the
    UI's tail picks it up.
    """
    import sys as _sys
    _self = _sys.modules[__name__]
    _knn = _self.knn

    index_dir = Path(index_dir)
    version_path = index_dir / "INDEX_VERSION.txt"
    if version_path.exists():
        # Read the version tag (first non-empty line).
        version_tag = next(
            (line.strip() for line in version_path.read_text().splitlines() if line.strip()),
            "unknown",
        )
    elif (index_dir / "faiss" / "index.index").exists():
        version_tag = "imageomics-official"
    else:
        raise FileNotFoundError(
            f"No KNN index found in {index_dir} — need either INDEX_VERSION.txt (custom) or faiss/index.index (imageomics official)"
        )

    if not asset_ids:
        return {
            "requested": 0, "cached": 0, "scanned": 0,
            "index_version": version_tag,
        }

    # Filter: only assets that have an embedding AND don't already have
    # a KNN row for this (version, k). The DB-side check keeps the
    # "already classified" count accurate per call. With `force`, we
    # drop the NOT EXISTS clause and recompute every asset that has an
    # embedding.
    placeholders = ",".join("?" for _ in asset_ids)
    if force:
        rows = conn.execute(
            f"""
            SELECT e.asset_id, e.vector
            FROM embeddings e
            WHERE e.asset_id IN ({placeholders})
            """,
            list(asset_ids),
        ).fetchall()
    else:
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


def _sweep_stale_scratch() -> None:
    """Remove `immich_bioclip_*` scratch dirs left by runs that died by
    SIGKILL/OOM — those bypass run_pipeline's finally rmtree. Called at
    run start (worker and CLI share this path). Age-gated to >24h so a
    concurrent live run's scratch is never touched: a live bbox stage
    rewrites its dir every chunk, while a hard-killed dir's mtime freezes
    at death. The end-of-run rmtree handles the clean-stop case."""
    cutoff = time.time() - 24 * 3600
    try:
        root = Path(tempfile.gettempdir())
        for d in root.glob("immich_bioclip_*"):
            try:
                if d.is_dir() and d.stat().st_mtime < cutoff:
                    shutil.rmtree(d, ignore_errors=True)
                    _log.info("[cleanup] removed stale scratch dir %s", d)
            except OSError:
                continue
    except OSError:
        pass


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
    # Per-run override for PLANT_BATCH_MAJORITY, the "stop criteria
    # threshold" exposed in the Run-page UI. Expressed as a percent
    # (0-100) of the batch that must score above PLANT_MIN_PROB for
    # the search to keep going. None falls back to the module default
    # (PLANT_BATCH_MAJORITY = 30%).
    stop_criteria_pct: float | None = None,
) -> RunResult:
    """Run a smart-search on Immich for `query`, embed + predict any assets
    not yet cached, recompute clusters, run the detector on uncached images,
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
    # Resolve the per-run stop-criteria override (None → module default
    # PLANT_BATCH_MAJORITY). Stored as a fraction internally so the
    # boundary check stays a straight comparison against `majority`.
    _stop_criteria = (
        float(stop_criteria_pct) / 100.0
        if stop_criteria_pct is not None
        else PLANT_BATCH_MAJORITY
    )
    _log.info(
        f"[run] starting  query={query!r}  n_requested={n_samples}  "
        f"rank={rank_name}  top_k={top_k}  batch_size={batch_size}  "
        f"stop_criteria={_stop_criteria * 100:.0f}%"
    )
    # Search's "chunk_total" is the photo target, not a batch count
    # — the search loop calls set_chunk_progress(len(accepted),
    # n_samples) per batch so the UI sees a steady advance instead
    # of "stage: search" with chunk_idx stuck at 0 for the 20+
    # seconds the per-batch DB-write loop takes on a large run.
    _stage_begin("search", chunk_total=n_samples)

    client = ImmichClient(base_url=base_url, api_key=api_key)
    tmp: Path | None = None
    try:
        with storage.connect(db_path) as conn:
            # Lazy classifier: only construct the TreeOfLifeClassifier
            # (and load BioCLIP weights + open_clip helpers) the first
            # time the kingdom stage has real BioCLIP work to do. The
            # BioCLIP model on CPU is ~5.5 GB resident — loading it for
            # an entirely-cached run wastes that until the process dies.
            # Both downstream stages gate on `if needs_bioclip:` /
            # `if to_predict_items:`, so the singleton is only created
            # when inference is genuinely about to run.
            clf = None

            run_id = storage.create_run(
                conn,
                query=query,
                n_requested=n_samples,
                rank=rank_name,
                top_k=top_k,
            )
            # Previews are fetched from Immich and decoded in memory —
            # raw photos are never written to disk. Crop scratch (for a
            # single predict() call) still uses `tmp`.
            _sweep_stale_scratch()
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
                decision_batch_size=decision_batch_size,
                stop_criterion=_stop_criteria,
                stop_event=stop_event,
            )
            _stage_end("search")

            needs_bioclip: list[ImmichAsset] = []
            needs_yolo: list[ImmichAsset] = []
            ordered_ids: list[str] = []

            # Stitch loop — populate / refresh the `assets` rows and
            # decide which assets still need a BioCLIP predict. Three
            # bulk SELECTs up front (existing rows + missing EXIF +
            # per-asset prediction count) replace the N+1 of the old
            # per-asset loop. For the common cached case (URLs match
            # AND captured_at is set AND predictions are complete) we
            # do ZERO writes — that was the 30-second silence between
            # "[search] reached target" and "query=... n_new=...".
            #
            # The user reported a still-suspicious gap between those
            # two lines even after that fix. The three log lines
            # below narrate what's happening in this section in plain
            # English, with a wall-clock split between the bulk
            # SELECTs and the in-Python loop, so a future slow run
            # can be attributed to the right step. Elapsed seconds
            # are measured from `_stage_end("search")` so they line
            # up with whatever gap a user sees in the file log.
            _stitch_t0 = time.time()
            _log.info(
                f"[stitch] starting: reconcile {len(assets)} Immich "
                f"search results against the local SQLite cache — "
                f"cache hits will skip both the EXIF fetch and the "
                f"BioCLIP predict"
            )
            asset_ids = [a.id for a in assets]
            _bulk_t0 = time.time()
            existing_assets = storage.bulk_asset_rows(
                conn, asset_ids=asset_ids
            )
            missing_captured_at = storage.bulk_assets_missing_captured_at(
                conn, asset_ids=asset_ids
            )
            cached_pred_counts = storage.bulk_fetch_prediction_counts(
                conn, asset_ids=asset_ids, rank=rank_name
            )
            # YOLO scan status is its OWN cache dimension — an asset can
            # have complete species predictions but never have been
            # detection-scanned (a run that stopped or crashed between
            # the species and yolo stages — e.g. the old YOLO-World
            # config-dir PermissionError). Gate the yolo worklist on
            # yolo_scanned_at, not on the predictions cache.
            yolo_scanned_ids = storage.bulk_yolo_scanned_assets(
                conn, asset_ids=asset_ids
            )
            _log.info(
                f"[stitch] bulk SELECTs done in "
                f"{time.time() - _bulk_t0:.2f}s — "
                f"loaded {len(existing_assets)} existing asset rows so "
                f"the loop can skip a URL refresh when URLs match, "
                f"found {len(missing_captured_at)} assets still missing "
                f"EXIF (those will trigger an Immich metadata fetch), "
                f"found {len(cached_pred_counts)} assets that already "
                f"have ≥{top_k} {rank_name} predictions (those can "
                f"skip BioCLIP if their URLs also match)"
            )

            n_stitched_skip = 0
            n_stitched_urls_only = 0
            n_stitched_exif_fetched = 0
            n_stitched_full_upsert = 0

            _loop_t0 = time.time()
            for asset in assets:
                original_url = f"{base_url}/api/assets/{asset.id}/original"
                thumbnail_url = (
                    f"{base_url}/api/assets/{asset.id}/thumbnail"
                    f"?size={THUMBNAIL_SIZE}"
                )
                existing = existing_assets.get(asset.id, {})
                urls_match = (
                    existing.get("original_file_name") == asset.original_file_name
                    and existing.get("original_url") == original_url
                    and existing.get("thumbnail_url") == thumbnail_url
                )
                has_exif = asset.id not in missing_captured_at
                if urls_match and has_exif:
                    # Cache hit on every dimension — skip the upsert
                    # entirely. This is the common case on a stable
                    # Immich install: same URLs, EXIF already
                    # captured, no-op in SQLite. Without this branch
                    # we were issuing n_samples no-op UPDATEs per
                    # run, ~30 s of silence on a 10k cached run.
                    n_stitched_skip += 1
                elif urls_match:
                    # URLs are fine but the row is missing EXIF.
                    # Fetch it once and write only the EXIF columns
                    # — don't clobber the URLs that already match.
                    # Metadata fetch errors are contained (skip the
                    # EXIF backfill for this asset) — one 500 from
                    # Immich must not abort the run this late.
                    try:
                        exif = ImmichClient.extract_exif(
                            client.get_asset_metadata(asset.id)
                        )
                    except Exception as e:
                        _log.warning(
                            "[stitch] %s: %s: %s — skipping EXIF backfill",
                            asset.id, type(e).__name__, e,
                        )
                        exif = None
                    if exif is not None:
                        n_stitched_exif_fetched += 1
                        storage.update_asset_exif(
                            conn,
                            asset_id=asset.id,
                            captured_at=exif["captured_at"],
                            latitude=exif["latitude"],
                            longitude=exif["longitude"],
                        )
                elif has_exif:
                    # Row exists with EXIF but URLs have drifted
                    # (Immich renamed the file, base_url changed,
                    # etc). Refresh URLs only — never overwrite EXIF
                    # with NULL.
                    n_stitched_urls_only += 1
                    storage.refresh_asset_urls(
                        conn,
                        asset_id=asset.id,
                        original_file_name=asset.original_file_name,
                        original_url=original_url,
                        thumbnail_url=thumbnail_url,
                    )
                else:
                    # Either a brand-new asset or a row with no EXIF
                    # AND drifted URLs. Fetch EXIF and do a full
                    # upsert (metadata errors contained as above).
                    try:
                        exif = ImmichClient.extract_exif(
                            client.get_asset_metadata(asset.id)
                        )
                        n_stitched_exif_fetched += 1
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
                        n_stitched_full_upsert += 1
                    except Exception as e:
                        _log.warning(
                            "[stitch] %s: %s: %s — skipping asset upsert",
                            asset.id, type(e).__name__, e,
                        )

                ordered_ids.append(asset.id)
                if cached_pred_counts.get(asset.id, 0) < top_k:
                    # Defer download until the chunk loop reads it.
                    needs_bioclip.append(asset)
                if asset.id not in yolo_scanned_ids:
                    # Detection scan is independent of the predictions
                    # cache — see yolo_scanned_ids above. Thumb download
                    # is deferred to the yolo chunk loop (the species
                    # loop also populates the shared path when it runs
                    # first, but cached-species assets never enter it).
                    needs_yolo.append(asset)

            n_new = len(needs_bioclip)
            n_cached = len(assets) - n_new

            # Two numbers worth keeping in the timeline: the in-
            # Python walk that is doing the per-asset decision (this
            # is the only thing the loop does), and the total time
            # spent in the [stitch] section. If they match closely,
            # the cost was in the loop; if there's a big gap, the
            # cost lives outside this section (DB writes, state
            # mirroring, etc.) and needs a different probe.
            _log.info(
                f"[stitch] reconcile loop walked {len(assets)} assets "
                f"in {time.time() - _loop_t0:.2f}s — decided that "
                f"{n_new} still need BioCLIP and {n_cached} are full "
                f"cache hits; total [stitch] section time "
                f"{time.time() - _stitch_t0:.2f}s"
            )
            # One log line that explains the whole stitch. Counts
            # that are 0 are still printed — the user wants to know
            # the work actually happened (or didn't), and zero is a
            # meaningful answer.
            _log.info(
                f"[stitch] {len(assets)} assets: "
                f"skip={n_stitched_skip}  urls_only={n_stitched_urls_only} "
                f"exif_fetched={n_stitched_exif_fetched} "
                f"full_upsert={n_stitched_full_upsert}"
            )
            # Flush the stitch's asset-row upserts (URLs + EXIF) so a
            # stop/crash right after search keeps them.
            conn.commit()
            _log.info(
                f"query={query!r}  n_requested={n_samples}  "
                f"n_new={n_new}  n_cached={n_cached}  "
                f"rank={rank_name}  top_k={top_k}  batch_size={batch_size}"
            )

            # ---- BioCLIP: embed + predict for uncached assets ----
            # Iterate by index throughout so we never read from a stale
            # slice. We mutate needs_bioclip/need_yolo in place as we go.
            species_total_chunks = (len(needs_bioclip) + batch_size - 1) // batch_size

            # _stage_begin is OUTSIDE the needs_bioclip branch so the
            # timeline always shows "[stage] species" even when cached.
            _stage_begin("species", chunk_total=species_total_chunks)

            if needs_bioclip:
                if clf is None:
                    clf = classifier if classifier is not None else _resolve_classifier()
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
                    import torch as _torch
                    chunk_indices: list[int] = []  # indices we just processed
                    chunk_asset_ids: list[str] = []

                    for k in range(i, j):
                        chunk_indices.append(k)
                        chunk_asset_ids.append(needs_bioclip[k].id)

                    # Reuse embeddings already persisted by the search
                    # stage (its kingdom scoring encodes every unseen
                    # asset once and stores the vector). Only assets
                    # still missing a vector of the CURRENT model's
                    # dimension get encoded here — dim guard so a DB
                    # built against a different BioCLIP major version
                    # is re-encoded rather than fed to the wrong text
                    # head.
                    expected_dim = int(clf.get_txt_embeddings().shape[0])
                    stored_vecs = {
                        aid: vec
                        for aid, vec in storage.bulk_fetch_embeddings(
                            conn, asset_ids=chunk_asset_ids
                        ).items()
                        if vec.shape[0] == expected_dim
                    }
                    # Per-asset download/decode with containment: one bad
                    # thumbnail (HTTP 500, truncated JPEG) must not abort
                    # the whole run — the run would otherwise die at the
                    # same asset on every retry, forever. Skip the asset
                    # this chunk; no cache rows are written for it, so the
                    # next run picks it up again.
                    dead: set = set()
                    pil_by_aid: dict = {}
                    todo = [needs_bioclip[k] for k in range(i, j)
                            if needs_bioclip[k].id not in stored_vecs]
                    got, dead_ids = fetch_thumbs(
                        client,
                        [(a.id, a.id) for a in todo],
                        "species")
                    pil_by_aid.update(got)
                    dead.update(dead_ids)
                    if dead:
                        chunk_indices = [
                            k for k in chunk_indices
                            if needs_bioclip[k].id not in dead
                        ]
                        chunk_asset_ids = [
                            aid for aid in chunk_asset_ids if aid not in dead
                        ]
                    if not chunk_indices:
                        i = j
                        continue
                    feats = _torch.empty(
                        (len(chunk_indices), expected_dim),
                        dtype=_torch.float32,
                    )
                    encode_positions: list[int] = []
                    for pos, k in enumerate(chunk_indices):
                        vec = stored_vecs.get(needs_bioclip[k].id)
                        if vec is not None:
                            feats[pos] = _torch.from_numpy(vec)
                        else:
                            encode_positions.append(pos)
                    if encode_positions:
                        pil_images = [
                            pil_by_aid[needs_bioclip[k].id]
                            for k in (chunk_indices[p] for p in encode_positions)
                        ]
                        try:
                            enc_feats = clf.create_image_features(
                                pil_images, normalize=True
                            )
                            feats[encode_positions] = enc_feats.detach().cpu()
                        finally:
                            # Free the PIL decodes — yolo.detect reopens.
                            for im in pil_images:
                                _close_image(im)
                    feats = feats.to(getattr(clf, "device", "cpu"))

                    # Classify from the features we already hold — same
                    # softmax clf.predict() computes, but WITHOUT a
                    # second ViT encode (predict() re-opens the JPEGs
                    # and re-runs encode_image internally — that was
                    # ~half the species stage's wall time on CPU).
                    raw = _classify_features(
                        clf, feats, chunk_asset_ids,
                        rank_enum=rank_enum, k=top_k,
                    )
                    preds_by_id = _group_predictions(raw, top_k)
                    feats_np = feats.detach().cpu().numpy().astype("<f4")

                    for pos, k in enumerate(chunk_indices):
                        asset = needs_bioclip[k]
                        if asset.id not in stored_vecs:
                            storage.upsert_embedding(
                                conn, asset_id=asset.id, vector=feats_np[pos]
                            )
                        rows = preds_by_id.get(asset.id, [])
                        if not rows:
                            rows = _rows_for_image(preds_by_id, asset.id)
                        storage.replace_predictions(
                            conn,
                            asset_id=asset.id,
                            rank=rank_name,
                            rows=rows,
                        )

                    del feats, feats_np, raw, preds_by_id
                    gc.collect()

                    # Commit this chunk's embeddings + predictions (the
                    # outer storage.connect() with-block won't commit
                    # until the whole run ends — see the search-loop
                    # commit for why that matters). This is what makes
                    # the "each chunk commits" claim above actually
                    # true.
                    conn.commit()

                    _chunk_log(
                        "species",
                        (i // batch_size) + 1,
                        species_total_chunks,
                        n_items=len(chunk_indices),
                        elapsed=time.monotonic() - species_t0,
                        items_done=j,
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
            # The run runs in a separate worker process; the parent's
            # caches are invalidated by app.py's reaper when it notices
            # the child exit. cluster_names are dropped by
            # pipeline_worker.py on success.
            _stage_begin("cluster")
            cluster_t0 = time.monotonic()
            newly_embedded_ids: list[str] = [a.id for a in needs_bioclip]
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
            conn.commit()  # cluster assignments survive a mid-run stop

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
            # skipping via taxon_scores_have_asset. Assets needing a
            # fresh encode fetch their preview at encode time — in
            # memory only.
            KINGDOM_TOP_K = 5
            import torch as _torch
            kingdom_assets: list[ImmichAsset] = []
            # Assets still missing kingdom scores. Their image features
            # are almost always already in the embeddings table (the
            # search stage persists one vector per asset it scores), so
            # kingdom classification normally needs ZERO image encodes
            # and ZERO thumbnail downloads — just the text-head matmul
            # over stored vectors. Only legacy assets (no embedding row,
            # or a vector from a different model version) fall back to
            # a fresh encode below.
            kingdom_missing = [
                asset
                for asset in assets
                if not storage.taxon_scores_have_asset(
                    conn, asset_id=asset.id, rank="kingdom"
                )
            ]
            kingdom_assets = list(kingdom_missing)
            kingdom_stored: dict[str, np.ndarray] = {}
            if kingdom_missing:
                if clf is None:
                    clf = classifier if classifier is not None else _resolve_classifier()
                expected_dim = int(clf.get_txt_embeddings().shape[0])
                kingdom_stored = {
                    aid: vec
                    for aid, vec in storage.bulk_fetch_embeddings(
                        conn, asset_ids=[a.id for a in kingdom_missing]
                    ).items()
                    if vec.shape[0] == expected_dim
                }
                # Assets WITHOUT a stored vector download their preview
                # at encode time below — in memory only.

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
                chunk_assets = kingdom_assets[i:j]
                # Assemble one feature matrix for the chunk: stored
                # vectors for assets already embedded (the common case),
                # fresh encodes only for the rest. This replaces a
                # clf.predict() call that re-encoded EVERY image.
                feats_rows: list = [None] * len(chunk_assets)
                encode_positions: list[int] = []
                for pos, asset in enumerate(chunk_assets):
                    vec = kingdom_stored.get(asset.id)
                    if vec is not None:
                        feats_rows[pos] = _torch.from_numpy(vec)
                    else:
                        encode_positions.append(pos)
                if encode_positions:
                    pos_of = {chunk_assets[p].id: p for p in encode_positions}
                    got, dead_aids = fetch_thumbs(
                        client,
                        [(aid, aid) for aid in pos_of],
                        "kingdom")
                    for aid in sorted(dead_aids, key=pos_of.get):
                        # Skip the asset, keep the run alive; nothing
                        # was cached for it so the next run retries.
                        encode_positions.remove(pos_of[aid])
                    pil_by_pos = {pos_of[aid]: pil
                                  for aid, pil in got.items()}
                    pil_images = [pil_by_pos[p] for p in encode_positions]
                if encode_positions:
                    try:
                        enc_feats = clf.create_image_features(
                            pil_images, normalize=True
                        )
                        enc_np = enc_feats.detach().cpu().numpy().astype("<f4")
                        for e, pos in enumerate(encode_positions):
                            feats_rows[pos] = enc_feats[e].detach().cpu()
                            # Persist the fresh vector so this fallback
                            # encode is never paid twice.
                            storage.upsert_embedding(
                                conn,
                                asset_id=chunk_assets[pos].id,
                                vector=enc_np[e],
                            )
                    finally:
                        for im in pil_images:
                            _close_image(im)
                feats = _torch.stack(feats_rows).to(
                    getattr(clf, "device", "cpu")
                )
                raw_kingdom = _classify_features(
                    clf, feats, [a.id for a in chunk_assets],
                    rank_enum=kingdom_rank, k=KINGDOM_TOP_K,
                )
                kingdom_by_id: dict[str, dict[str, float]] = {}
                for row in raw_kingdom:
                    f = row["file_name"]
                    d = kingdom_by_id.setdefault(f, {})
                    prev = d.get(row["kingdom"])
                    if prev is None or row["score"] > prev:
                        d[row["kingdom"]] = float(row["score"])
                for pos, aid in enumerate(a.id for a in chunk_assets):
                    asset = chunk_assets[pos]
                    scores = kingdom_by_id.get(aid) or {}
                    storage.upsert_taxon_scores(
                        conn,
                        asset_id=asset.id,
                        rank="kingdom",
                        scores=scores,
                    )
                del feats, raw_kingdom, kingdom_by_id
                gc.collect()
                _chunk_log(
                    "kingdom",
                    (i // batch_size) + 1,
                    kingdom_total_chunks,
                    n_items=len(chunk_assets),
                    elapsed=time.monotonic() - kingdom_t0,
                    items_done=j,
                )
                conn.commit()  # per-chunk checkpoint (see search-loop note)
                i = j
            if kingdom_total_chunks:
                _log.info(
                    f"[kingdom] done  {len(kingdom_assets)} assets in "
                    f"{_fmt_eta(time.monotonic() - kingdom_t0)}"
                )
            _stage_end("kingdom")
            del kingdom_assets

            # BioCLIP's stages are done (bbox-bioclip reloads it later).
            # Free the encoder before YOLO loads so the two models never
            # coexist — see _unload_classifier.
            if clf is not None:
                _log.info(
                    "[model] unloading %s classifier (done until bbox)",
                    type(clf).__name__,
                )
                _unload_classifier()
                clf = None

            # ---- Detector: detect on uncached assets (chunked) ----
            # Detection rows are persisted as we go; the bbox BioCLIP
            # pass below queries the detections table directly so it
            # handles cache-hit assets without needing tuples handed
            # off from here.
            yolo_total_chunks = (len(needs_yolo) + batch_size - 1) // batch_size
            yolo_t0 = time.monotonic()
            # Hand freed allocator arenas back to the OS before YOLO
            # loads its weights + CLIP text encoder (~1-2 GB on top of
            # BioCLIP's footprint). glibc keeps freed torch tensors in
            # its arenas, and on this memory-tight host the process was
            # OOM-killed twice exactly here — a hard SIGKILL that skips
            # every finally (run cleanup included).
            gc.collect()
            try:
                import ctypes

                ctypes.CDLL("libc.so.6").malloc_trim(0)
            except Exception:
                pass
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
                        todo: list[tuple[int, ImmichAsset]] = []
                        for k in range(i, j):
                            todo.append((k, needs_yolo[k]))
                        if not todo:
                            i = j
                            continue
                        # Lazy: don't load the detector until we actually
                        # need it.
                        if not yolo_model_loaded:
                            _log.info(
                                "[yolo] loading detection backend %s "
                                "(YOLO_BACKEND=%s)",
                                _yolo_backend_module(),
                                os.environ.get("YOLO_BACKEND", "rtmdet"),
                            )
                            _yolo.get_model()
                            yolo_model_loaded = True

                        # Detection is independent of the BioCLIP stages
                        # (this is what lets a run that died between
                        # species and yolo resume without re-running
                        # BioCLIP) — previews are fetched per chunk and
                        # decoded in memory. Per-asset errors skip just
                        # that asset (mark_yolo_scanned isn't reached,
                        # so the next run retries it).
                        got, dead = fetch_thumbs(
                            client,
                            [(asset.id, asset.id) for _, asset in todo],
                            "yolo", max_size=_yolo_decode_size())
                        todo_ok, pil_images = [], []
                        for _k, asset in todo:
                            if asset.id in dead:
                                continue
                            todo_ok.append((_k, asset))
                            pil_images.append(got[asset.id])
                        todo = todo_ok
                        detections = (
                            _yolo.detect(pil_images, conf=yolo_conf)
                            if pil_images else []
                        )
                        n_dets = 0
                        n_merged = 0
                        for (_k, asset), dets, pil_img in zip(
                            todo, detections, pil_images
                        ):
                            # Class-agnostic IoU merge: when YOLO returns
                            # multiple boxes for the same region under
                            # different classes (e.g. "plant" + "leaf"),
                            # collapse them so we only BioCLIP-classify
                            # one crop per region.
                            before = len(dets)
                            dets = _merge_overlapping_boxes(dets)
                            n_merged += before - len(dets)
                            # YuNet rescue before the tier split — see
                            # the helper. Appends `face` rows only when
                            # YOLO found a person box but the
                            # hand-suppression rule stamped it down.
                            dets = _augment_with_yunet_faces(dets, pil_img)
                            # ONE forward pass feeds everything: this
                            # model's label table already includes
                            # person/face/human/hand, so the boxes the
                            # person stage used to re-scan the same
                            # images for are right here. Persist all
                            # tiers from this pass:
                            #   plant  — every box; bbox-bioclip
                            #            re-verifies at BBOX_BIOCLIP_MIN
                            #   person — only >= PERSON_MIN_CONF, because
                            #            has_person is a UI-facing boolean
                            #            with no second-stage check
                            #   hand   — >= HAND_MIN_CONF; feeds the
                            #            has_real_person() suppression
                            #            rule below
                            #   face   — YuNet rescue rows only (YOLO's
                            #            face prompt never fires); feeds
                            #            has_real_person()'s face hatch
                            # Below-threshold person/hand boxes are
                            # dropped entirely (not lumped into plants).
                            plant_dets = [
                                d for d in dets
                                if d["label"] not in PERSON_LABELS
                                and d["label"] != HAND_LABEL
                            ]
                            person_dets = [
                                d for d in dets
                                if d["label"] in PERSON_LABELS
                                and d["label"] != "face"
                                and d["score"] >= PERSON_MIN_CONF
                            ]
                            hand_dets = [
                                d for d in dets
                                if d["label"] == HAND_LABEL
                                and d["score"] >= HAND_MIN_CONF
                            ]
                            face_dets = [
                                d for d in dets if d["label"] == "face"
                            ]
                            # Unscoped replace: this pass produced fresh
                            # boxes for EVERY label, so any earlier rows
                            # (e.g. person rows from a standalone
                            # backfill) are stale by definition. Their
                            # bbox_predictions children cascade-delete
                            # with them — also correct, since those
                            # detection_ids no longer exist.
                            new_ids = storage.replace_detections(
                                conn, asset_id=asset.id,
                                detections=plant_dets + person_dets
                                + hand_dets + face_dets,
                            )
                            n_dets += len(new_ids)
                            # Scan markers — set even when the asset had
                            # zero boxes, so a clean scan is
                            # distinguishable from "never scanned" and
                            # the asset never re-enters this worklist.
                            # has_person derives from the same boxes
                            # (hand-suppression rule inside
                            # has_real_person), which is what lets the
                            # person stage below skip every asset this
                            # stage has seen.
                            # Classifier veto — confident hand-only photos
                            # (p < PERSON_VETO_P) beat person-box evidence.
                            hp = apply_person_veto(
                                has_real_person(dets),
                                _person_clf_p(conn, asset.id),
                            )
                            storage.mark_yolo_scanned(
                                conn, asset_id=asset.id,
                                has_person=hp,
                            )
                        for im in pil_images:
                            _close_image(im)
                        _chunk_log(
                            "yolo",
                            (i // batch_size) + 1,
                            yolo_total_chunks,
                            n_items=len(todo),
                            elapsed=time.monotonic() - yolo_t0,
                            items_done=j,
                            note=f"{n_dets} new detections ({n_merged} merged)",
                        )
                        conn.commit()  # per-chunk checkpoint (see search-loop note)
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

            # ---- Person detection: residual sweep ----
            # The yolo stage above already derives has_person for every
            # asset it scans — same model, same forward pass, its label
            # table includes person/face/human/hand. Unlike that stage
            # (which only iterates plantae candidates from
            # `discover_plantae_assets`), this sweep covers EVERY asset
            # in the DB where `has_person IS NULL`: candidates left over
            # from earlier runs with different queries, or rows written
            # before the yolo stage started stamping has_person. That
            # gate makes this a no-op once the library is fully covered.
            # Uses `append_detections` (NOT `replace_detections`) so
            # existing plant detections and their `bbox_predictions` FK
            # CASCADE children stay cached — only person/face/human/hand
            # boxes are added.
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
                    todo_items: list[tuple[str, str]] = []
                    for asset_id, _fn, _ts in todo:
                        todo_items.append((asset_id, _ts))
                    # Process in chunks of `batch_size`; the previous
                    # yolo stage's finally already unloaded, so this
                    # reloads the detector session on first use.
                    person_t0 = time.monotonic()
                    for i in range(0, len(todo_items), batch_size):
                        check_stop(stop_event)
                        chunk = todo_items[i : i + batch_size]
                        pil_images = []
                        chunk_ok = []
                        for aid, _ts in chunk:
                            try:
                                pil_images.append(_decode_preview(
                                    client.download_bytes(
                                        f"/api/assets/{aid}/thumbnail"
                                        f"?size={THUMBNAIL_SIZE}"
                                    ),
                                    _yolo_decode_size(),
                                ))
                                chunk_ok.append((aid, _ts))
                            except Exception as e:
                                # Skipped assets stay unscanned
                                # (has_person stays NULL) so the next
                                # run retries them.
                                _log.warning(
                                    "[person] %s: %s: %s — skipping asset",
                                    aid, type(e).__name__, e,
                                )
                        chunk = chunk_ok
                        if not chunk:
                            continue
                        detections = _yolo.detect(pil_images, conf=yolo_conf)
                        for (asset_id, _ts), dets, pil_img in zip(
                            chunk, detections, pil_images
                        ):
                            # Class-agnostic IoU merge — same as yolo
                            # stage. A photo with overlapping person +
                            # face boxes collapses to one detection.
                            dets = _merge_overlapping_boxes(dets)
                            # YuNet rescue — see the helper. Fires when
                            # this pass found a person box but the
                            # hand-suppression rule would stamp it down.
                            dets = _augment_with_yunet_faces(dets, pil_img)
                            # Keep ONLY person/face/human/hand boxes —
                            # plant boxes for these assets, if any, were
                            # persisted by an earlier run's yolo stage,
                            # and appending plant boxes again would
                            # duplicate those rows. Hand is kept because
                            # has_real_person() cross-checks it to
                            # suppress hand-silhouette false positives.
                            # `face` rows are YuNet rescue output only
                            # (YOLO's face prompt never fires).
                            person_dets = [
                                d for d in dets
                                if d["label"] in PERSON_LABELS
                                and d["label"] != "face"
                                and d["score"] >= PERSON_MIN_CONF
                            ]
                            hand_dets = [
                                d for d in dets
                                if d["label"] == HAND_LABEL
                                and d["score"] >= HAND_MIN_CONF
                            ]
                            face_dets = [
                                d for d in dets if d["label"] == "face"
                            ]
                            if person_dets or hand_dets or face_dets:
                                # Append (no delete) — preserves any
                                # existing plant detections and keeps
                                # their bbox_predictions cache intact
                                # via FK CASCADE.
                                storage.append_detections(
                                    conn, asset_id=asset_id,
                                    detections=person_dets + hand_dets
                                    + face_dets,
                                )
                            # has_real_person applies the hand-suppression
                            # rule — see storage.has_real_person(); the
                            # classifier veto then overrides confident
                            # hand-only photos (p < PERSON_VETO_P).
                            storage.mark_person_scanned(
                                conn,
                                asset_id=asset_id,
                                has_person=apply_person_veto(
                                    has_real_person(dets),
                                    _person_clf_p(conn, asset_id),
                                ),
                            )
                        for im in pil_images:
                            _close_image(im)
                        del pil_images
                        gc.collect()
                    n_total += len(todo_items)
                    elapsed = time.monotonic() - person_t0
                    _log.info(
                        f"[person] page {page_num} "
                        f"items={len(todo_items)}  scanned_so_far={n_total}  "
                        f"elapsed={_fmt_eta(elapsed)}"
                    )
                    # Advance cursor to the last row of this page so a
                    # SIGKILL/OOM between pages resumes where we left off.
                    cursor_ts = todo_items[-1][1]
                    cursor_id = todo_items[-1][0]
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
            # Previews are fetched per chunk at crop time — in memory
            # only. Decoupled from the species-rank cache so cached
            # assets also get bbox classification on the next run.
            all_bbox_targets: list[tuple[str, int, dict]] = []
            for asset in assets:
                if not BBOX_BIOCLIP_ENABLED:
                    break
                dets = storage.fetch_detections(
                    conn, asset_id=asset.id, rank=rank_name
                )
                for d in dets:
                    if d["score"] < BBOX_BIOCLIP_MIN:
                        continue
                    # Skip person/face/human/hand boxes — BioCLIP is
                    # plant-focused, so re-classifying one of those
                    # crops would produce a meaningless Plantae score.
                    # (Hand boxes are persisted for has_person's
                    # suppression cross-check, not for classification.)
                    # Plant crops go through BioCLIP as before.
                    if d["label"] in PERSON_LABELS or d["label"] == HAND_LABEL:
                        continue
                    if storage.bbox_predictions_have_detection(
                        conn,
                        detection_id=d["id"],
                        rank=rank_name,
                        need=BBOX_BIOCLIP_TOP_K,
                    ):
                        continue
                    all_bbox_targets.append((asset.id, d["id"], d))

            if not BBOX_BIOCLIP_ENABLED:
                _log.info(
                    "[bbox-bioclip] disabled by BBOX_BIOCLIP_ENABLED=0 — "
                    "skipping the Linux teacher crop pass; use the Mac box "
                    "backfill to write predictions + bbox_embeddings"
                )

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
                    by_src: dict[str, Image.Image] = {}
                    try:
                        for src, det_id, d in chunk:
                            if src not in by_src:
                                try:
                                    blob = client.download_bytes(
                                        f"/api/assets/{src}/thumbnail"
                                        f"?size={THUMBNAIL_SIZE}"
                                    )
                                    by_src[src] = _decode_preview(blob)
                                except Exception as e:
                                    # Skip this source image's crops; the
                                    # detections stay uncached so the
                                    # next run retries them.
                                    _log.warning(
                                        "[bbox-bioclip] %s: %s: %s — skipping source",
                                        src, type(e).__name__, e,
                                    )
                                    continue
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
                    # The JPEGs are only needed as clf.predict() inputs.
                    # Unlink per-chunk so the scratch dir holds one batch
                    # at a time instead of every crop for the whole run.
                    for cp in crop_paths:
                        try:
                            os.unlink(cp)
                        except OSError:
                            pass
                    _chunk_log(
                        "bbox-bioclip",
                        (i // batch_size) + 1,
                        bbox_total_chunks,
                        n_items=n_crops,
                        elapsed=time.monotonic() - bbox_t0,
                        items_done=j,
                    )
                    conn.commit()  # per-chunk checkpoint (see search-loop note)
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
        # mkdtemp (unlike TemporaryDirectory) never cleans itself up —
        # without this every run leaks a scratch dir of crop jpgs in /tmp.
        # Log (don't swallow) partial-cleanup failures; the boot-time
        # _sweep_stale_scratch() reaps what survives.
        if tmp is not None:
            shutil.rmtree(
                tmp,
                onerror=lambda _f, p, _e: _log.warning(
                    "[cleanup] could not remove scratch entry %s", p
                ),
            )


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
    open-vocab label set, persists person-only boxes via `append_detections`,
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
                todo_items: list[tuple[str, str]] = []
                for asset_id, _fn, _ts in todo:
                    todo_items.append((asset_id, _ts))
                # Chunked YOLO forward pass.
                for i in range(0, len(todo_items), batch_size):
                    if _stop_requested():
                        break
                    chunk = todo_items[i : i + batch_size]
                    pil_images = []
                    chunk_ok = []
                    for aid, _ts in chunk:
                        try:
                            pil_images.append(_decode_preview(
                                client.download_bytes(
                                    f"/api/assets/{aid}/thumbnail"
                                    f"?size={THUMBNAIL_SIZE}"
                                ),
                                _yolo_decode_size(),
                            ))
                            chunk_ok.append((aid, _ts))
                        except Exception as e:
                            _log.warning(
                                "[person-backfill] %s: %s: %s — skipping asset",
                                aid, type(e).__name__, e,
                            )
                    chunk = chunk_ok
                    if not chunk:
                        continue
                    detections = _yolo.detect(pil_images, conf=yolo_conf)
                    for (asset_id, _ts), dets, pil_img in zip(
                        chunk, detections, pil_images
                    ):
                        dets = _merge_overlapping_boxes(dets)
                        # YuNet rescue — see the helper. Fires when this
                        # pass found a person box but the hand-suppression
                        # rule would stamp it down.
                        dets = _augment_with_yunet_faces(dets, pil_img)
                        person_dets = [
                            d for d in dets
                            if d["label"] in PERSON_LABELS
                            and d["label"] != "face"
                            and d["score"] >= PERSON_MIN_CONF
                        ]
                        hand_dets = [
                            d for d in dets
                            if d["label"] == HAND_LABEL
                            and d["score"] >= HAND_MIN_CONF
                        ]
                        face_dets = [
                            d for d in dets if d["label"] == "face"
                        ]
                        if person_dets or hand_dets or face_dets:
                            storage.append_detections(
                                conn, asset_id=asset_id,
                                detections=person_dets + hand_dets + face_dets,
                            )
                        # has_real_person applies the hand-suppression
                        # rule — see storage.has_real_person(); the
                        # classifier veto then overrides confident
                        # hand-only photos (p < PERSON_VETO_P).
                        has_p = apply_person_veto(
                            has_real_person(dets),
                            _person_clf_p(conn, asset_id),
                        )
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
                cursor_ts = todo_items[-1][1]
                cursor_id = todo_items[-1][0]
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


class _PipelineAbort(Exception):
    """Internal exception used to unwind the pipeline cleanly after we
    have already populated error_state() with a user-facing message.
    Callers (run.py CLI, pipeline_worker) catch it so the request or
    worker exits normally instead of surfacing a raw traceback."""


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


def _augment_with_yunet_faces(
    dets: list[dict], pil_img: Image.Image
) -> list[dict]:
    """YuNet rescue for hand-suppressed person photos.

    YOLO-World's "face" prompt never fires in practice, so a photo where
    every person box contains its own hand defeats has_real_person()'s
    has_separate_person hatch and gets suppressed as a hand silhouette —
    even when faces are clearly visible (d844f1f6: three people, zero
    face boxes). YuNet (face.py) finds those faces in ~3 ms at the same
    448px decode this pipeline already holds.

    Gating (both required — keeps YuNet off every other photo and makes
    foliage false positives structurally impossible):
      1. A person box >= PERSON_MIN_CONF already exists, AND
      2. has_real_person() is False (suppression fired, not absence).
    Kept faces must have their CENTER inside a person box — a face floats
    inside its body silhouette in every real-person photo we sampled.

    Returns dets with rescued `face` rows appended (input not mutated).
    `face` is a PERSON_LABELS member, so the appended rows (a) flip
    has_real_person() via its existing has_face hatch and (b) are skipped
    by the bbox-bioclip stage, which excludes PERSON_LABELS crops.
    """
    person_boxes = [
        d for d in dets
        if d["label"] in PERSON_LABELS and d["score"] >= PERSON_MIN_CONF
    ]
    if not person_boxes or has_real_person(dets):
        return dets
    from face import detect_faces

    face_dets = []
    for f in detect_faces(pil_img):
        cx = f["x"] + f["w"] / 2
        cy = f["y"] + f["h"] / 2
        if any(
            p["x"] <= cx <= p["x"] + p["w"]
            and p["y"] <= cy <= p["y"] + p["h"]
            for p in person_boxes
        ):
            face_dets.append(f)
    if not face_dets:
        return dets
    _log.info(
        "[person] yunet rescue: %d face(s) inside person box(es)",
        len(face_dets),
    )
    return dets + face_dets


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