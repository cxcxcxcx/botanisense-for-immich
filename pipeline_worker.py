"""Pipeline worker: the process that actually runs a pipeline.

Launched by app.py via subprocess.Popen (NOT multiprocessing) as
`python -m pipeline_worker` with its JSON args in the
PIPELINE_WORKER_ARGS env var (env, not argv: the args carry the Immich
API key, and argv is world-readable via /proc/<pid>/cmdline) — a plain
exec'd interpreter, no multiprocessing bootstrap. Every freshly-exec'd
python process benches the BioCLIP encode at full torch parallelism
(~0.33 s/img on 8 threads here), while children born from uvicorn's
multiprocessing spawn run the identical encode pinned to ~1 core
(~2.0 s/img) with the OpenMP pool spinning uselessly — regardless of
start method, uid, cgroup, rlimits, or import order. This module
exists so the run child is structurally the thing that works. See the
commit that introduced it for the diagnosis trail.

Stops arrive as SIGTERM from /api/run/stop (the parent no longer
shares an mp.Event with us). The handler arms pipeline.request_stop()
and the run unwinds at the next batch boundary through the normal
per-batch-commit path, so no work is lost — same semantics as the
old shared event.
"""

from __future__ import annotations

import json
import logging
import os
import signal
import sys
from pathlib import Path


def main() -> None:
    args = json.loads(os.environ["PIPELINE_WORKER_ARGS"])
    log = logging.getLogger("pipeline_worker")

    # Weights already cached? Skip HF Hub entirely. The load normally
    # takes ~9s from cache, but with the hub unreachable it hangs for
    # many minutes on unauthenticated retries — seen live as a run
    # frozen at "[model] loading BioCLIP-2" for 12+ minutes.
    _hf_cache = Path(
        os.environ.get("HF_HOME", "/app/data/huggingface")
    ) / "hub" / "models--imageomics--bioclip-2"
    if _hf_cache.is_dir():
        os.environ["HF_HUB_OFFLINE"] = "1"

    def _on_sigterm(signum, frame):  # noqa: ANN001
        # Don't raise from here: the handler may fire while native
        # torch/sqlite frames are on the stack. Arming the cooperative
        # flag unwinds at the next batch boundary instead.
        import pipeline

        log.info("[worker] SIGTERM — stopping at next batch boundary")
        pipeline.request_stop()

    signal.signal(signal.SIGTERM, _on_sigterm)

    import pipeline
    import storage

    pipeline.set_status_file(Path(args["status_path"]))
    # Route pipeline's own INFO lines ([search], [yolo], ...) into the
    # per-job file. A named handler here can't catch them — pipeline
    # logs under its own logger, and this exec'd process has no
    # inherited logging config to lean on.
    pipeline.attach_log_file(Path(args["log_path"]))
    # ALSO mirror into data/pipeline.log — the file /api/logs and the
    # SSE snapshot tail for the web UI's log panel. The old forked
    # child inherited the parent's handler for that file; this exec'd
    # worker doesn't, and without the mirror the panel goes silent
    # during runs.
    mirror = logging.FileHandler(
        Path(args["db_path"]).parent / "pipeline.log", encoding="utf-8"
    )
    mirror.setFormatter(
        logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S")
    )
    pipeline._log.addHandler(mirror)

    # Startup forensics: if the encode crawls again, this is the first
    # place to look — it prints what torch believes about its thread
    # pool and any env vars that could pinch it.
    import torch

    pipeline._log.info(
        "[worker] env: %s",
        {k: v for k, v in os.environ.items()
         if any(s in k.upper() for s in ("OMP", "MKL", "THREAD", "BLAS",
                                         "NUMEXPR", "GOMP", "KMP"))},
    )
    pipeline._log.info("[worker] torch parallel_info:\n%s",
                       torch.__config__.parallel_info())

    # Encode a handful of on-disk thumbs right here: if this comes out
    # at the bench number (~0.35 s/img on 8 threads), the encode path
    # is fine in this process and any crawl lives in the surrounding
    # pipeline work (downloads, decode, SQLite). If it crawls, the
    # process itself is throttled.
    try:
        import time as _time
        from PIL import Image as _Image

        _thumbs = sorted(Path("/app/data/thumbs").rglob("*.jpg"))[:8]
        if _thumbs:
            # Go through the module attr (PEP 562 hook) so the lazy
            # import registers into pipeline.globals, then construct
            # via _resolve_classifier so the SELF-BENCH instance IS the
            # run's instance. Two instances = ~12 GB RSS = swap death.
            torch.set_num_threads(min(8, os.cpu_count() or 8))
            getattr(pipeline, "TreeOfLifeClassifier")
            _clf = pipeline._resolve_classifier()
            _imgs = [_Image.open(p).convert("RGB") for p in _thumbs]
            _t0 = _time.time()
            _clf.create_image_features(_imgs, normalize=True)
            _dt = _time.time() - _t0
            pipeline._log.info(
                "[worker] self-bench: %d fresh encodes in %.2fs "
                "(%.2f s/img)",
                len(_thumbs), _dt, _dt / len(_thumbs),
            )
    except Exception:
        pipeline._log.exception("[worker] self-bench failed")

    try:
        # A SIGTERM during boot (imports / self-bench) arms the stop
        # flag while status is still "idle" — run_pipeline's
        # reset_state() would clear it and the run would proceed at
        # full speed until the parent's SIGKILL watchdog. Honor the
        # stop here instead. (The microseconds-wide race between this
        # check and reset_state is accepted; boot is ~10s, the window
        # is not.)
        if pipeline.is_stopped():
            pipeline._log.warning("[worker] stop requested during startup")
            pipeline.stopped_state("stopped by user (during startup)")
            log.info("[worker] stopped during startup")
            return
        pipeline.run_pipeline(
            base_url=args["base_url"],
            api_key=args["api_key"],
            query=args["query"],
            n_samples=args["n_samples"],
            top_k=args["top_k"],
            db_path=args["db_path"],
            classifier=None,
            batch_size=args["batch_size"],
            stop_criteria_pct=args.get("stop_criteria_pct"),
            stop_event=None,
        )
        # Cluster names may have changed (HDBSCAN re-assigns cluster_ids
        # on incremental new points). Drop the cached names in SQLite
        # so the next /api/clusters call rebuilds them.
        try:
            with storage.connect(args["db_path"]) as conn:
                storage.invalidate_cluster_names(conn)
        except Exception:
            log.exception("[worker] cluster-name invalidation failed")
        log.info("[worker] finished cleanly")
    except pipeline._PipelineStop:
        # Cooperative stop (SIGTERM → request_stop → check_stop at a
        # batch boundary). run_pipeline does NOT catch _PipelineStop
        # itself — its caller is expected to write the terminal state.
        # If we only log here, the status file still says "running" when
        # this process exits, and app.py's dead-process reaper rewrites
        # it to "worker died unexpectedly" — a clean user stop showing
        # up in the UI as a red crash. Write stopped_state ourselves.
        pipeline._log.warning("pipeline stopped by user")
        pipeline.stopped_state("stopped by user")
        log.info("[worker] stopped by request")
    except Exception as e:
        log.exception("[worker] run failed")
        try:
            pipeline.error_state(f"{type(e).__name__}: {e}")
        except Exception:
            pass


if __name__ == "__main__":
    main()
