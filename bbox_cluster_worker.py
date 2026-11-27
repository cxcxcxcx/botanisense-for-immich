"""Box-cluster refit worker: runs clusters.refit_bbox_clusters in its own
process.

Launched by app.py via subprocess.Popen as `python -m bbox_cluster_worker`
with JSON args in the BBOX_CLUSTER_WORKER_ARGS env var (same convention
as pipeline_worker). The refit loads every eligible box + photo vector,
builds a FAISS kNN graph and runs HDBSCAN — hundreds of MB of numpy /
FAISS / allocator arenas that glibc never hands back to the OS. Inside
uvicorn that memory stayed resident after every refit; in a child it
goes away with the process.

The result dict is written as JSON to `result_path`; the parent reads it
after a zero exit. Any failure exits non-zero with the error in the
result file (and the traceback in data/pipeline.log).
"""

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path


def main() -> int:
    args = json.loads(os.environ["BBOX_CLUSTER_WORKER_ARGS"])
    db_path = Path(args["db_path"])
    result_path = Path(args["result_path"])

    # Mirror into data/pipeline.log so the web UI's log panel shows the
    # refit, like it did when the refit ran inside uvicorn.
    handler = logging.FileHandler(db_path.parent / "pipeline.log", encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    log = logging.getLogger("bbox_cluster_worker")
    for name in ("clusters", "bbox_cluster_worker"):
        lg = logging.getLogger(name)
        lg.addHandler(handler)
        lg.setLevel(logging.INFO)

    try:
        import clusters
        import storage

        storage.init_db(db_path)  # idempotent; the app does it at boot too
        with storage.connect(db_path) as conn:
            res = clusters.refit_bbox_clusters(conn, **args["params"])
        result_path.write_text(json.dumps({"ok": True, "result": res}, default=str))
        return 0
    except Exception as e:  # noqa: BLE001 — report every failure to the parent
        log.exception("[bbox-clusters] refit failed")
        result_path.write_text(json.dumps({"ok": False, "error": f"{type(e).__name__}: {e}"}))
        return 1


if __name__ == "__main__":
    sys.exit(main())
