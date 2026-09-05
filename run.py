"""CLI: run the Immich smart-search + BioCLIP pipeline, persisting results.

Reads from .env (via python-dotenv) for IMMICH_BASE_URL, IMMICH_API_KEY,
N_SAMPLES (default 10000), TOP_K (default 5), DB_PATH (default data/state.db).

Usage:
    uv run run.py                       # run with defaults
    uv run run.py --run 7               # print a previous run from the DB
    N_SAMPLES=200 uv run run.py         # override count
    QUERY="red flowers" uv run run.py   # override query
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

import storage

DEFAULT_QUERY = "intentional plant photos"


def main() -> int:
    load_dotenv()
    # Import the heavy pipeline (pulls in torch/bioclip) only when we're
    # actually about to run, so `uv run run.py --run <id>` (read-only DB
    # dump) doesn't pay the cost.
    from pipeline import results_to_dicts, run_pipeline

    ap = argparse.ArgumentParser()
    ap.add_argument("--query", default=os.environ.get("QUERY", DEFAULT_QUERY))
    ap.add_argument("--db", default=os.environ.get("DB_PATH", "data/state.db"))
    ap.add_argument("--n", type=int, default=int(os.environ.get("N_SAMPLES", "10000")))
    ap.add_argument("--k", type=int, default=int(os.environ.get("TOP_K", "5")))
    ap.add_argument(
        "--batch-size",
        type=int,
        default=int(os.environ.get("BATCH_SIZE", "16")),
        help="Images per chunk for BioCLIP / YOLO inference (lower = less memory).",
    )
    ap.add_argument(
        "--run",
        type=int,
        default=None,
        help="Print results for a past run id instead of running the pipeline.",
    )
    ap.add_argument(
        "--reclassify-knn",
        action="store_true",
        help="After the regular pipeline, KNN-reclassify each asset in the "
             "run that has an embedding. Uses the cached BioCLIP image vector "
             "(no re-embed). Requires --inventory-dir.",
    )
    ap.add_argument(
        "--reclassify-all-knn",
        action="store_true",
        help="Skip Immich search entirely; KNN-reclassify every asset in the "
             "DB that has an embedding. Useful for one-shot reclassification "
             "after building a new index.",
    )
    ap.add_argument(
        "--knn-k",
        type=int,
        default=int(os.environ.get("KNN_K", "10")),
        help="K for the KNN vote (default 10).",
    )
    ap.add_argument(
        "--inventory-dir",
        default=os.environ.get("KNN_INVENTORY_DIR", ""),
        help="Directory containing tol200m_bioclip2.faiss + labels.parquet "
             "+ INDEX_VERSION.txt (or the official imageomics layout).",
    )
    args = ap.parse_args()

    if args.run is not None:
        storage.init_db(args.db)
        with storage.connect(args.db) as conn:
            run_row = storage.get_run(conn, args.run)
            if run_row is None:
                print(f"Run #{args.run} not found in {args.db}", file=sys.stderr)
                return 1
            results = storage.get_run_results(conn, run_id=args.run)
        _print_results(
            run_id=run_row["id"],
            query=run_row["query"],
            n_new=run_row["n_new"],
            n_cached=run_row["n_cached"],
            started_at=run_row["started_at"],
            finished_at=run_row["finished_at"],
            dicts=results,
        )
        return 0

    if args.reclassify_all_knn:
        # Skip Immich search entirely. Useful for one-shot reclassification
        # after building a new index. Just KNN every asset with an embedding.
        storage.init_db(args.db)
        from pipeline import run_knn_reclassify
        with storage.connect(args.db) as conn:
            ids = [
                r["asset_id"]
                for r in conn.execute(
                    "SELECT DISTINCT asset_id FROM embeddings"
                ).fetchall()
            ]
        if not ids:
            print(
                f"[reclassify-all-knn] no assets with embeddings in {args.db}",
                file=sys.stderr,
            )
            return 1
        print(
            f"[reclassify-all-knn] KNN over {len(ids)} assets  "
            f"k={args.knn_k}  inventory={args.inventory_dir}"
        )
        with storage.connect(args.db) as conn:
            stats = run_knn_reclassify(
                conn,
                asset_ids=ids,
                index_dir=__import__("pathlib").Path(args.inventory_dir),
                k=args.knn_k,
            )
        print(
            f"[reclassify-all-knn] done  "
            f"requested={stats['requested']}  cached={stats['cached']}  "
            f"scanned={stats['scanned']}  version={stats['index_version']}"
        )
        return 0

    base_url = os.environ.get("IMMICH_BASE_URL", "")
    api_key = os.environ.get("IMMICH_API_KEY")
    if not base_url or not api_key:
        print(
            "IMMICH_BASE_URL / IMMICH_API_KEY missing — set both in the "
            "environment (.env works too), or configure them once in the "
            "web app, which stores them in the same SQLite DB.",
            file=sys.stderr,
        )
        return 1

    print(
        f"[search] POST {base_url}/api/search/smart  query={args.query!r}  "
        f"size={args.n}"
    )
    try:
        out = run_pipeline(
            base_url=base_url,
            api_key=api_key,
            query=args.query,
            n_samples=args.n,
            top_k=args.k,
            db_path=args.db,
            batch_size=args.batch_size,
        )
    except Exception as e:
        # run_pipeline raises _PipelineAbort after populating error_state()
        # for user-facing failures (e.g. Immich 400). Print the message
        # instead of a stack trace so the CLI is usable.
        from pipeline import _PipelineAbort
        if isinstance(e, _PipelineAbort):
            print(f"error: {e}", file=sys.stderr)
            return 1
        raise

    # Optional KNN reclassify step. Runs after the regular pipeline so
    # we know every asset in the run has at least its cached BioCLIP
    # embedding (zero-shot predictions may be cached too).
    if args.reclassify_knn:
        from pathlib import Path as _Path
        from pipeline import run_knn_reclassify
        run_assets = [r.asset_id for r in out.results]
        if run_assets:
            print(
                f"[reclassify-knn] KNN over {len(run_assets)} assets  "
                f"k={args.knn_k}  inventory={args.inventory_dir}"
            )
            with storage.connect(args.db) as conn:
                stats = run_knn_reclassify(
                    conn,
                    asset_ids=run_assets,
                    index_dir=_Path(args.inventory_dir),
                    k=args.knn_k,
                )
            print(
                f"[reclassify-knn] done  "
                f"requested={stats['requested']}  cached={stats['cached']}  "
                f"scanned={stats['scanned']}  version={stats['index_version']}"
            )

    _print_results(
        run_id=out.run_id,
        query=out.query,
        n_new=out.n_new,
        n_cached=out.n_cached,
        started_at=None,
        finished_at=None,
        dicts=results_to_dicts(out.results),
    )
    return 0


def _print_results(
    *,
    run_id: int,
    query: str,
    n_new: int,
    n_cached: int,
    started_at: str | None,
    finished_at: str | None,
    dicts: list[dict],
) -> None:
    header = f"Run #{run_id}: query={query!r}  new={n_new}  cached={n_cached}"
    if started_at:
        header += f"  started={started_at}"
    if finished_at:
        header += f"  finished={finished_at}"
    print(header)
    n_with_dets = sum(1 for r in dicts if r.get("detections"))
    if n_with_dets:
        total_dets = sum(len(r.get("detections") or []) for r in dicts)
        header += f"  detections={n_with_dets} images, {total_dets} boxes"
    print(header)
    print()
    print(f"=== {len(dicts)} images, top-5 species ===\n")
    for r in dicts:
        print(f"• {r['original_file_name']}   ({r['asset_id']})")
        print(f"    original : {r['original_url']}")
        print(f"    thumb    : {r['thumbnail_url']}")
        if r["top_predictions"]:
            top1 = r["top_predictions"][0]
            print(
                f"    kingdom  : {top1.get('kingdom') or '?'}  "
                f"family={top1.get('family') or '?'}  "
                f"genus={top1.get('genus') or '?'}"
            )
        print("    top species:")
        for p in r["top_predictions"]:
            cn = p.get("common_name") or ""
            tail = f"  ({cn})" if cn else ""
            print(f"        {p['score']:.4f}  {p['species']}{tail}")
        print()


if __name__ == "__main__":
    raise SystemExit(main())