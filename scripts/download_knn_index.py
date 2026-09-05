"""Download the official imageomics KNN index (optional feature).

BotaniSense's KNN reclassification needs a FAISS index over BioCLIP-2
embeddings. You can build your own from a TreeOfLife-200M parquet
export (`python -m knn --help`, hours of CPU), or download the one the
imageomics project ships — this script does the latter:

    uv run scripts/download_knn_index.py                # ~20 GB total
    uv run scripts/download_knn_index.py --out /srv/botanisense/knn

It fetches from the HuggingFace model repo
`imageomics/bioclip-image-search-lite` and lays the files out exactly
the way knn_io.py expects:

    <out>/faiss/index.index         ~5.8 GB  (234M embeddings, IVFPQ)
    <out>/duckdb/metadata.duckdb   ~14 GB    (taxonomy + source URLs)

The metadata DuckDB is what maps a neighbor id back to species name
and reference-image URL. `--faiss-only` skips it to save the download,
but expect neighbor resolution to fail without it.

Point the server at the result by setting KNN_INVENTORY_DIR=/abs/path/to/out
(in the environment or compose.yml) and restarting.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ID = "imageomics/bioclip-image-search-lite"
# filename -> approximate size, for the pre-flight download summary
FILES = {
    "faiss/index.index": 5.8,
    "duckdb/metadata.duckdb": 14.0,
}


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Download the official imageomics KNN index."
    )
    ap.add_argument(
        "--out",
        default="knn-index",
        help="target directory (default ./knn-index); absolute paths are "
        "recommended so KNN_INVENTORY_DIR survives container mounts",
    )
    ap.add_argument(
        "--faiss-only",
        action="store_true",
        help="skip the ~14 GB metadata DuckDB (neighbor name/URL "
        "resolution will not work without it)",
    )
    args = ap.parse_args()

    wanted = {
        name: gib
        for name, gib in FILES.items()
        if not (args.faiss_only and name.startswith("duckdb/"))
    }
    if args.faiss_only:
        print(
            "WARNING: --faiss-only skips duckdb/metadata.duckdb, which "
            "knn_io.py needs to resolve neighbor ids to species names and "
            "reference-image URLs. KNN classification will likely fail."
        )

    total = sum(wanted.values())
    print(f"Downloading {len(wanted)} file(s), ~{total:.1f} GB total, from")
    print(f"  https://huggingface.co/{REPO_ID}")
    print(f"into {Path(args.out).resolve()}\n")

    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print(
            "huggingface_hub is not installed. It ships as a dependency "
            "of pybioclip, so inside the project venv `uv sync` is enough; "
            "otherwise: pip install huggingface_hub",
            file=sys.stderr,
        )
        return 1

    for name, gib in wanted.items():
        print(f"[{gib:.1f} GB] {name} ...", flush=True)
        hf_hub_download(
            repo_id=REPO_ID,
            filename=name,
            repo_type="model",
            local_dir=args.out,
        )

    # Post-condition: the exact layout knn_io.py's official-index
    # loader expects.
    missing = [n for n in wanted if not (Path(args.out) / n).exists()]
    if missing:
        print(f"\nERROR: download reported success but missing: {missing}",
              file=sys.stderr)
        return 1

    print(
        f"\nDone. Enable KNN with:\n"
        f"  KNN_INVENTORY_DIR={Path(args.out).resolve()}\n"
        f"then set KNN_INVENTORY_DIR and restart the server."
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
