#!/usr/bin/env python3
"""Smoke test for ENGINE=auto/coreml on macOS: resolve the classifier the
way a pipeline run does, then classify a directory of crops end to end
(create_image_features → create_probabilities → format_species_probs) and
time it. Prints the [model] log lines, the engine class and a few top-1s.

  ENGINE=auto python scripts/smoke_coreml_engine.py --crops-dir out/coreml_parity/crops
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--crops-dir", required=True)
    ap.add_argument("--n", type=int, default=128)
    args = ap.parse_args()

    import torch
    from PIL import Image

    import pipeline

    # Materialize the lazy import the way pipeline_worker does, so the
    # ENGINE=teacher fallback path can resolve it.
    pipeline.TreeOfLifeClassifier  # noqa: B018
    t0 = time.monotonic()
    clf = pipeline._resolve_classifier()
    print(f"engine={type(clf).__name__} load={time.monotonic() - t0:.1f}s", flush=True)

    paths = sorted(Path(args.crops_dir).glob("*.png"))[: args.n]
    crops = [Image.open(p).convert("RGB") for p in paths]
    t = time.perf_counter()
    with torch.no_grad():
        feats = clf.create_image_features(crops, normalize=True)
        probs = clf.create_probabilities(feats, clf.get_txt_embeddings())
    dt = time.perf_counter() - t
    print(f"{len(crops)} crops in {dt:.1f}s = {len(crops) / dt:.1f} img/s "
          f"(encode + 867k-label softmax)", flush=True)
    for i in range(3):
        top = clf.format_species_probs(str(i), probs[i], k=1)[0]
        print(paths[i].name, top.get("genus"), top.get("species"),
              round(float(top.get("score", 0)), 3), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
