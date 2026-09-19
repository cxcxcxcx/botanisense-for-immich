#!/usr/bin/env python3
"""Deploy-time golden gate for MobileEngine (the BioCLIP-Mobile path).

Runs the FULL engine path — PIL open → preprocess (short-side 256 /
center crop / /255) → ONNX session → cosine against the package's fp32
reference embeddings — for the 8 golden images bundled in the package.
This is the same gate the export scored (fp16-vs-fp32 cos mean
.999993); running it here proves the server-side preprocess + session
wiring before the first pipeline run, not the model itself.

Not on any startup path — run once at setup:

    python scripts/verify_mobile_engine.py [--package data/models/bioclip-mobile-v0]

Exit 0 iff cos mean >= .999 AND min >= .99 (the shipped gate).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from mobile_engine import MobileEngine  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--package", default="data/models/bioclip-mobile-v0")
    ap.add_argument(
        "--min-mean", type=float, default=0.999,
        help="cos mean gate (the export used .999; ANE's looser .995 is "
             "a device-gate concept, not an export gate)",
    )
    args = ap.parse_args()

    pkg = Path(args.package)
    if not pkg.is_absolute():
        pkg = REPO / pkg
    golden = np.load(pkg / "golden.npz", allow_pickle=True)
    imgs = sorted((pkg / "golden").glob("img*.jpg"))
    if len(imgs) != golden["x"].shape[0]:
        sys.exit(
            f"golden images ({len(imgs)}) != golden.npz rows "
            f"({golden['x'].shape[0]})"
        )

    engine = MobileEngine(pkg)
    print(f"engine version: {engine.version_key()}")

    # Also cross-check the engine's own preprocess against the recorded
    # preprocessed tensors — the contract IS the preprocess.
    prep_maxdiff = 0.0
    for p, x_ref in zip(imgs, golden["x"]):
        with Image.open(p) as im:
            x_ours = engine.preprocess(im)
        prep_maxdiff = max(prep_maxdiff, float(np.abs(x_ours - x_ref).max()))
    print(f"preprocess max|Δ| vs golden x: {prep_maxdiff:.4f}")

    feats = engine.create_image_features(
        [Image.open(p) for p in imgs], normalize=True
    )
    ref = golden["emb"].astype(np.float32)
    ref /= np.linalg.norm(ref, axis=1, keepdims=True)
    cos = (feats.numpy() * ref).sum(1)
    print(
        f"cos vs fp32 ref: mean {cos.mean():.6f}  min {cos.min():.6f}  "
        f"per-image {np.round(cos, 5).tolist()}"
    )
    # Exercise the classification head end-to-end (the label-table path
    # the species stage runs) on the first golden image.
    top1 = engine.format_species_probs(
        "probe", engine.create_probabilities(feats)[0], 1
    )[0]
    print(f"img0 top-1 via label table: {top1['species']} ({top1['score']:.4f})")

    ok = cos.mean() >= args.min_mean and cos.min() >= 0.99
    print("GATE:", "PASS ✅" if ok else "FAIL ❌")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
