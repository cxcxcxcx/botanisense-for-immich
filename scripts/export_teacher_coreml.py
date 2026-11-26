#!/usr/bin/env python3
"""Export the BioCLIP-2 teacher image encoder to Core ML (macOS only).

Recovered from the 2026-11-17 opencode session that produced
encoder.mlpackage / encoder_b16.mlpackage on the Mac mini (the code was run
inline over ssh and never saved). Same steps, same settings.

Two fixes were needed to make the ViT-L/14 trace convert:
  * coremltools 9's torch frontend requires numpy < 2.4 (use 1.26.4) —
    newer numpy breaks on the trace's aten::Int nodes. This is a converter
    limit only: the app's numpy 2.5 venv loads and runs the result fine.
  * disable PyTorch's fused MHA fastpath before tracing; otherwise the
    graph contains `_native_multi_head_attention`, which coremltools
    cannot convert:
        torch.backends.mha.set_fastpath_enabled(False)

Measured 2026-11-17 on Mac mini M4 (fp16, macOS15 target, 608 MB):
  batch 1,  ALL      20.4 img/s  cos mean .999844 / min .999349
  batch 1,  CPU_GPU  16.6 img/s  cos .999999
  batch 16, ALL      32.8 img/s  (recommended)
  batch 16, CPU_GPU  16.5 img/s  cos mean .999999 / min .999998
  (PyTorch MPS full teacher: 14.4 img/s; Linux CPU: 2.8 img/s)

The gate runs on a real-crop golden.npz (inputs already through the app's
bioclip `preprocess_img`, plus torch fp32 CPU reference embeddings), written
in the app venv by `verify_coreml_teacher.py --write-golden`. Without
--golden the gate falls back to synthetic solid-colour images (the
2026-11-17 check). A manifest.json with versions, sha256 and gate numbers is
written next to the package.

Usage (Mac, venv with coremltools 9 + numpy 1.26.4 + torch 2.7 + open_clip_torch):
  HF_HOME=... python export_teacher_coreml.py --out teacher/encoder_b16.mlpackage \
      --golden golden.npz
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import time
from pathlib import Path

import numpy as np
import torch

HF_MODEL = "hf-hub:imageomics/bioclip-2"
COS_MEAN_GATE = 0.999
COS_MIN_GATE = 0.998


def _sha256_dir(path: Path) -> str:
    h = hashlib.sha256()
    for f in sorted(p for p in path.rglob("*") if p.is_file()):
        h.update(str(f.relative_to(path)).encode())
        with open(f, "rb") as fh:
            for chunk in iter(lambda: fh.read(1 << 20), b""):
                h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True, help="output .mlpackage path")
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--golden", default=None, help="golden.npz (images, emb)")
    ap.add_argument("--bench-iters", type=int, default=20)
    args = ap.parse_args()

    import coremltools as ct
    import open_clip as oc
    from PIL import Image

    torch.backends.mha.set_fastpath_enabled(False)
    model, preprocess = oc.create_model_from_pretrained(
        HF_MODEL, pretrained=None, return_transform=True
    )
    model.eval()
    visual = model.visual.eval()

    b = args.batch
    if args.golden:
        g = np.load(args.golden)
        x, ref = torch.from_numpy(g["images"]), g["emb"]
        gate_set = f"golden:{Path(args.golden).name}"
    else:
        imgs = [
            Image.new("RGB", (224, 224), ((i * 7) % 255, (i * 13) % 255, (i * 29) % 255))
            for i in range(b)
        ]
        x = torch.stack([preprocess(im) for im in imgs])
        with torch.no_grad():
            ref = model.encode_image(x).numpy()
        gate_set = "synthetic"
    n = len(x) - len(x) % b
    x, ref = x[:n], ref[:n]

    with torch.no_grad():
        traced = torch.jit.trace(visual, torch.randn(b, 3, 224, 224))
    torch._C._jit_pass_inline(traced.graph)

    t0 = time.time()
    ml = ct.convert(
        traced,
        inputs=[ct.TensorType(name="images", shape=(b, 3, 224, 224), dtype=np.float32)],
        outputs=[ct.TensorType(name="emb")],
        minimum_deployment_target=ct.target.macOS15,
        compute_precision=ct.precision.FLOAT16,
    )
    ml.save(args.out)
    print(f"saved {args.out} (batch {b}) in {time.time() - t0:.1f} s", flush=True)

    results = {}
    for name, cu in (("ALL", ct.ComputeUnit.ALL), ("CPU_GPU", ct.ComputeUnit.CPU_AND_GPU)):
        m = ct.models.MLModel(args.out, compute_units=cu)
        got = np.concatenate([
            np.asarray(m.predict({"images": x[i:i + b].numpy()})["emb"], dtype=np.float32)
            for i in range(0, n, b)
        ])
        cos = (got * ref).sum(1) / (
            np.linalg.norm(got, axis=1) * np.linalg.norm(ref, axis=1)
        )
        xb = x[:b].numpy()
        for _ in range(5):
            m.predict({"images": xb})
        t = time.perf_counter()
        for _ in range(args.bench_iters):
            m.predict({"images": xb})
        ms = (time.perf_counter() - t) / args.bench_iters * 1000
        r = {
            "cos_mean": round(float(cos.mean()), 6),
            "cos_min": round(float(cos.min()), 6),
            "ms_per_batch": round(ms, 1),
            "img_per_s": round(b * 1000 / ms, 1),
        }
        r["pass"] = r["cos_mean"] >= COS_MEAN_GATE and r["cos_min"] >= COS_MIN_GATE
        results[name] = r
        print(name, json.dumps(r), flush=True)

    out = Path(args.out)
    manifest = {
        "source_model": HF_MODEL,
        "file": out.name,
        "sha256": _sha256_dir(out),
        "batch": b,
        "input": {"name": "images", "shape": [b, 3, 224, 224], "dtype": "float32",
                  "preprocess": "bioclip.predict.preprocess_img"},
        "output": {"name": "emb", "normalized": False},
        # coreml_engine never loads the torch weights; this is the one
        # model parameter the classifier still needs.
        "logit_scale": round(float(model.logit_scale.exp()), 6),
        "precision": "float16",
        "minimum_deployment_target": "macOS15",
        "versions": {"coremltools": ct.__version__, "torch": torch.__version__,
                     "open_clip": oc.__version__, "numpy": np.__version__,
                     "macos": platform.mac_ver()[0], "machine": platform.machine()},
        "gate": {"set": gate_set, "n": n, "cos_mean": COS_MEAN_GATE,
                 "cos_min": COS_MIN_GATE, "results": results},
    }
    (out.parent / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    ok = all(r["pass"] for r in results.values())
    print("GATE", "PASS" if ok else "FAIL", flush=True)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
