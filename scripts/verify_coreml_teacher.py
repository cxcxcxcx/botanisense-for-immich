#!/usr/bin/env python3
"""Parity check: CoreML BioCLIP-2 teacher encoder vs the PyTorch teacher.

Real crops, not synthetic images: samples plant/flower/tree detections from
a state.db, fetches each preview from a running app's /thumb, crops the box,
and encodes every crop with (a) open_clip fp32 on CPU — the reference — and
(b) the CoreML encoder under ALL and CPU_AND_GPU. Reports per-crop
embedding cosine and species top-1 / top-5 agreement against the 867k
TreeOfLife text table. Runs on macOS in the app's own venv + coremltools.

  python scripts/verify_coreml_teacher.py --db state.db \
      --base-url http://<linux-box>:8775 \
      --encoder encoder_b16.mlmodelc --n 64
"""

from __future__ import annotations

import argparse
import io
import json
import sqlite3
import time
import urllib.request

import numpy as np
import torch
from PIL import Image


def _crops(db: str, base_url: str, n: int, seed: int):
    conn = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    rows = conn.execute(
        "SELECT asset_id, x, y, w, h FROM detections "
        "WHERE label IN ('plant','flower','tree') AND w * h > 0.01"
    ).fetchall()
    rng = np.random.default_rng(seed)
    out = []
    for i in rng.permutation(len(rows)):
        if len(out) >= n:
            break
        aid, x, y, w, h = rows[i]
        try:
            with urllib.request.urlopen(
                f"{base_url}/thumb/{aid}?size=preview", timeout=30
            ) as r:
                im = Image.open(io.BytesIO(r.read())).convert("RGB")
        except Exception as e:  # noqa: BLE001 — skip unreachable previews
            print(f"skip {aid}: {e}", flush=True)
            continue
        W, H = im.size
        box = (int(x * W), int(y * H), max(int((x + w) * W), int(x * W) + 1),
               max(int((y + h) * H), int(y * H) + 1))
        out.append(im.crop(box))
    return out


def _cos(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    return (a * b).sum(1) / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", help="sample crops from this state.db (needs --base-url)")
    ap.add_argument("--base-url")
    ap.add_argument("--crops-dir", help="or: a directory of pre-cut crop images")
    ap.add_argument("--encoder", required=True, help=".mlmodelc or .mlpackage")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--report", default=None, help="write JSON results here")
    ap.add_argument("--write-golden", default=None,
                    help="save the first --golden-n inputs + torch reference "
                         "embeddings as golden.npz (export_teacher_coreml.py --golden)")
    ap.add_argument("--golden-n", type=int, default=32)
    args = ap.parse_args()

    import coremltools as ct
    from bioclip.predict import TreeOfLifeClassifier, preprocess_img

    if args.crops_dir:
        from pathlib import Path

        paths = sorted(Path(args.crops_dir).glob("*.png"))[: args.n]
        crops = [Image.open(p).convert("RGB") for p in paths]
    else:
        crops = _crops(args.db, args.base_url, args.n, args.seed)
    n = len(crops) - len(crops) % args.batch
    crops = crops[:n]
    print(f"crops={n}  numpy={np.__version__}  coremltools={ct.__version__}", flush=True)
    x = torch.stack([preprocess_img(c) for c in crops])

    clf = TreeOfLifeClassifier(device="cpu")
    with torch.no_grad():
        ref = clf.create_image_features(crops, normalize=True).numpy()
    if args.write_golden:
        g = args.golden_n
        np.savez_compressed(args.write_golden, images=x[:g].numpy(), emb=ref[:g])
        print(f"wrote {args.write_golden} ({g} crops)", flush=True)
    txt = clf.get_txt_embeddings().to("mps")
    scale = float(clf.model.logit_scale.exp())
    names = clf.get_current_txt_names()

    def top5(feats: np.ndarray) -> np.ndarray:
        f = torch.from_numpy(feats / np.linalg.norm(feats, axis=1, keepdims=True)).to("mps")
        return torch.topk(scale * f @ txt, 5, dim=1).indices.cpu().numpy()

    ref_top = top5(ref)
    report = {"n": n, "numpy": np.__version__, "coremltools": ct.__version__}
    compiled = args.encoder.rstrip("/").endswith(".mlmodelc")
    for label, cu in (("ALL", ct.ComputeUnit.ALL), ("CPU_GPU", ct.ComputeUnit.CPU_AND_GPU)):
        Model = ct.models.CompiledMLModel if compiled else ct.models.MLModel
        m = Model(args.encoder, compute_units=cu)
        t = time.perf_counter()
        got = np.concatenate([
            np.asarray(m.predict({"images": x[i:i + args.batch].numpy()})["emb"], dtype=np.float32)
            for i in range(0, n, args.batch)
        ])
        dt = time.perf_counter() - t
        c = _cos(got, ref)
        g = top5(got)
        r = {
            "cos_mean": round(float(c.mean()), 6),
            "cos_min": round(float(c.min()), 6),
            "top1_agree": round(float((g[:, 0] == ref_top[:, 0]).mean()), 4),
            "top1_in_ref_top5": round(float(np.mean([g[i, 0] in ref_top[i] for i in range(n)])), 4),
            "img_per_s": round(n / dt, 1),
        }
        flips = [
            (names[ref_top[i, 0]][0][5:], names[g[i, 0]][0][5:])
            for i in range(n) if g[i, 0] != ref_top[i, 0]
        ]
        r["top1_flips"] = flips
        report[label] = r
        print(label, json.dumps(r, ensure_ascii=False), flush=True)
    if args.report:
        with open(args.report, "w") as f:
            json.dump(report, f, indent=1, ensure_ascii=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
