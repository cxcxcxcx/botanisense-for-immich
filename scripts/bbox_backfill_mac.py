#!/usr/bin/env python3
"""Teacher BioCLIP box classification on Apple Silicon (CoreML + MPS).

The Linux pipeline's bbox stage is tied to a run's search results and uses
the PyTorch encoder. This worker is the full-library pass:

  detections(plant/flower/tree) -> preview -> tight crop -> CoreML encoder
  -> top-5 teacher species + 768-d embedding

It is designed to run on a Mac with the exported teacher encoder:
  encoder_b16.mlpackage  (batch 16, CoreML compute unit ALL)
The 867k teacher text table is loaded from the HF cache and evaluated on MPS.

Key Mac export/runtime notes (hard-won):
  * coremltools 9 torch frontend requires numpy < 2.4 (use 1.26.4).
  * tracing must disable the fused MHA fastpath:
        torch.backends.mha.set_fastpath_enabled(False)
  * batch 16 is much faster than batch 1 on the ANE/GPU (32.8 vs 20.4 img/s).

This script intentionally has no dependency on the repo's Python modules or
on pandas/bioclip: it writes the same SQLite tables the Linux app reads.

Usage:
  HF_HOME=/Volumes/X9_APFS/mac_big/home/hf-cache \
  /Volumes/X9_APFS/mac_big/home/bioclip_coreml/.venv/bin/python \
    bbox_backfill_mac.py \
      --db /Volumes/X9_APFS/mac_big/home/botanisense/bbox/state.db \
      --base-url http://127.0.0.1:18771 \
      --encoder /Volumes/X9_APFS/mac_big/home/bioclip_coreml/teacher/encoder_b16.mlpackage \
      --hf-cache /Volumes/X9_APFS/mac_big/home/hf-cache/hub \
      --batch-size 16

macOS Local Network privacy: once the launching ssh session closes, a
nohup'd python (even under tmux) gets EHOSTUNREACH ("No route to host")
for LAN addresses within seconds; curl is unaffected, loopback is exempt.
So reach the app through a reverse tunnel held open from the Linux box:

  ssh -N -o ServerAliveInterval=30 -R 127.0.0.1:18771:127.0.0.1:8771 <mac>

and pass --base-url http://127.0.0.1:18771. A LAN --base-url only works
while the launching ssh session stays attached.
"""

from __future__ import annotations

import argparse
import glob
import io
import itertools
import json
import sqlite3
import sys
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torchvision import transforms

MODEL_VERSION = "teacher:bioclip-2"
LABELS = ("plant", "flower", "tree")
MAX_CONSECUTIVE_FAIL = 20
PREPROCESS = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize((224, 224), antialias=True),
    transforms.Normalize(
        mean=(0.48145466, 0.4578275, 0.40821073),
        std=(0.26862954, 0.26130258, 0.27577711),
    ),
])


def _text_paths(hf_cache: Path) -> tuple[Path, Path]:
    root = hf_cache / "datasets--imageomics--TreeOfLife-200M" / "snapshots"
    npy = sorted(root.glob("*/embeddings/txt_emb_species.npy"))
    js = sorted(root.glob("*/embeddings/txt_emb_species.json"))
    if not npy or not js:
        raise SystemExit(f"TreeOfLife-200M cache not found under {root}")
    return npy[-1], js[-1]


def _load_text(hf_cache: Path):
    npy, js = _text_paths(hf_cache)
    print(f"text embeddings: {npy}", flush=True)
    txt = np.load(npy)
    if txt.ndim != 2 or txt.shape[0] != 768:
        raise SystemExit(f"unexpected text embedding shape {txt.shape}")
    txt_t = torch.from_numpy(np.ascontiguousarray(txt, dtype=np.float32)).to("mps")
    names = json.loads(js.read_text())
    print(f"text table {tuple(txt_t.shape)} names={len(names)}", flush=True)
    return txt_t, names


def _logit_scale() -> float:
    """Read the teacher's learned logit scale without keeping the encoder."""
    import open_clip as oc

    model, _ = oc.create_model_from_pretrained(
        "hf-hub:imageomics/bioclip-2", pretrained=None, return_transform=True
    )
    value = float(model.logit_scale.exp().item())
    del model
    print(f"logit_scale={value:.3f}", flush=True)
    return value


def _name_row(names, idx: int) -> dict:
    row = names[int(idx)]
    sci = row[0] if isinstance(row, list) and row else []
    common = row[1] if isinstance(row, list) and len(row) > 1 else ""
    species = ""
    if len(sci) >= 7:
        species = f"{sci[5]} {sci[6]}".strip()
    return {
        "species": species,
        "common_name": common or "",
        "kingdom": sci[0] if len(sci) > 0 else "",
        "family": sci[4] if len(sci) > 4 else "",
        "genus": sci[5] if len(sci) > 5 else "",
    }


def _db(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db), timeout=60.0)
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=60000")
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS bbox_embeddings (
          detection_id  INTEGER PRIMARY KEY REFERENCES detections(id) ON DELETE CASCADE,
          model_version TEXT NOT NULL,
          dim           INTEGER NOT NULL,
          vector        BLOB NOT NULL,
          computed_at   TEXT NOT NULL DEFAULT (datetime('now'))
        )
        """
    )
    conn.execute(
        "CREATE INDEX IF NOT EXISTS idx_bbox_embeddings_version "
        "ON bbox_embeddings(model_version)"
    )
    return conn


def _select(conn: sqlite3.Connection, min_score: float, force: bool):
    where_cache = "" if force else (
        "AND NOT EXISTS (SELECT 1 FROM bbox_embeddings e "
        "WHERE e.detection_id=d.id)"
    )
    sql = f"""
      SELECT d.id, d.asset_id, d.x, d.y, d.w, d.h, d.label, d.score
      FROM detections d
      WHERE d.label IN ({','.join('?' for _ in LABELS)})
        AND d.score >= ?
        {where_cache}
      ORDER BY d.asset_id, d.score DESC, d.id
    """
    return conn.execute(sql, (*LABELS, min_score)).fetchall()


def _preview(base_url: str, asset_id: str, retries: int = 5) -> Image.Image:
    """Network errors and 5xx retry with backoff (2+4+8+16+32 s); a 4xx
    from the server is a real per-asset answer and is not retried. 5xx is
    retried because the thumb path is a proxy onto Immich — an Immich
    restart surfaces here as a burst of 500s, not as a dead socket
    (2026-11-20: Immich restarted mid-run and aborted the pass at 72%).
    A persistent EHOSTUNREACH is the macOS Local Network block (see the
    module docstring), which no retry will fix — the abort catches it."""
    url = f"{base_url.rstrip('/')}/thumb/{urllib.parse.quote(asset_id, safe='')}?size=preview"
    for attempt in range(retries + 1):
        try:
            with urllib.request.urlopen(url, timeout=60) as resp:
                blob = resp.read()
            return Image.open(io.BytesIO(blob)).convert("RGB")
        except urllib.error.HTTPError as exc:
            if exc.code < 500 or attempt == retries:
                raise
        except (urllib.error.URLError, OSError):
            if attempt == retries:
                raise
        time.sleep(2 ** (attempt + 1))


def _write_batch(
    conn: sqlite3.Connection,
    batch: list[tuple[int, torch.Tensor]],
    model,
    txt_t: torch.Tensor,
    names,
    logit_scale: float,
    batch_size: int,
) -> int:
    """Run CoreML on one batch and write predictions + embeddings."""
    n = len(batch)
    tensors = [t for _, t in batch]
    # The CoreML graph has a static batch-16 shape. Pad the tail with the
    # last crop and slice the outputs; this avoids a second batch-1 model.
    if n < batch_size:
        tensors = tensors + [tensors[-1]] * (batch_size - n)
    inp = torch.stack(tensors).numpy()
    out = np.asarray(model.predict({"images": inp})["emb"], dtype=np.float32)[:n]
    norm = np.linalg.norm(out, axis=1, keepdims=True)
    feats = out / np.maximum(norm, 1e-12)

    feats_t = torch.from_numpy(feats).to("mps")
    logits = logit_scale * (feats_t @ txt_t)
    probs = torch.softmax(logits, dim=1)
    top = torch.topk(probs, k=5, dim=1)

    rows = []
    emb_rows = []
    for i, (det_id, _) in enumerate(batch):
        for pos, (idx, score) in enumerate(zip(top.indices[i], top.values[i])):
            meta = _name_row(names, int(idx))
            rows.append((
                det_id, "species", pos,
                meta["species"], meta["common_name"], float(score),
                meta["kingdom"], meta["family"], meta["genus"],
            ))
        emb_rows.append((
            det_id, MODEL_VERSION, feats.shape[1],
            feats[i].tobytes(),
        ))

    conn.executemany(
        """
        INSERT OR REPLACE INTO bbox_predictions
          (detection_id, rank, position, species, common_name, score,
           kingdom, family, genus)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    conn.executemany(
        """
        INSERT OR REPLACE INTO bbox_embeddings
          (detection_id, model_version, dim, vector, computed_at)
        VALUES (?, ?, ?, ?, datetime('now'))
        """,
        emb_rows,
    )
    conn.commit()
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--db", required=True)
    ap.add_argument("--base-url", default="http://192.168.84.113:8771")
    ap.add_argument("--encoder", required=True)
    ap.add_argument("--hf-cache", default="/Volumes/X9_APFS/mac_big/home/hf-cache/hub")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-score", type=float, default=0.25)
    ap.add_argument("--limit", type=int, default=0, help="0 = all detections")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    import coremltools as ct

    torch.backends.mha.set_fastpath_enabled(False)
    txt_t, names = _load_text(Path(args.hf_cache))
    logit_scale = _logit_scale()

    # An .mlpackage is compiled on load into the system temp dir on the
    # internal disk (TMPDIR is ignored), which fails once that disk is
    # nearly full (2026-11-23: 405 MB free vs ~580 MB of weights). A
    # precompiled .mlmodelc — `xcrun coremlcompiler compile
    # encoder_b16.mlpackage <dir>` — loads in place and skips that step.
    if args.encoder.rstrip("/").endswith(".mlmodelc"):
        model = ct.models.CompiledMLModel(
            args.encoder, compute_units=ct.ComputeUnit.ALL
        )
    else:
        model = ct.models.MLModel(args.encoder, compute_units=ct.ComputeUnit.ALL)
    conn = _db(Path(args.db))
    rows = _select(conn, args.min_score, args.force)
    if args.limit > 0:
        rows = rows[: args.limit]
    print(
        f"bbox_backfill: detections={len(rows)} model_version={MODEL_VERSION} "
        f"encoder={args.encoder} batch={args.batch_size}",
        flush=True,
    )
    if not rows:
        return 0

    started = time.monotonic()
    done = 0
    written = 0
    queue: list[tuple[int, torch.Tensor]] = []
    current_asset = None
    current_preview = None
    failed_assets = 0
    consecutive_fail = 0
    exit_code = 0

    try:
        for asset_id, group in itertools.groupby(rows, key=lambda r: r[1]):
            group = list(group)
            try:
                current_preview = _preview(args.base_url, asset_id)
            except Exception as exc:
                failed_assets += 1
                consecutive_fail += 1
                print(f"[asset] {asset_id}: {type(exc).__name__}: {exc} — skipped", flush=True)
                if consecutive_fail >= MAX_CONSECUTIVE_FAIL:
                    # The server is gone, not this asset. Stop so a rerun
                    # (cache-aware, no --force) resumes instead of the run
                    # "finishing" with most of the library skipped.
                    print(f"abort: {consecutive_fail} assets failed in a row", flush=True)
                    exit_code = 2
                    break
                continue
            consecutive_fail = 0
            W, H = current_preview.size
            for det_id, _aid, x, y, w, h, _label, _score in group:
                x0 = max(0, int(float(x) * W))
                y0 = max(0, int(float(y) * H))
                x1 = min(W, int((float(x) + float(w)) * W))
                y1 = min(H, int((float(y) + float(h)) * H))
                if x1 <= x0 or y1 <= y0:
                    continue
                crop = current_preview.crop((x0, y0, x1, y1))
                queue.append((int(det_id), PREPROCESS(crop)))
                if len(queue) >= args.batch_size:
                    written += _write_batch(
                        conn, queue[: args.batch_size], model, txt_t,
                        names, logit_scale, args.batch_size,
                    )
                    del queue[: args.batch_size]
                    done = written
                    elapsed = max(time.monotonic() - started, 1e-3)
                    rate = done / elapsed
                    eta = (len(rows) - done) / rate if rate > 0 else 0
                    print(
                        f"[{done}/{len(rows)}] {rate:.1f} img/s "
                        f"eta={eta/60:.1f}m failed_assets={failed_assets}",
                        flush=True,
                    )
            current_preview = None
    finally:
        if queue:
            written += _write_batch(
                conn, queue, model, txt_t, names, logit_scale, args.batch_size
            )
        conn.close()

    print(
        f"done: {written} crops, {(time.monotonic()-started)/60:.1f}m, "
        f"failed_assets={failed_assets}",
        flush=True,
    )
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
