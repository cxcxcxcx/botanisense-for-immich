#!/usr/bin/env python3
"""Check the bbox-bioclip stage on real crops from a state.db (opened
read-only) whose boxes were filled by scripts/bbox_backfill_mac.py, plus
its Immich.

  python scripts/verify_bbox_encode_once.py --db data/prod/state.db -n 64 \
      --out out/bbox_encode_once/report.json

old:  the previous stage — crop of the 448px draft decode -> JPEG q90 ->
      clf.predict(paths) (re-encodes)
new:  the current stage — crop of the full preview (_BBOX_DECODE_SIZE) ->
      create_image_features -> _classify_features
ref:  the DB's stored bbox_embeddings / bbox_predictions (Mac Core ML)

Reports embedding cosine and species top-1 agreement of old/new vs ref,
and peak RSS. Needs IMMICH_BASE_URL / IMMICH_API_KEY.
"""

from __future__ import annotations

import argparse
import json
import os
import resource
import sqlite3
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import pipeline  # noqa: E402
from immich_client import ImmichClient  # noqa: E402


def _rss_mb() -> float:
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024


def _crop(full, d):
    W, H = full.size
    return full.crop((max(0, int(d["x"] * W)), max(0, int(d["y"] * H)),
                      min(W, int((d["x"] + d["w"]) * W)), min(H, int((d["y"] + d["h"]) * H))))


def _agree(a: dict, b: dict, keys) -> dict:
    top1 = sum(a[k][0]["species"] == b[k][0]["species"] for k in keys)
    in5 = sum(a[k][0]["species"] in {r["species"] for r in b[k]} for k in keys)
    return {"top1": top1 / len(keys), "top1_in_ref_top5": in5 / len(keys)}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", required=True)
    ap.add_argument("-n", type=int, default=64)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    conn = sqlite3.connect(f"file:{args.db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    dets = conn.execute(
        """
        SELECT d.id, d.asset_id, d.x, d.y, d.w, d.h, e.vector, e.model_version
        FROM detections d JOIN bbox_embeddings e ON e.detection_id = d.id
        WHERE d.label IN ('plant', 'flower', 'tree') AND d.score >= 0.25
        ORDER BY d.id DESC LIMIT ?
        """,
        (args.n,),
    ).fetchall()
    keys = [str(d["id"]) for d in dets]
    ref = {str(d["id"]): [dict(r) for r in conn.execute(
        "SELECT species FROM bbox_predictions WHERE detection_id=? AND rank='species' "
        "ORDER BY position", (d["id"],))] for d in dets}
    ref_vecs = np.stack([np.frombuffer(d["vector"], dtype="<f4") for d in dets])

    client = ImmichClient(base_url=os.environ["IMMICH_BASE_URL"],
                          api_key=os.environ["IMMICH_API_KEY"])
    old_crops, new_crops = [], []
    blobs: dict = {}
    for d in dets:
        if d["asset_id"] not in blobs:
            blobs[d["asset_id"]] = client.download_bytes(
                f"/api/assets/{d['asset_id']}/thumbnail?size={pipeline.THUMBNAIL_SIZE}")
        blob = blobs[d["asset_id"]]
        old_crops.append(_crop(pipeline._decode_preview(blob), d))
        new_crops.append(_crop(pipeline._decode_preview(
            blob, max_size=pipeline._BBOX_DECODE_SIZE), d))
    del blobs
    rss_crops = _rss_mb()

    getattr(pipeline, "TreeOfLifeClassifier")  # PEP 562 lazy import hook
    clf = pipeline._resolve_classifier()
    from bioclip import Rank

    rank_enum = Rank.SPECIES
    k = pipeline.BBOX_BIOCLIP_TOP_K
    bs = pipeline.DEFAULT_BATCH_SIZE
    rss_model = _rss_mb()

    new_rows, feats_all = [], []
    for i in range(0, len(new_crops), bs):
        feats = clf.create_image_features(new_crops[i:i + bs], normalize=True)
        new_rows += pipeline._classify_features(clf, feats, keys[i:i + bs],
                                                rank_enum=rank_enum, k=k)
        feats_all.append(feats.detach().cpu().numpy())
    rss_new = _rss_mb()
    new = pipeline._group_predictions(new_rows, k)
    new_vecs = np.concatenate(feats_all)

    with tempfile.TemporaryDirectory(dir=Path(args.out).parent) as td:
        paths = []
        for key, crop in zip(keys, old_crops):
            p = Path(td) / f"bbox_{key}.jpg"
            crop.save(p, format="JPEG", quality=90)
            paths.append(str(p))
        old_rows = []
        for i in range(0, len(paths), bs):
            old_rows += clf.predict(images=paths[i:i + bs], rank=rank_enum, k=k)
        old_by_path = pipeline._group_predictions(old_rows, k)
        old = {key: old_by_path.get(p, []) for key, p in zip(keys, paths)}
        old_vecs = np.concatenate([
            clf.create_image_features([clf.ensure_rgb_image(p) for p in paths[i:i + bs]],
                                      normalize=True).detach().cpu().numpy()
            for i in range(0, len(paths), bs)
        ])

    def _cos(v):
        c = np.einsum("ij,ij->i", v, ref_vecs)
        return {"mean": float(c.mean()), "min": float(c.min())}

    report = {
        "n": len(keys),
        "ref_model_version": sorted({d["model_version"] for d in dets}),
        "crop_px_median": {
            "old": float(np.median([max(c.size) for c in old_crops])),
            "new": float(np.median([max(c.size) for c in new_crops])),
        },
        "cos_vs_ref": {"old": _cos(old_vecs), "new": _cos(new_vecs)},
        "species_vs_ref": {"old": _agree(old, ref, keys), "new": _agree(new, ref, keys)},
        "peak_rss_mb": {"crops_loaded": rss_crops, "model_loaded": rss_model,
                        "after_new_path": rss_new},
    }
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
