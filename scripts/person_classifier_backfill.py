"""Backfill has_person using the BioCLIP classifier.

Two phases:
  Phase A — Embed the unembedded rows that aren't Plantae candidates
            (these are the rows YOLO said no-person on but BioCLIP was
            never run on them). Use the existing thumbs in data/thumbs/.

  Phase B — Apply classifier flips:
              - p >= 0.5  AND current hp=0  →  hp=1
            (down-flip not implemented yet; needs more labels first)

Optionally limit to a candidate set with --filter-no-plantae.

Usage:
  PYTHONPATH=. python -u scripts/person_classifier_backfill.py [--dry-run]
        (use `python -u` to disable stdout buffering for live progress)
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import yaml
from PIL import Image

# Force unbuffered output even when piped
sys.stdout.reconfigure(line_buffering=True)
sys.stderr.reconfigure(line_buffering=True)

import storage


DB = Path("data/state.db")
MODEL_PATH = Path("data/person_classifier.joblib")
THUMB_DIR = Path("data/thumbs")


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def get_unembedded(conn, *, exclude_plantae: bool) -> list[tuple[str, str]]:
    """asset_id, original_file_name for rows without embeddings."""
    sql = """
        SELECT a.asset_id, a.original_file_name
        FROM assets a
        WHERE a.asset_id NOT IN (SELECT asset_id FROM embeddings)
    """
    if exclude_plantae:
        sql += """
          AND NOT EXISTS (
            SELECT 1 FROM taxon_scores t
            WHERE t.asset_id=a.asset_id AND t.taxon_name='Plantae' AND t.score>=0.5
          )
        """
    sql += " ORDER BY a.first_seen_at"
    return [(r[0], r[1]) for r in conn.execute(sql).fetchall()]


def has_thumb(aid: str) -> bool:
    return (THUMB_DIR / f"{aid}.jpg").exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--filter-no-plantae", action="store_true", default=False,
                    help="Skip Plantae candidates (default: False — the classifier "
                         "needs to see them too. Some Plantae photos are actually "
                         "hand-with-plant or person-in-garden photos that the "
                         "Plantae score misclassified; without re-embedding those, "
                         "the classifier can't overrule YOLO.)")
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit how many rows to embed (for testing)")
    args = ap.parse_args()

    blob = joblib.load(MODEL_PATH)
    coef = blob["coef"]
    intercept = float(blob["intercept"])

    with storage.connect(DB) as conn:
        todo = get_unembedded(conn, exclude_plantae=args.filter_no_plantae)
        if args.limit:
            todo = todo[:args.limit]
        n_total = len(todo)
        n_with_thumb = sum(1 for aid, _ in todo if has_thumb(aid))
        print(f"Phase A: {n_total} rows without embeddings", flush=True)
        if args.filter_no_plantae:
            print(f"  (excluding Plantae candidates)", flush=True)
        print(f"  with thumb on disk: {n_with_thumb}", flush=True)
        if not todo:
            return

        if args.dry_run:
            print("\n--dry-run: not running BioCLIP, not writing", flush=True)
            return

        # Lazy import: BioCLIP is heavy.
        print("\nLoading BioCLIP TreeOfLifeClassifier...", flush=True)
        from bioclip.predict import TreeOfLifeClassifier
        clf = TreeOfLifeClassifier()
        print("  loaded\n", flush=True)

        # Phase A: embed unembedded rows
        print(f"=== Phase A: embed {n_total} rows (batch_size={args.batch_size}) ===\n",
              flush=True)
        t0 = time.monotonic()
        embedded = 0
        skipped = 0
        # Commit every N batches so progress survives crashes
        CHECKPOINT_EVERY = 16
        last_checkpoint_embedded = 0
        for i in range(0, n_total, args.batch_size):
            chunk = todo[i:i+args.batch_size]
            # Resolve thumb paths, drop missing
            pil_images: list[Image.Image] = []
            kept: list[tuple[str, str]] = []
            for aid, fn in chunk:
                p = THUMB_DIR / f"{aid}.jpg"
                if not p.exists():
                    skipped += 1
                    continue
                try:
                    im = Image.open(p).convert("RGB")
                    pil_images.append(im)
                    kept.append((aid, fn))
                except Exception:
                    skipped += 1
            if not pil_images:
                continue
            feats = clf.create_image_features(pil_images, normalize=True)
            feats_np = feats.detach().cpu().numpy().astype("<f4")
            for (aid, _fn), vec in zip(kept, feats_np):
                storage.upsert_embedding(conn, asset_id=aid, vector=vec)
                embedded += 1
            # free memory
            for im in pil_images:
                try: im.close()
                except Exception: pass
            del feats, feats_np

            # Commit every CHECKPOINT_EVERY batches so progress is durable
            if (embedded - last_checkpoint_embedded) >= CHECKPOINT_EVERY * args.batch_size:
                conn.commit()
                last_checkpoint_embedded = embedded
                elapsed = time.monotonic() - t0
                rate = embedded / elapsed if elapsed > 0 else 0
                eta = (n_total - embedded) / rate if rate > 0 else 0
                print(f"  [{embedded}/{n_total}] "
                      f"elapsed {_fmt(elapsed)} rate={rate:.1f}/s eta={_fmt(eta)}",
                      flush=True)
        # Final commit at the end of the with-block
        print(f"\nEmbedded {embedded} rows in {_fmt(time.monotonic()-t0)} ({skipped} skipped)",
              flush=True)

        # Reload embeddings table after the commits
        all_ids, matrix = storage.all_embeddings(conn)
        id_to_idx = {a: i for i, a in enumerate(all_ids)}
        p = sigmoid(matrix @ coef + intercept)
        model_p = {a: float(p[i]) for i, a in enumerate(all_ids)}

        # Phase B: apply flips on rows where the classifier strongly says yes
        # but current has_person = 0
        cur_hp = {r[0]: r[1] for r in conn.execute(
            "SELECT asset_id, has_person FROM assets"
        ).fetchall()}

        to_flip = [a for a in all_ids
                   if cur_hp.get(a) == 0 and model_p.get(a, 0.0) >= 0.5]

        new_aids = {aid for aid, _ in todo}
        n_new_flips = sum(1 for a in to_flip if a in new_aids)

        print(f"\n=== Phase B: classifier flips on has_person=0 ===", flush=True)
        print(f"Total classifier flips (incl. from earlier runs): {len(to_flip)}", flush=True)
        print(f"  of which NEW embeddings from this run: {n_new_flips}", flush=True)

        if not to_flip:
            print("No flips to apply.", flush=True)
            return

        # Apply in batches
        BATCH = 500
        for i in range(0, len(to_flip), BATCH):
            chunk = to_flip[i:i+BATCH]
            qmarks = ",".join("?" * len(chunk))
            conn.execute(
                f"UPDATE assets SET has_person = 1 WHERE asset_id IN ({qmarks})",
                chunk,
            )
            print(f"  applied {min(i+BATCH, len(to_flip))}/{len(to_flip)}", flush=True)

        # Invalidate /api/results cache on the running uvicorn worker so the
        # flipped rows show up immediately (no restart needed).
        _post_cache_invalidate()

    print(f"\nDone. Flipped {len(to_flip)} rows from has_person=0 to 1.", flush=True)


def _post_cache_invalidate(base_url: str = "http://127.0.0.1:8765"):
    """POST /api/cache/invalidate so the running API worker drops its
    in-process results cache. Uses urllib to avoid a hard dependency on
    requests. Honors PORT env var if set.
    """
    import os
    port = os.environ.get("PORT", "8765")
    base_url = f"http://127.0.0.1:{port}"
    import json
    import urllib.request
    try:
        req = urllib.request.Request(
            f"{base_url}/api/cache/invalidate?prefix=all:",
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=5) as r:
            body = json.loads(r.read().decode("utf-8"))
            print(f"  cache invalidated: {body}", flush=True)
    except Exception as e:
        print(f"  WARN: cache invalidate failed: {type(e).__name__}: {e}",
              flush=True)
        print(f"  (the running uvicorn worker will still serve stale data "
              f"until restarted)", flush=True)


def _fmt(s: float) -> str:
    if s < 60:
        return f"{s:.1f}s"
    if s < 3600:
        return f"{int(s//60)}m{int(s%60):02d}s"
    return f"{int(s//3600)}h{int(s%3600)//60:02d}m"


if __name__ == "__main__":
    main()