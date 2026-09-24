# Fetch-once redesign (plan — NOT implemented)

Status: design accepted 2026-09-23, implementation deferred. Species/yolo
download parallelization (`fetch_thumbs`) + fp32 engine already landed
(species 4.8→8.5/s, yolo 9→11.3/s); this plan covers the remaining
structural win. Read this before touching `run_pipeline` stage loops.

## Problem

Every stage downloads the same thumbnail bytes again: species (decode 256)
→ yolo (decode 960) → kingdom (256, when active). Same URL
(`?size=preview`), same bytes — only the decode size differs. Measured
2026-09-23: end-to-end species 4.8/s vs 22/s pure inference; downloads were
~80% of wall time before parallelization, ~15% of yolo time after.

## Target: one fetch, many models, dispose

Per chunk, in order:

1. Fetch thumbnail bytes ONCE per asset (`fetch_thumbs` already parallel).
2. Decode per consumer size (256 BioCLIP, 960 yolo) from the SAME bytes.
3. Run BioCLIP batch + YOLO batch (+ kingdom fallback encode off the 256).
4. Persist everything this chunk owns (embeddings, predictions, detections,
   marks). Drop all PILs and bytes. Next chunk.

Stages stop being full-library sweeps with private fetch loops; they become
passes over per-chunk shared fetches. The chunk is the unit of download AND
of disposal — no run-scoped byte cache (rejected: 10K×100KB ≈ 1GB resident;
per-chunk disposal keeps memory flat).

## Stage-by-stage changes (`pipeline.py`)

- `run_pipeline`: replace sequential species → … → yolo sweeps with one
  fused loop. Compute the union work-list per chunk from existing gates
  (`needs_bioclip`, yolo `todo`, kingdom fallback set).
- species/kingdom/yolo fetch blocks → single `fetch_thumbs` call per chunk
  (helper exists; yolo passes `max_size=_yolo_decode_size()`).
- Keep per-stage chunk commits and resume marks EXACTLY as today
  (`mark_yolo_scanned`, `mark_person_scanned`, chunk_idx): a fused loop
  that loses resume correctness replays whole-library runs (the v2
  25-images incident class). Out of scope: search (460/s metadata, not a
  bottleneck), person/bbox-bioclip/cluster/stitch (consume stage outputs,
  need originals/crops — separate design).
- `fetch_thumbs` stays the only fetch path; no second fetcher.

## Risks (why this waited)

1. Resume-mark semantics across a fused loop — the highest-blast-radius
   part. Any doubt → keep stages separate.
2. Chunk-size coupling: species likes 16–64, yolo@960 likes smaller on CPU.
   Fused chunk must satisfy both or split internally.
3. Mixed failure modes per chunk (BioCLIP ok + YOLO dead asset) must keep
   per-asset containment (warn + skip, no rows).

## Validation before merge

1. `pytest` full suite green (repo rule, no exceptions).
2. Small-DB trial (≤500 assets, fresh shadow DB): kill -9 mid-run twice,
   verify resume completes with zero recompute of finished chunks
   (compare `n_new`/`n_cached` + row counts).
3. Rate check: species ≥12/s, yolo ≥12/s sustained on the deploy box;
   container CPU ≥600% during fused chunks (else still IO-bound).
4. `ab_compare` spot-run: fused-run DB vs sequential-run DB on shared
   assets — predictions must match (same bytes, same models).

## Non-goals

- GPU inference, yolo input-size reduction, fp16-vs-fp32 (done separately).
- Search-phase parallelization (metadata-bound, 460/s, leave it).
- Changing what is persisted (embeddings/scores/marks schema untouched).
