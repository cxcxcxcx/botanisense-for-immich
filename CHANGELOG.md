# Changelog

All notable changes to BotaniSense for Immich are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project does **not** follow Semantic Versioning until v1.0.

## 0.1.0 — 2026-09-15

First public release. The codebase has been under active private
development since the BotaniSense-for-Immich rebrand in mid-2026.

### Features

- **In-app setup — no config files.** The Immich URL and API key are
  entered in the web UI (Welcome screen), stored in the app's SQLite
  DB, and take effect immediately; edit them later via the health
  banner's "Fix connection" or the Run page's Sync section.
  `IMMICH_BASE_URL` / `IMMICH_API_KEY` environment variables still
  work and pin the connection. `config.toml` is gone; an existing
  file is migrated automatically on first boot of the new version
  (base URL → DB, KNN/CORS values → environment).
- **Immich smart-search ingestion.** Runs Immich's `/api/search/smart`
  with a configurable seed query and ingests the top-N matching assets
  (default 10 000).
- **BioCLIP-2 zero-shot classification.** Lazy-loaded, runs in a
  child process so uvicorn never holds the ~5 GB weights.
  Top-K predictions (default 5) per asset, with full taxonomy
  (kingdom → family → genus → species).
- **Hierarchical browse.** Cluster cards (HDBSCAN over BioCLIP
  embeddings, auto-refit on growth), plus two extra levels: a
  **related-clusters** strip on every cluster (nearest clusters by
  centroid cosine) and **super-groups** that merge genuinely-close
  clusters into plant families. The same cards exist per species,
  genus, and family.
- **People-aware photos.** A `has_person` flag drives a filter and
  the album-sync "with/without people" toggle. It combines three
  signals — YOLO-World person/hand boxes, a YuNet face-detection
  rescue pass (catches people the box model misses), and a
  BioCLIP-based classifier veto (a confident `p < 0.05` negative
  suppresses hand-holding-a-plant false positives). Card cover
  photos prefer person-free members; a people photo is used only
  when a whole group has none.
- **Live run updates.** Results stream in while a pipeline run is in
  progress (committed per batch, not once at the end), and the home
  page shows current run status; an optional in-app scheduler
  (`PIPELINE_CRON=HH:MM`) runs the pipeline daily.
- **Optional KNN reclassification.** Against a FAISS index over
  TreeOfLife-200M — download the official imageomics index
  (~20 GB, `scripts/download_knn_index.py`) or build your own from a
  parquet export — see README → "KNN reclassification". A debug /
  understanding tool, not recommended over zero-shot.
- **Immich album sync.** Per-cluster album sync (always-deletes on
  remove; **beta**), smart-album sync for Plantae and Fungi photos,
  and bulk remove.
- **FastAPI backend + React/Vite SPA.** SSE pipeline progress, light
  and dark themes, mobile-responsive, installable as a PWA
  (add-to-home-screen).

### Notes

- Cluster sync is **beta**. Per-cluster `Delete` and bulk `Remove
  synced clusters` always delete the Immich album alongside the local
  mapping (no orphans). Cluster titles may go stale after a cluster
  rename — re-sync to refresh.
- KNN reclassification is **opt-in** and requires an index (downloaded
  official, or self-built). Without it, the app still works for
  browsing and zero-shot predictions.
- Single-user, no auth. See README → "Known limits".

### Credits

- [BioCLIP-2](https://arxiv.org/abs/2311.18803) (Stevens et al.,
  CVPR 2024) — the underlying zero-shot species classifier.
- [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
  — the label set used by BioCLIP-2.
- [Immich](https://immich.app) — the photo library this app reads
  from and writes back to.
- Author: [Xing Chen](https://github.com/cxcxcxcx).