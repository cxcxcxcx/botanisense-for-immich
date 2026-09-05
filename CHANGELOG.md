# Changelog

All notable changes to BotaniSense for Immich are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project does **not** follow Semantic Versioning until v1.0.

## 0.1.0 — 2026-08-31

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
- **YOLO-World person detection.** Per-asset flag; pairs with the
  `has_person` filter and the album-sync "with/without people" toggle.
- **HDBSCAN clustering** of BioCLIP embeddings for browse-level
  organisation; auto-refit on growth.
- **Optional KNN reclassification.** Against a FAISS index over
  TreeOfLife-200M — download the official imageomics index
  (~20 GB, `scripts/download_knn_index.py`) or build your own from a
  parquet export — see README → "KNN reclassification".
- **Immich album sync.** Per-cluster album sync (always-deletes on
  remove; **beta**), smart-album sync for Plantae and Fungi photos,
  and bulk remove.
- **FastAPI backend + React/Vite SPA.** SSE pipeline progress, dark
  theme, mobile-responsive.

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