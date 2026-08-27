# Changelog

All notable changes to BotaniSense for Immich are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project does **not** follow Semantic Versioning until v1.0.

## 0.1.0 — 2026-08-27

First public release. The codebase has been under active private
development since the BotaniSense-for-Immich rebrand in mid-2026.

### Features

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
- **Optional KNN reclassification.** Against a pre-built
  `tol200m_bioclip2.faiss` index over TreeOfLife-200M. Index is built
  by the operator (separate job) — see "Known limits".
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
- KNN reclassification is **opt-in** and requires a separately-built
  FAISS index. Without it, the app still works for browsing and
  zero-shot predictions.
- Single-user, no auth. See README → "Known limits".

### Credits

- [BioCLIP-2](https://arxiv.org/abs/2311.18803) (Stevens et al.,
  CVPR 2024) — the underlying zero-shot species classifier.
- [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
  — the label set used by BioCLIP-2.
- [Immich](https://immich.app) — the photo library this app reads
  from and writes back to.
- Author: [Xing Chen](https://github.com/cxcxcxcx).