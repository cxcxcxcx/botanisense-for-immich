# Changelog

All notable changes to BotaniSense for Immich are documented here.
The format is loosely based on [Keep a Changelog](https://keepachangelog.com/),
and the project does **not** follow Semantic Versioning until v1.0.

## 0.1.0 — 2026-11-24

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
  (kingdom → family → genus → species). Thumbnails are fetched from
  Immich in parallel during the kingdom, species and detection stages.
- **Bundled plant/people detector.** A trained RTMDet-Ins-tiny model
  (`plant`, `person`, `hand`, `flower`, `tree`; 20.5 MB ONNX, CPU
  onnxruntime) is the default detection backend and ships in the repo
  and the Docker image — nothing to download. It replaces the
  YOLO-World open-vocabulary detector; the `ultralytics` dependency is
  gone and the image shrank from 2.53 to 2.17 GB. Each detected box
  is drawn on its own. See `docs/detector_backends.md`.
- **Similar boxes.** Click a detection box (or its Similar button) in
  the photo detail to see the most similar crops and photos from the
  rest of the library (`/api/bbox/{id}/neighbors`).
- **Hierarchical browse.** Cluster cards (HDBSCAN over BioCLIP
  embeddings, auto-refit on growth), plus two extra levels: a
  **related-clusters** strip on every cluster (nearest clusters by
  centroid cosine) and **super-groups** that merge genuinely-close
  clusters into plant families. The same cards exist per species,
  genus, and family. The near-duplicate threshold that collapses
  burst shots is configurable (`NEAR_DUP_THRESHOLD`, default 0.97).
- **People-aware photos.** A `has_person` flag drives a filter and
  the album-sync "with/without people" toggle. It combines three
  signals — the detector's person/hand boxes, a YuNet face-detection
  rescue pass (catches people the box model misses), and a
  BioCLIP-based classifier veto (a confident `p < 0.05` negative
  suppresses hand-holding-a-plant false positives). Card cover
  photos prefer person-free members; a people photo is used only
  when a whole group has none.
- **Map view.** New `#map` tab plots geotagged assets on an
  OpenStreetMap/Leaflet map, grouped into places by distance alone
  (150 m / 1 km / 10 km, optionally including single-photo places).
  Ported from the Solar app's location clustering. `GET /api/map/clusters`
  returns the place partition under the same filters as `/api/results`.
  A marker click opens a photo sheet over the map (thumbnails, "Load more",
  "Open in grid"); "Open in grid" uses exact membership (`place_id`) so the
  marker count and the grid total agree. Markers are uniform green sized by
  photo count; inferred positions are shown per photo (small amber dot in
  the place sheet), never as a cluster colour. The **Map tab is independent**
  of the Browse filters; the new map pin in a filtered grid opens the same
  component as a **scoped modal**. Switching
  radius keeps the current viewport; a Fit button re-frames. `/api/results`
  also gained `near_lat/near_lng/near_radius` disk filtering. Requires no
  tile API key.
- **Locally inferred locations.** For cameras without built-in GPS, the
  map can interpolate coordinates from GPS-tagged photos within ±15
  minutes / 1 km (`location_guesses` table; anchors recorded for audit).
  Local only: `assets.latitude/longitude` and Immich are never modified,
  real EXIF wins via `COALESCE`, inferred markers are amber, and the map
  panel can recompute or clear the set. Script:
  `scripts/guess_locations.py`.
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
  (add-to-home-screen). New app icon, and an About menu in the header
  (version, GitHub, bug report with the version pre-filled, feedback,
  license). Copy-to-clipboard falls back gracefully on plain-HTTP
  LAN setups.

### Notes

- **Licensing.** The code is MIT. The bundled detector weights
  (`models/obj6_rtmdet_ep97.onnx`) are licensed separately under
  **CC BY-NC 4.0** — see `models/README.md`.
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