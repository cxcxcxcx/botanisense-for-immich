# BotaniSense for Immich

> **Browse your Immich library by plant species.**
> BioCLIP-2 zero-shot classification + optional KNN reclassification,
> with smart-album sync back to Immich.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Immich compat ≥ 1.111](https://img.shields.io/badge/Immich-%E2%89%A5%201.111-blue)](https://immich.app)
[![Python ≥ 3.10](https://img.shields.io/badge/Python-%E2%89%A5%203.10-blue)](https://www.python.org)

BotaniSense for Immich is a self-hosted companion to
[Immich](https://immich.app) that classifies plant (and other
organism) photos using [BioCLIP-2](https://arxiv.org/abs/2311.18803),
then helps you organise the results — by cluster, species, genus, or
family — and optionally push curated albums back into Immich.

The ML engine inside is BioCLIP-2 (a CLIP model fine-tuned on the
TreeOfLife-200M label set). The app is "BotaniSense for Immich";
keep the two names straight when reading the code, comments, or this
README — see [DEVELOPMENT.md](DEVELOPMENT.md) for the dual-naming
convention.

---

## Features

- **Immich smart-search ingestion.** Run a seed query (e.g. "red
  flowers", "moss on rocks"); the app pulls the top-N matching assets
  from Immich's `/api/search/smart`.
- **BioCLIP-2 zero-shot species prediction.** Top-K (default 5) species
  per asset, with full taxonomy. Lazy-loaded in a child process — your
  backend process never holds the ~5 GB model weights.
- **YOLO-World person detection.** Per-asset flag (`has_person`),
  drives both a filter and the "with/without people" album-sync toggle.
- **HDBSCAN clustering** of BioCLIP embeddings for browse-level
  organisation; auto-refit on library growth.
- **Optional KNN reclassification** against a FAISS index over
  TreeOfLife-200M (~867k species labels) — download the official
  imageomics index or build your own; see
  [KNN reclassification](#knn-reclassification-optional).
- **Immich album sync.** Sync photos matching a Plantae or Fungi
  threshold as smart albums; per-cluster album sync (beta); bulk
  remove.
- **FastAPI backend + React/Vite SPA.** Live SSE pipeline progress,
  per-job status panels, dark theme, mobile-responsive.

---

## Architecture

```
Immich (REST)               React/Vite SPA
     │  /api/search/smart         │
     │  /api/albums, etc.         │  fetchJson → /api/*
     ▼                            ▼
     ┌─────── FastAPI backend (uvicorn :8765) ──────────┐
     │                                                   │
     │   storage.py      pipeline.py        knn.py       │
     │   (SQLite)        (BioCLIP-2 child    (FAISS,     │
     │                    process)           optional)   │
     └───────────────────────────────────────────────────┘
                  │             │             │
                  ▼             ▼             ▼
              state.db    BioCLIP-2      FAISS index
              (assets,    weights        (optional;
              embeddings, (~4.3 GB       download or
              predictions) in child)     build)
```

A 17 000-photo library is the realistic scale at which this app
shines. Below ~1 000 photos the run-once latency dominates; above
~50 000 the HDBSCAN refit starts to need attention.

---

## Prerequisites

- **Docker** (with Compose v2) — the only install path
- An **Immich instance** with an admin API key — see
  [Immich → Administration → Settings → API Keys][immich-keys]
- **~5 GB free RAM** during a pipeline run (BioCLIP-2 + species
  label table live in the child process)
- **~5 GB free disk** for the BioCLIP-2 artifacts (~4.3 GB: model
  weights + species label table; downloaded on first run into
  `data/huggingface/`) plus the FastAPI thumb cache (`data/thumbs/`)
- The BioCLIP-2 model itself, from
  [`imageomics/bioclip-2`][bioclip-hf] — downloaded automatically on
  first use

[immich-keys]: https://immich.app/docs/administration/users-and-access
[bioclip-hf]: https://huggingface.co/imageomics/bioclip-2

---

## Quickstart

```bash
git clone https://github.com/cxcxcxcx/botanisense-for-immich
cd botanisense-for-immich

# 1. Build + run
docker compose up --build -d

# 2. Open http://<this-host>:8765 — from any LAN device; the server
#    binds 0.0.0.0. (http://localhost:8765 works only on the host
#    itself.)
#    → the Welcome screen asks for your Immich URL and API key.
#      Click "Save & test connection" — it verifies immediately.
```

No `.env`, no config files. The Immich URL + API key you enter are
stored in the app's database (`./data/`) and take effect immediately;
you can change them any time via **Fix connection** in the health
banner or the **Edit** link on the Run page → Sync to IMMICH section.

Don't want to build locally? Uncomment the `image:` line in
`compose.yml` to pull the prebuilt image from GHCR instead
(published by CI on every release). Prefer not to type the key into
the web UI? Set `IMMICH_BASE_URL` / `IMMICH_API_KEY` in `compose.yml`
or a `.env` file instead — environment variables pin the connection.

The image contains the backend and the built web UI (~2 GB; CPU-only
torch). BioCLIP-2 downloads into `./data/` on the first pipeline run.
To enable the optional KNN reclassification, see
[KNN reclassification](#knn-reclassification-optional).

Contributing or developing from source (Python + Node)? See
[DEVELOPMENT.md](DEVELOPMENT.md).

---

## Configuration

Two surfaces, both optional:

1. **The web UI** — the Immich URL + API key, stored in the app's
   SQLite DB. This is the default path and needs no files.
2. **Environment variables** — for automation, or to pin values so
   UI edits can't change them (e.g. Immich reached over a Docker
   network hostname). Compose and `.env` land here. Precedence for
   the Immich connection is **environment > web UI**.

| Key | Default | Notes |
|---|---|---|
| `IMMICH_BASE_URL` | _(none — set in the web UI)_ | Where the server reaches Immich |
| `IMMICH_API_KEY` | _(none — set in the web UI)_ | Immich admin API key |
| `IMMICH_PUBLIC_URL` | = `IMMICH_BASE_URL` | Browser-side Immich URL for "Open in Immich" |
| `KNN_INVENTORY_DIR` | _(empty; KNN off)_ | Directory holding a KNN index |
| `CORS_ORIGINS` | _(any origin)_ | Comma-separated; empty = same-origin only |
| `QUERY` | `"intentional plant photos"` | Seed query |
| `N_SAMPLES` | `10000` | Max assets per run |
| `TOP_K` | `5` | Species predictions per asset |
| `BATCH_SIZE` | `16` | Inference batch size |
| `DB_PATH` | `data/state.db` | SQLite location |
| `HOST`, `PORT` | `0.0.0.0`, `8765` | Server bind |
| `RUN_ON_STARTUP` | `0` (off) | Run the pipeline once at boot |
| `PIPELINE_CRON` | _(off)_ | Daily pipeline run at HH:MM |
| `PLANTAE_GREEN`, `PLANTAE_YELLOW` | `0.60`, `0.35` | Confidence tier thresholds |
| `FUNGI_MIN_SCORE_DEFAULT` | `0.60` | Fungi sync threshold |

`IMMICH_PUBLIC_URL` matters in Docker setups where the server reaches
Immich over a hostname the browser can't resolve (e.g. both in one
compose network): the server keeps using `IMMICH_BASE_URL`, while
"Open in Immich" links use the public URL. `PIPELINE_CRON` (HH:MM)
runs the pipeline daily at that time. `CORS_ORIGINS` takes a
comma-separated origin list; set it empty for same-origin only.

---

## How it works

BioCLIP-2 ([Stevens et al., CVPR 2024][paper]) is a CLIP-style model
fine-tuned on the TreeOfLife-200M dataset. For each Immich asset, we
run it in zero-shot mode — no per-library training — and rank the
candidate species by cosine similarity to the asset's image embedding.
We surface the top-K (default 5) along with the full taxonomy
hierarchy (kingdom → family → genus → species) and a Plantae-tier
confidence score that drives the per-image green/yellow/red overlay.

### KNN reclassification (optional)

Zero-shot gets a photo to the right genus most of the time but can
miss the species. KNN mode closes that gap: each asset is matched
against a FAISS index of ~234M BioCLIP-2 reference embeddings, and the
species labels of the nearest neighbours vote on the answer —
dramatically more accurate for out-of-distribution photos.

Two ways to get an index:

1. **Download the official imageomics index (recommended).**

   ```bash
   # Run inside the container (huggingface_hub is already installed
   # there) so the index lands in the persistent ./data volume. ~20 GB.
   docker compose exec botanisense python scripts/download_knn_index.py \
       --out /app/data/knn-index
   ```

   then add `KNN_INVENTORY_DIR: /app/data/knn-index` to the
   `environment:` block in `compose.yml` and `docker compose up -d`
   again. (Alternatively download to any host directory and bind-mount
   it read-only at `/app/knn-index` — see the commented block in
   `compose.yml`.)

2. **Build your own** from a TreeOfLife-200M parquet export with
   `python -m knn` — hours of CPU. The expected layout contract is
   `tol200m_bioclip2.faiss` + `labels.parquet` + `INDEX_VERSION.txt`
   (see `knn_io.py`).

Without an index the feature is simply absent: the UI shows zero-shot
predictions only, and the KNN code paths never load. The Data tools
card on the Run page shows a short note when no index is configured —
zero-shot species accuracy is noticeably weaker than KNN.

[paper]: https://arxiv.org/abs/2311.18803

The frontend is a single-page React/Vite app. State is held in a
small Context + useReducer; server-driven data (results, clusters,
status, logs) is fetched per-component, with one SSE connection
(`/api/events`) for live progress.

---

## Known limits

- **Single-user, no auth.** Suitable for self-hosted LAN use. Don't
  expose the port to the public internet without adding auth in front
  (reverse proxy with basic-auth is a workable minimum).
- **Cluster sync is beta.** Per-cluster `Delete` and bulk `Remove
  synced clusters` always delete the Immich album alongside the local
  mapping (no orphans). Cluster titles may go stale after a cluster
  rename — re-sync to refresh.
- **KNN reclassification is opt-in.** Without an index (downloaded
  official, ~20 GB, or self-built — see
  [KNN reclassification](#knn-reclassification-optional)), only
  zero-shot predictions are available.
- **5 GB resident during a pipeline run.** The BioCLIP-2 child
  process holds the weights + label table for the duration; uvicorn
  itself stays at ~250 MB.
- **Tested against Immich ≥ 1.111.** Earlier versions may lack
  `/api/search/smart` and the album endpoints this app depends on.
- **No telemetry.** The app phones home to nobody — no update checks,
  no analytics, no crash reporting.

---

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for setup, test, and commit
guidelines. Be sure to read the "What not to commit" list — there are
some large runtime artifacts (`data/`, `weights/`, `data.state.db`)
that are easy to accidentally `git add`.

---

## Security

See [SECURITY.md](SECURITY.md). Please do **not** file public issues
for suspected security bugs — email the maintainer directly.

---

## License

[MIT](LICENSE) © 2026 Xing Chen.

The BioCLIP-2 model weights are subject to their own license; see the
[imageomics/bioclip-2 model card][bioclip-hf].

---

## Credits

- [BioCLIP-2](https://arxiv.org/abs/2311.18803) (Stevens et al., CVPR
  2024) — the zero-shot species classifier that powers this app.
- [TreeOfLife-200M](https://huggingface.co/datasets/imageomics/TreeOfLife-200M)
  — the label set used by BioCLIP-2.
- [Immich](https://immich.app) — the photo library this app reads
  from and writes back to.
- Author: [Xing Chen](https://github.com/cxcxcxcx).