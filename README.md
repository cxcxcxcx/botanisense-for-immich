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
- **Optional KNN reclassification** against a pre-built FAISS index
  over TreeOfLife-200M (~867k species labels). Index is built
  separately — see [Known limits](#known-limits).
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
              state.db    BioCLIP-2      tol200m_
              (assets,    weights        bioclip2.faiss
              embeddings,  (~5 GB          (~2 GB, optional)
              predictions) resident in
                          child only)
```

A 17 000-photo library is the realistic scale at which this app
shines. Below ~1 000 photos the run-once latency dominates; above
~50 000 the HDBSCAN refit starts to need attention.

---

## Prerequisites

- **Python ≥ 3.10** (3.11+ recommended; `tomllib` is stdlib there)
- **Node.js ≥ 18** (for the frontend build)
- An **Immich instance** with an admin API key — see
  [Immich → Administration → Settings → API Keys][immich-keys]
- **~5 GB free RAM** during a pipeline run (BioCLIP-2 + species
  label table live in the child process)
- **~6 GB free disk** for the BioCLIP-2 weights (~5 GB; downloaded on
  first run into `~/.cache/huggingface/`) and the FastAPI thumb cache
  (`data/thumbs/`)
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

# 1. Copy and edit your config (non-secret settings)
cp config.example.toml config.toml
$EDITOR config.toml     # set [immich].base_url

# 2. Copy and edit your secrets
cp .env.example .env
$EDITOR .env            # set IMMICH_API_KEY

# 3. Install + build
uv sync
cd frontend && npm install && npm run build && cd ..

# 4. Run
uvicorn app:app --host 0.0.0.0 --port 8765

# 5. Open http://localhost:8765
```

For day-to-day development (HMR on the frontend, hot-reload on the
backend), see [DEVELOPMENT.md](DEVELOPMENT.md).

---

## Configuration

Non-secret settings live in `config.toml`; secrets live in `.env`. Both
files are gitignored. The shipped templates (`config.example.toml` and
`.env.example`) document every key.

The merge precedence is **environment variable > config.toml** for
anything you might want to override per-launch (e.g. pointing the same
binary at a different Immich for a one-off test).

| Key | Where | Default | Required? |
|---|---|---|---|
| `IMMICH_BASE_URL` | `config.toml [immich].base_url` or env | _(none)_ | ✅ |
| `IMMICH_API_KEY` | `.env` only | _(none)_ | ✅ |
| `KNN_INVENTORY_DIR` | `config.toml [knn].inventory_dir` or env | _(empty)_ | — |
| `CORS_ORIGINS` | `config.toml [cors].origins` or env | _(empty)_ | — |
| `QUERY` | env | `"intentional plant photos"` | — |
| `N_SAMPLES` | env | `10000` | — |
| `TOP_K` | env | `5` | — |
| `BATCH_SIZE` | env | `16` | — |
| `DB_PATH` | env | `data/state.db` | — |
| `HOST`, `PORT` | env | `0.0.0.0`, `8765` | — |
| `RUN_ON_STARTUP` | env | `0` (off) | — |
| `PLANTAE_GREEN`, `PLANTAE_YELLOW` | env | `0.60`, `0.35` | — |
| `FUNGI_MIN_SCORE_DEFAULT` | env | `0.60` | — |

---

## How it works

BioCLIP-2 ([Stevens et al., CVPR 2024][paper]) is a CLIP-style model
fine-tuned on the TreeOfLife-200M dataset. For each Immich asset, we
run it in zero-shot mode — no per-library training — and rank the
candidate species by cosine similarity to the asset's image embedding.
We surface the top-K (default 5) along with the full taxonomy
hierarchy (kingdom → family → genus → species) and a Plantae-tier
confidence score that drives the per-image green/yellow/red overlay.

If you've built an optional [FAISS][faiss] index over a
TreeOfLife-200M parquet export (`KNN_INVENTORY_DIR` points to it),
you can additionally reclassify any asset against its K nearest
neighbours and see what fraction of those neighbours agree on a given
species. This is dramatically more accurate than zero-shot for
out-of-distribution photos, at the cost of an ~hours-long index build.

[paper]: https://arxiv.org/abs/2311.18803
[faiss]: https://github.com/facebookresearch/faiss

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
- **KNN reclassification is opt-in.** Without a pre-built
  `tol200m_bioclip2.faiss` index, only zero-shot predictions are
  available. Building the index requires a TreeOfLife-200M parquet
  export (large) and ~hours of CPU time.
- **5 GB resident during a pipeline run.** The BioCLIP-2 child
  process holds the weights + label table for the duration; uvicorn
  itself stays at ~250 MB.
- **Tested against Immich ≥ 1.111.** Earlier versions may lack
  `/api/search/smart` and the album endpoints this app depends on.

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