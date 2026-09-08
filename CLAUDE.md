# CLAUDE.md — BotaniSense for Immich

Guidance for AI agents (and humans) working in this repo.

## Running the app

The supported way to run BotaniSense is Docker:

```bash
docker compose up --build -d   # app at http://localhost:8765 (binds 0.0.0.0)
```

**Do not start the host dev servers just to "use" or smoke-test the app —
prefer the container.** The host uvicorn (`app:app` on :8765) and Vite dev
server (:5173) exist for active code iteration only; see DEVELOPMENT.md for
the full workflow (HMR, frontend rebuilds, restart rules). Before starting
anything on port 8765, check nothing already owns it (`ss -tlnp | grep 8765`),
and shut dev servers down when you're done iterating.

## Layout

- `app.py` — FastAPI app: API routes, `/thumb/*`, serves the built SPA from `frontend/dist/`
- `pipeline.py` / `pipeline_worker.py` — pipeline stages; a run executes in a
  separate worker process (Popen), never inside uvicorn — BioCLIP needs ~6 GB
- `storage.py` — SQLite state in `data/state.db`
- `immich_client.py` — Immich API client
- `yolo.py` — YOLO person/plant/hand detection
- `knn.py` / `knn_io.py` — optional FAISS KNN index (debug-only; not an accuracy upgrade)
- `frontend/` — React/TSX SPA; `npm run build` outputs to `frontend/dist/`
- `tests/` — pytest suite

## Conventions and invariants

- **Tests:** run `pytest` from the repo root. Tests stub env/config accessors
  (dotenv) — follow the existing fixtures rather than reading real config.
- **Config:** env vars > DB-stored Immich URL/API key (via `immich_base_url()` /
  `immich_api_key()`); the app boots unconfigured.
- **Models load lazily:** never construct `TreeOfLifeClassifier` at module
  import time — keep classifier construction behind `_resolve_classifier()`.
  Eager import silently inflates the server process by ~6 GB.
- **Encode once:** BioCLIP's `clf.predict()` re-encodes images internally;
  encode → persist embedding → classify from features.
- **Raw image previews are never persisted** — Immich is fetched on demand
  (local on typical setups); the DB holds embeddings/scores only.
- **Frontend changes ship as a build:** `cd frontend && npm run build` writes
  `frontend/dist/` (content-hashed); uvicorn serves the new build with no
  restart.
