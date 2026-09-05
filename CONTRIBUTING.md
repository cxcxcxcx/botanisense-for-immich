# Contributing

Thanks for your interest in BotaniSense for Immich.

## Development setup

```bash
# Backend (Python ≥3.10)
uv sync

# Frontend (Node ≥18)
cd frontend && npm install && cd ..
```

Run both servers during development (see `DEVELOPMENT.md` for the
full two-server dance):

```bash
# Terminal 1 — backend
uvicorn app:app --host 0.0.0.0 --port 8765

# Terminal 2 — frontend (Vite dev server with HMR + /api proxy)
cd frontend && npm run dev
```

## Running tests

```bash
uv run pytest tests/ -x
cd frontend && npm run typecheck && npm run build
```

Tests live in `tests/` (FastAPI `TestClient` + storage-level
regression tests). `uv run pytest tests/ -x` is what CI runs; the dev
dependency group provides pytest. New features should ship with at
least one regression test.

## Commit messages

Match the existing imperative mood: a subject line that reads as a
command ("Add …", "Fix …", "Drop …"). Examples from history:

```
Run page: restore the in-page KnnAlert banner, remove header KnnPill
Card: drop non-Plantae kingdom chips ("Fungi 0%" etc.)
Rebrand app to BotaniSense for Immich
```

Scope prefix (Browse / Run / Card / etc.) is optional but helpful.

## Pull request checklist

- [ ] Tests pass (`uv run pytest tests/ -x` and `npm run typecheck`)
- [ ] Frontend builds (`cd frontend && npm run build`)
- [ ] No new secrets in code or committed files (`.env` and `data/`
      are gitignored; never commit them)
- [ ] No hardcoded personal defaults (LAN IPs, filesystem paths,
      absolute `/media/...` mounts) — new settings flow through env
      vars (see `.env.example`) or the web-UI connection form
- [ ] If adding a new env var, update both `.env.example` and the
      "Configuration" table in `README.md`

## What not to commit

The following are gitignored but it's worth restating:

- `data/state.db`, `data.state.db`, `*.db` — runtime SQLite files
  containing photo metadata and Immich query history
- `data/thumbs/` — cached Immich thumbnails (~1.4M files for a17k library)
- `weights/` — downloaded ML model weights (~5 GB)
- `frontend/dist/` — built frontend assets
- `.env` — Immich API key / env-pinned settings
- `data/state.db` — SQLite DB (also holds the UI-saved connection)

If you accidentally `git add` one of these, `git rm --cached` will
untrack without deleting the working copy.

## Reporting issues

Use the GitHub issue templates (`.github/ISSUE_TEMPLATE/bug.yml` and
`feature.yml`). For suspected security bugs, see `SECURITY.md` —
**do not** file a public issue.