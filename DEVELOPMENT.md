# Development — how to run & refresh

This project runs as two servers during development. Knowing which one
serves what is the difference between "my change showed up instantly"
and "I've been staring at a stale bundle for an hour."

## The two servers

| Port | Server | Serves | Started by |
|---|---|---|---|
| **8765** | uvicorn (FastAPI) | `app.py` API + `/thumb/*` + the built SPA from `frontend/dist/` | `uvicorn app:app ...` |
| **5173** | Vite dev server | `frontend/src/` source (HMR), proxies `/api` and `/thumb` → 8765 | `npm run dev` in `frontend/` |

Both bind `0.0.0.0`, so on your phone use
`http://192.168.84.113:5173` (dev) or `http://192.168.84.113:8765` (built).

The running uvicorn is launched as:
```
.venv/bin/uvicorn app:app --host 0.0.0.0 --port 8765 \
  --reload-exclude=data/ --reload-exclude=frontend/ \
  --reload-exclude=*.db --reload-exclude=*.log
```

## Refresh cheatsheet

| You changed… | How to see it on **5173** | How to see it on **8765** |
|---|---|---|
| `frontend/src/**` (TSX/CSS) | auto, via HMR | `cd frontend && npm run build` (no restart) |
| `frontend/dist/**` manually | n/a | just refresh — `StaticFiles` reads from disk |
| `app.py` / `*.py` backend | n/a (proxied) | auto, via uvicorn `--reload` |
| `data/state.db` directly via SQL | n/a | **restart uvicorn** — see below |
| New pip dep / env var | n/a | restart uvicorn |

### Frontend on 8765 — build, don't restart

uvicorn mounts `frontend/dist/` with `StaticFiles` and serves
`index.html` with `Cache-Control: no-cache`. So after editing frontend
code:

```bash
cd frontend && npm run build
```

That's it — no uvicorn restart. The built JS/CSS use content hashes in
their filenames, so a plain browser refresh picks up the new bundle.

> ⚠️ The uvicorn command excludes `frontend/` from `--reload`, so
> editing (or rebuilding) the frontend **never** triggers a backend
> restart. This is intentional; don't expect one.

### Frontend on 5173 — HMR

```bash
cd frontend && npm run dev   # Vite on :5173
```

Edits to `src/` hot-reload. If HMR gets wedged (edits don't appear),
restart Vite:

```bash
pkill -f "vite"
cd frontend && nohup npm run dev -- --host 0.0.0.0 --port 5173 > /tmp/vite.log 2>&1 &
```

Use the system `npm`, not `.venv/bin/`. Hard-refresh the browser after
restarting.

### Backend Python — auto-reload

Editing any `*.py` file under the project (except the excluded paths
above) makes uvicorn restart automatically. If you ever need a manual
restart, kill the uvicorn PID and relaunch the command above.

### Direct SQLite writes — restart uvicorn

If you write to `data/state.db` by hand (sqlite3 CLI, external script),
the running uvicorn keeps stale connections/caches. `touch app.py`
does **not** help here because the watcher's `--reload-exclude=data/`
suppresses the data-dir mtime, and touching `app.py` alone is not
reliable. Restart the uvicorn process.

## Which should I use?

- **Actively iterating on UI (components, styles):** open **5173**. HMR
  gives instant feedback. The API/thumb calls are proxied to 8765, so
  the backend still needs to be running.
- **Testing on the phone, backend-only work, or normal use:** open
  **8765** and `npm run build` after each frontend change.
- **Both** can run at the same time on the same machine; they don't
  conflict.

## One-time setup

```bash
# backend
uv sync                                 # or: pip install -e .

# frontend
cd frontend && npm install
```

## Production-ish build

```bash
cd frontend && npm run build     # writes frontend/dist/
cd .. && uvicorn app:app --host 0.0.0.0 --port 8765
```

uvicorn automatically serves `dist/` at `/` when the directory exists
(see the SPA mount at the end of `app.py`). If `dist/` is missing, only
the API is served.
