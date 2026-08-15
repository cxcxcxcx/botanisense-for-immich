# Immich × BioCLIP — frontend

React + Vite + TypeScript SPA. Talks to the FastAPI backend over JSON.

## Dev (two terminals)

```bash
# Terminal A — backend on :8765
uv run app.py

# Terminal B — frontend on :5173 (proxies /api and /thumb to :8765)
cd frontend
npm install
npm run dev
```

Open http://localhost:5173.

## Prod

```bash
cd frontend
npm ci
npm run build      # writes frontend/dist/
cd ..
uv run app.py      # serves dist/ at / and the API at /api/*
```

Open http://localhost:8765.