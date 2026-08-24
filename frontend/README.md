# BotaniSense for Immich — frontend

> App name: **BotaniSense for Immich**.
> The ML engine inside it is BioCLIP-2 — don't confuse the two when
> reading this README or `package.json` (name: `botanisense-for-immich-frontend`).

React + Vite + TypeScript SPA. Talks to the FastAPI backend over JSON.

**👉 Full setup, run, and refresh instructions are in
[`../DEVELOPMENT.md`](../DEVELOPMENT.md)** — including the difference
between the Vite dev server (:5173) and the uvicorn-served build
(:8765), and how to see your changes on each.

Quick reference:

```bash
npm install
npm run dev        # Vite on :5173 with HMR (proxies /api, /thumb → :8765)
npm run build      # writes dist/, served by uvicorn at :8765
npm run typecheck  # tsc -b --noEmit
```

The backend must be running on :8765 for the dev server to work (Vite
proxies API and thumbnail requests to it).
