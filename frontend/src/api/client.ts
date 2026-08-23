// Thin wrapper around fetch that:
//   1. parses JSON,
//   2. throws an ApiError on non-2xx,
//   3. lets 409 propagate as a typed error so callers can render it inline
//      ("pipeline already running") instead of crashing.
//
// API base URL — important. In dev we run Vite on 5173 with a proxy to
// uvicorn (8765), but Vite's http-proxy buffers SSE chunked responses,
// which backpressures uvicorn's generator when a pipeline run is
// emitting state events. That wedges the click-Run request behind the
// SSE proxies in Vite's accept queue. Fix: when we're loaded from the
// Vite dev server, send /api/* directly to uvicorn, bypassing Vite's
// proxy entirely. Prod (uvicorn-served) keeps relative paths so the
// build artifact is portable. Detected via window.location.port: Vite
// runs on 5173 by default; if you change the dev port, update the
// match or set VITE_API_BASE in a .env.

const API_BASE: string =
  // explicit override via Vite's import.meta.env (set in .env.local)
  (import.meta.env.VITE_API_BASE as string | undefined) ??
  // dev heuristic: Vite serves on 5173 by default
  (typeof window !== 'undefined' && window.location?.port === '5173'
    ? `${window.location.protocol}//${window.location.hostname}:8765`
    : '');

export class ApiError extends Error {
  status: number;
  detail: string;
  constructor(status: number, detail: string) {
    super(`HTTP ${status}: ${detail}`);
    this.status = status;
    this.detail = detail;
    this.name = 'ApiError';
  }
}

export interface FetchOptions extends Omit<RequestInit, 'body'> {
  body?: unknown;
  query?: Record<string, string | number | boolean | undefined | null>;
}

function buildUrl(path: string, query?: FetchOptions['query']): string {
  // API_BASE only applies to /api/* paths — let other relative paths
  // (static assets, thumbnails) stay on the same origin.
  const base = path.startsWith('/api') ? API_BASE : '';
  const full = base + path;
  if (!query) return full;
  const sp = new URLSearchParams();
  for (const [k, v] of Object.entries(query)) {
    if (v === undefined || v === null || v === '') continue;
    sp.set(k, String(v));
  }
  const qs = sp.toString();
  return qs ? `${full}?${qs}` : full;
}

export async function fetchJson<T>(path: string, opts: FetchOptions = {}): Promise<T> {
  const url = buildUrl(path, opts.query);
  const init: RequestInit = {
    method: opts.method ?? 'GET',
    headers: {
      'Accept': 'application/json',
      ...(opts.body !== undefined ? { 'Content-Type': 'application/json' } : {}),
      ...(opts.headers ?? {}),
    },
    ...(opts.body !== undefined ? { body: JSON.stringify(opts.body) } : {}),
  };
  const res = await fetch(url, init);
  if (!res.ok) {
    let detail = res.statusText;
    try {
      const j = await res.json();
      detail = j?.detail ?? j?.error ?? JSON.stringify(j);
    } catch {
      /* fall through to statusText */
    }
    throw new ApiError(res.status, detail);
  }
  if (res.status === 204) return undefined as T;
  return res.json() as Promise<T>;
}