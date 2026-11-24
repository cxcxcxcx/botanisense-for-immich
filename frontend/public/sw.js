// Minimal service worker for BotaniSense. Strategy:
//   - precache the document shell (/) so the install dialog works
//     offline once cached on first visit
//   - network-first for everything else, falling back to cache
//
// We don't aggressively precache the JS/CSS chunks because Vite's
// content-hashed filenames bust the cache on every deploy; the
// runtime cache fills in once the user actually visits pages, and
// the next reload is served from cache.
//
// NOTE: this is intentionally minimal. A real PWA with full
// offline support would precache the route shell + offline page,
// handle fetch errors gracefully, and version the cache key.

const CACHE = 'botanisense-shell-v2';
const SHELL_PATHS = ['/', '/favicon.ico'];

self.addEventListener('install', (event) => {
  event.waitUntil(
    caches.open(CACHE).then((cache) => cache.addAll(SHELL_PATHS)),
  );
  // Activate immediately so the first install takes effect without a
  // second page load.
  self.skipWaiting();
});

self.addEventListener('activate', (event) => {
  event.waitUntil(
    caches.keys().then((keys) =>
      Promise.all(keys.filter((k) => k !== CACHE).map((k) => caches.delete(k))),
    ),
  );
  self.clients.claim();
});

self.addEventListener('fetch', (event) => {
  const req = event.request;
  if (req.method !== 'GET') return;
  const url = new URL(req.url);
  // Don't intercept API calls — uvicorn's response stream would
  // confuse the cache (and stale API responses are worse than no
  // response). Same for /thumb/* which routes through the FastAPI
  // proxy and returns per-asset images.
  if (url.pathname.startsWith('/api') || url.pathname.startsWith('/thumb')) {
    return;
  }
  // Network-first for navigation + assets; cache what comes back so
  // a subsequent offline launch can render the shell.
  event.respondWith(
    fetch(req)
      .then((res) => {
        if (res && res.status === 200) {
          const clone = res.clone();
          caches.open(CACHE).then((cache) => cache.put(req, clone));
        }
        return res;
      })
      .catch(() => caches.match(req)),
  );
});