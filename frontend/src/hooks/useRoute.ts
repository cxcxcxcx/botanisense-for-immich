// Minimal hash-route hook. We don't need a full router — only two
// views (browse / run) — and we want the back/forward buttons to work
// naturally. `window.location.hash` is browser-native, fires
// `hashchange` for free, and shares the same history entry lifecycle
// as the filter query string.

import { useEffect, useState } from 'react';

export type View = 'browse' | 'run';

export function parseHashRoute(hash: string = window.location.hash): View {
  const h = hash.replace(/^#/, '').trim();
  return h === 'run' ? 'run' : 'browse';
}

export function useRoute(): View {
  const [view, setView] = useState<View>(() => parseHashRoute());
  useEffect(() => {
    const onHash = () => setView(parseHashRoute());
    window.addEventListener('hashchange', onHash);
    return () => window.removeEventListener('hashchange', onHash);
  }, []);
  return view;
}

export function navigate(view: View): void {
  if (parseHashRoute() === view) return;
  // Setting hash fires hashchange synchronously; useRoute picks it up.
  window.location.hash = `#${view}`;
}