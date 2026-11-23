// Minimal hash-route hook. We don't need a full router — three views
// (browse-list / browse-content / run) — and we want the back/forward
// buttons to work naturally. `window.location.hash` is browser-native,
// fires `hashchange` for free, and shares the same history entry
// lifecycle as the filter query string.
//
//   #browse            → list of clusters / species / genus / family
//                        (filter cards, full-height)
//   #browse/content    → photo grid for the currently active filter
//                        (a cluster/taxon was selected on the list)
//   #map               → geotagged photos grouped into places on an
//                        OpenStreetMap view
//   #run               → params + status + logs (pipeline)

import { useEffect, useState } from 'react';
import { BOTANISENSE_ENABLED } from '../lib/features';

export type View = 'browse-list' | 'browse-content' | 'map' | 'run' | 'botanisense';

export type BrowseTab = 'browse-list' | 'browse-content';

/**
 * Parse the hash into a structured view.
 * `#browse/content` → 'browse-content'
 * `#browse`         → 'browse-list'
 * `#map`            → 'map'
 * `#run`            → 'run'
 * anything else     → 'browse-list' (default)
 */
export function parseHashRoute(hash: string = window.location.hash): View {
  const h = hash.replace(/^#/, '').trim();
  if (h === 'run') return 'run';
  if (h === 'map') return 'map';
  if (BOTANISENSE_ENABLED && h === 'botanisense') return 'botanisense';
  if (h === 'browse/content' || h === 'browse/content/') return 'browse-content';
  return 'browse-list';
}

/** Top-level view for the header tab strip. */
export function topLevel(view: View): BrowseTab | 'map' | 'run' | 'botanisense' {
  if (view === 'run') return 'run';
  if (view === 'map') return 'map';
  if (view === 'botanisense') return 'botanisense';
  return 'browse-list';
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
  const next = view === 'browse-list' ? '#browse'
             : view === 'browse-content' ? '#browse/content'
             : view === 'map' ? '#map'
             : view === 'botanisense' ? '#botanisense'
             : '#run';
  // Setting hash fires hashchange synchronously; useRoute picks it up.
  window.location.hash = next;
}

/**
 * Convenience navigation from the FilterCards → Grid transition.
 * Used by the back-to-filters button too.
 */
export function goToBrowseList(): void { navigate('browse-list'); }
export function goToBrowseContent(): void { navigate('browse-content'); }
