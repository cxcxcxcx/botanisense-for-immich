// URL ↔ filter state. The toolbar reads the current Filters via
// useFilters(); setFilters() pushes to history (no scroll) and updates
// localStorage for the persisted subset (group_by, sort, q).
//
// On mount the URL is parsed; popstate re-parses so browser back/forward
// just works.

import { useCallback } from 'react';
import { useApp, DEFAULT_FILTERS, type Filters, type GroupBy, type SortMode } from '../state/AppContext';

const PERSIST_KEYS = ['group_by', 'sort', 'q'] as const;

function readPersisted(fallback: Filters): Filters {
  try {
    const saved = JSON.parse(localStorage.getItem('filters:persist') ?? '{}');
    return { ...fallback, ...saved };
  } catch {
    return fallback;
  }
}

function writePersisted(f: Filters) {
  const partial: Partial<Filters> = {};
  for (const k of PERSIST_KEYS) {
    if (k === 'group_by') partial.group_by = f.group_by;
    else if (k === 'sort') partial.sort = f.sort;
    else if (k === 'q') partial.q = f.q;
  }
  try {
    localStorage.setItem('filters:persist', JSON.stringify(partial));
  } catch { /* quota / privacy mode — ignore */ }
}

function num(v: string | null): number | undefined {
  if (v === null || v === '') return undefined;
  const n = Number(v);
  return Number.isFinite(n) ? n : undefined;
}

export function parseUrlFilters(search: string = window.location.search): Filters {
  const sp = new URLSearchParams(search);
  const group_by = (sp.get('group_by') ?? 'cluster') as GroupBy;
  const sort = (sp.get('sort') ?? 'rank') as SortMode;
  const q = sp.get('q') ?? '';
  const page = num(sp.get('page')) ?? 0;
  const f: Filters = {
    group_by, sort, q, page,
    page_size: 60,
  };
  if (group_by === 'cluster')  f.cluster_id = num(sp.get('cluster_id'));
  if (group_by === 'species')  { const v = sp.get('species');  if (v !== null) f.species  = v; }
  if (group_by === 'genus')    { const v = sp.get('genus');    if (v !== null) f.genus    = v; }
  if (group_by === 'family')   { const v = sp.get('family');   if (v !== null) f.family   = v; }
  return f;
}

export function buildUrl(f: Filters): string {
  const sp = new URLSearchParams();
  sp.set('group_by', f.group_by);
  if (f.sort !== 'rank') sp.set('sort', f.sort);
  if (f.q) sp.set('q', f.q);
  if (f.group_by === 'cluster' && f.cluster_id != null) sp.set('cluster_id', String(f.cluster_id));
  if (f.group_by === 'species' && f.species) sp.set('species', f.species);
  if (f.group_by === 'genus'   && f.genus)   sp.set('genus',   f.genus);
  if (f.group_by === 'family'  && f.family)  sp.set('family',  f.family);
  if (f.page) sp.set('page', String(f.page));
  return `?${sp.toString()}`;
}

/** Initial filters — read from URL first, fall back to localStorage, fall back to defaults. */
export function initialFilters(): Filters {
  const fromUrl = parseUrlFilters();
  // If URL had nothing useful, try localStorage.
  if (fromUrl.group_by === 'cluster' && fromUrl.sort === 'rank' && !fromUrl.q && fromUrl.page === 0
      && fromUrl.cluster_id == null && !fromUrl.species && !fromUrl.genus && !fromUrl.family) {
    return readPersisted(DEFAULT_FILTERS);
  }
  return fromUrl;
}

export function useUrlFilters() {
  const { dispatch } = useApp();

  const setFilters = useCallback((patch: Partial<Filters>) => {
    dispatch({ type: 'SET_FILTERS', patch });
    // Compute the next Filters (we can't read the reducer's result here
    // without subscribing, so re-derive from a synthetic full object).
    // The next render will pull fresh state from useFilters().
    // We push the URL with a small optimistic merge: take the current
    // window.location.search and apply the patch.
    const next = { ...parseUrlFilters(), ...patch };
    if (patch.group_by && patch.group_by !== parseUrlFilters().group_by) {
      next.cluster_id = undefined; next.species = undefined;
      next.genus = undefined; next.family = undefined; next.page = 0;
    }
    writePersisted(next);
    const qs = buildUrl(next).slice(1); // strip leading '?'
    const url = qs ? `?${qs}` : window.location.pathname;
    window.history.pushState(null, '', url);
  }, [dispatch]);

  return setFilters;
}