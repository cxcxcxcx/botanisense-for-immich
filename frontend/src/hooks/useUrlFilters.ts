// URL ↔ filter state. The toolbar reads the current Filters via
// useFilters(); setFilters() pushes to history (no scroll) and updates
// localStorage for the persisted subset (group_by, sort, q).
//
// On mount the URL is parsed; popstate re-parses so browser back/forward
// just works.

import { useCallback } from 'react';
import { useApp, DEFAULT_FILTERS, type Filters, type GroupBy, type SortMode } from '../state/AppContext';

const PERSIST_KEYS = ['group_by', 'sort', 'q'] as const;

const PLANTAE_VALUES = ['plant', 'non_plant', 'uncertain'] as const;
type PlantaeValue = (typeof PLANTAE_VALUES)[number];

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
  const rawGroupBy = sp.get('group_by') ?? 'cluster';
  // Backwards-compat: map old `?group_by=bbox` or `?group_by=supergroup`
  // bookmarks seamlessly onto the unified `cluster` view.
  const group_by = (rawGroupBy === 'bbox' || rawGroupBy === 'supergroup' ? 'cluster' : rawGroupBy) as GroupBy;
  // Default sort lives on DEFAULT_FILTERS so a single source of truth
  // governs both URL parsing and localStorage seeding. Falling back
  // hard-coded here would silently diverge from the default pill.
  const sort = (sp.get('sort') ?? DEFAULT_FILTERS.sort) as SortMode;
  const q = sp.get('q') ?? '';
  const page = num(sp.get('page')) ?? 0;
  const f: Filters = {
    group_by, sort, q, page,
    page_size: 60,
  };
  if (group_by === 'cluster')  f.cluster_id = num(sp.get('cluster_id')) ?? num(sp.get('bbox_cluster_id'));
  if (group_by === 'supergroup') f.supergroup = num(sp.get('supergroup'));
  if (group_by === 'species')  { const v = sp.get('species');  if (v !== null) f.species  = v; }
  if (group_by === 'genus')    { const v = sp.get('genus');    if (v !== null) f.genus    = v; }
  if (group_by === 'family')   { const v = sp.get('family');   if (v !== null) f.family   = v; }
  // Person filter is a top-level facet, independent of group_by. Parse
  // as 0|1 (string is the URL transport); absent/unknown → undefined.
  const hp = sp.get('has_person');
  if (hp === '0' || hp === '1') f.has_person = hp === '1' ? 1 : 0;
  // Plantae-tier facet: independent of group_by. Absent/invalid → undefined.
  const pl = sp.get('plantae');
  if (pl && (PLANTAE_VALUES as readonly string[]).includes(pl)) {
    f.plantae = pl as PlantaeValue;
  }
  // Manual opt-in toggle for the project-wide plant-only default. Absent
  // means false (default) — only emit when explicitly enabled so a
  // bookmarked link with the flag off still serializes compactly.
  if (sp.get('include_non_plants') === '1' || sp.get('include_non_plants') === 'true') {
    f.include_non_plants = true;
  }
  // Map click-through. A partial triple is meaningless (a centre without a
  // radius would silently filter by a default box), so require lat + lng;
  // radius defaults to the fine place radius.
  const nearLat = num(sp.get('near_lat'));
  const nearLng = num(sp.get('near_lng'));
  if (nearLat != null && nearLng != null) {
    f.near_lat = nearLat;
    f.near_lng = nearLng;
    f.near_radius = num(sp.get('near_radius')) ?? 150;
  }
  // Exact place membership; only meaningful alongside the near triple.
  const place = sp.get('place');
  if (place && nearLat != null && nearLng != null) {
    f.place_id = place;
    const pg = sp.get('place_gran');
    f.place_granularity = pg === 'wide' ? 'wide' : pg === 'region' ? 'region' : 'fine';
    f.place_min_photos = num(sp.get('place_min')) ?? 2;
  }
  return f;
}

export function buildUrl(f: Filters): string {
  const sp = new URLSearchParams();
  sp.set('group_by', f.group_by);
  // Only emit `sort` when it differs from the project default — keeps
  // bookmarked URLs of the default view short (no `?sort=time` noise).
  if (f.sort !== DEFAULT_FILTERS.sort) sp.set('sort', f.sort);
  if (f.q) sp.set('q', f.q);
  if (f.group_by === 'cluster' && f.cluster_id != null) sp.set('cluster_id', String(f.cluster_id));
  // group_id 0 (Ungrouped) is a real filter — `!= null` is the gate.
  if (f.group_by === 'supergroup' && f.supergroup != null) sp.set('supergroup', String(f.supergroup));
  if (f.group_by === 'species' && f.species) sp.set('species', f.species);
  if (f.group_by === 'genus'   && f.genus)   sp.set('genus',   f.genus);
  if (f.group_by === 'family'  && f.family)  sp.set('family',  f.family);
  if (f.has_person != null) sp.set('has_person', String(f.has_person));
  if (f.plantae) sp.set('plantae', f.plantae);
  // Only emit when true — the default-off policy shouldn't pollute the
  // URL with a redundant `include_non_plants=0`.
  if (f.include_non_plants) sp.set('include_non_plants', '1');
  if (f.near_lat != null && f.near_lng != null) {
    sp.set('near_lat', String(f.near_lat));
    sp.set('near_lng', String(f.near_lng));
    sp.set('near_radius', String(f.near_radius ?? 150));
    if (f.place_id) {
      sp.set('place', f.place_id);
      sp.set('place_gran', f.place_granularity ?? 'fine');
      sp.set('place_min', String(f.place_min_photos ?? 2));
    }
  }
  if (f.page) sp.set('page', String(f.page));
  return `?${sp.toString()}`;
}

/** Initial filters — read from URL first, fall back to localStorage, fall back to defaults. */
export function initialFilters(): Filters {
  const fromUrl = parseUrlFilters();
  // If URL had nothing useful, try localStorage. The "nothing useful"
  // check must include EVERY top-level facet — including plantae, the
  // include_non_plants opt-in, AND the new default sort ('time') —
  // otherwise a URL like `?sort=rank` on a stale bookmark would be
  // silently upgraded to 'time' (the new default) when the user
  // expected their explicit choice to win.
  if (fromUrl.group_by === 'cluster'
      && fromUrl.sort === DEFAULT_FILTERS.sort
      && !fromUrl.q && fromUrl.page === 0
      && fromUrl.cluster_id == null && fromUrl.supergroup == null
      && !fromUrl.species && !fromUrl.genus && !fromUrl.family
      && fromUrl.has_person == null && fromUrl.plantae == null
      && !fromUrl.include_non_plants
      && fromUrl.near_lat == null && fromUrl.near_lng == null
      && fromUrl.place_id == null) {
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
      // Keys explicitly present in the patch survive the switch (mirrors
      // the reducer's clearTaxonomyFilter) — one setFilters call can
      // switch group_by AND set the new group's filter.
      if (!('cluster_id' in patch)) next.cluster_id = undefined;
      if (!('supergroup' in patch)) next.supergroup = undefined;
      if (!('species' in patch)) next.species = undefined;
      if (!('genus' in patch)) next.genus = undefined;
      if (!('family' in patch)) next.family = undefined;
      next.page = 0;
    }
    // Picking a taxonomy/cluster card leaves the map place — otherwise the
    // grid would stay scoped to the old neighbourhood and the new card
    // would look empty. A patch that explicitly sets a near triple (the
    // map click) is exempt.
    const groupKeys = ['cluster_id', 'supergroup', 'species', 'genus', 'family'] as const;
    const setsGroupKey = groupKeys.some((k) => k in patch);
    if (setsGroupKey && !('near_lat' in patch)) {
      next.near_lat = undefined;
      next.near_lng = undefined;
      next.near_radius = undefined;
      next.place_id = undefined;
      next.place_granularity = undefined;
      next.place_min_photos = undefined;
    }
    // has_person changes (including toggling off) re-paginate to 0 so
    // the user lands on page 1 of the new result set.
    if (Object.prototype.hasOwnProperty.call(patch, 'has_person')) {
      next.page = 0;
    }
    if (Object.prototype.hasOwnProperty.call(patch, 'plantae')) {
      next.page = 0;
    }
    // Toggling the non-plants opt-in changes the result-set size by a
    // large margin — re-paginate to 0 so the user lands on page 1 of
    // the new (larger or smaller) set rather than a now-empty page.
    if (Object.prototype.hasOwnProperty.call(patch, 'include_non_plants')) {
      next.page = 0;
    }
    writePersisted(next);
    const qs = buildUrl(next).slice(1); // strip leading '?'
    const search = qs ? `?${qs}` : '';
    // Preserve the route hash so filter changes inside #run / #browse
    // don't reset the user's current view.
    const hash = window.location.hash;
    window.history.pushState(null, '', `${window.location.pathname}${search}${hash}`);
  }, [dispatch]);

  return setFilters;
}