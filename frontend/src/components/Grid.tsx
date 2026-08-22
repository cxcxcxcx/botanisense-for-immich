import { useEffect, useMemo, useRef, useState } from 'react';
import { getResults } from '../api/endpoints';
import type { ResultRow, ResultsResponse } from '../api/types';
import { useFilters, type SortMode } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate } from '../hooks/useRoute';
import { Card } from './Card';
import { Pagination } from './Pagination';

// Photo-sort radio chips — live here (with the grid they sort) instead
// of in the Toolbar (which sits on the cluster/taxon list view). Labels
// use arrows so the direction is unambiguous; the matching `hint` is
// rendered inline next to the active chip so users see what each sort
// means without hovering for a tooltip.
const SORT_OPTIONS: { value: SortMode; label: string; hint: string }[] = [
  { value: 'rank',    label: 'default',    hint: 'order returned by the server (Immich rank)' },
  { value: 'plantae', label: 'Plantae ↓',  hint: 'highest Plantae kingdom confidence first (green badge)' },
  { value: 'score',   label: 'top-1 ↓',    hint: 'highest BioCLIP top-1 species score first (green chip)' },
  { value: 'time',    label: 'newest ↓',   hint: 'newest EXIF capture time first (nulls last)' },
  { value: 'added',   label: 'added ↓',    hint: 'newest first_seen_at first — most recently ingested into the DB (nulls last)' },
];

function sortRows(
  rows: ResultRow[],
  sort: 'rank' | 'plantae' | 'score' | 'time' | 'added',
): ResultRow[] {
  if (sort === 'plantae') {
    return [...rows].sort((a, b) => (b.taxon_scores?.Plantae ?? -1) - (a.taxon_scores?.Plantae ?? -1));
  }
  if (sort === 'score') {
    return [...rows].sort((a, b) => (b.top_predictions[0]?.score ?? -1) - (a.top_predictions[0]?.score ?? -1));
  }
  if (sort === 'time') {
    return [...rows].sort((a, b) => {
      const av = a.captured_at ?? '';
      const bv = b.captured_at ?? '';
      if (!av && bv) return 1;
      if (av && !bv) return -1;
      return bv.localeCompare(av);
    });
  }
  if (sort === 'added') {
    // Server already sorted by first_seen_at DESC when this is the
    // active sort (`sortRequests` below), but we still apply the same
    // comparator when the user re-renders (e.g. after a filter change)
    // so the chip stays consistent regardless of cache state.
    return [...rows].sort((a, b) => {
      const av = a.first_seen_at ?? '';
      const bv = b.first_seen_at ?? '';
      if (!av && bv) return 1;
      if (av && !bv) return -1;
      return bv.localeCompare(av);
    });
  }
  return rows;
}

export function Grid() {
  const filters = useFilters();
  const setFilters = useUrlFilters();
  const [data, setData] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const reqIdRef = useRef(0);

  const activeFilterLabel: string | null = (() => {
    const g = filters.group_by;
    if (g === 'cluster') return filters.cluster_id != null ? `cluster #${filters.cluster_id}` : null;
    if (g === 'species') return filters.species ?? null;
    if (g === 'genus') return filters.genus ?? null;
    if (g === 'family') return filters.family ?? null;
    return null;
  })();
  const clearActiveFilter = () => {
    if (filters.group_by === 'cluster') setFilters({ cluster_id: undefined, page: 0 });
    else if (filters.group_by === 'species') setFilters({ species: undefined, page: 0 });
    else if (filters.group_by === 'genus') setFilters({ genus: undefined, page: 0 });
    else if (filters.group_by === 'family') setFilters({ family: undefined, page: 0 });
    navigate('browse-list');
  };

  useEffect(() => {
    const reqId = ++reqIdRef.current;
    setLoading(true);
    // Drop the previous page's results so stale photos don't flash
    // under the new filter while the request is in flight. The render
    // branch then falls into the "loading && !data" → "Loading…"
    // path until the new response arrives.
    setData(null);
    setError(null);
    // Server-side sorts: 'time' and 'added' need to be sent to the
    // server so the page slice is the actual most-recent N photos
    // (not the top-N by rank re-sorted). Client-side sorts
    // ('plantae', 'score') and the default 'rank' don't need a param.
    const serverSort = filters.sort === 'time' || filters.sort === 'added' ? filters.sort : 'rank';
    getResults({
      page: filters.page,
      page_size: filters.page_size,
      cluster_id: filters.cluster_id,
      species: filters.species,
      genus: filters.genus,
      family: filters.family,
      // Top-level facet — independent of group_by. Pass through
      // unchanged; backend treats 0/1 as exact-match on assets.has_person
      // and excludes unscanned (NULL) rows from both.
      has_person: filters.has_person,
      // Plantae-tier facet — same idea, also top-level. Lets the user
      // narrow the grid to plants / uncertain / non-plants without
      // picking a cluster or taxon.
      plantae: filters.plantae,
      sort: serverSort,
    })
      .then(r => { if (reqId === reqIdRef.current) { setData(r); setError(null); } })
      .catch(e => { if (reqId === reqIdRef.current) setError(e); })
      .finally(() => { if (reqId === reqIdRef.current) setLoading(false); });
  }, [
    filters.page, filters.page_size,
    filters.cluster_id, filters.species, filters.genus, filters.family,
    filters.has_person, filters.plantae,
    filters.sort,
  ]);

  const sortedRows = useMemo(() => {
    if (!data) return [];
    return sortRows(data.results, filters.sort);
  }, [data, filters.sort]);

  const handleRowUpdated = async (assetId: string) => {
    void assetId;
  };

  return (
    <div className="px-2 py-0">
      <Breadcrumb
        groupBy={filters.group_by}
        activeFilterLabel={activeFilterLabel}
        // Optional-chain `?.` only short-circuits the IMMEDIATE property
        // (`pagination`), not the chained `.total` after it. When the
        // server returns the empty-filter sentinel `{error, results}`
        // there is no `pagination` field, so the unsafe chain would
        // throw "Cannot read properties of undefined (reading 'total')"
        // and crash the whole grid. Use `?.` on every segment.
        total={data?.pagination?.total ?? null}
        onBack={clearActiveFilter}
        sort={filters.sort}
        onSortChange={s => setFilters({ sort: s })}
      />
      {loading && !data ? (
        <div className="p-10 text-center text-fg-1 whitespace-pre-wrap">Loading…</div>
      ) : error && !data ? (
        <div className="p-10 text-center text-red-300 whitespace-pre-wrap">Failed to load: {error.message}</div>
      ) : !data || data.results.length === 0 ? (
        <div className="p-10 text-center text-fg-1 whitespace-pre-wrap">No results in this view.</div>
      ) : (
        <>
          {/* Responsive column count — packs more thumbnails on wider
              screens where they stay readable (each card is aspect-4/5
              with a compact meta footer). At 2xl (≥1536 px) each tile
              is ~192 px wide, which still leaves room for the
              predictions/species chips on the Card. */}
          <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 2xl:grid-cols-8 gap-1.5 items-start">
            {sortedRows.map(r => (
              <Card key={r.asset_id} row={r} onUpdated={handleRowUpdated} />
            ))}
          </div>
          <Pagination data={data.pagination} />
        </>
      )}
    </div>
  );
}

interface BreadcrumbProps {
  groupBy: string;
  activeFilterLabel: string | null;
  total: number | null;
  onBack: () => void;
  sort: SortMode;
  onSortChange: (s: SortMode) => void;
}
function Breadcrumb({ groupBy, activeFilterLabel, total, onBack, sort, onSortChange }: BreadcrumbProps) {
  return (
    <nav className="flex flex-wrap items-center gap-x-2 gap-y-1 py-1 pb-2.5 text-fg-0 text-[0.85rem]" aria-label="Photo grid context">
      <button
        type="button"
        className="bg-transparent text-accent border-0 p-0 cursor-pointer hover:text-accent-hover hover:underline"
        onClick={onBack}
        title={`Clear ${groupBy} filter`}
      >← back</button>
      <span className="text-fg-3">·</span>
      <span>
        {activeFilterLabel
          ? <><strong className="text-fg">{activeFilterLabel}</strong>{total != null && <> · <span className="tabular-nums">{total.toLocaleString()} asset{total === 1 ? '' : 's'}</span></>}</>
          : <>all photos{total != null && <> · <span className="tabular-nums">{total.toLocaleString()} asset{total === 1 ? '' : 's'}</span></>}</>}
      </span>
      <span className="text-fg-3">·</span>
      <span className="inline-flex items-center gap-1">
        <span className="text-fg-1">sort:</span>
        {SORT_OPTIONS.map((s, i) => (
          <span key={s.value} className="inline-flex items-center gap-1">
            <button
              type="button"
              className={`bg-transparent border-0 p-0 cursor-pointer m-0 ${
                sort === s.value ? 'text-[#5c93c0] font-medium' : 'text-fg-0 hover:text-fg'
              }`}
              onClick={() => onSortChange(s.value)}
              title={s.hint}
              aria-pressed={sort === s.value}
            >{s.label}</button>
            {i < SORT_OPTIONS.length - 1 && <span className="text-fg-3">/</span>}
          </span>
        ))}
        <span className="text-fg-1 text-[0.75rem] italic ml-1">
          {SORT_OPTIONS.find(o => o.value === sort)?.hint}
        </span>
      </span>
    </nav>
  );
}