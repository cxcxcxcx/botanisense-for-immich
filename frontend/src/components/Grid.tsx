import { useEffect, useMemo, useRef, useState } from 'react';
import { getResults } from '../api/endpoints';
import type { ResultRow, ResultsResponse } from '../api/types';
import { useFilters, type SortMode } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate } from '../hooks/useRoute';
import { Card } from './Card';
import { Pagination } from './Pagination';

// Photo-sort radio chips — live here (with the grid they sort) instead
// of in the Toolbar (which sits on the cluster/taxon list view).
// Labels are written to be self-explanatory at a glance (no inline
// tooltip / italic hint needed); a ⓘ button beside the chips opens a
// modal with the full glossary for users who want the exact criteria
// (see SortHelpButton below).
const SORT_OPTIONS: { value: SortMode; label: string }[] = [
  { value: 'rank',    label: 'Default order' },
  { value: 'plantae', label: 'Plantae confidence ↓' },
  { value: 'score',   label: 'ID confidence ↓' },
  { value: 'time',    label: 'Capture date ↓' },
  { value: 'added',   label: 'Date added ↓' },
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
      <ContextBar
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

// ContextBar — sits between the filter row and the photo grid.
// Replaces the old slash-list breadcrumb. Layout: left side holds the
// back link + current filter label + total count; right side holds the
// sort <select> (styled to match the chip aesthetic) plus a <details>
// disclosure with the full sort/filter glossary (replaces the old
// modal-based help).
interface ContextBarProps {
  groupBy: string;
  activeFilterLabel: string | null;
  total: number | null;
  onBack: () => void;
  sort: SortMode;
  onSortChange: (s: SortMode) => void;
}
function ContextBar({ groupBy, activeFilterLabel, total, onBack, sort, onSortChange }: ContextBarProps) {
  return (
    <div className="px-2 pt-2">
      <div className="context-bar">
        <div className="flex items-center gap-2 min-w-0 flex-1">
          <button
            type="button"
            className="context-back"
            onClick={onBack}
            title={`Clear ${groupBy} filter`}
          >← back</button>
          <div className="flex flex-col min-w-0">
            <strong className="truncate">
              {activeFilterLabel ?? 'all photos'}
            </strong>
            {total != null && (
              <span className="text-fg-1 text-[0.8rem] tabular-nums">
                {total.toLocaleString()} asset{total === 1 ? '' : 's'}
              </span>
            )}
          </div>
        </div>
        <div className="flex items-center gap-3 ml-auto">
          <label className="flex items-center gap-2 text-[0.85rem]">
            <span className="text-fg-1 uppercase tracking-wider text-[0.7rem]">Sort by</span>
            <select
              className="sort-select"
              value={sort}
              onChange={(e) => onSortChange(e.target.value as SortMode)}
              aria-label="Sort photos"
            >
              {SORT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
          <details className="relative">
            <summary
              className="inline-flex items-center justify-center w-[18px] h-[18px] rounded-full border border-line2 bg-transparent text-fg-1 text-[11px] leading-none cursor-pointer hover:bg-bg-2 hover:border-accent hover:text-fg list-none"
              title="What do the sort options and filters mean?"
              aria-label="Show sort and filter help"
            >ⓘ</summary>
            <div className="absolute right-0 top-6 z-50 w-[min(420px,90vw)] p-4 rounded-lg border border-line2 bg-bg-1 shadow-lg text-[13px] leading-[1.45]">
              <SortHelpBody />
            </div>
          </details>
        </div>
      </div>
    </div>
  );
}

// Help body — same content the old SortHelpBody modal had. Lives
// inside a <details> dropdown now (no modal needed). Plain <dl> for
// readability; each item pairs the user-facing label with the exact
// server-side criterion. Sections mirror the UI: sort options first,
// then filter facets below.
function SortHelpBody() {
  return (
    <div className="space-y-4">
      <section>
        <h4 className="m-0 mb-2 text-[#ad9] text-[0.95rem]">Sort criteria</h4>
        <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1.5 m-0">
          <dt className="font-medium">Default order</dt>
          <dd className="m-0 text-fg-1">Server's natural order — Immich rank, then cluster ID as tiebreak.</dd>
          <dt className="font-medium">ID confidence ↓</dt>
          <dd className="m-0 text-fg-1">Highest BioCLIP species identification confidence first (the green chip on each photo's top-1 prediction).</dd>
          <dt className="font-medium">Plantae confidence ↓</dt>
          <dd className="m-0 text-fg-1">Highest Plantae kingdom confidence first (the green badge).</dd>
          <dt className="font-medium">Capture date ↓</dt>
          <dd className="m-0 text-fg-1">Newest EXIF capture date first. Photos without a capture date sort last.</dd>
          <dt className="font-medium">Date added ↓</dt>
          <dd className="m-0 text-fg-1">Newest ingestion date first (when the asset was first indexed). Photos never ingested sort last.</dd>
        </dl>
      </section>

      <section>
        <h4 className="m-0 mb-2 text-[#ad9] text-[0.95rem]">Filters</h4>
        <dl className="grid grid-cols-[max-content_1fr] gap-x-3 gap-y-1.5 m-0">
          <dt className="font-medium">Group by</dt>
          <dd className="m-0 text-fg-1">Cluster / species / genus / family. Controls which list-view appears and which taxonomy chip is active.</dd>
          <dt className="font-medium">Search</dt>
          <dd className="m-0 text-fg-1">Narrows the cluster / taxonomy list by label substring. Doesn't affect the photo grid directly.</dd>
          <dt className="font-medium">Plantae tier</dt>
          <dd className="m-0 text-fg-1">
            Narrows both the filter cards and the photo grid:
            <span className="block">plants: Plantae ≥ 0.60</span>
            <span className="block">uncertain: Plantae 0.35–0.60</span>
            <span className="block">non-plants: Plantae &lt; 0.35</span>
          </dd>
          <dt className="font-medium">Person</dt>
          <dd className="m-0 text-fg-1">
            Narrows the photo grid by YOLO person detection. <em>any</em> includes unscanned; <em>has</em> / <em>no</em> exclude unscanned photos (we won't claim "no person" until we've actually looked).
          </dd>
        </dl>
      </section>
    </div>
  );
}