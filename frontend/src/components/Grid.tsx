import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { getResults } from '../api/endpoints';
import { ApiError } from '../api/client';
import type { ResultRow, ResultsResponse } from '../api/types';
import { useFilters, type SortMode } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate } from '../hooks/useRoute';
import { Card } from './Card';
import { Pagination } from './Pagination';
import { FilterModal } from './FilterModal';
import { useToasts } from './ToastProvider';

// Photo-grid sort options — surfaced as a small native <select> in
// the back-bar so the user can re-sort without leaving the photo
// view. Labels are self-explanatory at a glance.
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
  const push = useToasts();
  const [data, setData] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const [filterModalOpen, setFilterModalOpen] = useState(false);
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

  // Server-side sorts: 'time' and 'added' need to be sent to the
  // server so the page slice is the actual most-recent-N photos
  // (not the top-N by rank re-sorted). Client-side sorts
  // ('plantae', 'score') and the default 'rank' don't need a param.
  const serverSort = filters.sort === 'time' || filters.sort === 'added' ? filters.sort : 'rank';
  const runFetch = useCallback(() => {
    const reqId = ++reqIdRef.current;
    setLoading(true);
    setData(null);
    setError(null);
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
      .catch(e => {
        if (reqId !== reqIdRef.current) return;
        setError(e);
        // Surface 5xx as a structured toast so the user can retry
        // without scrolling back up to the grid. 401/403 are owned by
        // HealthBanner (it polls /api/health and renders a red banner
        // when the API key is invalid) so we don't double-toast.
        if (e instanceof ApiError && e.status >= 500) {
          push({
            kind: 'error',
            message: `Server error: ${e.detail || 'HTTP ' + e.status}`,
            action: { label: 'Retry', onClick: runFetch },
          }, 6000);
        }
      })
      .finally(() => { if (reqId === reqIdRef.current) setLoading(false); });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [
    filters.page, filters.page_size,
    filters.cluster_id, filters.species, filters.genus, filters.family,
    filters.has_person, filters.plantae,
    filters.sort,
  ]);

  useEffect(() => {
    runFetch();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [runFetch]);

  const sortedRows = useMemo(() => {
    if (!data) return [];
    return sortRows(data.results, filters.sort);
  }, [data, filters.sort]);

  const handleRowUpdated = async (assetId: string) => {
    void assetId;
  };

  const filtersActive = filters.plantae != null || filters.has_person != null;

  return (
    <div>
      {/* Sticky back-bar at the top of the photo view. Per the mock:
          Back button on the left, cluster/taxon name + count centered,
          a small Filter button + sort dropdown on the right. Stays
          visible while the user scrolls the photo grid. */}
      <div className="back-bar">
        <button
          type="button"
          onClick={clearActiveFilter}
          className="back-bar-back"
          title="Back to all clusters"
        >
          <BackArrowIcon />
          <span className="text-xs font-medium hidden sm:inline">Back</span>
        </button>

        <div className="back-bar-title">
          <h2 className="back-bar-name">
            {activeFilterLabel ?? 'all photos'}
          </h2>
          {data?.pagination?.total != null && (
            <p className="back-bar-count">
              {data.pagination.total.toLocaleString()} photo{data.pagination.total === 1 ? '' : 's'}
            </p>
          )}
        </div>

        <div className="flex items-center gap-1.5">
          <button
            type="button"
            onClick={() => setFilterModalOpen(true)}
            className="filter-trigger"
            title="Open filters"
            aria-label="Open filters"
          >
            <FilterIcon />
            {filtersActive && <span className="filter-trigger-dot" aria-hidden="true" />}
          </button>
          <label className="sort-select-wrapper">
            <select
              className="sort-select"
              value={filters.sort}
              onChange={(e) => setFilters({ sort: e.target.value as SortMode })}
              aria-label="Sort photos"
              title="Sort photos"
            >
              {SORT_OPTIONS.map(o => (
                <option key={o.value} value={o.value}>{o.label}</option>
              ))}
            </select>
          </label>
        </div>
      </div>

      {loading && !data ? (
        <div className="p-10 text-center text-fg-1 whitespace-pre-wrap">Loading…</div>
      ) : error && !data ? (
        // Smart error surface: 401/403 hint at the HealthBanner (which
        // already shows a red "API key invalid" banner in this state);
        // 5xx/other keep the inline message + a Retry button that
        // re-runs the same fetch via the runFetch closure.
        <div className="p-10 text-center text-red-300 whitespace-pre-wrap">
          {error instanceof ApiError && (error.status === 401 || error.status === 403) ? (
            <>
              <p className="m-0">Authorization failed.</p>
              <p className="m-0 mt-1 text-[12px] text-fg-1">
                Check the health banner at the top of the page for details.
              </p>
            </>
          ) : (
            <p className="m-0">Failed to load: {error.message}</p>
          )}
          <button
            type="button"
            className="mt-3 px-2.5 py-1 rounded border border-[#a44] bg-transparent text-red-300 text-sm hover:bg-[#3a1414]"
            onClick={runFetch}
          >
            Retry
          </button>
        </div>
      ) : !data || data.results.length === 0 ? (
        // Two distinct empty states: filtered-out (filters active, no
        // matches) vs unfiltered (DB empty). The filtered-out branch
        // shows why nothing matched and offers a one-click escape.
        <FilteredEmpty
          filters={filters}
          totalAll={data?.pagination?.total ?? 0}
          onClearFilters={() => {
            setFilters({
              cluster_id: undefined,
              species: undefined,
              genus: undefined,
              family: undefined,
              has_person: undefined,
              plantae: undefined,
              q: '',
              page: 0,
            });
            navigate('browse-list');
          }}
          onBackToAll={() => navigate('browse-list')}
        />
      ) : (
        <>
          {/* Responsive column count — packs more thumbnails on wider
              screens. Photo cards are now aspect-square (per the
              mock), so 8 columns fits comfortably at 2xl. */}
          <div className="p-0.5 sm:p-2">
            <div className="grid grid-cols-2 sm:grid-cols-4 md:grid-cols-5 lg:grid-cols-6 2xl:grid-cols-8 gap-0.5 sm:gap-2">
              {sortedRows.map(r => (
                <Card key={r.asset_id} row={r} onUpdated={handleRowUpdated} />
              ))}
            </div>
            <Pagination data={data.pagination} />
          </div>
        </>
      )}

      <FilterModal open={filterModalOpen} onClose={() => setFilterModalOpen(false)} />
    </div>
  );
}

function BackArrowIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <line x1="19" y1="12" x2="5" y2="12" />
      <polyline points="12 19 5 12 12 5" />
    </svg>
  );
}

// Empty-state block for the photo grid. Renders a context-aware
// message: when filters are active, tells the user *which* filter is
// hiding everything and offers a "Change filters" button. When no
// filters are active (i.e. the DB is empty), falls back to a generic
// "no photos" message and a back-to-all CTA.
function FilteredEmpty({
  filters,
  totalAll,
  onClearFilters,
  onBackToAll,
}: {
  filters: import('../state/AppContext').Filters;
  totalAll: number;
  onClearFilters: () => void;
  onBackToAll: () => void;
}) {
  const reasons: string[] = [];
  if (filters.cluster_id != null) reasons.push(`cluster #${filters.cluster_id}`);
  if (filters.species) reasons.push(`species "${filters.species}"`);
  if (filters.genus) reasons.push(`genus "${filters.genus}"`);
  if (filters.family) reasons.push(`family "${filters.family}"`);
  if (filters.has_person === 0) reasons.push('no people');
  if (filters.has_person === 1) reasons.push('with people');
  if (filters.plantae === 'plant') reasons.push('Plantae only');
  if (filters.plantae === 'non_plant') reasons.push('non-plants only');
  if (filters.plantae === 'uncertain') reasons.push('uncertain plants only');
  if (filters.q.trim()) reasons.push(`search "${filters.q.trim()}"`);

  if (reasons.length === 0) {
    return (
      <div className="p-10 text-center text-fg-1">
        <p className="m-0">No photos in the library yet.</p>
        <button
          type="button"
          className="mt-3 px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345] hover:border-accent"
          onClick={onBackToAll}
        >
          Back to all
        </button>
      </div>
    );
  }

  return (
    <div className="p-10 text-center text-fg-1">
      <p className="m-0">
        No photos match the current filter{reasons.length > 1 ? 's' : ''}:
      </p>
      <ul className="m-2 mb-4 list-none p-0 text-[12px] text-fg-1">
        {reasons.map((r, i) => (
          <li key={i}>
            <code className="font-mono">{r}</code>
          </li>
        ))}
      </ul>
      {totalAll > 0 && (
        <p className="m-0 mb-3 text-[11px] text-fg-2">
          {totalAll.toLocaleString()} photo{totalAll === 1 ? '' : 's'} in the library overall.
        </p>
      )}
      <button
        type="button"
        className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover"
        onClick={onClearFilters}
      >
        Change filters
      </button>
    </div>
  );
}

function FilterIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}