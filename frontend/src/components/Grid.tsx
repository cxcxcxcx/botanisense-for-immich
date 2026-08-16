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
const SORT_OPTIONS: { value: SortMode; label: string; hint: string }[] = [
  { value: 'rank',    label: 'default',        hint: 'order returned by the server' },
  { value: 'plantae', label: 'plants first',   hint: 'highest Plantae confidence on top' },
  { value: 'score',   label: 'most confident', hint: 'highest BioCLIP top-1 score on top' },
  { value: 'time',    label: 'newest first',   hint: 'newest EXIF capture time on top (nulls last)' },
];

function sortRows(rows: ResultRow[], sort: 'rank' | 'plantae' | 'score' | 'time'): ResultRow[] {
  if (sort === 'plantae') {
    return [...rows].sort((a, b) => (b.taxon_scores?.Plantae ?? -1) - (a.taxon_scores?.Plantae ?? -1));
  }
  if (sort === 'score') {
    return [...rows].sort((a, b) => (b.top_predictions[0]?.score ?? -1) - (a.top_predictions[0]?.score ?? -1));
  }
  if (sort === 'time') {
    // ISO 8601 strings sort lexicographically; nulls land at the end
    // via the `?? ''` fallback that we then push to the bottom.
    return [...rows].sort((a, b) => {
      const av = a.captured_at ?? '';
      const bv = b.captured_at ?? '';
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

  // Which subset is loaded. The grid toolbar carries the only
  // navigation affordance — clicking it returns to the filter cards.
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
    // Leave the photo grid and return to the filter cards view. The
    // route-sync effect used to do this automatically when the filter
    // cleared; that's gone now so the "All" card click can land on
    // #browse/content without being yanked back. This is the canonical
    // way to leave the content view.
    navigate('browse-list');
  };

  // Refresh on filter change. Re-read the entire page because filters
  // change which assets are in scope (and the server's `total` for
  // pagination depends on the filter).
  useEffect(() => {
    const reqId = ++reqIdRef.current;
    setLoading(true);
    getResults({
      page: filters.page,
      page_size: filters.page_size,
      cluster_id: filters.cluster_id,
      species: filters.species,
      genus: filters.genus,
      family: filters.family,
    })
      .then(r => { if (reqId === reqIdRef.current) { setData(r); setError(null); } })
      .catch(e => { if (reqId === reqIdRef.current) setError(e); })
      .finally(() => { if (reqId === reqIdRef.current) setLoading(false); });
  }, [
    filters.page, filters.page_size,
    filters.cluster_id, filters.species, filters.genus, filters.family,
  ]);

  const sortedRows = useMemo(() => {
    if (!data) return [];
    return sortRows(data.results, filters.sort);
  }, [data, filters.sort]);

  const handleRowUpdated = async (assetId: string) => {
    // Optimistic: leave the row alone — the user can re-run if they want
    // to see fresh KNN. We could call /api/asset/{id} here but it would
    // slow down rapid button clicks; keep it simple.
    void assetId;
  };

  return (
    <div className="grid-pane">
      <Breadcrumb
        groupBy={filters.group_by}
        activeFilterLabel={activeFilterLabel}
        total={data?.pagination.total ?? null}
        onBack={clearActiveFilter}
        sort={filters.sort}
        onSortChange={s => setFilters({ sort: s })}
      />
      {loading && !data ? (
        <div className="empty">Loading…</div>
      ) : error && !data ? (
        <div className="empty error">Failed to load: {error.message}</div>
      ) : !data || data.results.length === 0 ? (
        <div className="empty">No results in this view.</div>
      ) : (
        <>
          <div className="photo-grid">
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

// Breadcrumb above the photo grid:
//   ← back · species: Rosa canina · 142 assets · [sort: default | plants first | most confident]
// One tight strip, no big button, no bottom border. Hover the sort
// chips for the explanation.
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
    <nav className="grid-breadcrumb" aria-label="Photo grid context">
      <button type="button" className="crumb-back" onClick={onBack} title={`Clear ${groupBy} filter`}>
        ← back
      </button>
      <span className="crumb-sep">·</span>
      <span className="crumb-scope">
        {activeFilterLabel
          ? <><strong>{activeFilterLabel}</strong>{total != null && <> · <span className="crumb-count">{total.toLocaleString()} asset{total === 1 ? '' : 's'}</span></>}</>
          : <>all photos{total != null && <> · <span className="crumb-count">{total.toLocaleString()} asset{total === 1 ? '' : 's'}</span></>}</>}
      </span>
      <span className="crumb-sep">·</span>
      <span className="crumb-sort">
        <span className="crumb-sort-label">sort:</span>
        {SORT_OPTIONS.map((s, i) => (
          <span key={s.value} className="crumb-sort-group">
            <button
              type="button"
              className={`crumb-sort-chip ${sort === s.value ? 'active' : ''}`}
              onClick={() => onSortChange(s.value)}
              title={s.hint}
              aria-pressed={sort === s.value}
            >
              {s.label}
            </button>
            {i < SORT_OPTIONS.length - 1 && <span className="crumb-sort-sep">/</span>}
          </span>
        ))}
      </span>
    </nav>
  );
}
