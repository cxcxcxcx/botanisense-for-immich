import { useEffect, useMemo, useRef, useState } from 'react';
import { getResults } from '../api/endpoints';
import type { ResultRow, ResultsResponse } from '../api/types';
import { useFilters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { Card } from './Card';
import { Pagination } from './Pagination';

function sortRows(rows: ResultRow[], sort: 'rank' | 'plantae' | 'score'): ResultRow[] {
  if (sort === 'plantae') {
    return [...rows].sort((a, b) => (b.taxon_scores?.Plantae ?? -1) - (a.taxon_scores?.Plantae ?? -1));
  }
  if (sort === 'score') {
    return [...rows].sort((a, b) => (b.top_predictions[0]?.score ?? -1) - (a.top_predictions[0]?.score ?? -1));
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

  // The filter-card active-bar (in the Toolbar) is the source of truth
  // for which subset is loaded, but a dedicated back button here makes
  // the navigation obvious even when the user has scrolled past the
  // bar. Clicking it clears whichever cluster/taxon filter is set and
  // returns to the filter-cards home.
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

  // After a per-card reclassify, refetch only that one row's KNN.
  const handleRowUpdated = async (assetId: string) => {
    // Optimistic: leave the row alone — the user can re-run if they want
    // to see fresh KNN. We could call /api/asset/{id} here but it would
    // slow down rapid button clicks; keep it simple.
    void assetId;
  };

  if (loading && !data) {
    return (
      <div className="grid-pane">
        <GridToolbar activeFilterLabel={activeFilterLabel} onBack={clearActiveFilter} groupBy={filters.group_by} />
        <div className="empty">Loading…</div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="grid-pane">
        <GridToolbar activeFilterLabel={activeFilterLabel} onBack={clearActiveFilter} groupBy={filters.group_by} />
        <div className="empty error">Failed to load: {error.message}</div>
      </div>
    );
  }

  if (!data || data.results.length === 0) {
    return (
      <div className="grid-pane">
        <GridToolbar activeFilterLabel={activeFilterLabel} onBack={clearActiveFilter} groupBy={filters.group_by} />
        <div className="empty">No results in this view.</div>
      </div>
    );
  }

  return (
    <div className="grid-pane">
      <GridToolbar
        activeFilterLabel={activeFilterLabel}
        total={data.pagination.total}
        onBack={clearActiveFilter}
        groupBy={filters.group_by}
      />
      <div className="photo-grid">
        {sortedRows.map(r => (
          <Card key={r.asset_id} row={r} onUpdated={handleRowUpdated} />
        ))}
      </div>
      <Pagination data={data.pagination} />
    </div>
  );
}

// Slim nav strip above the photo grid: back-to-filters button + (in the
// populated state) a "showing X" reminder. The FilterCards active-bar in
// the Toolbar carries the same info, but the user can scroll past it; a
// dedicated bar here keeps the back affordance always reachable.
interface GridToolbarProps {
  activeFilterLabel: string | null;
  total?: number;
  onBack: () => void;
  groupBy: string;
}
function GridToolbar({ activeFilterLabel, total, onBack, groupBy }: GridToolbarProps) {
  return (
    <div className="grid-toolbar">
      <button
        type="button"
        className="secondary outline back-to-filters"
        onClick={onBack}
        title={`Clear ${groupBy} filter and return to filter cards`}
      >
        ← back to filters
      </button>
      {activeFilterLabel && (
        <span className="grid-active-label">
          showing <strong>{activeFilterLabel}</strong>
          {total != null && (
            <> ({total.toLocaleString()} asset{total === 1 ? '' : 's'})</>
          )}
        </span>
      )}
    </div>
  );
}