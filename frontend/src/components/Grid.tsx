import { useEffect, useMemo, useRef, useState } from 'react';
import { getResults } from '../api/endpoints';
import type { ResultRow, ResultsResponse } from '../api/types';
import { useApp, useFilters, useZoom } from '../state/AppContext';
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
  const zoomedAid = useZoom();
  const { dispatch } = useApp();
  const [data, setData] = useState<ResultsResponse | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const [loading, setLoading] = useState(true);
  const reqIdRef = useRef(0);

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
    if (!data) return;
    // Optimistic: leave the row alone — the user can re-run if they want
    // to see fresh KNN. We could call /api/asset/{id} here but it would
    // slow down rapid button clicks; keep it simple.
    void assetId;
  };

  // Escape closes the zoomed card.
  useEffect(() => {
    if (!zoomedAid) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dispatch({ type: 'ZOOM_OUT' });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [zoomedAid, dispatch]);

  // When the filter scope changes (page, sort, taxon pill), the zoomed
  // aid may no longer be in the visible page. Reset it so the grid
  // doesn't render a phantom "not found" state.
  useEffect(() => {
    if (!zoomedAid || !data) return;
    if (!sortedRows.some(r => r.asset_id === zoomedAid)) {
      dispatch({ type: 'ZOOM_OUT' });
    }
  }, [zoomedAid, sortedRows, data, dispatch]);

  // Zoomed-in mode: hide the grid and show only the active card at
  // near-fullscreen size, with a back button. We don't gate on data
  // here — if the page is mid-fetch we fall through to loading.
  if (zoomedAid && data) {
    const row = sortedRows.find(r => r.asset_id === zoomedAid);
    if (row) {
      return (
        <div className="grid-pane zoomed-pane">
          <Card
            key={row.asset_id}
            row={row}
            zoomed
            onZoomOut={() => dispatch({ type: 'ZOOM_OUT' })}
            onUpdated={handleRowUpdated}
          />
        </div>
      );
    }
  }

  if (loading && !data) {
    return (
      <div className="grid-pane">
        <div className="empty">Loading…</div>
      </div>
    );
  }

  if (error && !data) {
    return (
      <div className="grid-pane">
        <div className="empty error">Failed to load: {error.message}</div>
      </div>
    );
  }

  if (!data || data.results.length === 0) {
    return (
      <div className="grid-pane">
        <div className="empty">No results in this view.</div>
      </div>
    );
  }

  return (
    <div className="grid-pane">
      <div className="photo-grid">
        {sortedRows.map(r => (
          <Card key={r.asset_id} row={r} onUpdated={handleRowUpdated} />
        ))}
      </div>
      <Pagination data={data.pagination} />
    </div>
  );
}