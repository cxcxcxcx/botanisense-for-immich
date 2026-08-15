import { useEffect, useMemo, useState } from 'react';
import { getClusters, getResults } from '../api/endpoints';
import type { ClusterSummary, ResultsResponse } from '../api/types';
import { useFilters, useApp } from '../state/AppContext';
import { useDebounce } from '../hooks/useDebounce';
import { useUrlFilters } from '../hooks/useUrlFilters';

interface ClusterPill { key: string; label: string; count: number; tooltip: string; }

export function Pills() {
  const filters = useFilters();
  const { state } = useApp();
  const setFilters = useUrlFilters();
  const debouncedQ = useDebounce(filters.q, 300);

  const [clusterData, setClusterData] = useState<ClusterSummary[] | null>(null);
  const [resultsData, setResultsData] = useState<ResultsResponse | null>(null);

  // Load cluster summary when group_by == cluster.
  useEffect(() => {
    if (filters.group_by !== 'cluster') return;
    let cancelled = false;
    getClusters()
      .then(j => { if (!cancelled) setClusterData(j.clusters); })
      .catch(() => { /* keep null */ });
    return () => { cancelled = true; };
  }, [filters.group_by]);

  // Load taxonomy when group_by is species/genus/family.
  useEffect(() => {
    if (filters.group_by === 'cluster') return;
    let cancelled = false;
    getResults({ page: 0, page_size: 1 })
      .then(r => { if (!cancelled) setResultsData(r); })
      .catch(() => { /* keep null */ });
    return () => { cancelled = true; };
  }, [filters.group_by]);

  const pills = useMemo<{ entries: ClusterPill[]; active: string | number | null | undefined; total: number }>(() => {
    const q = debouncedQ.trim().toLowerCase();
    if (filters.group_by === 'cluster') {
      if (!clusterData) return { entries: [], active: filters.cluster_id, total: 0 };
      const entries: ClusterPill[] = clusterData.map(c => ({
        key: String(c.cluster_id),
        label: c.name || `cluster #${c.cluster_id}`,
        count: c.n,
        tooltip: c.name,
      }));
      const filtered = q ? entries.filter(p => p.label.toLowerCase().includes(q)) : entries;
      return { entries: filtered, active: filters.cluster_id, total: entries.reduce((a, b) => a + b.count, 0) };
    }
    if (!resultsData) return { entries: [], active: undefined, total: 0 };
    const tax = resultsData.taxonomy[filters.group_by] || [];
    const entries: ClusterPill[] = tax.slice(0, 50).map(t => ({
      key: t.name, label: t.name, count: t.n, tooltip: t.name,
    }));
    const filtered = q ? entries.filter(p => p.label.toLowerCase().includes(q)) : entries;
    const active =
      filters.group_by === 'species' ? filters.species :
      filters.group_by === 'genus'   ? filters.genus   :
      filters.group_by === 'family'  ? filters.family  : undefined;
    return { entries: filtered, active, total: entries.reduce((a, b) => a + b.count, 0) };
  }, [filters.group_by, filters.cluster_id, filters.species, filters.genus, filters.family,
      clusterData, resultsData, debouncedQ]);

  // Sticky chip when active filter is hidden by search.
  const activeHidden = useMemo(() => {
    if (pills.active == null) return null;
    if (pills.entries.some(p => p.key === String(pills.active))) return null;
    // Find the active in the unfiltered list.
    const all =
      filters.group_by === 'cluster'
        ? (clusterData ?? []).map(c => ({ key: String(c.cluster_id), label: c.name || `cluster #${c.cluster_id}`, count: c.n, tooltip: c.name }))
        : ((resultsData?.taxonomy[filters.group_by] ?? []).slice(0, 50).map(t => ({ key: t.name, label: t.name, count: t.n, tooltip: t.name })));
    return all.find(p => p.key === String(pills.active)) ?? null;
  }, [pills.entries, pills.active, filters.group_by, clusterData, resultsData]);

  const activeParamKey: keyof typeof filters =
    filters.group_by === 'cluster' ? 'cluster_id' :
    filters.group_by === 'species' ? 'species' :
    filters.group_by === 'genus'   ? 'genus'   :
    filters.group_by === 'family'  ? 'family'  : 'cluster_id';

  const clearFilter = () => setFilters({ [activeParamKey]: undefined, page: 0 } as Partial<typeof filters>);

  const handleClick = (key: string) => {
    const isActive = String(pills.active) === key;
    if (isActive) {
      clearFilter();
    } else {
      const v = activeParamKey === 'cluster_id' ? Number(key) : key;
      setFilters({ [activeParamKey]: v, page: 0 } as Partial<typeof filters>);
    }
  };

  return (
    <div className="cluster-pills" role="listbox" aria-label="Filter by">
      <button
        type="button"
        className={`cluster-pill ${pills.active == null ? 'on' : ''}`}
        onClick={clearFilter}
        title={`Show all ${pills.total} photos`}
      >
        all <small>{pills.total.toLocaleString()}</small>
      </button>
      {pills.entries.map(p => {
        const isActive = String(pills.active) === p.key;
        return (
          <button
            key={p.key}
            type="button"
            className={`cluster-pill ${isActive ? 'on' : ''}`}
            title={p.tooltip}
            onClick={() => handleClick(p.key)}
          >
            {p.label} <small>{p.count}</small>
          </button>
        );
      })}
      {activeHidden && (
        <button
          type="button"
          className="cluster-pill on"
          title={`${activeHidden.tooltip} (hidden by search)`}
          onClick={clearFilter}
        >
          {activeHidden.label} ✕
        </button>
      )}
    </div>
  );
}