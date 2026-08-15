// Filter cards — rich square tiles that replace the old text-only
// "cluster pills". Each card carries a sample thumbnail (the asset with
// the highest top-1 BioCLIP score in that group), the label, and the
// count. Sort modes (alpha / size / id) are persisted per group_by.

import { useEffect, useMemo, useState } from 'react';
import { getClusters, getResults } from '../api/endpoints';
import type { ClusterSummary, ResultsResponse, TaxonomyEntry } from '../api/types';
import { useFilters, type GroupBy } from '../state/AppContext';
import { useDebounce } from '../hooks/useDebounce';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { thumbUrl } from '../lib/thumb';

export type CardSort = 'alpha' | 'size' | 'id';
const SORTS: { value: CardSort; label: string }[] = [
  { value: 'size', label: 'size' },
  { value: 'alpha', label: 'A→Z' },
  { value: 'id', label: 'id' },
];

interface Card {
  key: string;
  label: string;
  count: number;
  sampleAssetId: string | null;
}

const SORT_KEY = (g: GroupBy) => `filtercards:sort:${g}`;

function readSort(g: GroupBy): CardSort {
  const v = localStorage.getItem(SORT_KEY(g));
  return v === 'alpha' || v === 'id' ? v : 'size';
}

export function FilterCards() {
  const filters = useFilters();
  const setFilters = useUrlFilters();
  const debouncedQ = useDebounce(filters.q, 300);

  const [clusterData, setClusterData] = useState<ClusterSummary[] | null>(null);
  const [resultsData, setResultsData] = useState<ResultsResponse | null>(null);
  const [sortMode, setSortMode] = useState<CardSort>(() => readSort(filters.group_by));

  // Persist sort per group_by so e.g. cluster-by-size + species-by-alpha
  // are independent.
  useEffect(() => {
    setSortMode(readSort(filters.group_by));
  }, [filters.group_by]);

  useEffect(() => {
    localStorage.setItem(SORT_KEY(filters.group_by), sortMode);
  }, [filters.group_by, sortMode]);

  // Fetch cluster summary when group_by == cluster.
  useEffect(() => {
    if (filters.group_by !== 'cluster') return;
    let cancelled = false;
    getClusters()
      .then(j => { if (!cancelled) setClusterData(j.clusters); })
      .catch(() => { /* keep null */ });
    return () => { cancelled = true; };
  }, [filters.group_by]);

  // Fetch taxonomy for species / genus / family. page_size=1 is enough —
  // the taxonomy block is global, not paginated.
  useEffect(() => {
    if (filters.group_by === 'cluster') return;
    let cancelled = false;
    getResults({ page: 0, page_size: 1 })
      .then(r => { if (!cancelled) setResultsData(r); })
      .catch(() => { /* keep null */ });
    return () => { cancelled = true; };
  }, [filters.group_by]);

  const allCards = useMemo<Card[]>(() => {
    if (filters.group_by === 'cluster') {
      if (!clusterData) return [];
      return clusterData.map(c => ({
        key: String(c.cluster_id),
        label: c.name || `cluster #${c.cluster_id}`,
        count: c.n,
        sampleAssetId: c.sample_asset_id,
      }));
    }
    if (!resultsData) return [];
    const tax: TaxonomyEntry[] = resultsData.taxonomy[filters.group_by] ?? [];
    return tax.map(t => ({
      key: t.name,
      label: t.name,
      count: t.n,
      sampleAssetId: t.sample_asset_id,
    }));
  }, [filters.group_by, clusterData, resultsData]);

  const sortedCards = useMemo<Card[]>(() => {
    const arr = [...allCards];
    if (sortMode === 'alpha') {
      arr.sort((a, b) => a.label.localeCompare(b.label));
    } else if (sortMode === 'id' && filters.group_by === 'cluster') {
      // Stable cluster_id sort (cluster keys are numeric strings).
      arr.sort((a, b) => Number(a.key) - Number(b.key));
    } else {
      // 'size' (default) and fallback for non-cluster group_by.
      arr.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
    }
    return arr;
  }, [allCards, sortMode, filters.group_by]);

  // Search-filter the visible cards. Empty search → show all.
  const visibleCards = useMemo<Card[]>(() => {
    const q = debouncedQ.trim().toLowerCase();
    if (!q) return sortedCards;
    return sortedCards.filter(c => c.label.toLowerCase().includes(q));
  }, [sortedCards, debouncedQ]);

  // Active filter key (cluster_id as string, or taxon name).
  const activeKey = useMemo<string | null>(() => {
    if (filters.group_by === 'cluster') return filters.cluster_id != null ? String(filters.cluster_id) : null;
    if (filters.group_by === 'species') return filters.species ?? null;
    if (filters.group_by === 'genus') return filters.genus ?? null;
    if (filters.group_by === 'family') return filters.family ?? null;
    return null;
  }, [filters.group_by, filters.cluster_id, filters.species, filters.genus, filters.family]);

  // Sticky-active card when search hides it.
  const stickyActive = useMemo<Card | null>(() => {
    if (activeKey == null) return null;
    if (visibleCards.some(c => c.key === activeKey)) return null;
    return allCards.find(c => c.key === activeKey) ?? null;
  }, [activeKey, visibleCards, allCards]);

  const activeParamKey: 'cluster_id' | 'species' | 'genus' | 'family' =
    filters.group_by === 'cluster' ? 'cluster_id'
    : filters.group_by === 'species' ? 'species'
    : filters.group_by === 'genus' ? 'genus'
    : 'family';

  const clearFilter = () =>
    setFilters({ [activeParamKey]: undefined, page: 0 } as Partial<typeof filters>);

  const handleClick = (card: Card) => {
    if (card.key === activeKey) {
      clearFilter();
      return;
    }
    const v = activeParamKey === 'cluster_id' ? Number(card.key) : card.key;
    setFilters({ [activeParamKey]: v, page: 0 } as Partial<typeof filters>);
  };

  const total = allCards.reduce((a, b) => a + b.count, 0);
  const isLoading =
    (filters.group_by === 'cluster' && !clusterData) ||
    (filters.group_by !== 'cluster' && !resultsData);

  return (
    <div className="filter-cards-wrap">
      <div className="filter-cards-header">
        <label>
          sort cards{' '}
          <select
            value={sortMode}
            onChange={e => setSortMode(e.target.value as CardSort)}
          >
            {SORTS.map(s => <option key={s.value} value={s.value}>{s.label}</option>)}
          </select>
        </label>
        <button
          type="button"
          className={`secondary outline filter-card-all-btn ${activeKey == null ? 'on' : ''}`}
          onClick={clearFilter}
        >
          all ({total.toLocaleString()})
        </button>
        <span className="filter-count">{visibleCards.length} shown</span>
      </div>

      <div className="filter-cards">
        {stickyActive && (
          <FilterCardTile
            card={stickyActive}
            rank={filters.group_by}
            active
            pinned
            onClick={clearFilter}
          />
        )}
        {isLoading && visibleCards.length === 0 && !stickyActive ? (
          <div className="empty" style={{ gridColumn: '1 / -1' }}>Loading filters…</div>
        ) : visibleCards.length === 0 && !stickyActive ? (
          <div className="empty" style={{ gridColumn: '1 / -1' }}>No matches.</div>
        ) : (
          visibleCards.map(c => (
            <FilterCardTile
              key={c.key}
              card={c}
              rank={filters.group_by}
              active={c.key === activeKey}
              onClick={() => handleClick(c)}
            />
          ))
        )}
      </div>
    </div>
  );
}

interface TileProps {
  card: Card;
  rank: GroupBy;
  active: boolean;
  pinned?: boolean;
  onClick: () => void;
}

function FilterCardTile({ card, rank, active, pinned, onClick }: TileProps) {
  const [imgError, setImgError] = useState(false);
  const showImg = card.sampleAssetId && !imgError;
  const fallbackChar = card.label.replace(/^["'(]|["')']$/g, '').charAt(0).toUpperCase() || '·';

  const classes = [
    'filter-card',
    rank,
    active ? 'on' : '',
    pinned ? 'pinned' : '',
  ].filter(Boolean).join(' ');

  return (
    <button
      type="button"
      className={classes}
      onClick={onClick}
      title={`${card.label} · ${card.count.toLocaleString()} asset${card.count === 1 ? '' : 's'}${pinned ? ' · click to clear' : ''}`}
    >
      <div className="thumb">
        {showImg ? (
          <img
            src={thumbUrl(card.sampleAssetId as string)}
            alt={card.label}
            loading="lazy"
            decoding="async"
            onError={() => setImgError(true)}
          />
        ) : (
          <span className="placeholder">{fallbackChar}</span>
        )}
      </div>
      <div className="meta">
        <span className="label">{card.label}</span>
        <span className="count">{card.count.toLocaleString()}</span>
      </div>
    </button>
  );
}