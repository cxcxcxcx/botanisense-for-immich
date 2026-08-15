// Filter cards — rich square tiles that replace the old text-only
// "cluster pills". Each card carries a sample thumbnail (the asset with
// the highest top-1 BioCLIP score in that group) with the label and
// count overlaid at the bottom of the photo.
//
// Once a filter is active the whole grid collapses behind a compact
// "Active: X · n photos · show / clear" bar to make room for the grid
// of matching photos.

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

  // Once a filter is active the grid auto-collapses; user can re-show
  // it via a "show filters" button. We persist that toggle per
  // group_by so switching views remembers it.
  const [forceShow, setForceShow] = useState(false);
  const COLLAPSE_KEY = (g: GroupBy) => `filtercards:collapsed:${g}`;

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
      arr.sort((a, b) => Number(a.key) - Number(b.key));
    } else {
      arr.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
    }
    return arr;
  }, [allCards, sortMode, filters.group_by]);

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

  // The active card object (for the collapse bar + sticky).
  const activeCard = useMemo<Card | null>(() => {
    if (activeKey == null) return null;
    return allCards.find(c => c.key === activeKey) ?? null;
  }, [activeKey, allCards]);

  // Sticky-active card when search hides it (only meaningful when the
  // grid is visible).
  const stickyActive = useMemo<Card | null>(() => {
    if (activeKey == null) return null;
    if (visibleCards.some(c => c.key === activeKey)) return null;
    return allCards.find(c => c.key === activeKey) ?? null;
  }, [activeKey, visibleCards, allCards]);

  // Auto-collapse when a filter is active; honor a user override that
  // persists per group_by.
  useEffect(() => {
    setForceShow(localStorage.getItem(COLLAPSE_KEY(filters.group_by)) === '1');
  }, [filters.group_by]);

  const toggleForceShow = () => {
    const next = !forceShow;
    setForceShow(next);
    localStorage.setItem(COLLAPSE_KEY(filters.group_by), next ? '1' : '0');
  };

  const collapsed = activeCard != null && !forceShow;

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

  // Collapsed view: just the active-bar with sample thumb + label +
  // count + clear/show buttons.
  if (collapsed && activeCard) {
    return (
      <div className="filter-cards-wrap">
        <div className="filter-active-bar">
          <button
            type="button"
            className={`filter-card filter-active-card ${filters.group_by}`}
            onClick={clearFilter}
            title={`${activeCard.label} · ${activeCard.count} asset${activeCard.count === 1 ? '' : 's'} — click to clear`}
          >
            <FilterCardOverlay card={activeCard} />
            <span className="filter-clear-x">✕</span>
          </button>
          <div className="filter-active-meta">
            <strong>{activeCard.label}</strong>
            <small>{activeCard.count.toLocaleString()} asset{activeCard.count === 1 ? '' : 's'} · {filters.group_by}</small>
          </div>
          <button
            type="button"
            className="secondary outline"
            onClick={toggleForceShow}
          >
            show all {allCards.length} filters
          </button>
        </div>
      </div>
    );
  }

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
        {activeCard && (
          <button
            type="button"
            className="secondary outline"
            onClick={toggleForceShow}
          >
            collapse
          </button>
        )}
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
      <FilterCardOverlay card={card} />
    </button>
  );
}

function FilterCardOverlay({ card }: { card: Card }) {
  const [imgError, setImgError] = useState(false);
  const showImg = card.sampleAssetId && !imgError;
  const fallbackChar = card.label.replace(/^["'(]|["')']$/g, '').charAt(0).toUpperCase() || '·';
  return (
    <>
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
      <div className="overlay">
        <span className="overlay-label">{card.label}</span>
        <span className="overlay-count">{card.count.toLocaleString()}</span>
      </div>
    </>
  );
}