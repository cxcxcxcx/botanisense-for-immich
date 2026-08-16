// Filter cards — rich square tiles that replace the old text-only
// "cluster pills". Each card carries a sample thumbnail (the asset with
// the highest top-1 BioCLIP score in that group) with the label and
// count overlaid at the bottom of the photo.

import { useEffect, useMemo, useState } from 'react';
import { getClusters, getResults } from '../api/endpoints';
import type { ClusterSummary, ResultsResponse, TaxonomyEntry } from '../api/types';
import { useApp, useFilters, type GroupBy } from '../state/AppContext';
import { useDebounce } from '../hooks/useDebounce';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate } from '../hooks/useRoute';
import { thumbUrl } from '../lib/thumb';

export type CardSort = 'alpha' | 'size' | 'id';
const SORTS: { value: CardSort; label: string; hint: string }[] = [
  { value: 'size',  label: 'size',  hint: 'largest cluster/taxon first' },
  { value: 'alpha', label: 'A→Z',   hint: 'alphabetical by label' },
  { value: 'id',    label: 'id',    hint: 'internal cluster id order (cluster only)' },
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
  const { dispatch } = useApp();
  const debouncedQ = useDebounce(filters.q, 300);

  const [clusterData, setClusterData] = useState<ClusterSummary[] | null>(null);
  const [resultsData, setResultsData] = useState<ResultsResponse | null>(null);
  const [sortMode, setSortMode] = useState<CardSort>(() => readSort(filters.group_by));
  const [forceShow, setForceShow] = useState(false);
  const COLLAPSE_KEY = (g: GroupBy) => `filtercards:collapsed:${g}`;

  useEffect(() => { setSortMode(readSort(filters.group_by)); }, [filters.group_by]);
  useEffect(() => { localStorage.setItem(SORT_KEY(filters.group_by), sortMode); }, [filters.group_by, sortMode]);

  useEffect(() => {
    if (filters.group_by !== 'cluster') return;
    let cancelled = false;
    getClusters().then(j => { if (!cancelled) setClusterData(j.clusters); }).catch(() => {});
    return () => { cancelled = true; };
  }, [filters.group_by]);

  useEffect(() => {
    if (filters.group_by === 'cluster') return;
    let cancelled = false;
    getResults({ page: 0, page_size: 1 }).then(r => { if (!cancelled) setResultsData(r); }).catch(() => {});
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
    if (sortMode === 'alpha') arr.sort((a, b) => a.label.localeCompare(b.label));
    else if (sortMode === 'id' && filters.group_by === 'cluster') arr.sort((a, b) => Number(a.key) - Number(b.key));
    else arr.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
    return arr;
  }, [allCards, sortMode, filters.group_by]);

  const visibleCards = useMemo<Card[]>(() => {
    const q = debouncedQ.trim().toLowerCase();
    if (!q) return sortedCards;
    return sortedCards.filter(c => c.label.toLowerCase().includes(q));
  }, [sortedCards, debouncedQ]);

  const activeKey = useMemo<string | null>(() => {
    if (filters.group_by === 'cluster') return filters.cluster_id != null ? String(filters.cluster_id) : null;
    if (filters.group_by === 'species') return filters.species ?? null;
    if (filters.group_by === 'genus') return filters.genus ?? null;
    if (filters.group_by === 'family') return filters.family ?? null;
    return null;
  }, [filters.group_by, filters.cluster_id, filters.species, filters.genus, filters.family]);

  const activeCard = useMemo<Card | null>(() => {
    if (activeKey == null) return null;
    return allCards.find(c => c.key === activeKey) ?? null;
  }, [activeKey, allCards]);

  const stickyActive = useMemo<Card | null>(() => {
    if (activeKey == null) return null;
    if (visibleCards.some(c => c.key === activeKey)) return null;
    return allCards.find(c => c.key === activeKey) ?? null;
  }, [activeKey, visibleCards, allCards]);

  useEffect(() => { setForceShow(localStorage.getItem(COLLAPSE_KEY(filters.group_by)) === '1'); }, [filters.group_by]);
  useEffect(() => { dispatch({ type: 'SET_FILTERS_COLLAPSED', collapsed: activeCard != null && !forceShow }); }, [activeCard, forceShow, dispatch]);

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

  const clearFilter = () => setFilters({ [activeParamKey]: undefined, page: 0 } as Partial<typeof filters>);

  const goToAll = () => { clearFilter(); navigate('browse-content'); };

  const handleClick = (card: Card) => {
    if (card.key === activeKey) { clearFilter(); navigate('browse-list'); return; }
    const v = activeParamKey === 'cluster_id' ? Number(card.key) : card.key;
    setFilters({ [activeParamKey]: v, page: 0 } as Partial<typeof filters>);
  };

  const total = allCards.reduce((a, b) => a + b.count, 0);
  const isLoading =
    (filters.group_by === 'cluster' && !clusterData) ||
    (filters.group_by !== 'cluster' && !resultsData);

  if (collapsed && activeCard) return null;

  return (
    <div className="pt-1">
      <div className="flex flex-wrap items-center gap-2 py-1 pb-2 text-[0.8rem] text-fg-1">
        <span className="inline-flex items-center gap-1" role="radiogroup" aria-label="Sort filter cards">
          <span className="text-fg-1 mr-[2px]">sort</span>
          {SORTS.map((s, i) => (
            <span key={s.value} className="inline-flex items-center gap-1">
              <button
                type="button"
                role="radio"
                aria-checked={sortMode === s.value}
                className={`bg-transparent border-0 p-0 m-0 cursor-pointer text-[0.85rem] leading-[22px] ${
                  sortMode === s.value ? 'text-accent font-medium' : 'text-fg-0 hover:text-fg'
                }`}
                onClick={() => setSortMode(s.value)}
                title={s.hint}
              >{s.label}</button>
              {i < SORTS.length - 1 && <span className="text-fg-3">/</span>}
            </span>
          ))}
        </span>
        <span className="ml-auto tabular-nums">{visibleCards.length} of {allCards.length} filters</span>
        {activeCard && (
          <button
            type="button"
            className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-[0.8rem] hover:bg-[#345] hover:border-accent"
            onClick={toggleForceShow}
          >
            collapse
          </button>
        )}
      </div>

      <div className="grid gap-2.5 py-1 grid-cols-[repeat(auto-fill,minmax(160px,1fr))]">
        <FilterAllCard
          total={total}
          groupBy={filters.group_by}
          active={activeKey == null}
          onClick={goToAll}
        />
        {stickyActive && (
          <FilterCardTile
            card={stickyActive}
            rank={filters.group_by}
            active
            pinned
            onClick={() => { clearFilter(); navigate('browse-list'); }}
          />
        )}
        {isLoading && visibleCards.length === 0 && !stickyActive ? (
          <div className="p-10 text-center text-fg-1 whitespace-pre-wrap" style={{ gridColumn: '1 / -1' }}>Loading filters…</div>
        ) : visibleCards.length === 0 && !stickyActive ? (
          <div className="p-10 text-center text-fg-1 whitespace-pre-wrap" style={{ gridColumn: '1 / -1' }}>No matches.</div>
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

interface AllCardProps {
  total: number;
  groupBy: GroupBy;
  active: boolean;
  onClick: () => void;
}

function FilterAllCard({ total, groupBy, active, onClick }: AllCardProps) {
  return (
    <button
      type="button"
      className={`filter-card relative aspect-square bg-bg-2 rounded-lg overflow-hidden cursor-pointer p-0 text-inherit border transition-colors hover:border-accent hover:-translate-y-px ${
        active ? 'border-accent shadow-[0_0_0_1px_#5cc078]' : 'border-line2'
      }`}
      onClick={onClick}
      title={`Show all ${total.toLocaleString()} assets (clear ${groupBy} filter)`}
    >
      <div className="absolute inset-0 bg-[#0a0a0a] flex items-center justify-center overflow-hidden" style={{ background: 'linear-gradient(135deg, #1a2a1a 0%, #142420 50%, #1a1a3a 100%)' }}>
        <span className={`text-5xl font-bold text-accent ${active ? 'opacity-100' : 'opacity-50'}`} style={{ textShadow: '0 0 12px rgba(92,192,120,0.4)' }}>∗</span>
      </div>
      <div className="absolute left-0 right-0 bottom-0 pt-[22px] pb-2 px-2 bg-gradient-to-b from-transparent to-black/85 text-fg flex flex-col gap-[3px] pointer-events-none text-left">
        <span className="font-semibold text-[13px]/[1.2] drop-shadow overflow-hidden text-ellipsis whitespace-nowrap">all</span>
        <span className="font-mono font-semibold text-[11px]/[1] text-fg-0">{total.toLocaleString()}</span>
      </div>
    </button>
  );
}

function FilterCardTile({ card, rank, active, pinned, onClick }: TileProps) {
  const borderCls = pinned
    ? 'border-accent'
    : active
      ? 'border-accent shadow-[0_0_0_1px_#5cc078]'
      : 'border-line2 hover:border-accent';
  return (
    <button
      type="button"
      className={`filter-card relative aspect-square bg-bg-2 rounded-lg overflow-hidden cursor-pointer p-0 text-inherit border transition-colors hover:border-accent hover:-translate-y-px ${borderCls}`}
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
      <div className="absolute inset-0 bg-[#0a0a0a] flex items-center justify-center overflow-hidden">
        {showImg ? (
          <img
            src={thumbUrl(card.sampleAssetId as string)}
            alt={card.label}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover"
            onError={() => setImgError(true)}
          />
        ) : (
          <span className="text-[2.25rem] font-semibold uppercase text-[#444]">{fallbackChar}</span>
        )}
      </div>
      <div className="absolute left-0 right-0 bottom-0 pt-[22px] pb-2 px-2 bg-gradient-to-b from-transparent to-black/85 text-fg flex flex-col gap-[3px] pointer-events-none text-left">
        <span className="font-semibold italic text-[13px]/[1.2] drop-shadow overflow-hidden text-ellipsis whitespace-nowrap">{card.label}</span>
        <span className="font-mono font-semibold text-[11px]/[1] text-fg-0">{card.count.toLocaleString()}</span>
      </div>
    </>
  );
}