// FilterCards — Level 1 of the two-level browse layout.
//   • sticky toolbar at the top: search pill + filter-trigger button
//   • quick bar below: Group + Sort segmented controls (horizontal
//     scroll on overflow)
//   • count line
//   • cluster/taxon grid (square cards, rounded-xl per the mock)
//
// Plantae tier and Person filters moved into FilterModal.tsx —
// triggered by the filter-trigger button. Plantae tier stays in scope
// here so the cluster / taxonomy counts are filtered server-side
// (they're a backend parameter on /api/results and /api/clusters).

import { useEffect, useMemo, useState } from 'react';
import { getClusters, getResults } from '../api/endpoints';
import type { ClusterSummary, ResultsResponse, TaxonomyEntry } from '../api/types';
import { useApp, useFilters, type GroupBy } from '../state/AppContext';
import { useDebounce } from '../hooks/useDebounce';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate, useRoute } from '../hooks/useRoute';
import { thumbUrl } from '../lib/thumb';
import { FilterModal } from './FilterModal';

export type CardSort = 'alpha' | 'size' | 'id' | 'latest';
const SORTS: { value: CardSort; label: string }[] = [
  { value: 'size',   label: 'Size ↓' },
  { value: 'alpha',  label: 'A→Z' },
  { value: 'latest', label: 'Latest' },
  { value: 'id',     label: 'ID' },
];

const GROUP_BY_CHIPS: Array<{ v: GroupBy; label: string }> = [
  { v: 'cluster', label: 'Cluster' },
  { v: 'species', label: 'Species' },
  { v: 'genus',   label: 'Genus' },
  { v: 'family',  label: 'Family' },
];

interface Card {
  key: string;
  label: string;
  count: number;
  latest: string | null;
  sampleAssetId: string | null;
}

const SORT_KEY = (g: GroupBy) => `filtercards:sort:${g}`;

function readSort(g: GroupBy): CardSort {
  const v = localStorage.getItem(SORT_KEY(g));
  return v === 'alpha' || v === 'id' || v === 'latest' ? v : 'size';
}

export function FilterCards() {
  const filters = useFilters();
  const setFilters = useUrlFilters();
  const { dispatch } = useApp();
  const debouncedQ = useDebounce(filters.q, 200);
  const view = useRoute();

  const [clusterData, setClusterData] = useState<ClusterSummary[] | null>(null);
  const [resultsData, setResultsData] = useState<ResultsResponse | null>(null);
  const [sortMode, setSortMode] = useState<CardSort>(() => readSort(filters.group_by));
  const [searchInput, setSearchInput] = useState(filters.q);
  const [filterModalOpen, setFilterModalOpen] = useState(false);
  const [forceShow, setForceShow] = useState(false);
  const COLLAPSE_KEY = (g: GroupBy) => `filtercards:collapsed:${g}`;

  useEffect(() => { setSortMode(readSort(filters.group_by)); }, [filters.group_by]);
  useEffect(() => { localStorage.setItem(SORT_KEY(filters.group_by), sortMode); }, [filters.group_by, sortMode]);
  useEffect(() => { setSearchInput(filters.q); }, [filters.q]);

  // Push search into the URL once the user stops typing. Local state
  // mirrors the input so typing is responsive; debouncing here avoids
  // hammering the URL on every keystroke.
  useEffect(() => {
    if (debouncedQ !== filters.q) setFilters({ q: debouncedQ, page: 0 });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedQ]);

  // Cluster list re-fetches whenever group_by flips to cluster OR the
  // plantae tier changes. Both must be in the deps array.
  useEffect(() => {
    if (filters.group_by !== 'cluster') return;
    let cancelled = false;
    getClusters(filters.plantae).then(j => { if (!cancelled) setClusterData(j.clusters); }).catch(() => {});
    return () => { cancelled = true; };
  }, [filters.group_by, filters.plantae]);

  // /api/results is fetched for the "all" tile total. We need it on
  // BOTH the cluster view AND the taxonomy view (the previous behavior
  // only fetched it for taxonomy, which made the cluster "all" tile
  // sum local card counts and quietly disagree with the grid).
  useEffect(() => {
    let cancelled = false;
    getResults({ page: 0, page_size: 1, plantae: filters.plantae })
      .then(r => { if (!cancelled) setResultsData(r); })
      .catch(() => {});
    return () => { cancelled = true; };
  }, [filters.group_by, filters.plantae]);

  const allCards = useMemo<Card[]>(() => {
    if (filters.group_by === 'cluster') {
      if (!clusterData) return [];
      return clusterData.map(c => ({
        key: String(c.cluster_id),
        label: c.name || `cluster #${c.cluster_id}`,
        count: c.n,
        latest: c.latest_captured_at ?? null,
        sampleAssetId: c.sample_asset_id,
      }));
    }
    if (!resultsData) return [];
    const tax: TaxonomyEntry[] = resultsData.taxonomy[filters.group_by] ?? [];
    return tax.map(t => ({
      key: t.name,
      label: t.name,
      count: t.n,
      latest: t.latest_captured_at ?? null,
      sampleAssetId: t.sample_asset_id,
    }));
  }, [filters.group_by, clusterData, resultsData]);

  const sortedCards = useMemo<Card[]>(() => {
    const arr = [...allCards];
    if (sortMode === 'alpha') {
      arr.sort((a, b) => a.label.localeCompare(b.label));
    } else if (sortMode === 'latest') {
      arr.sort((a, b) => {
        if (!a.latest && b.latest) return 1;
        if (a.latest && !b.latest) return -1;
        if (!a.latest && !b.latest) return b.count - a.count || a.label.localeCompare(b.label);
        return b.latest!.localeCompare(a.latest!);
      });
    } else if (sortMode === 'id' && filters.group_by === 'cluster') {
      arr.sort((a, b) => Number(a.key) - Number(b.key));
    } else {
      arr.sort((a, b) => b.count - a.count || a.label.localeCompare(b.label));
    }
    return arr;
  }, [allCards, sortMode, filters.group_by]);

  const visibleCards = useMemo<Card[]>(() => {
    const q = filters.q.trim().toLowerCase();
    if (!q) return sortedCards;
    return sortedCards.filter(c => c.label.toLowerCase().includes(q));
  }, [sortedCards, filters.q]);

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

  // The "all" tile total = the grid's pagination.total (raw file count
  // for the active plantae tier). Sum of card counts is shown
  // elsewhere as a sanity check (it should match). Fall back to the
  // card sum while resultsData is loading so the tile doesn't flash 0.
  // Same optional-chain trap as Grid.tsx — `?.pagination.total`
  // short-circuits only `.pagination`, so a server response without a
  // `pagination` field (e.g. the empty-filter sentinel) would throw
  // when reading `.total`. Chain `?.` on every segment.
  const totalAll = resultsData?.pagination?.total ?? allCards.reduce((a, b) => a + b.count, 0);
  const isLoading =
    (filters.group_by === 'cluster' && !clusterData) ||
    (filters.group_by !== 'cluster' && !resultsData);

  const filtersActive = filters.plantae != null || filters.has_person != null;

  // On browse-content (photo grid) we only render the chrome —
  // clusters don't make sense there. The sticky toolbar / quick bar
  // / grid are Level-1 only.
  const isPhotoView = view === 'browse-content';

  return (
    <div className={isPhotoView ? '' : 'flex flex-col'}>
      {!isPhotoView && (
        <div className="sticky-toolbar">
          {/* Row 1: search pill + filter trigger. */}
          <div className="sticky-toolbar-pad">
            <div className="search-pill">
              <SearchIcon />
              <input
                type="text"
                placeholder="Filter by name..."
                className="search-pill-input"
                autoComplete="off"
                value={searchInput}
                onChange={(e) => setSearchInput(e.target.value)}
                title="Hide cards whose label doesn't match (sub-filter; doesn't change the active filter)"
              />
            </div>
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
          </div>

          {/* Row 2: quick bar (Group + Sort segmented). Horizontal
              scroll on overflow for narrow viewports. */}
          <div className="quick-bar hide-scrollbar">
            <div className="flex items-center gap-2">
              <span className="quick-bar-label">Group</span>
              <div className="quick-bar-group">
                {GROUP_BY_CHIPS.map(g => (
                  <button
                    key={g.v}
                    type="button"
                    className={filters.group_by === g.v ? 'seg-chip seg-chip-active' : 'seg-chip seg-chip-inactive'}
                    onClick={() => setFilters({ group_by: g.v, page: 0 })}
                    title={`Bucket photos by ${g.label.toLowerCase()}`}
                  >{g.label}</button>
                ))}
              </div>
            </div>
            <span className="quick-bar-divider" aria-hidden="true" />
            <div className="flex items-center gap-2">
              <span className="quick-bar-label">Sort</span>
              <div className="quick-bar-group">
                {SORTS.map(s => (
                  <button
                    key={s.value}
                    type="button"
                    className={sortMode === s.value ? 'seg-chip seg-chip-active' : 'seg-chip seg-chip-inactive'}
                    onClick={() => setSortMode(s.value)}
                  >{s.label}</button>
                ))}
              </div>
            </div>
            {activeCard && (
              <button
                type="button"
                onClick={toggleForceShow}
                className="seg-chip seg-chip-inactive ml-auto"
                title={forceShow ? 'Hide filter cards on the grid view' : 'Show filter cards on the grid view'}
              >{forceShow ? 'hide on grid' : 'show on grid'}</button>
            )}
          </div>
        </div>
      )}

      {/* Cards grid only on browse-list. Hide entirely only when the
          user explicitly collapsed an active card on browse-list. */}
      {!isPhotoView && !collapsed && (
        <div className="p-3">
          <div className="count-line">
            <span>{visibleCards.length} of {allCards.length} cards</span>
          </div>
          <div className="cluster-grid">
            <ClusterAllCard
              total={totalAll}
              active={activeKey == null}
              onClick={goToAll}
            />
            {stickyActive && (
              <ClusterCardTile
                card={stickyActive}
                active
                pinned
                onClick={() => { clearFilter(); navigate('browse-list'); }}
              />
            )}
            {isLoading && visibleCards.length === 0 && !stickyActive ? (
              <div className="p-10 text-center text-fg-1 whitespace-pre-wrap col-span-full">Loading filters…</div>
            ) : visibleCards.length === 0 && !stickyActive ? (
              <div className="p-10 text-center text-fg-1 whitespace-pre-wrap col-span-full">No matches.</div>
            ) : (
              visibleCards.map(c => (
                <ClusterCardTile
                  key={c.key}
                  card={c}
                  active={c.key === activeKey}
                  onClick={() => handleClick(c)}
                />
              ))
            )}
          </div>
        </div>
      )}

      {/* Active-filter pill — only on browse-content when a card is
          selected. Surfaces the "you've narrowed to X" state so the
          user has a one-click path back to All without scrolling up
          to the Level 1 toolbar. */}
      {!isPhotoView && false /* keep prop consistent with the prior design — no active pill on browse-list */}

      <FilterModal open={filterModalOpen} onClose={() => setFilterModalOpen(false)} />
    </div>
  );
}

function ClusterAllCard({ total, active, onClick }: { total: number; active: boolean; onClick: () => void }) {
  return (
    <button
      type="button"
      className={active ? 'cluster-card cluster-card-all' : 'cluster-card'}
      onClick={onClick}
      title={`Show all ${total.toLocaleString()} assets (clear filter)`}
    >
      <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-[#1a2a1a] via-[#142420] to-[#1a1a3a]">
        <span className={`text-4xl text-accent font-light mt-2 ${active ? 'opacity-100' : 'opacity-50'}`} style={{ textShadow: '0 0 12px rgba(92,192,120,0.4)' }}>*</span>
      </div>
      <div className="cluster-card-overlay" aria-hidden="true" />
      <div className="absolute bottom-0 left-0 right-0 p-2">
        <h3 className={`cluster-card-label ${active ? 'cluster-card-name-all' : 'cluster-card-name-all'}`}>all</h3>
        <p className="cluster-card-count">{total.toLocaleString()} items</p>
      </div>
    </button>
  );
}

function ClusterCardTile({ card, active, pinned, onClick }: { card: Card; active: boolean; pinned?: boolean; onClick: () => void }) {
  const borderCls = pinned ? 'border-accent' : active ? 'border-accent ring-1 ring-accent' : 'border-line2 hover:border-accent';
  return (
    <button
      type="button"
      className={`cluster-card ${borderCls}`}
      onClick={onClick}
      title={`${card.label} · ${card.count.toLocaleString()} asset${card.count === 1 ? '' : 's'}${pinned ? ' · click to clear' : ''}`}
    >
      <ClusterCardOverlay card={card} />
    </button>
  );
}

function ClusterCardOverlay({ card }: { card: Card }) {
  const [imgError, setImgError] = useState(false);
  const showImg = card.sampleAssetId && !imgError;
  const fallbackChar = card.label.replace(/^["'(]|["')']$/g, '').charAt(0).toUpperCase() || '·';
  return (
    <>
      <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
        {showImg ? (
          <img
            src={thumbUrl(card.sampleAssetId as string)}
            alt={card.label}
            loading="lazy"
            decoding="async"
            className="w-full h-full object-cover transition-transform duration-500 group-hover:scale-105"
            onError={() => setImgError(true)}
          />
        ) : (
          <span className="text-[2.25rem] font-semibold uppercase text-[#444]">{fallbackChar}</span>
        )}
      </div>
      <div className="cluster-card-overlay" aria-hidden="true" />
      <div className="absolute bottom-0 left-0 right-0 p-2">
        <h3 className="cluster-card-label cluster-card-name">{card.label}</h3>
        <p className="cluster-card-count">{card.count.toLocaleString()} items</p>
      </div>
    </>
  );
}

function SearchIcon() {
  return (
    <svg className="search-pill-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="11" cy="11" r="7" />
      <path d="m21 21-4.3-4.3" />
    </svg>
  );
}

function FilterIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polygon points="22 3 2 3 10 12.46 10 19 14 21 14 12.46 22 3" />
    </svg>
  );
}