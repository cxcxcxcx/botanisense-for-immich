// FilterCards — Level 1 of the two-level browse layout.
//   • sticky toolbar at the top: search pill only
//   • quick bar below: Group segmented control + Sort menu
//   • count line
//   • cluster/taxon grid (square cards, rounded-xl per the mock)
//
// Plantae tier + Person filters are NOT surfaced on Level 1 — the
// Filter trigger button was removed because the chips + button
// combination looked unused and the URL params alone provide enough
// filtering for the cluster list. The same filters stay reachable
// from Level 2 via Grid.tsx's back-bar (same FilterModal component).
// Plantae tier stays in scope here so /api/clusters is server-side
// filtered when the user arrives on Level 1 with plantae= in the URL.

import { useEffect, useMemo, useRef, useState } from 'react';
import { getClusters, getResults } from '../api/endpoints';
import type { ClusterSummary, ResultsResponse, TaxonomyEntry } from '../api/types';
import { useFilters, type GroupBy } from '../state/AppContext';
import { useDebounce } from '../hooks/useDebounce';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate, useRoute } from '../hooks/useRoute';
import { thumbUrl } from '../lib/thumb';

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
  const view = useRoute();

  const [clusterData, setClusterData] = useState<ClusterSummary[] | null>(null);
  const [resultsData, setResultsData] = useState<ResultsResponse | null>(null);
  const [sortMode, setSortMode] = useState<CardSort>(() => readSort(filters.group_by));
  const [searchInput, setSearchInput] = useState(filters.q);

  // Debounce the LOCAL input (not the URL-derived filters.q) so the
  // input stays responsive while typing. The previous version
  // debounced filters.q, which meant typing updated the local input
  // but never pushed back to the URL — so the search filter looked
  // stuck at whatever value the URL had when the page loaded.
  const debouncedSearch = useDebounce(searchInput, 200);

  useEffect(() => { setSortMode(readSort(filters.group_by)); }, [filters.group_by]);
  useEffect(() => { localStorage.setItem(SORT_KEY(filters.group_by), sortMode); }, [filters.group_by, sortMode]);

  // Sync the local input when the URL changes externally (back/forward,
  // a deep-link, or another component). Don't clobber a user keystroke
  // mid-edit — only mirror when the URL value actually differs.
  useEffect(() => {
    setSearchInput(prev => (prev === filters.q ? prev : filters.q));
  }, [filters.q]);

  // Push the debounced search value into the URL. The guard ensures
  // we don't loop with the sync effect above (if filters.q already
  // matches the debounced value, this is a no-op).
  useEffect(() => {
    if (debouncedSearch !== filters.q) setFilters({ q: debouncedSearch, page: 0 });
    // setFilters is stable from useUrlFilters; excluding it from deps
    // avoids re-running on every parent re-render.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [debouncedSearch]);

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

  // FilterCards is Level-1 only now (Toolbar mounts it on browse-list;
  // Grid owns its own Filter button on browse-content). Kept the
  // isPhotoView guard so this component is safe to mount on either
  // view, even though current routing only mounts it on browse-list.
  const isPhotoView = view === 'browse-content';

  return (
    <div className={isPhotoView ? '' : 'flex flex-col'}>
      {!isPhotoView && (
        <div className="sticky-toolbar">
          {/* Row 1: search pill only on Level 1.
              The Filter trigger button was removed — the Plantae /
              Person filters are still reachable from Level 2's
              back-bar (Grid.tsx mounts the same FilterModal). The
              URL params persist, so server-side counts and taxonomy
              stays correctly filtered. */}
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
          </div>

          {/* Row 2: quick bar.
                - Group chips: full segmented control on every viewport
                  (no "Group" label — the chip text is self-explanatory;
                  4 chips fit without scrolling on a 375 px screen).
                - Sort: full segmented control on sm+, collapses to an
                  icon button with a dropdown menu on narrow viewports
                  so we don't overflow horizontally. */}
            <div className="quick-bar">
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
              <span className="quick-bar-divider" aria-hidden="true" />
              {/* Sort: full segmented control on sm+ ... */}
              <div className="hidden sm:flex quick-bar-group">
                {SORTS.map(s => (
                  <button
                    key={s.value}
                    type="button"
                    className={sortMode === s.value ? 'seg-chip seg-chip-active' : 'seg-chip seg-chip-inactive'}
                    onClick={() => setSortMode(s.value)}
                  >{s.label}</button>
                ))}
              </div>
              {/* ... icon button + dropdown on narrow viewports */}
              <SortMenu
                value={sortMode}
                onChange={setSortMode}
              />
            </div>
        </div>
      )}

      {/* Cards grid only on browse-list. Always renders on
          browse-list — the prior `&& !collapsed` gate was hiding the
          grid after navigating back from a cluster (race between
          setFilters and navigate left `activeCard` set on first
          paint). Now that FilterCards is Level-1 only, the cluster
          grid should always be visible here. */}
      {!isPhotoView && (
        <div className="p-2 sm:p-3">
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
              <div className="p-10 text-center text-fg-1 whitespace-pre-wrap col-span-full">Loading…</div>
            ) : visibleCards.length === 0 && !stickyActive ? (
              // Empty state for Level 1. The most common cause is the
              // local `q=` search pill filtering out every card; show
              // which search and offer a one-click clear so the user
              // doesn't have to hunt for the input.
              <div className="p-10 text-center text-fg-1 whitespace-pre-wrap col-span-full">
                {filters.q.trim() ? (
                  <>
                    <p className="m-0">No {filters.group_by}s match <code className="font-mono">"{filters.q.trim()}"</code>.</p>
                    <button
                      type="button"
                      className="mt-3 px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345] hover:border-accent"
                      onClick={() => {
                        setSearchInput('');
                        setFilters({ q: '', page: 0 });
                      }}
                    >
                      Clear search
                    </button>
                  </>
                ) : (
                  <p className="m-0">No {filters.group_by}s in the library.</p>
                )}
              </div>
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
      <div className="absolute inset-0 flex items-center justify-center bg-gradient-to-br from-emerald-900/40 via-teal-900/30 to-indigo-900/40 dark-gradient">
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
      className={`cluster-card group ${borderCls}`}
      onClick={onClick}
      title={`${card.label} · ${card.count.toLocaleString()} asset${card.count === 1 ? '' : 's'}${pinned ? ' · click to clear' : ''}`}
    >
      <ClusterCardOverlay card={card} />
    </button>
  );
}

function ClusterCardOverlay({ card }: { card: Card }) {
  const [imgError, setImgError] = useState(false);
  const [imgLoaded, setImgLoaded] = useState(false);
  const showImg = card.sampleAssetId && !imgError;
  const fallbackChar = card.label.replace(/^["'(]|["')']$/g, '').charAt(0).toUpperCase() || '·';
  return (
    <>
      <div className="absolute inset-0 flex items-center justify-center overflow-hidden">
        {showImg && !imgLoaded && (
          <div className="absolute inset-0 shimmer" aria-hidden="true" />
        )}
        {showImg ? (
          <img
            src={thumbUrl(card.sampleAssetId as string)}
            alt={card.label}
            loading="lazy"
            decoding="async"
            onLoad={() => setImgLoaded(true)}
            onError={() => setImgError(true)}
            className={`w-full h-full object-cover transition-transform duration-500 group-hover:scale-105 transition-opacity duration-200 ${imgLoaded ? 'opacity-100' : 'opacity-0'}`}
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

// SortMenu — icon button + dropdown shown on narrow viewports. The
// Sort segmented control renders on sm+ instead. Owns its own open
// state and click-outside-to-close; the parent only passes the
// current sort value + a setter.
function SortMenu({ value, onChange }: { value: CardSort; onChange: (v: CardSort) => void }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);

  // Click-outside closes the popover. Only attach the listener while
  // open so we don't pay the document-level handler cost when the
  // menu isn't visible.
  useEffect(() => {
    if (!open) return;
    const handler = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener('click', handler);
    return () => document.removeEventListener('click', handler);
  }, [open]);

  const activeLabel = SORTS.find(s => s.value === value)?.label ?? '';
  return (
    <div className="relative sm:hidden" ref={ref}>
      <button
        type="button"
        className="sort-menu-trigger"
        onClick={() => setOpen(o => !o)}
        title="Sort options"
        aria-haspopup="listbox"
        aria-expanded={open}
      >
        <SortAscIcon />
        <span className="hidden xs:inline">{activeLabel}</span>
        <ChevronDownIcon />
      </button>
      {open && (
        <div className="sort-menu-pop" role="listbox">
          {SORTS.map(s => (
            <button
              key={s.value}
              type="button"
              role="option"
              aria-selected={value === s.value}
              className={value === s.value ? 'sort-menu-item sort-menu-item-active' : 'sort-menu-item'}
              onClick={() => { onChange(s.value); setOpen(false); }}
            >{s.label}</button>
          ))}
        </div>
      )}
    </div>
  );
}

function SortAscIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M3 6h18" />
      <path d="M7 12h10" />
      <path d="M10 18h4" />
    </svg>
  );
}

function ChevronDownIcon() {
  return (
    <svg width="10" height="10" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="6 9 12 15 18 9" />
    </svg>
  );
}