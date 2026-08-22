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
import { navigate, useRoute } from '../hooks/useRoute';
import { thumbUrl } from '../lib/thumb';

export type CardSort = 'alpha' | 'size' | 'id' | 'latest';
const SORTS: { value: CardSort; label: string; hint: string }[] = [
  { value: 'size',   label: 'size',   hint: 'largest cluster/taxon first' },
  { value: 'alpha',  label: 'A→Z',    hint: 'alphabetical by label' },
  { value: 'latest', label: 'latest', hint: 'most recent captured_at in the group first (nulls last)' },
  { value: 'id',     label: 'id',     hint: 'internal cluster id order (cluster only)' },
];

// Plantae-tier chips render here (and only here) because FilterCards
// mounts on BOTH browse-list and browse-content (App.tsx), whereas
// Toolbar only mounts on browse-list. Showing the chips on the grid
// view lets the user filter plants/non-plants without first clicking
// back to the card list.
const PLANTAE_OPTS: Array<{ v: 'plant' | 'non_plant' | 'uncertain' | undefined; label: string; color: string; title: string }> = [
  { v: undefined,  label: 'all',      color: 'transparent',  title: 'Show every photo (plants + uncertain + non-plants)' },
  { v: 'plant',    label: 'plants',   color: '#1a3a1a',     title: 'Plantae ≥ 0.60' },
  { v: 'uncertain', label: 'uncertain', color: '#3a3a1a',    title: 'Plantae 0.35–0.60' },
  { v: 'non_plant', label: 'non-plants', color: '#3a1a1a',   title: 'Plantae < 0.35 (or unknown)' },
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
  const debouncedQ = useDebounce(filters.q, 300);
  // The `person` chip is a top-level facet that only applies to the
  // photo grid (it gets passed to /api/results via Grid.tsx). Cluster
  // and taxonomy counts use unfiltered aggregations on the backend
  // (storage.cluster_summary / get_global_taxonomy_summary), so
  // showing the chip on the cluster grid would be misleading — it'd
  // look like it filters the cluster counts when it doesn't. Hide it
  // on browse-list and show only on browse-content.
  const view = useRoute();
  const showPersonChip = view === 'browse-content';

  const [clusterData, setClusterData] = useState<ClusterSummary[] | null>(null);
  const [resultsData, setResultsData] = useState<ResultsResponse | null>(null);
  const [sortMode, setSortMode] = useState<CardSort>(() => readSort(filters.group_by));
  const [forceShow, setForceShow] = useState(false);
  const COLLAPSE_KEY = (g: GroupBy) => `filtercards:collapsed:${g}`;

  useEffect(() => { setSortMode(readSort(filters.group_by)); }, [filters.group_by]);
  useEffect(() => { localStorage.setItem(SORT_KEY(filters.group_by), sortMode); }, [filters.group_by, sortMode]);

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
      // Sort by captured_at DESC, NULLs last. ISO 8601 UTC strings sort
      // chronologically lexicographically, so a string compare with a
      // null-first sentinel gives DESC with nulls at the bottom.
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

  // The "all" tile total = the grid's pagination.total (raw file count
  // for the active plantae tier). Sum of card counts is shown
  // elsewhere as a sanity check (it should match). Fall back to the
  // card sum while resultsData is loading so the tile doesn't flash 0.
  // Same optional-chain trap as Grid.tsx — `?.pagination.total`
    // short-circuits only `.pagination`, so a server response without
    // a `pagination` field (e.g. the empty-filter sentinel) would
    // throw when reading `.total`. Chain `?.` on every segment.
    const totalAll = resultsData?.pagination?.total ?? allCards.reduce((a, b) => a + b.count, 0);
  const isLoading =
    (filters.group_by === 'cluster' && !clusterData) ||
    (filters.group_by !== 'cluster' && !resultsData);

  // Render the toolbar (sort + plantae + person chips) on BOTH
  // browse-list and browse-content. Only render the cards grid on
  // browse-list (where it makes sense). Hide entirely only when the
  // user explicitly collapsed an active card on browse-list.
  if (collapsed && activeCard && !showPersonChip) return null;

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
        {/* Plantae-tier chip — top-level facet that narrows both the
            filter cards AND the photo grid. Lives in FilterCards
            (not Toolbar) so it's visible on both views. The colored
            squares match the LegendBar tiers. "all" clears the facet
            (sets to undefined). Clicking any chip resets page to 0. */}
        <span className="inline-flex items-center gap-1 ml-2" role="radiogroup" aria-label="Plantae tier filter">
          <span className="text-fg-1 mr-[2px]">plants</span>
          {PLANTAE_OPTS.map((opt, i, arr) => (
            <span key={opt.label} className="inline-flex items-center gap-1">
              <button
                type="button"
                role="radio"
                aria-checked={filters.plantae === opt.v}
                className={`inline-flex items-center gap-1 bg-transparent border-0 p-0 m-0 cursor-pointer text-[0.85rem] leading-[22px] ${
                  filters.plantae === opt.v ? 'text-accent font-medium' : 'text-fg-0 hover:text-fg'
                }`}
                onClick={() => setFilters({ plantae: opt.v, page: 0 })}
                title={opt.title}
              >
                <span
                  className="inline-block w-2 h-2 rounded-sm"
                  style={{ background: opt.color, border: opt.color === 'transparent' ? '1px solid #456' : undefined }}
                />
                {opt.label}
              </button>
              {i < arr.length - 1 && <span className="text-fg-3">/</span>}
            </span>
          ))}
        </span>
        {/* Person filter — top-level facet that only applies to the
            photo grid (only shown when the user is on browse-content).
            Three-state chip group: any / has person / no person. Backed
            by /api/results?has_person= (server-side). Unscanned rows
            (NULL) are excluded by both 0 and 1, so neither shows them
            — that's the intentional "don't claim 'no person' until
            we've actually checked" semantics. */}
        {showPersonChip && (
          <span className="inline-flex items-center gap-1 ml-2" role="radiogroup" aria-label="Person filter">
            <span className="text-fg-1 mr-[2px]">person</span>
            {([
              { v: undefined, label: 'any' },
              { v: 1 as const, label: 'has' },
              { v: 0 as const, label: 'no' },
            ]).map((opt, i, arr) => (
              <span key={opt.label} className="inline-flex items-center gap-1">
                <button
                  type="button"
                  role="radio"
                  aria-checked={filters.has_person === opt.v}
                  className={`bg-transparent border-0 p-0 m-0 cursor-pointer text-[0.85rem] leading-[22px] ${
                    filters.has_person === opt.v ? 'text-accent font-medium' : 'text-fg-0 hover:text-fg'
                  }`}
                  onClick={() => setFilters({ has_person: opt.v, page: 0 })}
                  title={
                    opt.v === undefined
                      ? 'Show all photos (scanned + unscanned)'
                      : opt.v === 1
                        ? 'Show only photos with a person detected'
                        : 'Show only photos scanned and confirmed without a person'
                  }
                >{opt.label}</button>
                {i < arr.length - 1 && <span className="text-fg-3">/</span>}
              </span>
            ))}
          </span>
        )}
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

      {/* Cards grid only on browse-list. On browse-content we already
          rendered the toolbar above (sort + chips); the cards
          themselves make no sense when looking at the photo grid. */}
      {!showPersonChip && (
      <div className="grid gap-2.5 py-1 grid-cols-[repeat(auto-fill,minmax(160px,1fr))]">
        <FilterAllCard
          total={totalAll}
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
      )}
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
        active ? 'border-accent shadow-[0_0_0_1px#5cc078]' : 'border-line2'
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
      ? 'border-accent shadow-[0_0_0_1px#5cc078]'
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
