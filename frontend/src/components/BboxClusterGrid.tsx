// BboxClusterGrid — Level 2 view for a plant-box cluster.
// Multiple boxes from the same photo are merged into ONE photo card:
//   - If a photo has MULTIPLE boxes in this cluster: show the whole photo
//     with all member bounding boxes drawn on top via browser SVG overlay.
//   - If a photo has ONE box in this cluster: show the zoomed-in crop filling
//     the card, with a Picture-in-Picture (PiP) whole-photo inset in the
//     bottom-right corner highlighting where the crop sits in the full image.

import { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { SortPicker } from './SortPicker';
import { getBboxClusterItems, getClusterGroupItems, getClusterGroups } from '../api/endpoints';
import type { BboxClusterBox, BboxClusterItem, BboxClusterItemsResponse, ClusterGroup } from '../api/types';
import { useApp, useFilters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { navigate } from '../hooks/useRoute';
import { formatPct } from '../lib/format';
import { thumbUrl } from '../lib/thumb';
import { useIntersectionObserver } from '../hooks/useIntersectionObserver';
import { GroupMembersStrip, RelatedClusters } from './RelatedClusters';

type BboxSort = 'score' | 'time' | 'area' | 'yolo';

const SORT_OPTIONS: { value: BboxSort; label: string }[] = [
  { value: 'score', label: 'ID confidence ↓' },
  { value: 'time',  label: 'Newest first' },
  { value: 'area',  label: 'Largest crop' },
  { value: 'yolo',  label: 'Detector score ↓' },
];

export function BboxClusterGrid() {
  const filters = useFilters();
  const setFilters = useUrlFilters();
  const { dispatch } = useApp();

  const isSuper = filters.group_by === 'supergroup' && filters.supergroup != null;
  const targetId = isSuper ? filters.supergroup : filters.cluster_id;
  const [sort, setSort] = useState<BboxSort>('score');
  const [data, setData] = useState<BboxClusterItemsResponse | null>(null);
  const [groupsData, setGroupsData] = useState<ClusterGroup[] | null>(null);
  const [items, setItems] = useState<BboxClusterItem[]>([]);
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const loadingMoreRef = useRef(false);
  const reqIdRef = useRef(0);

  useEffect(() => {
    if (!isSuper) { setGroupsData(null); return; }
    let cancelled = false;
    getClusterGroups()
      .then(j => {
        if (cancelled) return;
        setGroupsData(j.ungrouped ? [...j.groups, j.ungrouped] : j.groups);
      })
      .catch(() => { if (!cancelled) setGroupsData([]); });
    return () => { cancelled = true; };
  }, [isSuper]);

  const activeGroup = useMemo<ClusterGroup | null>(() => {
    if (!isSuper || filters.supergroup == null) return null;
    return groupsData?.find(g => g.group_id === filters.supergroup) ?? null;
  }, [groupsData, isSuper, filters.supergroup]);

  const fetchPage = useCallback(
    (page: number, pageSize: number) => {
      if (targetId == null) return Promise.reject(new Error('missing id'));
      return isSuper
        ? getClusterGroupItems(targetId, { page, page_size: pageSize, sort })
        : getBboxClusterItems(targetId, { page, page_size: pageSize, sort, dedupe_assets: true });
    },
    [isSuper, targetId, sort],
  );

  const runFetch = useCallback(() => {
    if (targetId == null) return;
    const reqId = ++reqIdRef.current;
    setLoading(true);
    setError(null);
    setItems([]);
    fetchPage(0, 60)
      .then(r => {
        if (reqId !== reqIdRef.current) return;
        setData(r);
        setItems(r.items);
      })
      .catch(e => {
        if (reqId !== reqIdRef.current) return;
        setError(e instanceof Error ? e.message : String(e));
      })
      .finally(() => {
        if (reqId === reqIdRef.current) setLoading(false);
      });
  }, [targetId, fetchPage]);

  useEffect(() => {
    runFetch();
  }, [runFetch]);

  const loadMore = useCallback(() => {
    if (targetId == null || !data || loadingMoreRef.current) return;
    const nextPage = data.pagination.page + 1;
    if (nextPage >= data.pagination.total_pages) return;
    loadingMoreRef.current = true;
    setLoadingMore(true);
    const reqId = reqIdRef.current;
    fetchPage(nextPage, data.pagination.page_size)
      .then(r => {
        if (reqId !== reqIdRef.current) return;
        setData(r);
        setItems(prev => [...prev, ...r.items]);
      })
      .catch(() => {})
      .finally(() => {
        loadingMoreRef.current = false;
        setLoadingMore(false);
      });
  }, [targetId, data, fetchPage]);

  const [sentinelRef, isSentinelVisible] = useIntersectionObserver<HTMLDivElement>();

  useEffect(() => {
    if (isSentinelVisible && !loading) loadMore();
  }, [isSentinelVisible, loading, loadMore]);

  const handleBack = () => {
    if (isSuper) setFilters({ supergroup: undefined, page: 0 });
    else setFilters({ cluster_id: undefined, page: 0 });
    navigate('browse-list');
  };

  if (targetId == null) return null;

  const meta = data?.cluster;
  const isUnclustered = !isSuper && targetId === -1;
  const title = isUnclustered
    ? 'Unclustered'
    : (isSuper ? activeGroup?.name : null) || meta?.name || `${isSuper ? 'Super' : 'Cluster'} #${targetId}`;

  return (
    <div className="flex flex-col">
      {/* Sticky back-bar */}
      <div className="sticky-toolbar">
        <div className="flex items-center gap-2 px-3 py-1.5">
          <button
            type="button"
            onClick={handleBack}
            className="inline-flex items-center gap-1 px-2.5 py-1 rounded-md border border-line2 bg-bg-1 text-fg text-xs hover:border-accent"
          >
            {isSuper ? '← Super' : '← Clusters'}
          </button>
          <div className="min-w-0 flex-1">
            <div className="flex flex-wrap items-baseline gap-x-2 gap-y-0">
              <h2 className="m-0 text-sm sm:text-base font-semibold text-fg truncate">
                {title}
              </h2>
              {!isUnclustered && (
                <span className="text-xs text-fg-2 tabular-nums">
                  #{targetId}
                </span>
              )}
              {meta && (
                <span className="text-xs text-fg-1 tabular-nums">
                  {meta.n_assets.toLocaleString()} photos
                  {isSuper && activeGroup ? ` · ${activeGroup.n_clusters} clusters` : ''}
                </span>
              )}
            </div>
          </div>

          <div className="flex items-center gap-2 ml-auto">
            <SortPicker value={sort} options={SORT_OPTIONS} onChange={setSort} />
          </div>
        </div>
        {/* Top species on its own full-width line (max 2 lines) — squeezed
            between the back button and the sort select it truncated after
            the first name on phones. */}
        {meta && meta.top_species.length > 0 && (
          <div className="px-3 pb-1.5 -mt-0.5 text-[11px] text-fg-2 line-clamp-2">
            {meta.top_species
              .map(s => `${s.common_name || s.species} (${s.count})`)
              .join(' · ')}
          </div>
        )}
      </div>

      {/* Super-group member chips or similar clusters strip */}
      {isSuper ? (
        activeGroup && <GroupMembersStrip group={activeGroup} />
      ) : !isUnclustered ? (
        <RelatedClusters />
      ) : null}

      {/* Photo-merged grid */}
      <div className="p-2 sm:p-3">
        {loading && (
          <div className="p-10 text-center text-fg-1 text-sm">Loading photos…</div>
        )}
        {error && (
          <div className="p-6 text-center text-red-300 text-sm">
            Failed to load box cluster: {error}
          </div>
        )}
        {!loading && !error && items.length === 0 && (
          <div className="p-10 text-center text-fg-1 text-sm">No photos in this cluster.</div>
        )}
        {!loading && !error && items.length > 0 && (
          <>
            <div className="grid grid-cols-2 sm:grid-cols-3 md:grid-cols-4 lg:grid-cols-5 xl:grid-cols-6 gap-2 sm:gap-2.5">
              {items.map(item => (
                <BboxPhotoCard
                  key={item.asset_id}
                  item={item}
                  onOpen={() => dispatch({ type: 'OPEN_MODAL', assetId: item.asset_id })}
                />
              ))}
            </div>
            {data && data.pagination.page + 1 < data.pagination.total_pages && (
              <div ref={sentinelRef} className="py-6 text-center text-xs text-fg-2">
                {loadingMore ? 'Loading more photos…' : 'Scroll for more'}
              </div>
            )}
          </>
        )}
      </div>
    </div>
  );
}

/**
 * Pure browser-side photo + subtle SVG viewfinder overlay.
 * Uses a real DOM <img> (so native loading="lazy" works reliably) wrapped
 * in an aspect-matched box with delicate 1px translucent rounded frames
 * so the photo stays the hero while the boxes give quiet visual context.
 */
function PhotoWithBoxes({
  src,
  boxes,
  variant = 'card',
  alt = '',
}: {
  src: string;
  boxes: BboxClusterBox[];
  variant?: 'card' | 'pip';
  alt?: string;
}) {
  const [nat, setNat] = useState<{ w: number; h: number } | null>(null);
  const [err, setErr] = useState(false);

  const isPip = variant === 'pip';

  return (
    <div className="relative flex items-center justify-center w-full h-full bg-bg-0">
      {!nat && !err && <div className="absolute inset-0 shimmer" aria-hidden="true" />}
      <div
        className="relative flex items-center justify-center max-w-full max-h-full"
        style={nat ? { aspectRatio: `${nat.w} / ${nat.h}` } : { width: '100%', height: '100%' }}
      >
        <img
          src={src}
          alt={alt}
          loading="lazy"
          decoding="async"
          onLoad={(e) => {
            const el = e.currentTarget;
            if (el.naturalWidth && el.naturalHeight) {
              setNat({ w: el.naturalWidth, h: el.naturalHeight });
            }
          }}
          onError={() => setErr(true)}
          className={`block max-w-full max-h-full w-full h-full object-contain transition-opacity duration-200 ${nat ? 'opacity-100' : 'opacity-0'}`}
        />
        {nat && boxes.length > 0 && (
          <svg
            className={`absolute inset-0 w-full h-full pointer-events-none transition-opacity duration-200 ${
              isPip ? 'opacity-90' : 'opacity-75 group-hover:opacity-100'
            }`}
            viewBox={`0 0 ${nat.w} ${nat.h}`}
            preserveAspectRatio="none"
            style={{ filter: 'drop-shadow(0 0.5px 1.5px rgba(0, 0, 0, 0.55))' }}
          >
            {boxes.map(b => {
              const x = b.x * nat.w;
              const y = b.y * nat.h;
              const w = b.w * nat.w;
              const h = b.h * nat.h;
              const r = Math.min(w, h) * (isPip ? 0.08 : 0.05);
              return (
                <rect
                  key={b.detection_id}
                  x={x}
                  y={y}
                  width={w}
                  height={h}
                  rx={r}
                  ry={r}
                  fill={isPip ? 'rgba(255, 255, 255, 0.14)' : 'rgba(255, 255, 255, 0.04)'}
                  stroke={isPip ? 'rgba(255, 255, 255, 0.85)' : 'rgba(255, 255, 255, 0.78)'}
                  strokeWidth={1}
                  vectorEffect="non-scaling-stroke"
                />
              );
            })}
          </svg>
        )}
      </div>
    </div>
  );
}

function BboxPhotoCard({ item, onOpen }: { item: BboxClusterItem; onOpen: () => void }) {
  const [cropLoaded, setCropLoaded] = useState(false);
  const boxes = item.boxes?.length ? item.boxes : [];
  const isWholePhoto = item.kind === 'photo' || boxes.length === 0;
  const isMulti = !isWholePhoto && boxes.length > 1;
  const title = item.common_name || item.species || item.label;
  const dateStr = item.captured_at ? item.captured_at.slice(0, 10) : '';
  const photoSrc = thumbUrl(item.asset_id, 'thumbnail');

  return (
    <button
      type="button"
      onClick={onOpen}
      className="group text-left rounded-lg border border-line2 bg-bg-1 overflow-hidden hover:border-accent transition-colors flex flex-col"
      title={`${item.species || item.label}${item.common_name ? ` (${item.common_name})` : ''} · ${isWholePhoto ? 'whole-photo fallback' : `${boxes.length} box${boxes.length === 1 ? '' : 'es'} in this cluster`} · ${item.original_file_name}${item.is_new_asset ? ' · Not in any whole-image cluster' : ''}`}
    >
      <div className="relative aspect-square w-full overflow-hidden bg-bg-0">
        {isWholePhoto ? (
          /* Whole-photo fallback (no qualifying plant box): show clean full photo */
          <>
            {!cropLoaded && <div className="absolute inset-0 shimmer" aria-hidden="true" />}
            <img
              src={photoSrc}
              alt={title}
              loading="lazy"
              decoding="async"
              onLoad={() => setCropLoaded(true)}
              className={`h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03] ${cropLoaded ? 'opacity-100' : 'opacity-0'}`}
            />
          </>
        ) : isMulti ? (
          /* Multi-box photo: show full photo with delicate 1px viewfinder frames */
          <PhotoWithBoxes src={photoSrc} boxes={boxes} variant="card" alt={title} />
        ) : (
          /* Single-box photo: zoomed-in crop + subtle corner PiP of the full photo */
          <>
            {!cropLoaded && <div className="absolute inset-0 shimmer" aria-hidden="true" />}
            <img
              src={item.thumb_url}
              alt={title}
              loading="lazy"
              decoding="async"
              onLoad={() => setCropLoaded(true)}
              className={`h-full w-full object-cover transition-transform duration-300 group-hover:scale-[1.03] ${cropLoaded ? 'opacity-100' : 'opacity-0'}`}
            />
            <div
              className="absolute bottom-1.5 right-1.5 w-[30%] h-[30%] rounded-md overflow-hidden border border-white/30 shadow-[0_2px_8px_rgba(0,0,0,0.55)] bg-black/75 opacity-90 group-hover:opacity-100 transition-all duration-200 group-hover:scale-110 origin-bottom-right"
              title="Full photo context"
            >
              <PhotoWithBoxes src={photoSrc} boxes={boxes} variant="pip" />
            </div>
          </>
        )}

        <span className="absolute top-1.5 right-1.5 z-[1] px-1.5 py-0.5 rounded bg-black/45 text-[9px] uppercase tracking-wider text-white/80 leading-none backdrop-blur-md">
          {isWholePhoto ? 'photo' : isMulti ? `${boxes.length} boxes` : item.label}
        </span>
      </div>
      <div className="p-1.5 flex flex-col gap-0.5 min-w-0">
        <div className="truncate text-xs font-medium text-fg">{title}</div>
        <div className="flex items-center justify-between gap-1 text-[10px] text-fg-2">
          <span className="truncate">{dateStr || item.original_file_name}</span>
          {item.species_score != null && (
            <span className="tabular-nums text-accent flex-shrink-0">
              {formatPct(item.species_score)}
            </span>
          )}
        </div>
      </div>
    </button>
  );
}
