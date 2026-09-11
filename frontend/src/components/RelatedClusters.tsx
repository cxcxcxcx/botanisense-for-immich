// RelatedClusters — chip strip under the Level-2 back-bar. Two variants
// share the same chip UI:
//
//   <RelatedClusters />     — nearest clusters by centroid cosine
//                             (/api/clusters/related), shown while a
//                             CLUSTER filter is active: browse "sideways"
//                             through embedding space.
//   <GroupMembersStrip />   — member clusters of the active SUPER-GROUP
//                             (data fetched by Grid, passed as a prop):
//                             browse DOWN the hierarchy into a family.
//
// Both swap the filter in place, so neither navigation round-trips
// through the Level-1 card grid.

import { useEffect, useState } from 'react';
import { getRelatedClusters } from '../api/endpoints';
import type { ClusterGroup, RelatedCluster } from '../api/types';
import { useFilters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { thumbUrl } from '../lib/thumb';

// Presentational chip row. `chips` carry display data only — selection
// behavior is decided by the variant wrapper below.
interface Chip {
  key: string;
  label: string;
  meta: string;
  title: string;
  assetId: string | null;
  onClick: () => void;
}

function ClusterChipStrip({ label, chips, loading }: { label: string; chips: Chip[]; loading?: boolean }) {
  return (
    <div className="related-bar" data-testid="related-clusters">
      <span className="related-bar-label">{label}</span>
      {loading
        ? Array.from({ length: 4 }, (_, i) => (
            <div key={i} className="related-chip" aria-hidden="true">
              <span className="related-chip-thumb shimmer" />
              <span className="py-1.5 space-y-1">
                <span className="related-chip-name block w-16 h-2 rounded bg-bg-2" />
                <span className="block w-10 h-2 rounded bg-bg-2" />
              </span>
            </div>
          ))
        : chips.map(c => (
            <button key={c.key} type="button" className="related-chip group" title={c.title} onClick={c.onClick}>
              <ChipThumb assetId={c.assetId} fallback={c.label} />
              <span className="min-w-0 py-1">
                <span className="related-chip-name block group-hover:text-accent">{c.label}</span>
                <span className="related-chip-meta block">{c.meta}</span>
              </span>
            </button>
          ))}
    </div>
  );
}

export function RelatedClusters() {
  const filters = useFilters();
  const setFilters = useUrlFilters();
  const activeCluster =
    filters.group_by === 'cluster' && filters.cluster_id != null
      ? filters.cluster_id
      : null;

  const [related, setRelated] = useState<RelatedCluster[] | null>(null);
  // Which cluster the `related` payload belongs to. While a new fetch is
  // in flight after a chip click, loadedFor lags activeCluster and the
  // strip shows shimmer chips instead of the previous cluster's
  // neighbors (which would be misleading for a heartbeat).
  const [loadedFor, setLoadedFor] = useState<number | null>(null);

  useEffect(() => {
    if (activeCluster == null) {
      setRelated(null);
      setLoadedFor(null);
      return;
    }
    let cancelled = false;
    getRelatedClusters(activeCluster)
      .then(r => {
        if (cancelled) return;
        setRelated(r.related);
        setLoadedFor(r.cluster_id);
      })
      .catch(() => {
        // Empty (not null) — render the strip away rather than leaving
        // a stuck loading state when the endpoint errors.
        if (!cancelled) { setRelated([]); setLoadedFor(activeCluster); }
      });
    return () => { cancelled = true; };
  }, [activeCluster]);

  if (activeCluster == null) return null;

  const loading = related == null || loadedFor !== activeCluster;
  // The server returns [] both for "no clusters yet" and "this cluster
  // has no embedded assets" — either way there's nothing useful to show.
  if (!loading && related.length === 0) return null;

  return (
    <ClusterChipStrip
      label="Related"
      loading={loading}
      chips={loading ? [] : related.map(c => ({
        key: String(c.cluster_id),
        label: c.name || `cluster #${c.cluster_id}`,
        meta: `${c.n.toLocaleString()} · ${(c.similarity * 100).toFixed(0)}%`,
        title: `${c.name || `cluster #${c.cluster_id}`} · ${c.n.toLocaleString()} photos · similarity ${(c.similarity * 100).toFixed(0)}%`,
        assetId: c.sample_asset_id,
        onClick: () => setFilters({ cluster_id: c.cluster_id, page: 0 }),
      }))}
    />
  );
}

// Member clusters of the active super-group. Grid owns the
// /api/cluster-groups fetch (it needs the family name for the back-bar)
// and hands the resolved group down. Renders nothing until Grid has both
// the group data AND a matching active supergroup filter — so a filter
// change can't show the previous family's members. `members` is capped
// server-side (the Ungrouped bucket holds hundreds of clusters).
export function GroupMembersStrip({ group }: { group: ClusterGroup | null }) {
  const setFilters = useUrlFilters();
  if (!group) return null;
  const capped = group.n_clusters > group.members.length;
  return (
    <ClusterChipStrip
      label="Members"
      chips={group.members.map(m => ({
        key: String(m.cluster_id),
        label: m.name || `cluster #${m.cluster_id}`,
        meta: m.n.toLocaleString(),
        title: `${m.name || `cluster #${m.cluster_id}`} · ${m.n.toLocaleString()} photos (member of ${group.name})`,
        assetId: null,
        onClick: () => setFilters({ group_by: 'cluster', cluster_id: m.cluster_id, page: 0 }),
      })).concat(capped ? [{
        key: 'more',
        label: `+${group.n_clusters - group.members.length} more`,
        meta: `${group.n.toLocaleString()} photos total`,
        title: `${group.n_clusters - group.members.length} more clusters in ${group.name} — all their photos are shown in this grid`,
        assetId: null,
        onClick: () => { /* informational — the grid already shows the whole group */ },
      }] : [])}
    />
  );
}

// Thumb with the same imgError fallback pattern as FilterCards'
// ClusterCardOverlay — Immich thumbs can 404 for purged assets and a
// broken <img> icon inside a tiny pill looks worse than a letter.
function ChipThumb({ assetId, fallback }: { assetId: string | null; fallback: string }) {
  const [imgError, setImgError] = useState(false);
  const showImg = assetId && !imgError;
  const fallbackChar = (fallback || '·').replace(/^["'(]|["')']$/g, '').charAt(0).toUpperCase() || '·';
  return showImg ? (
    <img
      src={thumbUrl(assetId)}
      alt=""
      loading="lazy"
      decoding="async"
      onError={() => setImgError(true)}
      className="related-chip-thumb"
    />
  ) : (
    <span className="related-chip-thumb flex items-center justify-center text-[11px] font-semibold uppercase text-fg-1">
      {fallbackChar}
    </span>
  );
}
