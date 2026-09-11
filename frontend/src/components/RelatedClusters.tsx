// RelatedClusters — chip strip under the Level-2 back-bar, shown only
// while a cluster filter is active. Each chip is a cluster whose BioCLIP
// centroid is close to the open cluster's (cosine similarity — see
// /api/clusters/related); clicking one swaps the cluster filter in place,
// so browse "sideways" through embedding space without going back to the
// card grid.
//
// Deliberately NOT rendered for species/genus/family views — the
// similarity is defined over HDBSCAN cluster centroids, so a "related
// species" strip would be a different (undefined) metric.

import { useEffect, useState } from 'react';
import { getRelatedClusters } from '../api/endpoints';
import type { RelatedCluster } from '../api/types';
import { useFilters } from '../state/AppContext';
import { useUrlFilters } from '../hooks/useUrlFilters';
import { thumbUrl } from '../lib/thumb';

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
    <div className="related-bar" data-testid="related-clusters">
      <span className="related-bar-label">Related</span>
      {loading ? (
        Array.from({ length: 4 }, (_, i) => (
          <div key={i} className="related-chip" aria-hidden="true">
            <span className="related-chip-thumb shimmer" />
            <span className="py-1.5 space-y-1">
              <span className="related-chip-name block w-16 h-2 rounded bg-bg-2" />
              <span className="block w-10 h-2 rounded bg-bg-2" />
            </span>
          </div>
        ))
      ) : (
        related.map(c => (
          <button
            key={c.cluster_id}
            type="button"
            className="related-chip group"
            title={`${c.name || `cluster #${c.cluster_id}`} · ${c.n.toLocaleString()} photos · similarity ${(c.similarity * 100).toFixed(0)}%`}
            onClick={() => setFilters({ cluster_id: c.cluster_id, page: 0 })}
          >
            <RelatedChipThumb clusterId={c.cluster_id} assetId={c.sample_asset_id} name={c.name} />
            <span className="min-w-0 py-1">
              <span className="related-chip-name block group-hover:text-accent">
                {c.name || `cluster #${c.cluster_id}`}
              </span>
              <span className="related-chip-meta block">
                {c.n.toLocaleString()} · {(c.similarity * 100).toFixed(0)}%
              </span>
            </span>
          </button>
        ))
      )}
    </div>
  );
}

// Thumb with the same imgError fallback pattern as FilterCards'
// ClusterCardOverlay — Immich thumbs can 404 for purged assets and a
// broken <img> icon inside a tiny pill looks worse than a letter.
function RelatedChipThumb({ clusterId, assetId, name }: { clusterId: number; assetId: string | null; name: string }) {
  const [imgError, setImgError] = useState(false);
  const showImg = assetId && !imgError;
  const fallbackChar = (name || String(clusterId)).charAt(0).toUpperCase() || '·';
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
