// KnnAlert — red-tinted warning shown at the top of the Run page
// when the KNN reference index is missing. Renders nothing while
// loading or when the index exists. The sole KNN-unavailable
// indicator on the Run view (the header KnnPill was removed).

import { useEffect, useState } from 'react';
import { getKnnStatus } from '../api/endpoints';
import type { KnnStatus } from '../api/types';

export function KnnAlert() {
  const [data, setData] = useState<KnnStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    getKnnStatus()
      .then(d => { if (!cancelled) setData(d); })
      .catch(() => { /* keep null — silently no alert */ });
    return () => { cancelled = true; };
  }, []);

  // Loading or build currently in flight → no alert.
  // While building the JobPanel shows live progress; stacking the
  // "missing" warning on top of an active build is misleading
  // (looks like the build is doing nothing).
  if (!data || data.building) return null;

  // Built but failing validation: worse than missing, because the
  // predictions look plausible. Say so explicitly.
  if (data.index_built && data.healthy === false) {
    return (
      <div className="knn-alert" role="alert">
        <svg className="knn-alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
          <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
          <line x1="12" y1="9" x2="12" y2="13" />
          <line x1="12" y1="17" x2="12.01" y2="17" />
        </svg>
        <div>
          <h4 className="knn-alert-title">KNN: index failed validation</h4>
          <p className="knn-alert-body">
            The reference index does not contain L2-normalized vectors, so
            its nearest neighbours are effectively arbitrary — species
            shown from KNN are <strong>not trustworthy</strong>. Rebuild
            it from the Run page → <strong>Build KNN index</strong>.
          </p>
          {data.health_reason && (
            <p className="knn-alert-body text-[10px] text-fg-1 mt-1">
              <code className="font-mono">{data.health_reason}</code>
            </p>
          )}
        </div>
      </div>
    );
  }

  if (data.index_built) return null;

  return (
    <div className="knn-alert" role="alert">
      <svg className="knn-alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      <div>
        <h4 className="knn-alert-title">KNN: reference index missing</h4>
        <p className="knn-alert-body">
          Build a KNN index to enable re-classification. Open the Run
          page → <strong>Build KNN index</strong> → pick a TreeOfLife-200M
          parquet file. Builds take hours to days depending on size.
          Until built, KNN reclassification is disabled.
        </p>
        {data.inventory_dir && (
          <p className="knn-alert-body text-[10px] text-fg-1 mt-1">
            Expected at: <code className="font-mono">{data.inventory_dir}</code>
          </p>
        )}
      </div>
    </div>
  );
}