// KnnStatusNote (rendered by <KnnAlert />) — inline KNN status shown
// INSIDE the Data tools card on the Run page, right above the
// "Build KNN index" button. Deliberately not a top-of-page banner:
// a missing index is a quality upgrade waiting to happen, not an
// error — so the missing state is a muted amber note that says the
// zero-shot results aren't great yet and points at the button below.
// Only a built-but-unhealthy index keeps the red alert treatment
// (that one IS a correctness problem: neighbours are arbitrary).

import { useEffect, useState } from 'react';
import { getKnnStatus } from '../api/endpoints';
import type { KnnStatus } from '../api/types';

export function KnnAlert({ refreshKey = 0 }: { refreshKey?: number }) {
  const [data, setData] = useState<KnnStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    getKnnStatus()
      .then(d => { if (!cancelled) setData(d); })
      .catch(() => { /* keep null — silently no note */ });
    return () => { cancelled = true; };
  }, [refreshKey]);

  // Loading or build currently in flight → no note.
  // While building the JobPanel shows live progress; stacking the
  // "missing" note on top of an active build is misleading (looks
  // like the build is doing nothing).
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
            it with <strong>Build KNN index</strong> below.
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
    <div className="knn-note">
      <svg className="knn-note-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <circle cx="12" cy="12" r="10" />
        <line x1="12" y1="16" x2="12" y2="12" />
        <line x1="12" y1="8" x2="12.01" y2="8" />
      </svg>
      <div>
        <h4 className="knn-note-title">KNN reclassification: not set up</h4>
        <p className="knn-note-body">
          Predictions are zero-shot only, and species-level accuracy
          isn&apos;t great yet. Build a KNN index (button below) to
          re-classify against ~234M reference embeddings — builds take
          hours to days depending on size.
        </p>
        {data.inventory_dir && (
          <p className="knn-note-body text-[10px] text-fg-1 mt-1">
            Looks for an index at: <code className="font-mono">{data.inventory_dir}</code>
          </p>
        )}
      </div>
    </div>
  );
}
