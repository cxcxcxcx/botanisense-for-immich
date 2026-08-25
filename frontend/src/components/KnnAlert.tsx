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

  // Loading or healthy → no alert.
  if (!data || data.index_built) return null;

  return (
    <div className="knn-alert" role="alert">
      <svg className="knn-alert-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
        <path d="M10.29 3.86 1.82 18a2 2 0 0 0 1.71 3h16.94a2 2 0 0 0 1.71-3L13.71 3.86a2 2 0 0 0-3.42 0z" />
        <line x1="12" y1="9" x2="12" y2="13" />
        <line x1="12" y1="17" x2="12.01" y2="17" />
      </svg>
      <div>
        <h4 className="knn-alert-title">KNN: reference index missing</h4>
        <p className="knn-alert-body">reclassify will return 503 error.</p>
      </div>
    </div>
  );
}