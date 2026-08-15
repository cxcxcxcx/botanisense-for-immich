import { useEffect, useState } from 'react';
import { getKnnStatus } from '../api/endpoints';
import type { KnnStatus } from '../api/types';
import { cls } from '../lib/format';

export function KnnPill() {
  const [data, setData] = useState<KnnStatus | null>(null);

  useEffect(() => {
    let cancelled = false;
    getKnnStatus()
      .then(d => { if (!cancelled) setData(d); })
      .catch(() => { /* keep null */ });
    return () => { cancelled = true; };
  }, []);

  const label = !data
    ? 'KNN: …'
    : !data.index_built
      ? 'KNN: reference index missing — reclassify will 503'
      : `KNN: ${data.version || 'v?'} · ${data.n_rows.toLocaleString()} ref · ${data.cached_assets.toLocaleString()} cached`;

  const className = cls(
    'knn-pill',
    data && !data.index_built && 'unavailable',
    data?.index_built && 'ready',
  );

  const title = !data
    ? 'KNN status loading…'
    : !data.index_built
      ? `KNN index missing at ${data.inventory_dir}\nBuild with: python -m knn --parquet-glob … --index-dir ${data.inventory_dir} --tag v1 build`
      : `inventory_dir: ${data.inventory_dir}\nversion: ${data.version}\nn_rows: ${data.n_rows}\ncached: ${data.cached_assets}`;

  return <p id="knn-pill" className={className} title={title}>{label}</p>;
}