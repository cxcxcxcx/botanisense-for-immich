import type { ResultRow } from '../api/types';
import { reclassify } from '../api/endpoints';
import { formatPct } from '../lib/format';
import { useToasts } from './ToastProvider';

interface Props {
  row: ResultRow;
  onUpdated?: (assetId: string) => void;
}

export function KnnBlock({ row, onUpdated }: Props) {
  const push = useToasts();
  const knn = row.knn ?? null;

  const handleRerun = async (e: React.MouseEvent) => {
    e.stopPropagation();
    try {
      await reclassify([row.asset_id]);
      push({ kind: 'success', message: 'KNN reclassify queued' }, 2000);
      onUpdated?.(row.asset_id);
    } catch (err) {
      push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 4000);
    }
  };

  return (
    <div className="knn-block">
      {knn?.top_species ? (
        <>
          <div className="knn-header">
            <span>
              KNN top-1: {knn.top_species}
              {knn.common_name ? ` (${knn.common_name})` : ''} · {formatPct(knn.score)}
            </span>
            <small>{knn.kingdom}{knn.family ? ` · ${knn.family}` : ''}</small>
          </div>
          {(knn.neighbors ?? []).slice(0, 3).map((n, i) => (
            <div key={`${n.species}-${i}`} className="pred">
              <div className="sp">
                {n.species}
                <div className="cn">{n.common_name ?? ''}</div>
              </div>
              <div className="sc">{formatPct(n.sim, 1)}</div>
            </div>
          ))}
        </>
      ) : (
        <div className="knn-empty">No KNN result yet.</div>
      )}
      <button
        type="button"
        className={`reclassify-btn ${knn?.top_species ? '' : 'primary'}`}
        onClick={handleRerun}
      >
        {knn?.top_species ? 'Re-run KNN' : 'Reclassify with KNN'}
      </button>
    </div>
  );
}