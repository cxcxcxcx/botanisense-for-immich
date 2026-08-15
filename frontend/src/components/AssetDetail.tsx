import { useEffect, useState } from 'react';
import { getAsset, reclassifyBbox, reclassify } from '../api/endpoints';
import type { AssetDetail } from '../api/types';
import { thumbUrl } from '../lib/thumb';
import { Predictions } from './Predictions';
import { LibraryNeighbors } from './LibraryNeighbors';
import { DetectionBlock } from './DetectionBlock';
import { useToasts } from './ToastProvider';
import { plantaeColorClass } from '../lib/plantae';
import { formatPct } from '../lib/format';

interface Props {
  assetId: string;
}

export function AssetDetail({ assetId }: Props) {
  const [data, setData] = useState<AssetDetail | null>(null);
  const [error, setError] = useState<Error | null>(null);
  const push = useToasts();

  useEffect(() => {
    let cancelled = false;
    setData(null);
    setError(null);
    getAsset(assetId)
      .then(d => { if (!cancelled) setData(d); })
      .catch(e => { if (!cancelled) setError(e); });
    return () => { cancelled = true; };
  }, [assetId]);

  const handleReclassifyAll = async () => {
    try {
      await reclassify([assetId]);
      push({ kind: 'success', message: 'Reclassify queued' }, 2000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    }
  };

  const handleReclassifyBbox = async (bboxIndex: number) => {
    try {
      await reclassifyBbox({ asset_id: assetId, bbox_index: bboxIndex });
      push({ kind: 'info', message: 'Per-crop reclassify is a no-op stub (use Reclassify with KNN)' }, 3000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    }
  };

  if (error) return <div className="error">Failed to load asset: {error.message}</div>;
  if (!data) return <p aria-busy="true">Loading…</p>;

  const fullTop1 = data.top_predictions[0];
  const plantaeScore = data.taxon_scores?.Plantae;
  const plantaeTier = plantaeColorClass(plantaeScore);

  return (
    <div className="asset-detail">
      <figure className="hero-photo">
        <img
          src={thumbUrl(data.asset_id, 'fullsize')}
          alt={data.original_file_name}
          loading="eager"
          decoding="async"
          onError={e => {
            // Fall back to preview if fullsize is rejected (rare — Immich
            // is supposed to support it, but some assets 404).
            const img = e.currentTarget;
            if (!img.dataset.fallback) {
              img.dataset.fallback = '1';
              img.src = thumbUrl(data.asset_id, 'preview');
            }
          }}
        />
      </figure>

      <p>
        <strong>{data.original_file_name}</strong>
        {data.cluster_id != null && <small> · cluster {data.cluster_id}</small>}
      </p>

      <nav className="taxon-row" aria-label="taxonomy hierarchy">
        {fullTop1?.kingdom && <span className="taxon-rank kingdom">{fullTop1.kingdom}</span>}
        {fullTop1?.family && (
          <>
            <span className="taxon-sep">›</span>
            <span className="taxon-rank family">{fullTop1.family}</span>
          </>
        )}
        {fullTop1?.genus && (
          <>
            <span className="taxon-sep">›</span>
            <span className="taxon-rank genus">{fullTop1.genus}</span>
          </>
        )}
        {fullTop1?.species && (
          <>
            <span className="taxon-sep">›</span>
            <span className="taxon-rank species">{fullTop1.species}</span>
          </>
        )}
        {plantaeScore != null && (
          <span
            className={`plantae ${plantaeTier}`}
            style={{ position: 'static', display: 'inline-block', marginLeft: 'auto' }}
          >
            Plantae {(plantaeScore * 100).toFixed(1)}%
          </span>
        )}
      </nav>

      <LibraryNeighbors items={data.library_neighbors} />

      <h5>Top-{data.top_predictions.length} predictions</h5>
      <div className="preds">
        <Predictions items={(data.top_predictions ?? []).slice(0, 5)} />
      </div>

      {data.knn?.top_species ? (
        <>
          <h5>KNN (re-classified)</h5>
          <div className="knn-block" style={{ borderTop: 0, padding: 0 }}>
            <div className="knn-header">
              <span>
                {data.knn.top_species}
                {data.knn.common_name ? ` (${data.knn.common_name})` : ''} · {formatPct(data.knn.score)}
              </span>
              <small>{data.knn.kingdom}{data.knn.family ? ` · ${data.knn.family}` : ''}</small>
            </div>
            {(data.knn.neighbors ?? []).map((n, i) => (
              <div key={`${n.species}-${i}`} className="pred">
                <div className="sp">
                  {n.species}
                  <div className="cn">{n.common_name ?? ''}</div>
                </div>
                <div className="sc">{formatPct(n.sim, 1)}</div>
              </div>
            ))}
          </div>
        </>
      ) : null}
      <button type="button" className="reclassify-btn" onClick={handleReclassifyAll}>
        Re-run KNN
      </button>

      {(data.detections ?? []).filter(d => d.score >= 0.10).length > 0 && (
        <>
          <h5>YOLO detections (per-crop)</h5>
          {data.detections
            .filter(d => d.score >= 0.10)
            .map((d, i) => (
              <div key={d.id}>
                <DetectionBlock det={d} fullTop1={fullTop1} />
                <button type="button" className="reclassify-btn" onClick={() => handleReclassifyBbox(i)}>
                  Reclassify crop
                </button>
              </div>
            ))}
        </>
      )}
    </div>
  );
}