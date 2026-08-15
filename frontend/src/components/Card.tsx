import { useState } from 'react';
import type { ResultRow } from '../api/types';
import { useApp } from '../state/AppContext';
import { thumbUrl } from '../lib/thumb';
import { BboxOverlay } from './BboxOverlay';
import { Predictions } from './Predictions';
import { DetectionBlock } from './DetectionBlock';
import { NearDupStrip } from './NearDupStrip';
import { KnnBlock } from './KnnBlock';
import { plantaeColorClass } from '../lib/plantae';

interface Props {
  row: ResultRow;
  onUpdated: (assetId: string) => void;
}

export function Card({ row, onUpdated }: Props) {
  const { dispatch } = useApp();
  const [primaryAid, setPrimaryAid] = useState(row.asset_id);

  const fullTop1 = row.top_predictions[0];
  const plantaeScore = row.taxon_scores?.Plantae;
  const plantaeTier = plantaeColorClass(plantaeScore);

  const handleOpen = (e: React.MouseEvent) => {
    // Open modal when clicking anywhere outside an interactive child.
    const target = e.target as HTMLElement;
    if (target.closest('button, a, input, select, textarea, label')) return;
    dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id });
  };

  return (
    <article
      className="card"
      id={`card-${row.asset_id}`}
      onClick={handleOpen}
    >
      <div className="img-wrap">
        <img src={thumbUrl(primaryAid)} alt={row.original_file_name} loading="lazy" decoding="async" />
        {row.cluster_id != null && (
          <span className="cluster-tag">cluster {row.cluster_id}</span>
        )}
        {row.rank != null && (
          <span className="rank-tag">#{row.rank + 1}</span>
        )}
        {row.near_dups && row.near_dups.length > 0 && (
          <span className="dup-badge" title={`${row.near_dups.length} similar photo(s)`}>
            {row.near_dups.length}×
          </span>
        )}
        <BboxOverlay detections={row.detections} />
        {plantaeScore != null && (
          <span className={`plantae ${plantaeTier}`}>
            Plantae {(plantaeScore * 100).toFixed(1)}%
          </span>
        )}
      </div>

      <div className="meta-card">
        <div className="name">{row.original_file_name}</div>
        <div>
          <span className="badge">{fullTop1?.kingdom || '?'}</span>
          <span className="badge">{fullTop1?.family || '?'}</span>
          <span className="badge">{fullTop1?.genus || '?'}</span>
          {row.taxon_scores && (
            <small className="muted">
              {Object.entries(row.taxon_scores)
                .filter(([k]) => k !== 'Plantae')
                .sort((a, b) => b[1] - a[1])
                .map(([k, v], i, arr) => (
                  <span key={k}>
                    {i > 0 ? ' · ' : ''}{k}={(v * 100).toFixed(0)}
                  </span>
                ))}
            </small>
          )}
        </div>
        <div className="urls">
          <a href={row.original_url} target="_blank" rel="noopener noreferrer">original →</a>
          <a href={row.thumbnail_url} target="_blank" rel="noopener noreferrer">thumb →</a>
        </div>
      </div>

      <div className="preds">
        <Predictions items={row.top_predictions.slice(0, 5)} />
      </div>

      {(row.detections ?? []).filter(d => d.score >= 0.10).length > 0 && (
        <div className="dets">
          {row.detections
            .filter(d => d.score >= 0.10)
            .map(d => (
              <DetectionBlock key={d.id} det={d} fullTop1={fullTop1} />
            ))}
        </div>
      )}

      <NearDupStrip
        row={row}
        primaryAid={primaryAid}
        onPickPrimary={setPrimaryAid}
      />

      <KnnBlock row={row} onUpdated={onUpdated} />
    </article>
  );
}