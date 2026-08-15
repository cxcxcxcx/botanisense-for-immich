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
import { formatPct } from '../lib/format';

interface Props {
  row: ResultRow;
  /** When true, render the zoomed-in variant: hide the grid behind us,
   *  show a back button, overlay predictions on a dominant photo. */
  zoomed?: boolean;
  onZoomOut?: () => void;
  onUpdated: (assetId: string) => void;
}

export function Card({ row, zoomed = false, onZoomOut, onUpdated }: Props) {
  const { dispatch } = useApp();
  const [primaryAid, setPrimaryAid] = useState(row.asset_id);

  const fullTop1 = row.top_predictions[0];
  const plantaeScore = row.taxon_scores?.Plantae;
  const plantaeTier = plantaeColorClass(plantaeScore);

  const handleOpen = (e: React.MouseEvent) => {
    if (zoomed) return;
    const target = e.target as HTMLElement;
    if (target.closest('button, a, input, select, textarea, label')) return;
    // Single click on the photo: zoom in (not the modal — the modal
    // has library_neighbors etc., that's for double-click / "open in
    // dialog"). Holding shift while clicking opens the dialog instead.
    if (e.shiftKey) {
      dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id });
    } else {
      dispatch({ type: 'ZOOM_IN', aid: row.asset_id });
    }
  };

  const classes = [
    'card',
    zoomed ? 'zoomed' : '',
  ].filter(Boolean).join(' ');

  return (
    <article className={classes} id={`card-${row.asset_id}`} onClick={handleOpen}>
      {zoomed && onZoomOut && (
        <button
          type="button"
          className="card-back-btn"
          onClick={e => { e.stopPropagation(); onZoomOut(); }}
          title="Back to grid (Esc)"
        >
          ← back
        </button>
      )}

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
        {/* Predictions overlay — translucent strip across the bottom of
            the photo so the grid card stays compact (no separate meta
            block for filename, urls, etc. in cluster view). */}
        <div className="img-overlay">
          {fullTop1 && (
            <div className="overlay-top">
              <span className="overlay-species">{fullTop1.species}</span>
              <span className="overlay-score">{formatPct(fullTop1.score)}</span>
            </div>
          )}
          {row.taxon_scores && (
            <div className="overlay-scores">
              {Object.entries(row.taxon_scores)
                .filter(([k]) => k !== 'Plantae')
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4)
                .map(([k, v]) => (
                  <span key={k} className="overlay-score-chip">
                    {k} {(v * 100).toFixed(0)}%
                  </span>
                ))}
            </div>
          )}
        </div>
      </div>

      {/* Optional full taxonomy row — only in zoomed view (room to
          breathe). In compact grid mode the chips are condensed into
          the overlay above. */}
      {zoomed && (
        <div className="meta-card">
          <div className="name">{row.original_file_name}</div>
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
          </nav>
          <div className="urls">
            <a href={row.original_url} target="_blank" rel="noopener noreferrer">original →</a>
            <a href={row.thumbnail_url} target="_blank" rel="noopener noreferrer">thumb →</a>
          </div>
        </div>
      )}

      {/* Predictions / detections / near-dup / knn only in zoomed view.
          In compact grid mode the top prediction is already in the
          overlay. */}
      {zoomed && (
        <>
          <div className="preds">
            <Predictions items={(row.top_predictions ?? []).slice(0, 5)} />
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
        </>
      )}
    </article>
  );
}