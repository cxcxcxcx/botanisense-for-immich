import type { ResultRow } from '../api/types';
import { useApp } from '../state/AppContext';
import { thumbUrl } from '../lib/thumb';
import { BboxOverlay } from './BboxOverlay';
import { plantaeColorClass } from '../lib/plantae';
import { formatPct } from '../lib/format';

interface Props {
  row: ResultRow;
  onUpdated: (assetId: string) => void;
}

export function Card({ row }: Props) {
  const { dispatch } = useApp();

  const fullTop1 = row.top_predictions[0];
  const plantaeScore = row.taxon_scores?.Plantae;
  const plantaeTier = plantaeColorClass(plantaeScore);

  // Single click on the photo opens the asset detail popup (modal).
  // The popup carries library_neighbors + KNN + detections, so it's
  // the canonical "look at this photo" surface. Avoid opening from
  // interactive children (buttons / links).
  const handleOpen = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('button, a, input, select, textarea, label')) return;
    dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id });
  };

  return (
    <article className="card" id={`card-${row.asset_id}`} onClick={handleOpen}>
      <div className="img-wrap">
        <img src={thumbUrl(row.asset_id)} alt={row.original_file_name} loading="lazy" decoding="async" />
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
      {/* Textual info lives in its own meta block under the photo:
         filename + top-1 species + score + per-kingdom chips. This
         makes the card naturally taller than a bare square so the
         text breathes; the in-photo overlay was cramped and unreadable. */}
      <div className="meta-card">
        <div className="meta-name" title={row.original_file_name}>
          {row.original_file_name}
        </div>
        {fullTop1 && (
          <div className="meta-top">
            <span className="meta-species" title={fullTop1.species}>
              {fullTop1.species}
            </span>
            <span className="meta-score">{formatPct(fullTop1.score)}</span>
          </div>
        )}
        {row.taxon_scores && (
          <div className="meta-scores">
            {Object.entries(row.taxon_scores)
              .filter(([k]) => k !== 'Plantae')
              .sort((a, b) => b[1] - a[1])
              .slice(0, 4)
              .map(([k, v]) => (
                <span key={k} className="meta-score-chip">
                  {k} {(v * 100).toFixed(0)}%
                </span>
              ))}
          </div>
        )}
      </div>
    </article>
  );
}