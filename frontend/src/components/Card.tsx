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

  const handleOpen = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('button, a, input, select, textarea, label')) return;
    dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id });
  };

  return (
    <article
      id={`card-${row.asset_id}`}
      onClick={handleOpen}
      className="bg-black overflow-hidden flex flex-col cursor-pointer relative rounded-sm border border-line2 hover:border-[#3a8a52] transition-colors m-0 p-0"
    >
      <div className="relative bg-black aspect-[4/5] overflow-hidden">
        <img
          src={thumbUrl(row.asset_id)}
          alt={row.original_file_name}
          loading="lazy"
          decoding="async"
          className="w-full h-full block object-cover object-center"
        />
        {row.cluster_id != null && (
          <span className="absolute top-1.5 left-1.5 bg-black/60 text-[#cfb] font-mono text-[11px] py-0.5 px-1.5 rounded">
            cluster {row.cluster_id}
          </span>
        )}
        {row.rank != null && (
          <span className="absolute top-1.5 right-1.5 bg-bg-2 text-white font-mono text-[11px] py-0.5 px-1.5 rounded shadow">
            #{row.rank + 1}
          </span>
        )}
        {plantaeScore != null && (
          <span className={`absolute top-[34px] right-1.5 font-mono text-[11px] py-0.5 px-1.5 rounded shadow font-bold ${plantaeTier}`}>
            Plantae {(plantaeScore * 100).toFixed(1)}%
          </span>
        )}
        {row.near_dups && row.near_dups.length > 0 && (
          <span
            className="absolute top-1.5 left-1.5 bg-score-red text-white font-mono font-bold text-[11px] py-0.5 px-1.5 rounded shadow"
            title={`${row.near_dups.length} similar photo(s)`}
          >{row.near_dups.length}×</span>
        )}
        <BboxOverlay detections={row.detections} />
        {/* Bottom-of-photo overlay: top-1 prediction + score + kingdom
           chips on a translucent gradient. No filename — it lives in
           the asset-detail modal. */}
        <div className="absolute left-0 right-0 bottom-0 px-2.5 pt-8 pb-3.5 bg-gradient-to-b from-transparent to-black/85 text-fg flex flex-col gap-1 pointer-events-none">
          {fullTop1 && (
            <div className="flex justify-between items-baseline gap-2">
              <span className="italic font-semibold text-[13px]/[1.2] drop-shadow overflow-hidden text-ellipsis whitespace-nowrap">
                {fullTop1.species}
              </span>
              <span className="font-mono font-semibold text-[12px]/[1] bg-accent-dark text-white py-0.5 px-1.5 rounded flex-shrink-0">
                {formatPct(fullTop1.score)}
              </span>
            </div>
          )}
          {row.taxon_scores && (
            <div className="flex flex-wrap gap-1">
              {Object.entries(row.taxon_scores)
                .filter(([k]) => k !== 'Plantae')
                .sort((a, b) => b[1] - a[1])
                .slice(0, 4)
                .map(([k, v]) => (
                  <span
                    key={k}
                    className="font-mono text-[10px] bg-white/10 text-fg-0 py-px px-1.5 rounded"
                  >{k} {(v * 100).toFixed(0)}%</span>
                ))}
            </div>
          )}
        </div>
      </div>
    </article>
  );
}