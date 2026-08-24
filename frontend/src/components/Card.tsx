import type { ResultRow } from '../api/types';
import { useApp } from '../state/AppContext';
import { thumbUrl } from '../lib/thumb';
import { plantaeTier } from '../lib/plantae';
import { formatPct } from '../lib/format';

interface Props {
  row: ResultRow;
  onUpdated: (assetId: string) => void;
}

export function Card({ row }: Props) {
  const { dispatch } = useApp();

  const fullTop1 = row.top_predictions[0];
  const plantaeScore = row.taxon_scores?.Plantae;
  // Per "no need to show low conf plants": red-tier BioCLIP results
  // suppress the whole bottom overlay (species, tier swatch, conf
  // pill) — the photo stands alone. Green/yellow tiers show the
  // tier swatch + species name + confidence pill.
  const tier = plantaeTier(plantaeScore);
  const hideOverlay = tier === 'red';

  const handleOpen = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('button, a, input, select, textarea, label')) return;
    dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id });
  };

  return (
    <article
      id={`card-${row.asset_id}`}
      onClick={handleOpen}
      className="photo-card"
    >
      <img
        src={thumbUrl(row.asset_id, 'thumbnail')}
        alt={row.original_file_name}
        loading="lazy"
        decoding="async"
        className="photo-card-image"
      />
      {/* Bottom overlay: tier swatch + species name (wraps to 2 lines
          so long binomials aren't truncated) + confidence pill on
          the right. Suppressed entirely for red-tier BioCLIP results
          so low-confidence predictions don't crowd the grid. */}
      <div className="photo-card-overlay">
        {!hideOverlay && fullTop1 && (
          <div className="flex items-end justify-between gap-2 w-full">
            <span className="inline-flex items-start gap-1.5 min-w-0 flex-1">
              <span
                className={`tier-dot tier-dot-${tier} mt-1.5 shrink-0`}
                aria-hidden="true"
              />
              <span
                className="italic font-semibold text-[13px]/[1.2] drop-shadow break-words line-clamp-2"
                title={fullTop1.species}
              >
                {fullTop1.species}
              </span>
            </span>
            <span
              className="font-mono font-semibold text-[12px]/[1] bg-accent-dark text-white py-0.5 px-1.5 rounded shrink-0 backdrop-blur-sm"
              title={`${fullTop1.species} · ${formatPct(fullTop1.score)}`}
            >
              {formatPct(fullTop1.score)}
            </span>
          </div>
        )}
      </div>
      {/* Near-dup strip — kept from the prior design so the user can
          reach hidden burst-shot siblings from the grid. Renders
          below the photo at the card's bottom edge. */}
      {row.near_dups && row.near_dups.length > 0 && (
        <div className="flex flex-wrap gap-1 p-1 bg-bg-1 border-t border-line2">
          {row.near_dups.slice(0, 6).map((sib) => (
            <button
              key={sib.asset_id}
              type="button"
              onClick={(e) => {
                e.stopPropagation();
                dispatch({ type: 'OPEN_MODAL', assetId: sib.asset_id });
              }}
              title={sib.original_file_name ?? 'similar photo'}
              className="block w-12 h-12 overflow-hidden rounded-sm border border-line2 hover:border-accent"
            >
              <img
                src={thumbUrl(sib.asset_id, 'thumbnail')}
                alt={sib.original_file_name ?? 'similar photo'}
                loading="lazy"
                decoding="async"
                className="w-full h-full object-cover"
              />
            </button>
          ))}
          {row.near_dups.length > 6 && (
            <span className="self-center text-[10px] text-fg-1 px-1">
              +{row.near_dups.length - 6} more
            </span>
          )}
        </div>
      )}
    </article>
  );
}