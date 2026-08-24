import type { ResultRow } from '../api/types';
import { useApp } from '../state/AppContext';
import { thumbUrl } from '../lib/thumb';
import { plantaeTier } from '../lib/plantae';
import { formatPct } from '../lib/format';

interface Props {
  row: ResultRow;
  onUpdated: (assetId: string) => void;
}

// Confidence pill tier — used to color the pill at the bottom-right of
// each photo card. Mirrors the mock's two-tone scheme but adds an
// explicit red tier for low-confidence BioCLIP results so the user can
// spot them at a glance without expanding the card.
function confidencePillClass(score: number | undefined): string {
  if (score == null) return 'confidence-pill-yellow';
  if (score >= 0.80) return 'confidence-pill-green';
  if (score >= 0.35) return 'confidence-pill-yellow';
  return 'confidence-pill-red';
}

export function Card({ row }: Props) {
  const { dispatch } = useApp();

  const fullTop1 = row.top_predictions[0];
  const confidenceScore = fullTop1?.score;

  const handleOpen = (e: React.MouseEvent) => {
    const target = e.target as HTMLElement;
    if (target.closest('button, a, input, select, textarea, label')) return;
    dispatch({ type: 'OPEN_MODAL', assetId: row.asset_id });
  };

  // Per the mock redesign: photo cards show ONLY the confidence pill
  // — no species name, no tier swatch, no top badges. The species
  // identity is implied by the active cluster/taxon filter; full
  // kingdom + species detail lives in the asset-detail modal.
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
      {/* Bottom-right confidence pill. Color-coded by tier: emerald
          for high (≥80%), amber for medium (35-80%), red for low
          (<35% or null). The pill is the only overlay text on the
          card — keeps the photo as the primary visual. */}
      {confidenceScore != null && (
        <div className="photo-card-overlay">
          <span
            className={`confidence-pill ${confidencePillClass(confidenceScore)}`}
            title={
              fullTop1
                ? `${fullTop1.species} · ${formatPct(confidenceScore)}`
                : `confidence ${formatPct(confidenceScore)}`
            }
          >
            {formatPct(confidenceScore)}
          </span>
        </div>
      )}
      {/* Near-dup strip — kept from the prior design so the user can
          reach hidden burst-shot siblings from the grid. Renders
          below the photo at the card's bottom edge; the photo area
          stays square (the strip adds height below the 1:1 frame). */}
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
      {/* suppress unused-var lint for plantaeTier (still imported by
          other code paths and the lib module) — declared but not
          referenced here because the mock dropped the per-card tier
          swatch. */}
      {void plantaeTier}
    </article>
  );
}