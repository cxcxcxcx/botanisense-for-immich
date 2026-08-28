import { useState } from 'react';
import type { ResultRow } from '../api/types';
import { useApp, useSelection } from '../state/AppContext';
import { thumbUrl } from '../lib/thumb';
import { plantaeTier } from '../lib/plantae';
import { formatPct } from '../lib/format';

interface Props {
  row: ResultRow;
  onUpdated: (assetId: string) => void;
}

export function Card({ row }: Props) {
  const { dispatch } = useApp();
  const selection = useSelection();
  const selected = selection.has(row.asset_id);
  // Show a shimmer under the image until it loads. Resets when the
  // asset_id changes (a remount would too, but the same component
  // can be re-keyed when the user re-sorts).
  const [imgLoaded, setImgLoaded] = useState(false);

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

  // Multi-select checkbox (UX-17). Visible whenever the user has
  // started selecting OR has the cursor over the card — keeps the
  // grid clean by default but one click away from a multi-pick
  // workflow. stopPropagation so clicking the checkbox doesn't also
  // open the asset detail modal.
  const showCheckbox = selected || selection.size > 0;

  return (
    <article
      id={`card-${row.asset_id}`}
      onClick={handleOpen}
      className={`photo-card ${selected ? 'ring-2 ring-accent' : ''}`}
    >
      {/* Shimmer placeholder under the real image. Sized the same as
          the image so layout doesn't shift when the load completes. */}
      {!imgLoaded && <div className="absolute inset-0 shimmer" aria-hidden="true" />}
      <img
        src={thumbUrl(row.asset_id, 'thumbnail')}
        alt={row.original_file_name}
        loading="lazy"
        decoding="async"
        className={`photo-card-image transition-opacity duration-200 ${imgLoaded ? 'opacity-100' : 'opacity-0'}`}
        onLoad={() => setImgLoaded(true)}
      />
      {/* Multi-select checkbox overlay — top-left corner. Wrapped in
          a <label> so the entire checkbox area is the click target;
          the click is intercepted by handleOpen via the
          closest('button, label') check. */}
      <label
        className={`absolute top-1.5 left-1.5 w-6 h-6 rounded border-2 flex items-center justify-center cursor-pointer transition-opacity ${
          showCheckbox ? 'opacity-100' : 'opacity-0 group-hover:opacity-100'
        } ${selected ? 'bg-accent border-accent' : 'bg-bg/80 border-fg-1 hover:border-accent'}`}
        onClick={e => e.stopPropagation()}
        title={selected ? 'Unselect' : 'Select'}
      >
        <input
          type="checkbox"
          checked={selected}
          onChange={() => dispatch({ type: 'TOGGLE_SELECT', assetId: row.asset_id })}
          className="sr-only"
          aria-label={selected ? `Unselect ${row.original_file_name}` : `Select ${row.original_file_name}`}
        />
        {selected && (
          <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="3" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
            <polyline points="20 6 9 17 4 12" />
          </svg>
        )}
      </label>
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