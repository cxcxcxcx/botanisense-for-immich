import type { Detection, Prediction } from '../api/types';
import { Predictions } from './Predictions';
import { formatPct } from '../lib/format';

export function DetectionBlock({ det, fullTop1 }: { det: Detection; fullTop1: Prediction | undefined }) {
  const bp = (det.bbox_predictions ?? []).slice(0, 3);
  if (bp.length === 0) return null;
  return (
    <div className="mt-2">
      <div className="text-[11px] text-[#ad9] mb-0.5 font-mono">crop: {det.label} {formatPct(det.score)}</div>
      {bp.map((p, i) => {
        const beats = !!fullTop1 && p.species === fullTop1.species && p.score > fullTop1.score;
        return (
          <div key={`${p.species}-${i}`} className="flex justify-between gap-2.5 py-1 border-t border-[#222] text-[13px] first:border-t-0">
            <div className="flex-1">
              {p.species}
              <div className="text-fg-1 text-[11px]">{p.common_name || ''}</div>
            </div>
            <div className="text-[#ad9] tabular-nums min-w-[56px] text-right">{formatPct(p.score)}{beats ? ' ↑' : ''}</div>
          </div>
        );
      })}
    </div>
  );
}