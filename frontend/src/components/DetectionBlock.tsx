import type { Detection, Prediction } from '../api/types';
import { Predictions } from './Predictions';
import { formatPct } from '../lib/format';

export function DetectionBlock({ det, fullTop1 }: { det: Detection; fullTop1: Prediction | undefined }) {
  const bp = (det.bbox_predictions ?? []).slice(0, 3);
  if (bp.length === 0) return null;
  return (
    <div className="det-block">
      <div className="det-header">crop: {det.label} {formatPct(det.score)}</div>
      {bp.map((p, i) => {
        const beats = !!fullTop1 && p.species === fullTop1.species && p.score > fullTop1.score;
        return (
          <div key={`${p.species}-${i}`} className="pred">
            <div className="sp">
              {p.species}
              <div className="cn">{p.common_name || ''}</div>
            </div>
            <div className="sc">{formatPct(p.score)}{beats ? ' ↑' : ''}</div>
          </div>
        );
      })}
    </div>
  );
}