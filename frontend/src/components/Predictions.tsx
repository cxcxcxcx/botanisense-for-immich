import type { Prediction } from '../api/types';
import { formatPct } from '../lib/format';

export function Predictions({ items, showArrow = false }: { items: Prediction[]; showArrow?: boolean }) {
  return (
    <>
      {items.map((p, i) => (
        <div key={`${p.species}-${i}`} className="flex justify-between gap-2.5 py-1 border-t border-line text-[13px] first:border-t-0">
          <div className="flex-1">
            {p.species}
            <div className="text-fg-1 text-[11px]">{p.common_name || ''}</div>
          </div>
          <div className="text-leaf tabular-nums min-w-[56px] text-right">{formatPct(p.score)}</div>
        </div>
      ))}
    </>
  );
}