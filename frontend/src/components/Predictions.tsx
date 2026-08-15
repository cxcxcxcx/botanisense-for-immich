import type { Prediction } from '../api/types';
import { formatPct } from '../lib/format';

export function Predictions({ items, showArrow = false }: { items: Prediction[]; showArrow?: boolean }) {
  return (
    <>
      {items.map((p, i) => (
        <div key={`${p.species}-${i}`} className="pred">
          <div className="sp">
            {p.species}
            <div className="cn">{p.common_name || ''}</div>
          </div>
          <div className="sc">{formatPct(p.score)}</div>
        </div>
      ))}
    </>
  );
}