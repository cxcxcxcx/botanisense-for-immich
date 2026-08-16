import { Fragment } from 'react';
import type { Detection } from '../api/types';

export function BboxOverlay({ detections }: { detections: Detection[] | undefined | null }) {
  const visible = (detections ?? []).filter(d => d.score >= 0.10);
  if (visible.length === 0) return null;
  return (
    <svg className="absolute inset-0 w-full h-full pointer-events-none" viewBox="0 0 100 100" preserveAspectRatio="none">
      {visible.map(d => (
        <Fragment key={d.id}>
          <rect
            x={d.x * 100}
            y={d.y * 100}
            width={d.w * 100}
            height={d.h * 100}
            fill="none"
            stroke="#5f5"
            strokeWidth="2"
            vectorEffect="non-scaling-stroke"
          />
          <text x={d.x * 100} y={Math.max(0, d.y * 100 - 0.5)} fill="#5f5" fontSize="11" fontFamily="ui-monospace, monospace" style={{ paintOrder: 'stroke', stroke: '#000', strokeWidth: 3 }}>
            {d.label} {Math.round(d.score * 100)}%
          </text>
        </Fragment>
      ))}
    </svg>
  );
}