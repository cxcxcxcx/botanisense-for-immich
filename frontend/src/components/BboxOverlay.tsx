import { Fragment } from 'react';
import type { Detection } from '../api/types';

export function BboxOverlay({ detections }: { detections: Detection[] | undefined | null }) {
  const visible = (detections ?? []).filter(d => d.score >= 0.10);
  if (visible.length === 0) return null;
  return (
    <svg className="bbox" viewBox="0 0 100 100" preserveAspectRatio="none">
      {visible.map(d => (
        <Fragment key={d.id}>
          <rect
            x={d.x * 100}
            y={d.y * 100}
            width={d.w * 100}
            height={d.h * 100}
          />
          <text x={d.x * 100} y={Math.max(0, d.y * 100 - 0.5)}>
            {d.label} {Math.round(d.score * 100)}%
          </text>
        </Fragment>
      ))}
    </svg>
  );
}