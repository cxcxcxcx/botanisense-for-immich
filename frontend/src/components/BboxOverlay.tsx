import { Fragment } from 'react';
import type { Detection } from '../api/types';

// Color-code by label so the user can tell at a glance what YOLO
// saw. Person-flavored boxes (person / face / human) get the accent
// color (high signal); hands get yellow (visible-but-not-person);
// plants/flowers default to green.
const COLOR_FOR: Record<string, string> = {
  person: '#ff7a45',  // accent — high signal
  face:   '#5fa8ff',  // light blue
  human:  '#5fc8c8',  // teal
  hand:   '#ffd84d',  // yellow — visible but not "has person"
  plant:  '#5f5',     // green
  leaf:   '#5f5',
  flower: '#5f5',
  tree:   '#5f5',
  shrub:  '#5f5',
  succulent: '#5f5',
  cactus: '#5f5',
  herb: '#5f5',
  grass: '#5f5',
  stem: '#5f5',
  branch: '#5f5',
  bloom: '#5f5',
  petal: '#5f5',
  seedling: '#5f5',
};
const DEFAULT_COLOR = '#888';

export function BboxOverlay({ detections }: { detections: Detection[] | undefined | null }) {
  const visible = (detections ?? []).filter(d => d.score >= 0.50);
  if (visible.length === 0) return null;
  return (
    <svg
      className="absolute inset-0 w-full h-full pointer-events-none overflow-hidden"
      viewBox="0 0 100 100"
      preserveAspectRatio="none"
    >
      {/* The labels live INSIDE the box top-left with a solid colored
          chip background. The chip is clamped against the right/bottom
          edges so it never overflows the image (and the SVG itself
          clips anything that does). Reasons:
          1. Anchoring with `dy="-N"` in user units is unpredictable
             after `preserveAspectRatio="none"` — the y-shift is in
             fractions of the image height, while the text size is in
             CSS pixels. The two coordinate spaces fight each other
             and the label ends up half-inside / half-outside the
             box, or clipped at the top edge of the image.
          2. A `<rect>` backdrop eliminates the need for the
             `paint-order: stroke` text halo, which was the
             biggest contributor to "huge ugly text" — the halo
             around a 10px label is 2.5px on each side, so the
             effective text block was ~15px tall in dark scenes.
          3. The chip is in user units (scales with the image),
             the text inside is in CSS pixels (always 10px), so
             the chip gets bigger on big images but the text
             stays the same — same proportions as before but
             visually predictable.

          `dominant-baseline="hanging"` puts the text baseline at
          the top of the glyph, so the chip rect can be exactly
          tall enough to fit the text without padding math. */}
      {visible.map(d => {
        const color = COLOR_FOR[d.label] ?? DEFAULT_COLOR;
        const label = `${d.label} ${Math.round(d.score * 100)}%`;
        // Chip dimensions: ~6.5 chars × 5.5px + 4px padding each side
        // ≈ 40px wide, 14px tall in CSS px. Convert to user units
        // (out of 100) so the chip scales with the image. Width and
        // height are clamped against the right/bottom edges so the
        // chip never extends past x=100 / y=100 — at x=93, chipW
        // collapses to 7 instead of overflowing to 102.
        const CHIP_W = 9;  // user units (9% of image width)
        const CHIP_H = 4;  // user units (4% of image height — fits ~14px text)
        const chipX = d.x * 100;
        const chipY = d.y * 100;
        const w = Math.min(CHIP_W, 100 - chipX);
        const h = Math.min(CHIP_H, 100 - chipY);
        return (
          <Fragment key={d.id}>
            <rect
              x={d.x * 100}
              y={d.y * 100}
              width={d.w * 100}
              height={d.h * 100}
              fill="none"
              stroke={color}
              strokeWidth="2"
              vectorEffect="non-scaling-stroke"
            />
            <rect
              x={chipX}
              y={chipY}
              width={w}
              height={h}
              fill={color}
              opacity="0.9"
            />
            <text
              x={chipX + 0.4}
              y={chipY + 0.2}
              dominantBaseline="hanging"
              fill="#000"
              fontFamily="ui-monospace, SFMono-Regular, Menlo, monospace"
              fontSize="10"
              fontWeight="600"
            >{label}</text>
          </Fragment>
        );
      })}
    </svg>
  );
}