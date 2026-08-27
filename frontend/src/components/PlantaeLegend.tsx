import { useState } from 'react';

// Always-visible legend for the Plantae confidence tiers (green /
// yellow / red). Sits in the bottom-left of every Browse view so
// users don't have to navigate to the Run page to learn what the
// card-tier dots mean.
//
// Dismissable per session via localStorage. The `dismissible` prop
// suppresses the close button on the Run page where the equivalent
// info already lives next to the slider controls.

interface Props {
  dismissible?: boolean;
}

export function PlantaeLegend({ dismissible = true }: Props) {
  const [hidden, setHidden] = useState(() =>
    typeof window !== 'undefined' && localStorage.getItem('plantae-legend-dismissed') === '1',
  );
  if (hidden) return null;

  return (
    <div
      role="region"
      aria-label="Plantae confidence tiers"
      className="fixed bottom-3 left-3 z-40 px-3 py-2 rounded-md bg-bg-1/95 border border-line2 backdrop-blur text-[11px] text-fg-1 shadow-md flex items-center gap-3"
    >
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="tier-dot tier-dot-green" />
        <span>Plantae ≥60%</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="tier-dot tier-dot-yellow" />
        <span>≥35%</span>
      </span>
      <span className="flex items-center gap-1.5">
        <span aria-hidden="true" className="tier-dot tier-dot-red" />
        <span>&lt;35%</span>
      </span>
      {dismissible && (
        <button
          type="button"
          aria-label="Hide Plantae legend"
          className="text-fg-1 hover:text-fg leading-none px-1"
          onClick={() => {
            localStorage.setItem('plantae-legend-dismissed', '1');
            setHidden(true);
          }}
        >
          ×
        </button>
      )}
    </div>
  );
}