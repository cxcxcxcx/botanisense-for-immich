import { useState } from 'react';
import { useApp, useSelection } from '../state/AppContext';
import { batchSetPlantaeOverride } from '../api/endpoints';
import { useToasts } from './ToastProvider';

interface Props {
  // Called after a successful batch action so the parent can refetch
  // the current page (the cached payload's plantae_score / cluster
  // counts are now stale until the server's results cache invalidates).
  onApply?: () => void;
}

// Floating action bar rendered when selection.size > 0. Sits at the
// bottom-center of the viewport so it's reachable from any card
// without scrolling. First batch action: "Mark as non-plant" —
// the most common triage operation (BioCLIP mis-classifies a flower
// as non-plant often enough to make this the obvious action).
//
// Future actions (e.g. "Reclassify with KNN", "Move to cluster X")
// can be added as siblings here without touching the rest of the
// app.

export function SelectionActionBar({ onApply }: Props) {
  const { dispatch } = useApp();
  const selection = useSelection();
  const push = useToasts();
  const [busy, setBusy] = useState(false);
  if (selection.size === 0) return null;

  const apply = async (value: 0 | 1 | null, label: string) => {
    if (busy) return;
    setBusy(true);
    const ids = Array.from(selection);
    try {
      const r = await batchSetPlantaeOverride(ids, value);
      push({
        kind: 'success',
        message: `${label}: ${r.updated} photo${r.updated === 1 ? '' : 's'}.`,
      }, 2500);
      dispatch({ type: 'CLEAR_SELECTION' });
      onApply?.();
    } catch (e) {
      push({
        kind: 'error',
        message: `${label} failed: ${e instanceof Error ? e.message : String(e)}`,
      }, 4000);
    } finally {
      setBusy(false);
    }
  };

  return (
    <div
      role="region"
      aria-label={`${selection.size} photo${selection.size === 1 ? '' : 's'} selected`}
      className="fixed bottom-6 left-1/2 -translate-x-1/2 z-40 px-4 py-2 rounded-lg bg-bg-1 border border-accent shadow-[0_8px_24px_rgba(0,0,0,0.4)] flex items-center gap-3 text-[0.85rem]"
    >
      <span className="text-fg-1">
        <strong className="text-fg">{selection.size}</strong>
        {` `}selected
      </span>
      <span aria-hidden="true" className="text-line2">|</span>
      <button
        type="button"
        disabled={busy}
        className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-50"
        onClick={() => apply(0, 'Marked as non-plant')}
        title="Force plantae_override = 0 on every selected asset (excluded from Plantae albums)"
      >
        Mark as non-plant
      </button>
      <button
        type="button"
        disabled={busy}
        className="px-2.5 py-1 rounded border border-line2 text-fg text-sm hover:border-accent disabled:opacity-50"
        onClick={() => apply(1, 'Marked as plant')}
        title="Force plantae_override = 1 on every selected asset (included in Plantae albums)"
      >
        Mark as plant
      </button>
      <button
        type="button"
        disabled={busy}
        className="px-2.5 py-1 rounded border border-line2 text-fg-1 text-sm hover:border-accent disabled:opacity-50"
        onClick={() => apply(null, 'Cleared plantae override')}
      >
        Clear override
      </button>
      <span aria-hidden="true" className="text-line2">|</span>
      <button
        type="button"
        className="px-2.5 py-1 rounded border border-[#a44] text-red-300 text-sm hover:bg-[#3a1414] hover:border-[#a44]"
        onClick={() => dispatch({ type: 'CLEAR_SELECTION' })}
      >
        Cancel
      </button>
    </div>
  );
}