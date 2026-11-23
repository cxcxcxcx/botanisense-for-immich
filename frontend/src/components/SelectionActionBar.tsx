import { useState } from 'react';
import { useApp, useSelection } from '../state/AppContext';
import { batchSetPlantaeOverride, getAsset } from '../api/endpoints';
import { BOTANISENSE_ENABLED } from '../lib/features';
import { thumbUrl } from '../lib/thumb';
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
  const [syncing, setSyncing] = useState(false);
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

  const sendToBotanisense = async () => {
    if (!BOTANISENSE_ENABLED) return;
    // Imported on demand so a build with the switch off never bundles
    // firebase.
    const [{ importPhotoToBotanisense }, { botanisenseAuth, toBotanisenseDetections }] =
      await Promise.all([import('../api/botanisense'), import('../lib/botanisense')]);
    // The session is restored from IndexedDB asynchronously after a reload.
    await botanisenseAuth.authStateReady();
    const user = botanisenseAuth.currentUser;
    if (!user) {
      push({
        kind: 'error',
        message: 'Connect BotaniSense first: open the Diary tab and sign in.',
      }, 4500);
      return;
    }
    if (busy || syncing) return;

    setSyncing(true);
    const ids = Array.from(selection);
    let sent = 0;
    let deduped = 0;
    let skippedPeople = 0;
    let failed = 0;
    let firstError = '';

    for (const assetId of ids) {
      try {
        const detail = await getAsset(assetId);
        if (detail.has_person === true) {
          skippedPeople += 1;
          continue;
        }

        const response = await fetch(thumbUrl(assetId, 'preview'));
        if (!response.ok) {
          throw new Error(`thumbnail HTTP ${response.status}`);
        }
        const blob = await response.blob();
        const baseName =
          detail.original_file_name?.replace(/\.[^./\\]+$/, '') || assetId;
        const result = await importPhotoToBotanisense(user, {
          file: blob,
          filename: `${baseName}.jpg`,
          clientRef: `immich:${assetId}`,
          capturedAt: detail.captured_at,
          latitude: detail.latitude,
          longitude: detail.longitude,
          detections: toBotanisenseDetections(detail),
        });
        if (result.deduped) deduped += 1;
        else sent += 1;
      } catch (error) {
        failed += 1;
        if (!firstError) {
          firstError = error instanceof Error ? error.message : String(error);
        }
        console.warn('[botanisense] import failed', assetId, error);
      }
    }

    const parts: string[] = [];
    if (sent) parts.push(`${sent} sent`);
    if (deduped) parts.push(`${deduped} already there`);
    if (skippedPeople) parts.push(`${skippedPeople} with people skipped`);
    if (failed) parts.push(`${failed} failed`);
    push(
      {
        kind: failed > 0 ? 'error' : 'success',
        message:
          parts.length > 0
            ? `BotaniSense: ${parts.join(', ')}.${firstError ? ` First error: ${firstError}` : ''}`
            : 'Nothing to send.',
      },
      failed > 0 ? 7000 : 4000,
    );

    if (failed === 0) dispatch({ type: 'CLEAR_SELECTION' });
    setSyncing(false);
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
      {BOTANISENSE_ENABLED && (
        <>
          <button
            type="button"
            disabled={busy || syncing}
            className="px-2.5 py-1 rounded border border-accent text-accent text-sm hover:bg-accent hover:text-bg disabled:opacity-50"
            onClick={sendToBotanisense}
            title="Upload the selected photos and their BioCLIP predictions to your BotaniSense account"
          >
            {syncing ? 'Sending…' : 'Send to BotaniSense'}
          </button>
          <span aria-hidden="true" className="text-line2">|</span>
        </>
      )}
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