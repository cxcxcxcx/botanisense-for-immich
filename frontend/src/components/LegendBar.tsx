import { useEffect, useState } from 'react';
import { backfillExif, getResults, reclassifyVisible } from '../api/endpoints';
import type { ResultsResponse } from '../api/types';
import { useApp } from '../state/AppContext';
import { useToasts } from './ToastProvider';

export function LegendBar() {
  const { dispatch } = useApp();
  const push = useToasts();
  const [plantae, setPlantae] = useState<{ green: number; yellow: number }>({ green: 0.60, yellow: 0.35 });

  // Read the canonical thresholds from the server on first paint.
  useEffect(() => {
    let cancelled = false;
    getResults({ page: 0, page_size: 1 })
      .then((r: ResultsResponse) => {
        if (cancelled) return;
        setPlantae({ green: r.plantae_green, yellow: r.plantae_yellow });
      })
      .catch(() => { /* keep defaults */ });
    return () => { cancelled = true; };
  }, []);

  const handleSyncAll = () => dispatch({ type: 'OPEN_JOB', kind: 'sync-all', payload: {} });
  const handlePlants  = () => dispatch({ type: 'OPEN_JOB', kind: 'plants', payload: {} });
  const handleFungi   = () => dispatch({ type: 'OPEN_JOB', kind: 'fungi', payload: {} });
  // Maintenance op — destructive, so it sits on its own row beneath the
  // enrichment actions. The confirm modal handles the actual safety
  // check (Immich trash retention is not silent).
  const handlePurgeMissing = () => dispatch({ type: 'OPEN_JOB', kind: 'purge-missing', payload: {} });

  const handleReclassifyVisible = async () => {
    try {
      const r = await reclassifyVisible();
      push({ kind: 'info', message: 'KNN reclassify queued for visible assets' }, 2000);
      console.debug('reclassify-visible queued', r);
    } catch (err) {
      push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 4000);
    }
  };

  // One-shot backfill — pulls EXIF (capture time + GPS) for every
  // DB row that doesn't have it yet. Synchronous on the server
  // (~30s for the user's 17k-row DB), no progress bar; just a
  // success toast when it lands.
  const handleBackfillExif = async () => {
    push({ kind: 'info', message: 'Backfilling EXIF from Immich — this takes ~30s for a 17k library' }, 2500);
    try {
      const r = await backfillExif();
      push({
        kind: r.updated_count > 0 ? 'success' : 'info',
        message: r.updated_count > 0
          ? `Backfilled EXIF on ${r.updated_count.toLocaleString()} assets`
          : 'Nothing to backfill — every asset already has capture time',
      }, 4000);
    } catch (err) {
      push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 4000);
    }
  };

  return (
    <nav className="legend-bar">
      <div className="legend-bar-row">
        <span className="plantae-legend" title="Plantae confidence tiers">
          Plantae <span className="swatch" style={{ background: '#1a3a1a' }}></span>≥{plantae.green.toFixed(2)}{' '}
          <span className="swatch" style={{ background: '#3a3a1a' }}></span>≥{plantae.yellow.toFixed(2)}{' '}
          <span className="swatch" style={{ background: '#3a1a1a' }}></span>&lt;{plantae.yellow.toFixed(2)}
        </span>
      </div>
      <div className="legend-bar-actions">
        <button className="secondary outline" onClick={handleSyncAll}>
          ⟳ Sync all clusters as albums
        </button>
        <button className="secondary outline" onClick={handlePlants}>
          Tag as Plants album
        </button>
        <button className="secondary outline" onClick={handleFungi}>
          Tag as Fungi album
        </button>
        <button className="secondary outline" onClick={handleReclassifyVisible}>
          Reclassify visible with KNN
        </button>
      </div>
      <div className="legend-bar-actions legend-bar-maintenance">
        <button
          className="secondary outline"
          onClick={handleBackfillExif}
          title="Pull EXIF capture time + GPS from Immich for every asset that doesn't have it yet. ~30s for 17k assets."
        >
          🕒 Backfill EXIF
        </button>
        <button
          className="secondary outline danger"
          onClick={handlePurgeMissing}
          title="Scan Immich for assets missing from the server, then delete them from the local DB after confirmation."
        >
          🗑 Purge assets deleted in Immich
        </button>
      </div>
    </nav>
  );
}