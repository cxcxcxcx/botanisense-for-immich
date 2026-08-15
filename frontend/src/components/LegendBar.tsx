import { useEffect, useState } from 'react';
import { getResults, reclassifyVisible } from '../api/endpoints';
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

  const handleReclassifyVisible = async () => {
    try {
      const r = await reclassifyVisible();
      push({ kind: 'info', message: 'KNN reclassify queued for visible assets' }, 2000);
      console.debug('reclassify-visible queued', r);
    } catch (err) {
      push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 4000);
    }
  };

  return (
    <nav className="legend-bar">
      <span className="plantae-legend" title="Plantae confidence tiers">
        Plantae <span className="swatch" style={{ background: '#1a3a1a' }}></span>≥{plantae.green.toFixed(2)}{' '}
        <span className="swatch" style={{ background: '#3a3a1a' }}></span>≥{plantae.yellow.toFixed(2)}{' '}
        <span className="swatch" style={{ background: '#3a1a1a' }}></span>&lt;{plantae.yellow.toFixed(2)}
      </span>
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
    </nav>
  );
}