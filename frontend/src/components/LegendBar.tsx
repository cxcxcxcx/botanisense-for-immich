import { useEffect, useState } from 'react';
import { getResults, reclassifyVisible } from '../api/endpoints';
import type { ResultsResponse } from '../api/types';
import { useApp } from '../state/AppContext';
import { useToasts } from './ToastProvider';
import { KnnBuildModal } from './KnnBuildModal';
import { BackfillExifModal } from './BackfillExifModal';

export function LegendBar() {
  const { dispatch } = useApp();
  const push = useToasts();
  const [plantae, setPlantae] = useState<{ green: number; yellow: number }>({ green: 0.60, yellow: 0.35 });
  const [knnOpen, setKnnOpen] = useState(false);
  const [backfillOpen, setBackfillOpen] = useState(false);

  useEffect(() => {
    let cancelled = false;
    getResults({ page: 0, page_size: 1 })
      .then((r: ResultsResponse) => {
        if (!cancelled) setPlantae({ green: r.plantae_green, yellow: r.plantae_yellow });
      })
      .catch(() => { /* keep defaults */ });
    return () => { cancelled = true; };
  }, []);

  const handleSyncAll = () => dispatch({ type: 'OPEN_JOB', kind: 'sync-all', payload: {} });
  const handlePlants  = () => dispatch({ type: 'OPEN_JOB', kind: 'plants', payload: {} });
  const handleFungi   = () => dispatch({ type: 'OPEN_JOB', kind: 'fungi', payload: {} });
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

  // Shared button class — Pico's "secondary outline" button look,
  // translated to Tailwind utility classes.
  const btnCls =
    'px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-[0.8rem] hover:bg-[#345] hover:border-accent disabled:opacity-50';
  const btnDanger = 'text-red-300 border-[#5a3030] hover:bg-[#5a3030] hover:text-white hover:border-red-300';

  return (
    <nav className="flex flex-col gap-1 py-1 border-b border-line text-[0.85rem] w-full">
      <div className="flex flex-wrap items-center gap-2">
        <span className="inline-flex gap-1.5 items-center text-fg-1" title="Plantae confidence tiers">
          Plantae <span className="w-2 h-2 rounded-sm" style={{ background: '#1a3a1a' }}></span>≥{plantae.green.toFixed(2)}{' '}
          <span className="w-2 h-2 rounded-sm" style={{ background: '#3a3a1a' }}></span>≥{plantae.yellow.toFixed(2)}{' '}
          <span className="w-2 h-2 rounded-sm" style={{ background: '#3a1a1a' }}></span>&lt;{plantae.yellow.toFixed(2)}
        </span>
      </div>
      <div className="flex flex-wrap items-center gap-2">
        <button className={btnCls} onClick={handleSyncAll}>⟳ Sync all clusters as albums</button>
        <button className={btnCls} onClick={handlePlants}>Tag as Plants album</button>
        <button className={btnCls} onClick={handleFungi}>Tag as Fungi album</button>
        <button className={btnCls} onClick={handleReclassifyVisible}>Reclassify visible with KNN</button>
      </div>
      <div className="flex flex-wrap items-center gap-2 pt-1 mt-0.5 border-t border-dashed border-line2">
        <button
          className={btnCls}
          onClick={() => setKnnOpen(true)}
          title="Build the FAISS KNN index from TreeOfLife-200M parquet embeddings. Heavy: minutes-to-hours; logs stream into a tail viewer."
        >🧠 Build KNN index</button>
        <button
          className={btnCls}
          onClick={() => setBackfillOpen(true)}
          title="Pull EXIF capture time + GPS from Immich for every asset that doesn't have it yet. ~30s for 17k assets."
        >🕒 Backfill EXIF</button>
        <button
          className={`${btnCls} ${btnDanger}`}
          onClick={handlePurgeMissing}
          title="Scan Immich for assets missing from the server, then delete them from the local DB after confirmation."
        >🗑 Purge assets deleted in Immich</button>
      </div>
      <KnnBuildModal open={knnOpen} onClose={() => setKnnOpen(false)} />
      <BackfillExifModal open={backfillOpen} onClose={() => setBackfillOpen(false)} />
    </nav>
  );
}