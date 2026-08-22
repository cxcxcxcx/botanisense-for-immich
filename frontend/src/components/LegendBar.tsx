// Data tools shown on the Run view. These are the actions that DON'T
// push to IMMICH (those live in <ImmichSync />): build the KNN index,
// pull EXIF from Immich, purge assets that were removed from Immich,
// reclassify the visible assets via KNN, and run a YOLO person scan.
//
// The previous version of this component also exposed the three
// IMMICH-sync actions (Sync all clusters / Tag as Plants album /
// Tag as Fungi album) and a Plantae confidence legend. Both moved:
//   - the sync actions duplicate <ImmichSync /> and were removed
//   - the Plantae legend is a Browse-view concern and would mislead
//     users on the Run view, so it's gone
//
// Rendered as a section in the Run view's main column (not in the
// sticky header) so it doesn't compete with the sync section for
// visual space.

import { useState } from 'react';
import { useApp } from '../state/AppContext';
import { useToasts } from './ToastProvider';
import { reclassifyVisible } from '../api/endpoints';
import { KnnBuildModal } from './KnnBuildModal';
import { BackfillExifModal } from './BackfillExifModal';

export function LegendBar() {
  const { dispatch } = useApp();
  const push = useToasts();
  const [knnOpen, setKnnOpen] = useState(false);
  const [backfillOpen, setBackfillOpen] = useState(false);

  const handlePurgeMissing = () => dispatch({ type: 'OPEN_JOB', kind: 'purge-missing', payload: {} });
  const handleDetectPeople = () => dispatch({ type: 'OPEN_JOB', kind: 'person-backfill', payload: {} });

  const handleReclassifyVisible = async () => {
    try {
      await reclassifyVisible();
      push({ kind: 'info', message: 'KNN reclassify queued for visible assets' }, 2000);
    } catch (err) {
      push({ kind: 'error', message: err instanceof Error ? err.message : String(err) }, 4000);
    }
  };

  const btnCls =
    'px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-[0.85rem] hover:bg-[#345] hover:border-accent disabled:opacity-50';
  const btnDanger = 'text-red-300 border-[#5a3030] hover:bg-[#5a3030] hover:text-white hover:border-red-300';

  return (
    <section className="border border-line2 rounded-lg bg-bg-1 p-4 flex flex-col gap-3 w-full">
      <header className="flex items-center justify-between gap-2">
        <h3 className="m-0 text-base">Data tools</h3>
        <small className="text-fg-2">local DB maintenance — doesn't touch Immich</small>
      </header>
      <div className="flex flex-wrap gap-2">
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
          className={btnCls}
          onClick={handleReclassifyVisible}
          title="Queue KNN reclassification for every asset currently visible in the browse grid."
        >Reclassify visible with KNN</button>
        <button
          className={btnCls}
          onClick={handleDetectPeople}
          title="Run YOLO over every asset where has_person is unknown. No BioCLIP — fast relative to a full pipeline run. ~70 min for ~19k photos."
        >👤 Detect people</button>
        <button
          className={`${btnCls} ${btnDanger}`}
          onClick={handlePurgeMissing}
          title="Scan Immich for assets missing from the server, then delete them from the local DB after confirmation."
        >🗑 Purge assets deleted in Immich</button>
      </div>
      <KnnBuildModal open={knnOpen} onClose={() => setKnnOpen(false)} />
      <BackfillExifModal open={backfillOpen} onClose={() => setBackfillOpen(false)} />
    </section>
  );
}