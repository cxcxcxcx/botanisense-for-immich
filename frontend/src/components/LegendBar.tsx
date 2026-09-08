// Data tools shown on the Run view. These are the actions that DON'T
// push to IMMICH (those live in <ImmichSync />): build the KNN index,
// pull EXIF from Immich, purge assets that were removed from Immich,
// reclassify the visible assets via KNN, and run a YOLO person scan.
//
// Styled per the mock's Run page: rounded-2xl outer card, header with
// a Database icon + "Data tools" title + "Local DB" tag on the right,
// and a 2-column grid of icon-prefixed action buttons below. The
// destructive purge action uses the red-tinted button variant.

import { useState } from 'react';
import { useApp } from '../state/AppContext';
import { useToasts } from './ToastProvider';
import { reclassifyVisible } from '../api/endpoints';
import { KnnAlert } from './KnnAlert';
import { KnnBuildModal } from './KnnBuildModal';
import { BackfillExifModal } from './BackfillExifModal';

export function LegendBar() {
  const { dispatch } = useApp();
  const push = useToasts();
  const [knnOpen, setKnnOpen] = useState(false);
  const [backfillOpen, setBackfillOpen] = useState(false);
  // Bumped when a KNN build finishes so the status note refetches
  // without a remount (getKnnStatus is fetched once per refreshKey).
  const [knnRefresh, setKnnRefresh] = useState(0);

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

  return (
    <section className="run-card" aria-label="Data tools">
      <header className="run-card-header">
        <h3 className="run-card-title">
          <DatabaseIcon />
          Data tools
        </h3>
        <span className="run-card-tag">Local DB</span>
      </header>

      {/* Inline KNN status — lives here (not as a top-of-page banner)
          because the action it points at is this card's build button. */}
      <KnnAlert refreshKey={knnRefresh} />

      <div className="data-tool-grid">
        <button
          className="data-tool-btn"
          onClick={() => setKnnOpen(true)}
          title="Build the FAISS KNN index from TreeOfLife-200M parquet embeddings. Debugging tool, not recommended: in the current setup KNN is less accurate than zero-shot and the index is very memory-hungry. Heavy: minutes-to-hours; logs stream into a tail viewer."
        >
          <BrainIcon />
          Build KNN index
        </button>
        <button
          className="data-tool-btn"
          onClick={() => setBackfillOpen(true)}
          title="Pull EXIF capture time + GPS from Immich for every asset that doesn't have it yet. ~30s for 17k assets."
        >
          <ClockIcon />
          Backfill EXIF
        </button>
        <button
          className="data-tool-btn"
          onClick={handleReclassifyVisible}
          title="Queue KNN reclassification for every asset currently visible in the browse grid. Debugging only — KNN is expected to be less accurate than the zero-shot predictions here."
        >
          <ActivityIcon />
          Reclassify with KNN
        </button>
        <button
          className="data-tool-btn"
          onClick={handleDetectPeople}
          title="Run YOLO over every asset where has_person is unknown. No BioCLIP — fast relative to a full pipeline run. ~70 min for ~19k photos."
        >
          <UsersIcon />
          Detect people
        </button>
        <button
          className="data-tool-btn data-tool-btn-danger sm:col-span-2"
          onClick={handlePurgeMissing}
          title="Scan Immich for assets missing from the server, then delete them from the local DB after confirmation."
        >
          <TrashIcon />
          Purge assets deleted in Immich
        </button>
      </div>

      <KnnBuildModal
        open={knnOpen}
        onClose={() => setKnnOpen(false)}
        onBuilt={() => setKnnRefresh(k => k + 1)}
      />
      <BackfillExifModal open={backfillOpen} onClose={() => setBackfillOpen(false)} />
    </section>
  );
}

// ---- icons (inline SVG; the codebase doesn't pull in lucide-react) ----

function DatabaseIcon() {
  return (
    <svg className="run-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <ellipse cx="12" cy="5" rx="9" ry="3" />
      <path d="M3 5v14a9 3 0 0 0 18 0V5" />
      <path d="M3 12a9 3 0 0 0 18 0" />
    </svg>
  );
}

function BrainIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-pink-400 shrink-0" aria-hidden="true">
      <path d="M12 5a3 3 0 1 0-5.997.142 4 4 0 0 0-2.526 5.77 4 4 0 0 0 .556 6.588A4 4 0 0 0 12 18Z" />
      <path d="M12 5a3 3 0 1 1 5.997.142 4 4 0 0 1 2.526 5.77 4 4 0 0 1-.556 6.588A4 4 0 0 1 12 18Z" />
      <path d="M15 13a4.5 4.5 0 0 1-3-4 4.5 4.5 0 0 1-3 4" />
      <path d="M17.6 9.8a3.5 3.5 0 0 0 0 4.4" />
      <path d="M6.4 9.8a3.5 3.5 0 0 1 0 4.4" />
    </svg>
  );
}

function ClockIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-blue-400 shrink-0" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <polyline points="12 6 12 12 16 14" />
    </svg>
  );
}

function ActivityIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-emerald-400 shrink-0" aria-hidden="true">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}

function UsersIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="text-purple-400 shrink-0" aria-hidden="true">
      <path d="M16 21v-2a4 4 0 0 0-4-4H6a4 4 0 0 0-4 4v2" />
      <circle cx="9" cy="7" r="4" />
      <path d="M22 21v-2a4 4 0 0 0-3-3.87" />
      <path d="M16 3.13a4 4 0 0 1 0 7.75" />
    </svg>
  );
}

function TrashIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" className="shrink-0" aria-hidden="true">
      <polyline points="3 6 5 6 21 6" />
      <path d="M19 6v14a2 2 0 0 1-2 2H7a2 2 0 0 1-2-2V6m3 0V4a2 2 0 0 1 2-2h4a2 2 0 0 1 2 2v2" />
      <line x1="10" y1="11" x2="10" y2="17" />
      <line x1="14" y1="11" x2="14" y2="17" />
    </svg>
  );
}