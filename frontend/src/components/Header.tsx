import { KnnPill } from './KnnPill';
import { StatusPanel } from './StatusPanel';
import { LegendBar } from './LegendBar';
import { navigate, type View } from '../hooks/useRoute';

interface Props {
  view: View;
}

export function Header({ view }: Props) {
  void navigate; // reserved for future header-level actions
  return (
    <header className="app-header">
      <div className="view-tabs" role="tablist" aria-label="View">
        <button
          role="tab"
          aria-selected={view === 'browse'}
          className={`view-tab ${view === 'browse' ? 'active' : ''}`}
          onClick={() => navigate('browse')}
        >
          Browse
        </button>
        <button
          role="tab"
          aria-selected={view === 'run'}
          className={`view-tab ${view === 'run' ? 'active' : ''}`}
          onClick={() => navigate('run')}
        >
          Run
        </button>
      </div>
      <hgroup>
        <h1>Immich × BioCLIP</h1>
        <KnnPill />
        {/* The pipeline params + server logs live inline on the Run
            view (see App.tsx run-view section) — no header buttons
            here. The browse view shouldn't surface pipeline chrome at
            all; the Run tab in .view-tabs is how you get there. */}
      </hgroup>
      {/* Compact status on browse (just a badge + ETA), full status
          (with stage bar + legend) on run. LegendBar lives only on
          run — the action buttons belong with the pipeline runner. */}
      {view === 'run' ? <StatusPanel /> : <StatusPanel compact />}
      {view === 'run' && <LegendBar />}
    </header>
  );
}