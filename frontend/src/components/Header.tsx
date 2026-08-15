import { KnnPill } from './KnnPill';
import { StatusPanel } from './StatusPanel';
import { LegendBar } from './LegendBar';
import { navigate, topLevel, type View } from '../hooks/useRoute';

interface Props {
  view: View;
}

export function Header({ view }: Props) {
  // Both browse-list and browse-content share the "Browse" tab (we're
  // still browsing; the content view is just deeper into it).
  const active = topLevel(view);
  return (
    <header className="app-header">
      <div className="view-tabs" role="tablist" aria-label="View">
        <button
          role="tab"
          aria-selected={active === 'browse-list'}
          className={`view-tab ${active === 'browse-list' ? 'active' : ''}`}
          onClick={() => navigate('browse-list')}
        >
          Browse
        </button>
        <button
          role="tab"
          aria-selected={active === 'run'}
          className={`view-tab ${active === 'run' ? 'active' : ''}`}
          onClick={() => navigate('run')}
        >
          Run
        </button>
      </div>
      <hgroup>
        <h1>Immich × BioCLIP</h1>
      </hgroup>
      {/* The KNN pill only matters where reclassification is exposed
          (the Run view + per-asset reclassify buttons). On browse the
          user is looking at already-classified photos, so the
          "KNN: not built" state is just noise. */}
      {active === 'run' && <KnnPill />}
      {active === 'run' ? <StatusPanel /> : <StatusPanel compact />}
      {active === 'run' && <LegendBar />}
    </header>
  );
}
