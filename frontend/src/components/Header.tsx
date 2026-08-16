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
    <header className="sticky top-0 z-50 bg-bg flex flex-wrap items-center gap-3 py-1.5 px-4 border-b border-line">
      <div className="inline-flex gap-1 p-[3px] bg-bg-1 border border-line2 rounded-lg" role="tablist" aria-label="View">
        <button
          role="tab"
          aria-selected={active === 'browse-list'}
          className={`bg-transparent text-fg-0 border border-transparent rounded-md cursor-pointer py-[3px] px-3 text-[0.8rem] hover:text-fg hover:border-[#444] ${active === 'browse-list' ? 'bg-accent-dark text-white border-accent' : ''}`}
          onClick={() => navigate('browse-list')}
        >
          Browse
        </button>
        <button
          role="tab"
          aria-selected={active === 'run'}
          className={`bg-transparent text-fg-0 border border-transparent rounded-md cursor-pointer py-[3px] px-3 text-[0.8rem] hover:text-fg hover:border-[#444] ${active === 'run' ? 'bg-accent-dark text-white border-accent' : ''}`}
          onClick={() => navigate('run')}
        >
          Run
        </button>
      </div>
      <hgroup className="flex items-center gap-2 m-0">
        <h1 className="m-0 text-[1.1rem]">Immich × BioCLIP</h1>
      </hgroup>
      {/* KNN pill + StatusPanel only on Run view; compact status in
          browse view. LegendBar buttons (sync / backfill / purge)
          only show on Run view. */}
      {active === 'run' && <KnnPill />}
      {active === 'run' ? <StatusPanel /> : <StatusPanel compact />}
      {active === 'run' && <LegendBar />}
    </header>
  );
}