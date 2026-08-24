import { KnnPill } from './KnnPill';
import { navigate, topLevel, type View } from '../hooks/useRoute';
import { useApp } from '../state/AppContext';

interface Props {
  view: View;
}

export function Header({ view }: Props) {
  // Both browse-list and browse-content share the "Browse" tab (we're
  // still browsing; the content view is just deeper into it).
  const active = topLevel(view);
  // Clicking "Browse" while a filter is active should land on the
  // photo grid (browse-content), not the empty filter-card list
  // (browse-list). The list view collapses the cards behind an
  // active-card summary when a filter is set, so without this jump
  // the user gets a blank main area below the banner — see the
  // `if (collapsed && activeCard && !showPersonChip) return null;`
  // early-return in FilterCards. Deep-link navigation is still
  // handled by the matching useEffect in App.tsx.
  const { state } = useApp();
  const hasActiveFilter =
    state.filters.cluster_id != null ||
    !!state.filters.species ||
    !!state.filters.genus ||
    !!state.filters.family;
  const browseTarget: View = hasActiveFilter ? 'browse-content' : 'browse-list';
  return (
    <header className="sticky top-0 z-50 bg-bg flex flex-wrap items-center gap-3 py-1.5 px-4 border-b border-line">
      <div className="inline-flex gap-1 p-[3px] bg-bg-1 border border-line2 rounded-lg" role="tablist" aria-label="View">
        <button
          role="tab"
          aria-selected={active === 'browse-list'}
          className={`bg-transparent text-fg-0 border border-transparent rounded-md cursor-pointer py-[3px] px-3 text-[0.8rem] hover:text-fg hover:border-[#444] ${active === 'browse-list' ? 'bg-accent-dark text-white border-accent' : ''}`}
          onClick={() => navigate(browseTarget)}
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
        <h1 className="m-0 text-[1.1rem]">BotaniSense for Immich</h1>
      </hgroup>
      {/* KNN pill on Run view. The full StatusPanel lives in the Run
          page's main column (one instance total), not the sticky header —
          having it at the top meant two status bars on the Run page.
          The Browse view doesn't get a status bar; if you want to see
          run progress, switch to the Run tab. */}
      {active === 'run' && <KnnPill />}
    </header>
  );
}