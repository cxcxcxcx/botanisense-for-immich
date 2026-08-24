import { KnnPill } from './KnnPill';
import { navigate, topLevel, type View } from '../hooks/useRoute';
import { useApp } from '../state/AppContext';

// Inline BotaniSense brand mark — same fan-of-blades SVG used by
// `frontend/public/favicon.svg`, drawn here so the header doesn't have
// to round-trip through Vite's asset pipeline. Keep these two in sync
// when the brand evolves.
function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      width={size}
      height={size}
      className="brand-mark"
      aria-hidden="true"
    >
      <g transform="translate(16 24)">
        <path d="M0 0 C -1 -7 -7 -13 -12 -10 C -8 -5 -4 -2 0 0 Z" fill="#c8201f" />
        <path d="M0 0 C -1 -10 -3 -17 -6 -16 C -5 -10 -2 -4 0 0 Z" fill="#ee7d18" />
        <path d="M0 0 C 0 -12 -1 -19 -3 -18 C -2 -12 -1 -5 0 0 Z" fill="#f3c623" />
        <path d="M0 0 C 2 -11 4 -17 7 -16 C 5 -11 3 -5 0 0 Z" fill="#6b3a8a" />
        <path d="M0 0 C 4 -9 10 -12 12 -8 C 8 -5 4 -2 0 0 Z" fill="#1f6fb4" />
        <path d="M0 0 C 7 2 13 1 14 -2 C 9 -2 4 -2 0 0 Z" fill="#1f8a3d" />
      </g>
    </svg>
  );
}

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
    <header className="sticky top-0 z-50 bg-bg/95 backdrop-blur flex flex-wrap items-center gap-3 py-1.5 px-4 border-b border-line">
      {/* Brand identity: the fan-of-blades SVG mark + a two-line
          wordmark ("BotaniSense" + "for Immich" subtitle) on mobile.
          On `md:` and up the wordmark collapses into one horizontal
          row ("BotaniSense · for Immich") so the tabs sit to the
          right at the same baseline. */}
      <a
        href="#browse"
        className="flex items-center gap-2 no-underline text-inherit"
        title="BotaniSense for Immich — home"
      >
        <BrandMark size={28} />
        <span className="flex flex-col leading-tight">
          <span className="wordmark">BotaniSense</span>
          <span className="wordmark-sub hidden sm:inline">for Immich</span>
        </span>
        {/* On `sm:` and below, show the subtitle inline next to the
            brand name as a small middot-separated tag. On `md:` and
            up, the subtitle moves to its own line above for a
            cleaner two-row identity. */}
        <span className="sm:hidden text-fg-1 text-[0.7rem] uppercase tracking-wider ml-1">
          · for Immich
        </span>
      </a>

      {/* Tab strip — single rounded pill containing Browse/Run.
          Active = filled accent (clear against dark bg). */}
      <div className="inline-flex gap-1 p-[3px] bg-bg-1 border border-line2 rounded-lg order-3 sm:order-2 ml-auto sm:ml-0" role="tablist" aria-label="View">
        <button
          role="tab"
          aria-selected={active === 'browse-list'}
          className={`border border-transparent rounded-md cursor-pointer py-[3px] px-3 text-[0.8rem] hover:text-fg hover:border-[#444] ${active === 'browse-list' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0'}`}
          onClick={() => navigate(browseTarget)}
        >
          Browse
        </button>
        <button
          role="tab"
          aria-selected={active === 'run'}
          className={`border border-transparent rounded-md cursor-pointer py-[3px] px-3 text-[0.8rem] hover:text-fg hover:border-[#444] ${active === 'run' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0'}`}
          onClick={() => navigate('run')}
        >
          Run
        </button>
      </div>

      {/* KNN pill on Run view only. Sits to the right of the tabs and
          wraps below them on narrow screens. The full StatusPanel
          lives in the Run page's main column (one instance total). */}
      <div className="order-2 sm:order-3 ml-auto sm:ml-0">
        {active === 'run' && <KnnPill />}
      </div>
    </header>
  );
}