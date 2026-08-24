import { KnnPill } from './KnnPill';
import { navigate, topLevel, type View } from '../hooks/useRoute';
import { useApp } from '../state/AppContext';

// Inline BotaniSense brand mark — same fan-of-blades SVG used by
// `frontend/public/favicon.svg`. Drawn here so the header doesn't
// have to round-trip through Vite's asset pipeline. The mark sits
// inside an emerald-cyan gradient circle (per the mock) so the
// brand reads as a single icon, not six separate blades.
function BrandMark({ size = 16 }: { size?: number }) {
  return (
    <svg
      xmlns="http://www.w3.org/2000/svg"
      viewBox="0 0 32 32"
      width={size}
      height={size}
      aria-hidden="true"
    >
      <g transform="translate(16 24)">
        <path d="M0 0 C -1 -7 -7 -13 -12 -10 C -8 -5 -4 -2 0 0 Z" fill="#0a2e1a" />
        <path d="M0 0 C -1 -10 -3 -17 -6 -16 C -5 -10 -2 -4 0 0 Z" fill="#0a2e1a" />
        <path d="M0 0 C 0 -12 -1 -19 -3 -18 C -2 -12 -1 -5 0 0 Z" fill="#0a2e1a" />
        <path d="M0 0 C 2 -11 4 -17 7 -16 C 5 -11 3 -5 0 0 Z" fill="#0a2e1a" />
        <path d="M0 0 C 4 -9 10 -12 12 -8 C 8 -5 4 -2 0 0 Z" fill="#0a2e1a" />
        <path d="M0 0 C 7 2 13 1 14 -2 C 9 -2 4 -2 0 0 Z" fill="#0a2e1a" />
      </g>
    </svg>
  );
}

interface Props {
  view: View;
}

export function Header({ view }: Props) {
  // Both browse-list and browse-content share the "Browse" tab.
  const active = topLevel(view);
  // Clicking "Browse" while a filter is active should land on the
  // photo grid (browse-content) rather than the empty Level-1
  // toolbar. See the comment in App.tsx for the matching useEffect.
  const { state } = useApp();
  const hasActiveFilter =
    state.filters.cluster_id != null ||
    !!state.filters.species ||
    !!state.filters.genus ||
    !!state.filters.family;
  const browseTarget: View = hasActiveFilter ? 'browse-content' : 'browse-list';
  return (
    <header className="shrink-0 z-20 bg-bg/90 backdrop-blur-md border-b border-line px-3 py-2.5 flex items-center justify-between">
      {/* Brand identity: gradient circle with the fan-of-blades mark,
          plus the wordmark + "FOR IMMICH" tag. Single horizontal row
          on every viewport — the wordmark never wraps. */}
      <div className="flex items-center gap-2 min-w-0">
        <div className="brand-circle">
          <BrandMark size={16} />
        </div>
        <span className="font-bold text-[15px] tracking-tight truncate">BotaniSense</span>
        <span className="brand-tag hidden sm:inline">FOR IMMICH</span>
      </div>

      {/* Tab strip — Browse / Run. Each tab carries an icon and (on
          sm:+) a label. Active = bg-bg-1 text-accent; inactive =
          text-fg-1 hover:text-fg. */}
      <div className="flex bg-bg-1/80 p-1 rounded-lg border border-line">
        <button
          onClick={() => navigate(browseTarget)}
          aria-selected={active === 'browse-list'}
          role="tab"
          className={active === 'browse-list' ? 'tab-button tab-button-active' : 'tab-button tab-button-inactive'}
        >
          <BrowseIcon />
          <span className="hidden sm:inline">Browse</span>
        </button>
        <button
          onClick={() => navigate('run')}
          aria-selected={active === 'run'}
          role="tab"
          className={active === 'run' ? 'tab-button tab-button-active' : 'tab-button tab-button-inactive'}
        >
          <RunIcon />
          <span className="hidden sm:inline">Run / Sync</span>
        </button>
      </div>

      {/* KNN pill on Run view only. Sits at the far right of the
          header on Run view; hidden entirely on Browse. */}
      <div className="ml-2 hidden md:block">
        {active === 'run' && <KnnPill />}
      </div>
    </header>
  );
}

function BrowseIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

function RunIcon() {
  return (
    <svg width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  );
}