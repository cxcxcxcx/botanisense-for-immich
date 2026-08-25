import { navigate, topLevel, type View } from '../hooks/useRoute';
import { useApp } from '../state/AppContext';

// Inline BotaniSense brand mark — same fan-of-blades SVG used by
// `frontend/public/favicon.svg`. Drawn at 28×28 in the header so the
// brand is immediately readable without the gradient circle wrapper.
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
    <header className="sticky top-0 z-50 bg-bg/95 backdrop-blur flex flex-wrap items-center gap-3 py-1.5 px-4 border-b border-line">
      {/* Brand identity: bare fan-of-blades SVG (no gradient ring).
          Wordmark + "for Immich" subtitle on mobile, one line on sm:
          and up. Sits on the left, takes only as much space as it
          needs. */}
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
        <span className="sm:hidden text-fg-1 text-[0.7rem] uppercase tracking-wider ml-1">
          · for Immich
        </span>
      </a>

      {/* Right-side group: tab strip only.
          The KNN-unavailable indicator lives in the Run page (KnnAlert
          banner at the top), not here — keeping the header simple. */}
      <div className="flex items-center gap-2 ml-auto">
        <div className="inline-flex gap-1 p-[3px] bg-bg-1 border border-line2 rounded-lg" role="tablist" aria-label="View">
          <button
            role="tab"
            aria-selected={active === 'browse-list'}
            className={`flex items-center gap-1.5 border border-transparent rounded-md cursor-pointer py-[3px] px-2.5 sm:px-3 text-[0.8rem] sm:text-xs ${active === 'browse-list' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0 hover:text-fg hover:border-[#444]'}`}
            onClick={() => navigate(browseTarget)}
          >
            <BrowseIcon />
            <span className="hidden sm:inline">Browse</span>
          </button>
          <button
            role="tab"
            aria-selected={active === 'run'}
            className={`flex items-center gap-1.5 border border-transparent rounded-md cursor-pointer py-[3px] px-2.5 sm:px-3 text-[0.8rem] sm:text-xs ${active === 'run' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0 hover:text-fg hover:border-[#444]'}`}
            onClick={() => navigate('run')}
          >
            <RunIcon />
            <span className="hidden sm:inline">Run / Sync</span>
          </button>
        </div>
      </div>
    </header>
  );
}

function BrowseIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <rect x="3" y="3" width="18" height="18" rx="2" ry="2" />
      <circle cx="8.5" cy="8.5" r="1.5" />
      <polyline points="21 15 16 10 5 21" />
    </svg>
  );
}

function RunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="4 17 10 11 4 5" />
      <line x1="12" y1="19" x2="20" y2="19" />
    </svg>
  );
}