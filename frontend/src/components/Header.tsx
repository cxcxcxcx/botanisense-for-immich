import { navigate, topLevel, type View } from '../hooks/useRoute';
import { useApp } from '../state/AppContext';
import { getHealth } from '../api/endpoints';
import type { HealthEngine } from '../api/types';
import { useEffect, useRef, useState } from 'react';
import { BOTANISENSE_ENABLED } from '../lib/features';
import {
  APP_VERSION,
  BOTANISENSE_APP_URL,
  FEEDBACK_URL,
  REPO_URL,
  bugReportUrl,
} from '../lib/links';

// App icon (watercolor bird-of-paradise + Immich mark), the same image
// as the favicon/manifest icons; icon-64.png keeps it sharp at 2× DPR.
// Source: docs/brand/icon.svg. The earlier fan-of-blades mark is kept
// at docs/brand/fan-mark.svg.
function BrandMark({ size = 28 }: { size?: number }) {
  return (
    <img
      src="/icon-64.png"
      width={size}
      height={size}
      className="brand-mark"
      alt=""
      aria-hidden="true"
    />
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
    state.filters.supergroup != null ||
    !!state.filters.species ||
    !!state.filters.genus ||
    !!state.filters.family ||
    state.filters.near_lat != null;
  const browseTarget: View = hasActiveFilter ? 'browse-content' : 'browse-list';

  // Light/dark theme toggle. Initial value mirrors the inline
  // bootstrap script in index.html — same precedence: localStorage >
  // prefers-color-scheme > 'dark'.
  const [theme, setTheme] = useState<'light' | 'dark'>(() => {
    if (typeof document !== 'undefined') {
      const attr = document.documentElement.getAttribute('data-theme');
      if (attr === 'light' || attr === 'dark') return attr;
    }
    return 'dark';
  });
  useEffect(() => {
    document.documentElement.setAttribute('data-theme', theme);
    try { localStorage.setItem('botanisense:theme', theme); } catch { /* ignore */ }
  }, [theme]);

  // Engine badge (A/B instances). Fetched once — the engine is fixed
  // for the life of the container. Teacher (production default) renders
  // nothing; a mobile/A-B instance gets a small marker so two browser
  // tabs (:8770 vs :8771) are distinguishable at a glance.
  const [engine, setEngine] = useState<HealthEngine | null>(null);
  useEffect(() => {
    let cancelled = false;
    getHealth().then(
      (h) => { if (!cancelled) setEngine(h.engine ?? null); },
      () => { /* health unreachable — badge simply stays off */ },
    );
    return () => { cancelled = true; };
  }, []);
  const isMobileEngine = !!engine?.id && engine.id.startsWith('mobile');

  // Publish the header's live height as --header-h so the page-level
  // sticky bars (.sticky-toolbar / .back-bar) stick just below it
  // instead of sliding underneath (the header is also sticky, z-50).
  const headerRef = useRef<HTMLElement>(null);
  useEffect(() => {
    const el = headerRef.current;
    if (!el) return;
    const root = document.documentElement;
    const ro = new ResizeObserver(() => {
      root.style.setProperty('--header-h', `${el.getBoundingClientRect().height}px`);
    });
    ro.observe(el);
    return () => ro.disconnect();
  }, []);

  return (
    <header ref={headerRef} className="sticky top-0 z-50 bg-bg/95 backdrop-blur flex flex-nowrap items-center gap-2 sm:gap-3 py-1.5 px-3 sm:px-4 border-b border-line">
      {/* Brand identity: app icon spanning the two-line stacked wordmark
          (BotaniSense on top, smaller "for Immich" underneath) so both
          left and right icons vertically span the two text lines on one row. */}
      <a
        href="#browse"
        className="flex items-center gap-2 no-underline text-inherit min-w-0 shrink-0"
        title="BotaniSense for Immich — home"
      >
        <BrandMark size={30} />
        <span className="flex flex-col justify-center leading-none">
          <span className="wordmark">BotaniSense</span>
          <span className="wordmark-sub">for Immich</span>
        </span>
      </a>

      {/* Right-side group: tab strip + theme + info, vertically centered
          across the same two-line height as the brand lockup. */}
      <div className="flex items-center gap-1.5 sm:gap-2 ml-auto shrink-0">
        {isMobileEngine && (
          <span
            className="hidden sm:inline-flex items-center gap-1 text-[0.65rem] uppercase tracking-wider text-accent border border-accent/40 rounded px-1.5 py-[2px]"
            title={`Engine: ${engine?.id}${engine?.package ? ` (${engine.package})` : ''} · detector: ${engine?.yolo_backend ?? 'world'}`}
          >
            S0 mobile
          </span>
        )}
        <div className="inline-flex gap-0.5 sm:gap-1 p-[3px] bg-bg-1 border border-line2 rounded-lg" role="tablist" aria-label="View">
          <button
            role="tab"
            aria-selected={active === 'browse-list'}
            className={`flex items-center gap-1.5 border border-transparent rounded-md cursor-pointer py-[3px] px-2 sm:px-3 text-[0.8rem] sm:text-xs ${active === 'browse-list' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0 hover:text-fg hover:border-[#444]'}`}
            onClick={() => navigate(browseTarget)}
          >
            <BrowseIcon />
            <span className="hidden sm:inline">Browse</span>
          </button>
          <button
            role="tab"
            aria-selected={active === 'map'}
            className={`flex items-center gap-1.5 border border-transparent rounded-md cursor-pointer py-[3px] px-2 sm:px-3 text-[0.8rem] sm:text-xs ${active === 'map' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0 hover:text-fg hover:border-[#444]'}`}
            onClick={() => navigate('map')}
          >
            <MapIcon />
            <span className="hidden sm:inline">Map</span>
          </button>
          {BOTANISENSE_ENABLED && (
            <button
              role="tab"
              aria-selected={active === 'botanisense'}
              title="BotaniSense diary"
              className={`flex items-center gap-1.5 border border-transparent rounded-md cursor-pointer py-[3px] px-2 sm:px-3 text-[0.8rem] sm:text-xs ${active === 'botanisense' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0 hover:text-fg hover:border-[#444]'}`}
              onClick={() => navigate('botanisense')}
            >
              <DiaryIcon />
              <span className="hidden sm:inline">Diary</span>
            </button>
          )}
          <button
            role="tab"
            aria-selected={active === 'run'}
            className={`flex items-center gap-1.5 border border-transparent rounded-md cursor-pointer py-[3px] px-2 sm:px-3 text-[0.8rem] sm:text-xs ${active === 'run' ? 'bg-accent text-bg border-accent' : 'bg-transparent text-fg-0 hover:text-fg hover:border-[#444]'}`}
            onClick={() => navigate('run')}
          >
            <RunIcon />
            <span className="hidden sm:inline">Run / Sync</span>
          </button>
        </div>
        {/* Theme toggle — sun/moon icon flips data-theme between light
            and dark. State is mirrored to <html data-theme> via the
            useEffect above and persisted in localStorage. Inline
            bootstrap in index.html reads the same key on first paint
            to avoid a flash of the wrong theme. */}
        <button
          type="button"
          aria-label={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          title={theme === 'dark' ? 'Switch to light theme' : 'Switch to dark theme'}
          className="p-1.5 rounded border border-line2 text-fg-1 hover:text-fg hover:border-accent"
          onClick={() => setTheme(t => (t === 'dark' ? 'light' : 'dark'))}
        >
          {theme === 'dark' ? <SunIcon /> : <MoonIcon />}
        </button>
        <AboutMenu
          detail={engine ? `engine ${engineLabel(engine)}, detector ${engine.yolo_backend ?? '?'}` : undefined}
        />
      </div>
    </header>
  );
}

// "auto → Core ML" etc. for ENGINE=auto/coreml, so a Mac user can see
// whether the accelerated encoder is actually in use.
function engineLabel(e: HealthEngine): string {
  if (e.id !== 'auto' && e.id !== 'coreml') return e.id ?? '?';
  const used = e.loaded?.engine ?? e.resolved;
  const name = used === 'coreml' ? 'Core ML' : 'BioCLIP-2 torch';
  const why = used !== 'coreml' && (e.loaded?.fallback_reason ?? e.unavailable_reason);
  return `${e.id} → ${name}${why ? ` (${why})` : ''}`;
}

// About / feedback popover: version, GitHub, issue links and the
// BotaniSense app. Closes on outside click, Escape, or following a link.
function AboutMenu({ detail }: { detail?: string }) {
  const [open, setOpen] = useState(false);
  const ref = useRef<HTMLDivElement>(null);
  useEffect(() => {
    if (!open) return;
    const onDown = (e: MouseEvent) => {
      if (ref.current && !ref.current.contains(e.target as Node)) setOpen(false);
    };
    const onKey = (e: KeyboardEvent) => { if (e.key === 'Escape') setOpen(false); };
    document.addEventListener('mousedown', onDown);
    document.addEventListener('keydown', onKey);
    return () => {
      document.removeEventListener('mousedown', onDown);
      document.removeEventListener('keydown', onKey);
    };
  }, [open]);

  const item = 'flex items-center justify-between gap-3 px-3 py-1.5 text-[12.5px] text-fg-0 no-underline hover:bg-bg-2 hover:text-fg';
  return (
    <div ref={ref} className="relative">
      <button
        type="button"
        aria-label="About and feedback"
        aria-expanded={open}
        aria-haspopup="menu"
        title="About and feedback"
        className="p-1.5 rounded border border-line2 text-fg-1 hover:text-fg hover:border-accent"
        onClick={() => setOpen(o => !o)}
      >
        <InfoIcon />
      </button>
      {open && (
        <div
          role="menu"
          className="absolute right-0 mt-1.5 w-64 py-1 rounded-lg border border-line2 bg-bg-1 shadow-lg z-50"
          onClick={e => { if ((e.target as HTMLElement).closest('a')) setOpen(false); }}
        >
          <div className="px-3 pt-1.5 pb-2 border-b border-line">
            <div className="text-[13px] text-fg font-semibold">BotaniSense for Immich</div>
            <div className="text-[11px] text-fg-2">
              v{APP_VERSION} · plant ID for your Immich library
            </div>
          </div>
          <a role="menuitem" className={item} href={REPO_URL} target="_blank" rel="noreferrer">
            GitHub <span aria-hidden="true">↗</span>
          </a>
          <a role="menuitem" className={item} href={bugReportUrl(detail)} target="_blank" rel="noreferrer">
            Report a problem <span aria-hidden="true">↗</span>
          </a>
          <a role="menuitem" className={item} href={FEEDBACK_URL} target="_blank" rel="noreferrer">
            Ideas &amp; feedback <span aria-hidden="true">↗</span>
          </a>
          <div className="my-1 border-t border-line" />
          <a role="menuitem" className={item} href={BOTANISENSE_APP_URL} target="_blank" rel="noreferrer">
            BotaniSense app <span aria-hidden="true">↗</span>
          </a>
          <a role="menuitem" className={item} href={`${REPO_URL}#license`} target="_blank" rel="noreferrer">
            License <span aria-hidden="true">↗</span>
          </a>
        </div>
      )}
    </div>
  );
}

function InfoIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="10" />
      <line x1="12" y1="16" x2="12" y2="12" />
      <line x1="12" y1="8" x2="12.01" y2="8" />
    </svg>
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

function MapIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 10c0 7-9 13-9 13s-9-6-9-13a9 9 0 0 1 18 0z" />
      <circle cx="12" cy="10" r="3" />
    </svg>
  );
}

function DiaryIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M11 20A7 7 0 0 1 9.8 6.1C15.5 5 17 4.48 19 2c1 2 2 4.18 2 8 0 5.5-4.78 10-10 10Z" />
      <path d="M2 21c0-3 1.85-5.36 5.08-6C9.5 14.52 12 13 13 12" />
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

function SunIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <circle cx="12" cy="12" r="4" />
      <path d="M12 2v2" />
      <path d="M12 20v2" />
      <path d="M4.93 4.93l1.41 1.41" />
      <path d="M17.66 17.66l1.41 1.41" />
      <path d="M2 12h2" />
      <path d="M20 12h2" />
      <path d="M4.93 19.07l1.41-1.41" />
      <path d="M17.66 6.34l1.41-1.41" />
    </svg>
  );
}

function MoonIcon() {
  return (
    <svg width="14" height="14" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <path d="M21 12.79A9 9 0 1 1 11.21 3 7 7 0 0 0 21 12.79z" />
    </svg>
  );
}