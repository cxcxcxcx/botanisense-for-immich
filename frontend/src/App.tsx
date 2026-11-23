import { lazy, Suspense, useEffect, useState } from 'react';
import { Header } from './components/Header';
import { Toolbar } from './components/Toolbar';
import { ClusterAlbumBar } from './components/ClusterAlbumBar';
import { Grid } from './components/Grid';
import { LocationMapView } from './components/LocationMapView';
import { Modal } from './components/Modal';
import { AssetDetail } from './components/AssetDetail';
import { ParamsForm } from './components/ParamsForm';
import { LogTail } from './components/LogTail';
import { JobPanel } from './components/JobPanel';
import { ImmichSync } from './components/ImmichSync';
import { LegendBar } from './components/LegendBar';
import { StatusPanel } from './components/StatusPanel';
import { ErrorBoundary } from './components/ErrorBoundary';
import { Welcome } from './components/Welcome';
import { HealthBanner } from './components/HealthBanner';
import { PlantaeLegend } from './components/PlantaeLegend';
import { ShortcutsHelpModal } from './components/ShortcutsHelpModal';
import { SelectionActionBar } from './components/SelectionActionBar';
import { ConnectionModal } from './components/ConnectionModal';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import { useApp, useFilters, useModal } from './state/AppContext';
import { initialFilters, parseUrlFilters } from './hooks/useUrlFilters';
import { useRoute, navigate } from './hooks/useRoute';
import { getConfig } from './api/endpoints';
import { BOTANISENSE_ENABLED } from './lib/features';

// Lazy so a build with the switch off never bundles firebase.
const BotanisensePage = BOTANISENSE_ENABLED
  ? lazy(() => import('./components/BotanisensePage').then(m => ({ default: m.BotanisensePage })))
  : null;

export function App() {
  const { dispatch } = useApp();
  const modal = useModal();
  const filters = useFilters();
  const view = useRoute();
  // Hoisted to the App so the "Edit params" toggle sits in the
  // Pipeline card header (per the mock) instead of inside the form.
  // Default to open on first visit so the most consequential knob
  // (`query`) is visible without an extra click; remember the user's
  // preference in localStorage so it survives reloads.
  const [showShortcuts, setShowShortcuts] = useState(false);

  // Keyboard shortcuts — single keys for the two main views + help.
  // Disabled while typing in an input (handled by the hook itself).
  useKeyboardShortcuts([
    { key: 'b', label: 'b', action: () => navigate('browse-list') },
    { key: 'm', label: 'm', action: () => navigate('map') },
    { key: 'r', label: 'r', action: () => navigate('run') },
    ...(BOTANISENSE_ENABLED
      ? [{ key: 't', label: 't', action: () => navigate('botanisense') }]
      : []),
    { key: '?', label: '?', action: () => setShowShortcuts(s => !s) },
  ]);

  const [paramsOpen, setParamsOpen] = useState<boolean>(() => {
    if (typeof window === 'undefined') return true;
    const stored = localStorage.getItem('botanisense:params-open');
    return stored === null ? true : stored === '1';
  });

  // Whether the current URL filters narrow to a specific subset of
  // assets. This is the trigger for switching from the list view
  // (filter cards, full-height) to the content view (photo grid).
  const hasActiveFilter =
    filters.cluster_id != null ||
    filters.supergroup != null ||
    !!filters.species ||
    !!filters.genus ||
    !!filters.family ||
    filters.near_lat != null;

  // On mount, hydrate the reducer from URL (and fall back to localStorage).
  useEffect(() => {
    dispatch({ type: 'SET_FILTERS', patch: initialFilters() });
  }, [dispatch]);

  // Fetch the non-secret config once (connection form prefill + the
  // env-pinned notice); refreshed after the connection form saves.
  useEffect(() => {
    let cancelled = false;
    getConfig()
      .then(cfg => { if (!cancelled) dispatch({ type: 'SET_CONFIG', config: cfg }); })
      .catch(() => { /* form falls back to an empty prefill */ });
    return () => { cancelled = true; };
  }, [dispatch]);

  // Back/forward navigation: re-parse the URL. Both popstate (history
  // navigation) and hashchange (route changes) trigger this; on a
  // hash-only change the diff is a no-op for filters, which is fine.
  useEffect(() => {
    const onChange = () => {
      dispatch({ type: 'SET_FILTERS', patch: { ...parseUrlFilters() } });
    };
    window.addEventListener('popstate', onChange);
    window.addEventListener('hashchange', onChange);
    return () => {
      window.removeEventListener('popstate', onChange);
      window.removeEventListener('hashchange', onChange);
    };
  }, [dispatch]);

  // Ensure a stable hash on first load so back/forward always works
  // and the URL bar shows the current view.
  useEffect(() => {
    const h = window.location.hash.replace(/^#/, '').trim();
    if (h !== 'browse' && h !== 'browse/content' && h !== 'map' && h !== 'run' && !(BOTANISENSE_ENABLED && h === 'botanisense')) {
      window.location.hash = '#browse';
    }
  }, []);

  // Keep the route aligned with the filter state.
  //   - filter becomes active while on the list view → jump to content
  //     (so deep-links like ?cluster_id=2 land on the photo grid).
  //   - filter is cleared while on content view → STAY on content.
  //     The user navigated there explicitly (clicked "All" or a card),
  //     and the breadcrumb "back" link is the canonical way to leave
  //     the content view. Auto-jumping back here would override that
  //     and leave the "All" card click stranded on the filter cards.
  useEffect(() => {
    const h = window.location.hash.replace(/^#/, '').trim();
    if (hasActiveFilter && h === 'browse') {
      navigate('browse-content');
    }
  }, [hasActiveFilter]);

  // Close the modal on Escape (the dialog handles this natively, but we
  // also need to flip the React state).
  useEffect(() => {
    if (!modal.open) return;
    const onKey = (e: KeyboardEvent) => {
      if (e.key === 'Escape') dispatch({ type: 'CLOSE_MODAL' });
    };
    window.addEventListener('keydown', onKey);
    return () => window.removeEventListener('keydown', onKey);
  }, [modal.open, dispatch]);

  // No asset-id title in the modal header — the body shows the
  // filename + capture time prominently, and a truncated UUID adds
  // noise without telling the user anything they don't already see.
  const modalTitle = '';

  return (
    <>
      <Header view={view} />
      {/* ErrorBoundary wraps the page body so a render exception in
          Grid / Modal / etc. doesn't take the whole SPA down. The
          Header above stays rendered so the user can still navigate. */}
      <ErrorBoundary>
      <HealthBanner />
      <main className={`pb-8 ${view === 'run' ? 'max-w-[1100px] mx-auto py-4 px-3 sm:px-6' : ''}`}>
        {view === 'run' ? (
          <section className="run-section">
            {/* Run-page sections, top-to-bottom:
                  1. Pipeline run — params + Run button + live progress
                  2. Sync to IMMICH — clusters / Plantae / Fungi / Remove
                  3. Data tools — KNN status + build / EXIF backfill / purge / detect people
                  4. Logs — pipeline + worker output */}
            <section className="run-card" aria-label="Pipeline">
              <header className="run-card-header">
                <h3 className="run-card-title">
                  <PipelineIcon />
                  Pipeline
                </h3>
                <button
                  type="button"
                  className="run-card-edit-btn"
                  onClick={() => setParamsOpen(o => {
                    const next = !o;
                    localStorage.setItem('botanisense:params-open', next ? '1' : '0');
                    return next;
                  })}
                  aria-expanded={paramsOpen}
                >
                  {paramsOpen ? 'Hide params' : 'Edit params'}
                </button>
              </header>
              <ParamsForm paramsOpen={paramsOpen} />
              <div className="mt-3">
                <StatusPanel />
              </div>
            </section>
            <ImmichSync />
            <LegendBar />
            <section className="pipeline-log-card" aria-label="Pipeline log">
              <header className="pipeline-log-header">
                <span className="pipeline-log-title">PIPELINE LOG</span>
                <span className="text-fg-2 font-sans text-[10px]">tail</span>
              </header>
              <div className="pipeline-log-body">
                <LogTail active={true} />
              </div>
            </section>
          </section>
        ) : view === 'map' ? (
          <LocationMapView />
        ) : view === 'botanisense' && BotanisensePage ? (
          <Suspense fallback={null}>
            <BotanisensePage />
          </Suspense>
        ) : view === 'browse-content' ? (
          <>
            <ClusterAlbumBar />
            <Grid />
          </>
        ) : (
          <>
            <Welcome />
            <Toolbar />
          </>
        )}
      </main>
      <Modal
        open={modal.open}
        onClose={() => dispatch({ type: 'CLOSE_MODAL' })}
        title={modalTitle}
      >
        {modal.kind === 'asset' && modal.assetId && <AssetDetail assetId={modal.assetId} />}
      </Modal>
      <JobPanel />
      {/* PlantaeLegend — bottom-left on every view. The Run page
          already has its own tier indicator next to the slider, so
          we pass `dismissible={false}` there to avoid a duplicate. */}
      <PlantaeLegend dismissible={view !== 'run'} />
      <SelectionActionBar />
      <ShortcutsHelpModal open={showShortcuts} onClose={() => setShowShortcuts(false)} />
      <ConnectionModal />
      </ErrorBoundary>
    </>
  );
}

function PipelineIcon() {
  return (
    <svg className="run-card-icon" viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="2" strokeLinecap="round" strokeLinejoin="round" aria-hidden="true">
      <polyline points="22 12 18 12 15 21 9 3 6 12 2 12" />
    </svg>
  );
}
