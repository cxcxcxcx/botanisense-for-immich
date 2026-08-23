import { useEffect } from 'react';
import { Header } from './components/Header';
import { Toolbar } from './components/Toolbar';
import { ClusterAlbumBar } from './components/ClusterAlbumBar';
import { Grid } from './components/Grid';
import { FilterCards } from './components/FilterCards';
import { Modal } from './components/Modal';
import { AssetDetail } from './components/AssetDetail';
import { ParamsForm } from './components/ParamsForm';
import { LogTail } from './components/LogTail';
import { JobPanel } from './components/JobPanel';
import { ImmichSync } from './components/ImmichSync';
import { LegendBar } from './components/LegendBar';
import { StatusPanel } from './components/StatusPanel';
import { useApp, useFilters, useModal } from './state/AppContext';
import { initialFilters, parseUrlFilters } from './hooks/useUrlFilters';
import { useRoute, navigate } from './hooks/useRoute';

export function App() {
  const { dispatch } = useApp();
  const modal = useModal();
  const filters = useFilters();
  const view = useRoute();

  // Whether the current URL filters narrow to a specific subset of
  // assets. This is the trigger for switching from the list view
  // (filter cards, full-height) to the content view (photo grid).
  const hasActiveFilter =
    filters.cluster_id != null ||
    !!filters.species ||
    !!filters.genus ||
    !!filters.family;

  // On mount, hydrate the reducer from URL (and fall back to localStorage).
  useEffect(() => {
    dispatch({ type: 'SET_FILTERS', patch: initialFilters() });
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
    if (h !== 'browse' && h !== 'browse/content' && h !== 'run') {
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
      <main className={`pb-8 ${view === 'run' ? 'max-w-[1100px] mx-auto py-4 px-6' : ''}`}>
        {view === 'run' ? (
          <section className="flex flex-col gap-4">
            {/* Run-page sections, top-to-bottom:
                  1. Pipeline run — query / sample size / batch + live progress
                  2. Sync to IMMICH — clusters / Plantae / Fungi / Remove
                  3. Data tools — KNN build / EXIF backfill / purge / detect people
                  4. Logs — pipeline + worker output */}
            <ParamsForm />
            <StatusPanel />
            <ImmichSync />
            <LegendBar />
            <LogTail active={true} />
          </section>
        ) : view === 'browse-content' ? (
          <>
            <ClusterAlbumBar />
            <FilterCards />
            <Grid />
          </>
        ) : (
          <Toolbar />
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
    </>
  );
}
