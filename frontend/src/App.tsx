import { useEffect } from 'react';
import { Header } from './components/Header';
import { Toolbar } from './components/Toolbar';
import { ClusterAlbumBar } from './components/ClusterAlbumBar';
import { Grid } from './components/Grid';
import { Modal } from './components/Modal';
import { AssetDetail } from './components/AssetDetail';
import { ParamsForm } from './components/ParamsForm';
import { LogTail } from './components/LogTail';
import { JobPanel } from './components/JobPanel';
import { useApp, useFilters, useModal } from './state/AppContext';
import { initialFilters, parseUrlFilters } from './hooks/useUrlFilters';
import { useRoute } from './hooks/useRoute';

export function App() {
  const { dispatch } = useApp();
  const modal = useModal();
  const filters = useFilters();
  const view = useRoute();

  // Browse home: only filter cards. Once a cluster/taxon filter is
  // active, swap to the photo grid with a back affordance. This keeps
  // the first paint lightweight (no per-asset fetches) and lets the
  // user choose what subset of photos to render.
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
    if (!window.location.hash || window.location.hash === '#') {
      window.location.hash = '#browse';
    }
  }, []);

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

  const modalTitle = modal.kind === 'asset'
    ? (modal.assetId ? `Asset ${modal.assetId.slice(0, 8)}…` : 'Asset detail')
    : '';

  return (
    <>
      <Header view={view} />
      <main className={`app-main view-${view}`}>
        {view === 'browse' ? (
          hasActiveFilter ? (
            <>
              <Toolbar />
              <ClusterAlbumBar />
              <Grid />
            </>
          ) : (
            <Toolbar />
          )
        ) : (
          <section className="run-view">
            <ParamsForm />
            <LogTail active={true} />
          </section>
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