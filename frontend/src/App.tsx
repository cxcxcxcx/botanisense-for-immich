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

  const modalTitle = modal.kind === 'asset'
    ? (modal.assetId ? `Asset ${modal.assetId.slice(0, 8)}…` : 'Asset detail')
    : '';

  return (
    <>
      <Header view={view} />
      <main className={`app-main view-${view}`}>
        {view === 'run' ? (
          <section className="run-view">
            <ParamsForm />
            <LogTail active={true} />
          </section>
        ) : view === 'browse-content' ? (
          <>
            <ClusterAlbumBar />
            <Grid />
          </>
        ) : (
          // browse-list — full-height filter cards. Toolbar is the
          // owner of FilterCards and supplies the sort/group-by chrome.
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
