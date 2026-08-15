import { useEffect } from 'react';
import { Header } from './components/Header';
import { Toolbar } from './components/Toolbar';
import { ClusterAlbumBar } from './components/ClusterAlbumBar';
import { Grid } from './components/Grid';
import { Modal } from './components/Modal';
import { AssetDetail } from './components/AssetDetail';
import { JobPanel } from './components/JobPanel';
import { useApp, useModal } from './state/AppContext';
import { initialFilters, parseUrlFilters, useUrlFilters } from './hooks/useUrlFilters';

export function App() {
  const { dispatch } = useApp();
  const modal = useModal();
  const setFilters = useUrlFilters();

  // On mount, hydrate the reducer from URL (and fall back to localStorage).
  useEffect(() => {
    dispatch({ type: 'SET_FILTERS', patch: initialFilters() });
  }, [dispatch]);

  // Back/forward navigation: re-parse the URL.
  useEffect(() => {
    const onPop = () => {
      const next = parseUrlFilters();
      // Reset the reducer to the URL's state.
      dispatch({ type: 'SET_FILTERS', patch: { ...next } });
    };
    window.addEventListener('popstate', onPop);
    return () => window.removeEventListener('popstate', onPop);
  }, [dispatch]);

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

  // Touch `setFilters` so eslint doesn't flag unused (the hook is here for
  // the linke above).
  void setFilters;

  return (
    <>
      <Header />
      <Toolbar />
      <ClusterAlbumBar />
      <Grid />
      <Modal
        open={modal.open}
        onClose={() => dispatch({ type: 'CLOSE_MODAL' })}
        title={modal.assetId ? `Asset ${modal.assetId.slice(0, 8)}…` : 'Asset detail'}
      >
        {modal.assetId && <AssetDetail assetId={modal.assetId} />}
      </Modal>
      <JobPanel />
    </>
  );
}