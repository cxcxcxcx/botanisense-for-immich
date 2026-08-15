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

  void setFilters;

  // The Modal is a single <dialog> slot reused for three different
  // dialogs (asset detail / params / logs) — discriminated by modal.kind.
  const modalTitle = modal.kind === 'asset'
    ? (modal.assetId ? `Asset ${modal.assetId.slice(0, 8)}…` : 'Asset detail')
    : modal.kind === 'params' ? 'Pipeline parameters'
    : modal.kind === 'logs' ? 'Server logs'
    : '';

  return (
    <>
      <Header />
      <Toolbar />
      <ClusterAlbumBar />
      <Grid />
      <Modal
        open={modal.open}
        onClose={() => dispatch({ type: 'CLOSE_MODAL' })}
        title={modalTitle}
      >
        {modal.kind === 'asset' && modal.assetId && <AssetDetail assetId={modal.assetId} />}
        {modal.kind === 'params' && <ParamsForm />}
        {modal.kind === 'logs' && <LogTail active={modal.open} />}
      </Modal>
      <JobPanel />
    </>
  );
}