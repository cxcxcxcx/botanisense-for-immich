import { useEffect, useState } from 'react';
import { listClusterAlbums, deleteClusterAlbum, previewClusterAlbum, renameCluster } from '../api/endpoints';
import type { ClusterAlbumMapping } from '../api/types';
import { useApp, useFilters } from '../state/AppContext';
import { useToasts } from './ToastProvider';
import { ConfirmModal } from './ConfirmModal';
import { BetaInfoModal } from './BetaInfoModal';
import { Modal } from './Modal';

export function ClusterAlbumBar() {
  const filters = useFilters();
  const { dispatch, state } = useApp();
  const push = useToasts();

  const [albums, setAlbums] = useState<ClusterAlbumMapping[] | null>(null);
  const [busy, setBusy] = useState(false);
  // Two-step delete: the in-app ConfirmModal replaces window.confirm so
  // the destructive flow matches the rest of the design. Set when the
  // user clicks Delete; cleared by either confirm or cancel.
  const [pendingDelete, setPendingDelete] = useState(false);
  // Shared beta-info modal — opened by the inline beta tag.
  const [showBetaInfo, setShowBetaInfo] = useState(false);
  // Manual cluster rename — opened by the "Rename" button. name + busy
  // state live here so the inline input can flip to a spinner.
  const [renameOpen, setRenameOpen] = useState(false);
  const [renameValue, setRenameValue] = useState('');
  const [renaming, setRenaming] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listClusterAlbums().then(r => { if (!cancelled) setAlbums(r.albums); }).catch(() => {});
    return () => { cancelled = true; };
  }, []);

  if (filters.group_by !== 'cluster' || filters.cluster_id == null) return null;
  if (state.filtersCollapsed) return null;
  const clusterId = filters.cluster_id;

  const album = albums?.find(a => a.cluster_id === clusterId) ?? null;

  const handleSync = async () => {
    try {
      const prev = await previewClusterAlbum(clusterId);
      if (!prev.n_assets) {
        push({ kind: 'info', message: `Cluster #${clusterId} has no assets` }, 3000);
        return;
      }
      dispatch({ type: 'OPEN_JOB', kind: 'cluster', payload: { clusterId } });
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    }
  };

  const handleDelete = async () => {
    if (!album) return;
    setBusy(true);
    try {
      const r = await deleteClusterAlbum(clusterId);
      setAlbums(prev => (prev ?? []).filter(a => a.cluster_id !== clusterId));
      push({
        kind: 'success',
        message: r.remote_deleted
          ? 'Mapping dropped · Immich album deleted'
          : 'Mapping dropped · (Immich album was already gone)',
      }, 3000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    } finally {
      setBusy(false);
    }
  };

  const btnCls = 'px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-[0.8rem] hover:bg-[#345] hover:border-accent disabled:opacity-50';

  // Cluster rename — opens an in-app Modal with a single input
  // pre-filled with the current cluster id. PATCHes /api/clusters/{id}
  // and surfaces a toast on success/error. Empty input = clear the
  // override (revert to the computed name).
  const openRename = () => {
    setRenameValue('');
    setRenameOpen(true);
  };
  const submitRename = async (e: React.FormEvent) => {
    e.preventDefault();
    if (renaming) return;
    setRenaming(true);
    try {
      await renameCluster(clusterId, renameValue.trim() || null);
      push({ kind: 'success', message: `Cluster #${clusterId} renamed.` }, 2500);
      setRenameOpen(false);
      // Force the parent (and FilterCards) to refetch by toggling a
      // no-op filter patch. The cluster_summary cache is server-side
      // and was invalidated by the backend.
      dispatch({ type: 'SET_FILTERS_COLLAPSED', collapsed: state.filtersCollapsed });
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    } finally {
      setRenaming(false);
    }
  };
  // Clickable beta tag — opens the shared BetaInfoModal with the full
  // explanation. Cluster sync may produce stale titles on rename and
  // doesn't currently PATCH the album name.
  const betaTag = (
    <button
      type="button"
      onClick={() => setShowBetaInfo(true)}
      className="text-[0.65rem] uppercase tracking-wider px-1.5 py-0.5 rounded border border-[#a87] text-[#d8a] bg-[#2a1a2a] cursor-pointer hover:bg-[#3a2a3a]"
      title="Click for details on what's beta and why"
      aria-label="Beta feature: click for details"
    >
      beta
    </button>
  );

  return (
    <nav className="py-1 px-4 border-b border-line2 text-[0.85rem]">
      <div className="flex flex-wrap items-center gap-2">
        <strong>Cluster #{clusterId}</strong>
        {betaTag}
        {album ? (
          <>
            <span className="text-fg-1">Album:</span>
            <a href={album.album_url} target="_blank" rel="noopener noreferrer" className="text-[#7ad] hover:underline">
              {album.album_id.slice(0, 8)}…
            </a>
            <small className="text-fg-1">({album.n_assets} photos{album.reused_existing ? ', reused' : ''})</small>
            <button type="button" className={btnCls} onClick={handleSync}>↻ Re-sync</button>
            <button type="button" className={btnCls} onClick={openRename}>Rename</button>
            <button type="button" className={btnCls} onClick={() => setPendingDelete(true)} disabled={busy}>
              Delete album
            </button>
          </>
        ) : (
          <>
            <span className="text-fg-1">No Immich album yet.</span>
            <button type="button" className={btnCls} onClick={handleSync}>+ Sync this cluster as album</button>
            <button type="button" className={btnCls} onClick={openRename}>Rename</button>
          </>
        )}
      </div>
      <ConfirmModal
        open={pendingDelete}
        onClose={() => setPendingDelete(false)}
        onConfirm={handleDelete}
        // Cluster sync is beta; Immich album always goes with the local
        // mapping ("never leave orphans"). The confirm message makes
        // the cascade explicit so a click-by-accident is recoverable
        // via re-sync (cluster_names persists across runs).
        title={`Delete cluster #${clusterId}'s album?`}
        body={`This deletes the Immich album AND drops the local mapping. You can re-sync after this — the album will be recreated with the same #BotaniSense title.`}
        confirmLabel="Delete album"
        danger
      />
      <BetaInfoModal open={showBetaInfo} onClose={() => setShowBetaInfo(false)} />
      <Modal open={renameOpen} onClose={() => setRenameOpen(false)} title={`Rename cluster #${clusterId}`}>
        <form onSubmit={submitRename} className="flex flex-col gap-3 text-[0.85rem]">
          <p className="m-0 text-fg-1">
            Override the auto-generated name (e.g. <em>"Oak trees in yard"</em>).
            Leave blank to revert to the computed name. Override persists
            across pipeline runs.
          </p>
          <input
            type="text"
            value={renameValue}
            onChange={e => setRenameValue(e.target.value)}
            placeholder="Cluster name (optional)"
            autoFocus
            className="bg-bg-2 text-fg border border-line2 px-2.5 py-1.5 rounded text-[0.85rem] focus:outline-none focus:border-accent"
          />
          <footer className="flex gap-2 justify-end">
            <button
              type="button"
              className="px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-sm hover:bg-[#345]"
              onClick={() => setRenameOpen(false)}
            >
              Cancel
            </button>
            <button
              type="submit"
              className="px-2.5 py-1 rounded bg-accent-dark border border-accent text-white text-sm hover:bg-accent-hover disabled:opacity-50"
              disabled={renaming}
            >
              {renaming ? 'Saving…' : 'Save'}
            </button>
          </footer>
        </form>
      </Modal>
    </nav>
  );
}