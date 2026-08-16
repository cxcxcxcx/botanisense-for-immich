import { useEffect, useState } from 'react';
import { listClusterAlbums, deleteClusterAlbum, previewClusterAlbum } from '../api/endpoints';
import type { ClusterAlbumMapping } from '../api/types';
import { useApp, useFilters } from '../state/AppContext';
import { useToasts } from './ToastProvider';

export function ClusterAlbumBar() {
  const filters = useFilters();
  const { dispatch, state } = useApp();
  const push = useToasts();

  const [albums, setAlbums] = useState<ClusterAlbumMapping[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleteRemote, setDeleteRemote] = useState(false);

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
    if (!confirm(`Drop mapping for cluster #${clusterId}?`)) return;
    setBusy(true);
    try {
      const r = await deleteClusterAlbum(clusterId, deleteRemote);
      setAlbums(prev => (prev ?? []).filter(a => a.cluster_id !== clusterId));
      push({
        kind: 'success',
        message: deleteRemote && r.remote_deleted ? 'Mapping dropped · Immich album deleted'
          : deleteRemote ? 'Mapping dropped · (Immich album was already gone)'
          : 'Mapping dropped',
      }, 3000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    } finally {
      setBusy(false);
    }
  };

  const btnCls = 'px-2.5 py-1 rounded border border-[#456] bg-transparent text-fg text-[0.8rem] hover:bg-[#345] hover:border-accent disabled:opacity-50';

  return (
    <nav className="py-1 px-4 border-b border-line2 text-[0.85rem]">
      <div className="flex flex-wrap items-center gap-2">
        <strong>Cluster #{clusterId}</strong>
        {album ? (
          <>
            <span className="text-fg-1">Album:</span>
            <a href={album.album_url} target="_blank" rel="noopener noreferrer" className="text-[#7ad] hover:underline">
              {album.album_id.slice(0, 8)}…
            </a>
            <small className="text-fg-1">({album.n_assets} photos{album.reused_existing ? ', reused' : ''})</small>
            <button type="button" className={btnCls} onClick={handleSync}>↻ Re-sync</button>
            <label className="text-fg-1 flex items-center gap-1">
              <input
                type="checkbox"
                checked={deleteRemote}
                onChange={e => setDeleteRemote(e.target.checked)}
                className="m-0"
              />
              also delete in Immich
            </label>
            <button type="button" className={btnCls} onClick={handleDelete} disabled={busy}>
              Delete mapping
            </button>
          </>
        ) : (
          <>
            <span className="text-fg-1">No Immich album yet.</span>
            <button type="button" className={btnCls} onClick={handleSync}>+ Sync this cluster as album</button>
          </>
        )}
      </div>
    </nav>
  );
}