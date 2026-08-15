import { useEffect, useState } from 'react';
import { listClusterAlbums, deleteClusterAlbum, previewClusterAlbum } from '../api/endpoints';
import type { ClusterAlbumMapping } from '../api/types';
import { useApp, useFilters } from '../state/AppContext';
import { useToasts } from './ToastProvider';

export function ClusterAlbumBar() {
  const filters = useFilters();
  const { dispatch, state } = useApp();
  const push = useToasts();

  // All hooks MUST be called unconditionally — early return would change
  // the hook count between renders and React would throw #310.
  const [albums, setAlbums] = useState<ClusterAlbumMapping[] | null>(null);
  const [busy, setBusy] = useState(false);
  const [deleteRemote, setDeleteRemote] = useState(false);

  useEffect(() => {
    let cancelled = false;
    listClusterAlbums()
      .then(r => { if (!cancelled) setAlbums(r.albums); })
      .catch(() => { /* leave null */ });
    return () => { cancelled = true; };
  }, []);

  // Render nothing when no cluster pill is active.
  if (filters.group_by !== 'cluster' || filters.cluster_id == null) return null;
  // Skip when FilterCards is collapsed — the active-bar already shows
  // "Cluster #N · count · group_by" with a ✕ to clear, so this bar
  // would duplicate the cluster identity.
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
        message: deleteRemote
        && r.remote_deleted ? 'Mapping dropped · Immich album deleted'
          : deleteRemote ? 'Mapping dropped · (Immich album was already gone)'
          : 'Mapping dropped',
      }, 3000);
    } catch (e) {
      push({ kind: 'error', message: e instanceof Error ? e.message : String(e) }, 4000);
    } finally {
      setBusy(false);
    }
  };

  return (
    <nav className="cluster-album-bar">
      <div className="cluster-album-controls">
        <strong>Cluster #{clusterId}</strong>
        {album ? (
          <>
            <span className="muted">Album:</span>
            <a href={album.album_url} target="_blank" rel="noopener noreferrer">
              {album.album_id.slice(0, 8)}…
            </a>
            <small className="muted">({album.n_assets} photos{album.reused_existing ? ', reused' : ''})</small>
            <button type="button" className="secondary outline" onClick={handleSync}>↻ Re-sync</button>
            <label className="muted" style={{ display: 'flex', alignItems: 'center', gap: '0.25rem' }}>
              <input
                type="checkbox"
                checked={deleteRemote}
                onChange={e => setDeleteRemote(e.target.checked)}
              />
              also delete in Immich
            </label>
            <button type="button" className="secondary outline" onClick={handleDelete} disabled={busy}>
              Delete mapping
            </button>
          </>
        ) : (
          <>
            <span className="muted">No Immich album yet.</span>
            <button type="button" className="secondary outline" onClick={handleSync}>+ Sync this cluster as album</button>
          </>
        )}
      </div>
    </nav>
  );
}