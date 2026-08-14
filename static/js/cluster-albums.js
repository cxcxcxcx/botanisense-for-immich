// Per-cluster Immich album sync/delete.
//
// One click on a cluster pill surfaces a contextual toolbar row below
// the cluster sidebar: "Sync this cluster as album" if no album exists
// yet, or the existing-album chip with [Re-sync] [Delete] if it does.
// The actual work runs server-side in a background job (POST
// /api/album/cluster/{id}/sync, GET /api/album/cluster/job/{id}) so the
// UI can show a live progress bar while the bulk-add is in flight.
//
// This module mirrors the Plants-album flow (./plants.js) closely — same
// confirm → running → done/error state machine, same 400ms poller, same
// data-action delegation. The only difference is the scope: per-cluster
// instead of per-kingdom.
import { state } from './state.js';
import { safeFetch, escapeHtml, toast } from './utils.js';

const POLL_MS = 400;

// In-flight job id. Non-null while a job is being polled for the
// current cluster; used to ignore stale poll responses and to dedupe
// accidental double-clicks.
let activeJobId = null;
let pollTimer = null;

// Server-side mapping cluster_id -> {album_id, album_url, n_assets, ...}
// Populated by loadClusterAlbums() on init and refreshed after every
// sync/delete.
let clusterAlbumsMap = new Map();

export function wireClusterAlbumBar() {
  ensurePanel();
  // Single delegated handler for the panel's data-action buttons.
  document.addEventListener('click', (e) => {
    const t = e.target.closest('#cluster-album-panel [data-action]');
    if (!t) return;
    const action = t.getAttribute('data-action');
    const clusterIdAttr = t.getAttribute('data-cluster-id');
    const clusterId = clusterIdAttr !== null ? Number(clusterIdAttr) : null;
    switch (action) {
      // per-cluster sync
      case 'sync-start':
        syncClusterAlbum(clusterId, t);
        break;
      case 'sync-confirm':
        startJob(clusterId);
        break;
      case 'sync-cancel':
      case 'sync-dismiss':
        hidePanel();
        break;
      case 'resync':
        syncClusterAlbum(clusterId, t);
        break;
      case 'delete-start':
        deleteClusterAlbum(clusterId, t);
        break;
      case 'delete-confirm': {
        const checkbox = document.querySelector(
          '#cluster-album-panel input[data-delete-in-immich]'
        );
        const alsoDelete = checkbox ? checkbox.checked : false;
        performDelete(clusterId, alsoDelete);
        break;
      }
      case 'delete-cancel':
        renderClusterAlbumControls();
        break;
    }
  });
}

// Legend-bar "Sync all clusters" button. Separate panel that lives next
// to the button (mirrors the Plants/Fungi pattern), so the user can
// trigger a bulk sync regardless of group-by mode.
export function wireSyncAllClustersBtn() {
  const btn = document.getElementById('sync-all-clusters-btn');
  if (!btn) return;
  ensureSyncAllPanel();
  btn.addEventListener('click', onSyncAllBtnClick);
}

function ensureSyncAllPanel() {
  if (document.getElementById('sync-all-panel')) return;
  const btn = document.getElementById('sync-all-clusters-btn');
  if (!btn) return;
  const panel = document.createElement('div');
  panel.id = 'sync-all-panel';
  panel.className = 'plants-panel'; // reuse the same styling
  panel.hidden = true;
  btn.insertAdjacentElement('afterend', panel);
}

function paintSyncAllPanel(state, data) {
  const panel = document.getElementById('sync-all-panel');
  if (!panel) return;
  panel.hidden = false;
  switch (state) {
    case 'confirm':
      panel.innerHTML = `
        <div class="plants-msg">
          Sync <strong>${data.nClusters.toLocaleString()}</strong> cluster${
            data.nClusters === 1 ? '' : 's'
          } as Immich albums
          (<strong>${data.nAssets.toLocaleString()}</strong> total photo${
            data.nAssets === 1 ? '' : 's'
          })?
        </div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="sync-all-cancel">Cancel</button>
          <button class="reclassify-btn primary" data-action="sync-all-confirm">Confirm</button>
        </div>`;
      break;
    case 'running': {
      const pct = data.nClusters > 0
        ? Math.min(100, (data.clustersDone / data.nClusters) * 100)
        : 0;
      panel.innerHTML = `
        <div class="plants-msg">
          Syncing <strong>${(data.clustersDone || 0).toLocaleString()}</strong>
          / <strong>${data.nClusters.toLocaleString()}</strong> clusters
          (<strong>${(data.photosAdded || 0).toLocaleString()}</strong>
          / <strong>${data.nAssetsTotal.toLocaleString()}</strong> photos)${
            data.currentClusterId !== null && data.currentClusterId !== undefined
              ? ` — current: cluster ${data.currentClusterId}`
              : ''
          }…
        </div>
        <div class="plants-progress"><div class="bar" style="width:${pct}%"></div></div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="sync-all-dismiss-running"
            title="Hide this panel — the job keeps running in the background and the chips will appear on completion">Hide</button>
        </div>`;
      break;
    }
    case 'done':
      panel.innerHTML = `
        <div class="plants-msg success">
          ✓ Done — <strong>${data.nClusters.toLocaleString()}</strong> cluster${
            data.nClusters === 1 ? '' : 's'
          } synced,
          <strong>${(data.photosAdded || 0).toLocaleString()}</strong> photos added
          to <a href="${escapeHref(BASE_URL_FOR_ALBUMS)}" target="_blank" rel="noopener">Immich</a>.
        </div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="sync-all-dismiss">Dismiss</button>
        </div>`;
      break;
    case 'error':
      panel.innerHTML = `
        <div class="plants-msg error">✕ ${escapeHtml(data.error || 'unknown error')}</div>
        <div class="plants-actions">
          <button class="reclassify-btn primary" data-action="sync-all-retry">Retry</button>
          <button class="reclassify-btn" data-action="sync-all-dismiss">Dismiss</button>
        </div>`;
      break;
  }
}

function hideSyncAllPanel() {
  const panel = document.getElementById('sync-all-panel');
  if (panel) {
    panel.hidden = true;
    panel.innerHTML = '';
  }
}

const BASE_URL_FOR_ALBUMS = ''; // Album IDs are surfaced via /api/album/cluster; no direct Immich link per cluster.

function escapeHref(s) {
  return String(s ?? '').replace(/"/g, '%22');
}

async function onSyncAllBtnClick() {
  if (activeJobId) {
    toast('Cluster album job already running — see the panel.', 'info', 2500);
    return;
  }
  // Fetch cluster count + total photo count for the confirm panel.
  let nClusters = 0;
  let nAssets = 0;
  try {
    const data = await safeFetch('/api/album/cluster');
    const syncedCount = (data.albums || []).length;
    const summary = await safeFetch('/api/clusters');
    const all = summary.clusters || [];
    // For the preview, count every cluster regardless of whether
    // it's already synced. The sync-all worker reuses existing
    // albums, so re-running is idempotent and safe.
    nClusters = all.length;
    if (nClusters === 0) nClusters = syncedCount;
    nAssets = all.reduce((s, c) => s + (Number(c.n) || 0), 0);
  } catch (e) {
    toast(`Could not preview sync-all: ${e.message}`, 'error', 4000);
    return;
  }
  if (nClusters === 0) {
    toast('No clusters to sync.', 'info', 3500);
    return;
  }
  paintSyncAllPanel('confirm', { nClusters, nAssets });
}

// Sync-all delegated handler. Separate from the cluster-album-panel
// handler so the buttons in each panel can't accidentally cross-fire.
document.addEventListener('click', (e) => {
  const t = e.target.closest('#sync-all-panel [data-action]');
  if (!t) return;
  const action = t.getAttribute('data-action');
  switch (action) {
    case 'sync-all-confirm':
      startSyncAllJobFromLegend();
      break;
    case 'sync-all-cancel':
    case 'sync-all-dismiss':
      hideSyncAllPanel();
      break;
    case 'sync-all-dismiss-running':
      document.getElementById('sync-all-panel').hidden = true;
      break;
    case 'sync-all-retry':
      onSyncAllBtnClick();
      break;
  }
});

async function startSyncAllJobFromLegend() {
  let job;
  try {
    job = await safeFetch('/api/album/cluster/sync-all', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
  } catch (e) {
    paintSyncAllPanel('error', { error: e.message });
    return;
  }
  activeJobId = job.job_id;
  paintSyncAllPanel('running', {
    nClusters: job.n_clusters,
    clustersDone: 0,
    photosAdded: 0,
    nAssetsTotal: job.n_assets_total,
    currentClusterId: null,
  });
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(pollSyncAllJob, POLL_MS);
}

// Poll handler for the legend-bar sync-all job. Same backend endpoint
// as the per-cluster sync (/api/album/cluster/job/{job_id}), different
// payload shape (n_clusters / clusters_done / photos_added), different
// panel (#sync-all-panel).
async function pollSyncAllJob() {
  if (!activeJobId) return;
  let job;
  try {
    job = await safeFetch(
      `/api/album/cluster/job/${encodeURIComponent(activeJobId)}`
    );
  } catch (e) {
    if (e && /HTTP 404/.test(e.message)) {
      finalizeSyncAll(null);
      return;
    }
    paintSyncAllPanel('error', { error: e.message });
    finalizeSyncAll(null);
    return;
  }
  if (job.status === 'running') {
    paintSyncAllPanel('running', {
      nClusters: job.n_clusters || 0,
      clustersDone: job.clusters_done || 0,
      photosAdded: job.photos_added || 0,
      nAssetsTotal: job.n_assets_total || 0,
      currentClusterId: job.current_cluster_id,
    });
    if (pollTimer) clearTimeout(pollTimer);
    pollTimer = setTimeout(pollSyncAllJob, POLL_MS);
    return;
  }
  if (job.status === 'done') {
    finalizeSyncAll({
      nClusters: job.n_clusters || 0,
      photosAdded: job.photos_added || 0,
    });
    return;
  }
  if (job.status === 'error') {
    finalizeSyncAll({ error: job.error || 'unknown error' });
    return;
  }
}

function finalizeSyncAll(result) {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  activeJobId = null;
  if (result === null) {
    hideSyncAllPanel();
    toast('Sync-all job expired — check Immich.', 'info', 4000);
    loadClusterAlbums();
    return;
  }
  if (result.error) {
    paintSyncAllPanel('error', { error: result.error });
    toast(`Sync-all failed: ${result.error}`, 'error', 5000);
    return;
  }
  paintSyncAllPanel('done', {
    nClusters: result.nClusters,
    photosAdded: result.photosAdded,
  });
  toast(
    `Done — ${result.nClusters.toLocaleString()} cluster${
      result.nClusters === 1 ? '' : 's'
    } synced, ${result.photosAdded.toLocaleString()} photos added`,
    'success',
    4000
  );
  // Refresh per-cluster chips so newly-created albums show up.
  loadClusterAlbums();
}

// Fetch every cluster->album mapping on init. The toolbar paints from
// this map so a reload preserves the chip state without an extra click.
export async function loadClusterAlbums() {
  try {
    const data = await safeFetch('/api/album/cluster');
    clusterAlbumsMap = new Map(
      (data.albums || []).map((a) => [Number(a.cluster_id), a])
    );
  } catch (e) {
    // Silent — the panel will render "No clusters synced yet" on miss.
    clusterAlbumsMap = new Map();
  }
  renderClusterAlbumControls();
}

// Public hook so other modules (clusters.js) can ask us to re-render
// when the user picks a different cluster.
export function renderClusterAlbumControls() {
  const panel = document.getElementById('cluster-album-panel');
  if (!panel) return;
  const cid = state.currentGroupFilter;
  // `currentGroupFilter` is null in 'all' mode; in taxonomy modes it's
  // a string (species / genus / family) so we only show the panel for
  // cluster mode with a specific cluster selected. Bulk "Sync all"
  // lives in the legend-bar button (always visible), not here.
  if (state.currentGroupBy !== 'cluster' || cid === null) {
    panel.innerHTML = '';
    panel.hidden = true;
    return;
  }
  panel.hidden = false;
  const album = clusterAlbumsMap.get(Number(cid));
  if (album) {
    panel.innerHTML = renderExistingChip(Number(cid), album);
  } else {
    panel.innerHTML = renderSyncPrompt(Number(cid));
  }
}

// ---- DOM scaffolding ----------------------------------------------------

function ensurePanel() {
  if (document.getElementById('cluster-album-panel')) return;
  const bar = document.getElementById('cluster-album-bar');
  if (!bar) return;
  // Insert as a child of the bar so the toolbar's flex layout still
  // works. Hidden by default; renderClusterAlbumControls() unhides
  // it when the user has selected a cluster.
  const panel = document.createElement('div');
  panel.id = 'cluster-album-panel';
  panel.className = 'cluster-album-panel';
  panel.hidden = true;
  bar.appendChild(panel);
}

function renderSyncPrompt(clusterId) {
  return `
    <div class="cluster-album-msg">
      <span style="color:#888">Selected cluster ${clusterId} —</span>
      <button class="reclassify-btn primary" data-action="sync-start" data-cluster-id="${clusterId}">
        [+ Sync this cluster as album]
      </button>
    </div>`;
}

function renderExistingChip(clusterId, album) {
  const shortId = String(album.album_id).slice(0, 8);
  return `
    <div class="cluster-album-chip">
      <span class="cluster-album-chip-label">
        Album for cluster ${clusterId}:
        <a href="${escapeHtml(album.album_url)}" target="_blank" rel="noopener">${shortId}…</a>
        <small style="color:#888">(${album.n_assets} photo${album.n_assets === 1 ? '' : 's'}${album.reused_existing ? ', reused' : ''})</small>
      </span>
      <span class="cluster-album-actions">
        <button class="reclassify-btn" data-action="resync" data-cluster-id="${clusterId}" title="Re-add this cluster's assets to the album (idempotent)">Re-sync</button>
        <button class="reclassify-btn" data-action="delete-start" data-cluster-id="${clusterId}" title="Remove the cluster→album mapping (optionally also delete in Immich)">Delete</button>
      </span>
    </div>`;
}

function renderConfirmSync(clusterId, nAssets, savedAlbumId) {
  return `
    <div class="cluster-album-msg">
      Tag <strong>${nAssets.toLocaleString()}</strong> photo${nAssets === 1 ? '' : 's'}
      from cluster ${clusterId} as an Immich album${savedAlbumId ? ' (will reuse existing)' : ''}?
    </div>
    <div class="cluster-album-actions">
      <button class="reclassify-btn" data-action="sync-cancel" data-cluster-id="${clusterId}">Cancel</button>
      <button class="reclassify-btn primary" data-action="sync-confirm" data-cluster-id="${clusterId}">Confirm</button>
    </div>`;
}

function renderRunning(clusterId, nAdded, nAssets, chunksDone, nChunks) {
  return `
    <div class="cluster-album-msg">
      Adding <strong>${(nAdded || 0).toLocaleString()}</strong>
      / <strong>${nAssets.toLocaleString()}</strong> photos
      (chunk ${chunksDone}/${nChunks})…
    </div>
    <div class="cluster-album-progress"><div class="bar" style="width:0%"></div></div>
    <div class="cluster-album-actions">
      <button class="reclassify-btn" data-action="sync-dismiss" data-cluster-id="${clusterId}" title="Hide this panel — the job keeps running in the background and the chip will appear on the next sync">Hide</button>
    </div>`;
}

function renderError(clusterId, errMsg) {
  return `
    <div class="cluster-album-msg error">✕ ${escapeHtml(errMsg || 'unknown error')}</div>
    <div class="cluster-album-actions">
      <button class="reclassify-btn primary" data-action="sync-start" data-cluster-id="${clusterId}">Retry</button>
      <button class="reclassify-btn" data-action="sync-dismiss" data-cluster-id="${clusterId}">Dismiss</button>
    </div>`;
}

function renderDeleteConfirm(clusterId) {
  return `
    <div class="cluster-album-msg">
      Drop the cluster→album mapping for cluster ${clusterId}?
    </div>
    <label class="cluster-album-checkbox">
      <input type="checkbox" data-delete-in-immich>
      Also delete the album in Immich
    </label>
    <div class="cluster-album-actions">
      <button class="reclassify-btn" data-action="delete-cancel" data-cluster-id="${clusterId}">Cancel</button>
      <button class="reclassify-btn primary" data-action="delete-confirm" data-cluster-id="${clusterId}">Confirm</button>
    </div>`;
}

function hidePanel() {
  const panel = document.getElementById('cluster-album-panel');
  if (panel) {
    panel.innerHTML = '';
    panel.hidden = true;
  }
}

// ---- click handlers ---------------------------------------------------

async function syncClusterAlbum(clusterId, _btn) {
  if (activeJobId) {
    toast('Cluster album job already running — see the panel.', 'info', 2500);
    return;
  }
  // Fetch the preview count so the confirm panel can show "Tag N
  // photos as cluster album". No job is started on this GET — that
  // happens on Confirm.
  let preview;
  try {
    preview = await safeFetch(
      `/api/album/cluster/${encodeURIComponent(clusterId)}/preview`
    );
  } catch (e) {
    toast(`Could not preview cluster album: ${e.message}`, 'error', 4000);
    return;
  }
  if (!preview.n_assets) {
    toast(
      `Cluster ${clusterId} has no assets to sync.`,
      'info',
      3500
    );
    return;
  }
  paintConfirm(clusterId, preview);
}

async function startJob(clusterId) {
  let job;
  try {
    job = await safeFetch(
      `/api/album/cluster/${encodeURIComponent(clusterId)}/sync`,
      {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify({}),
      }
    );
  } catch (e) {
    paintError(clusterId, e.message);
    return;
  }
  activeJobId = job.job_id;
  paintRunning(clusterId, {
    n_added: 0,
    n_assets: job.n_assets,
    chunks_done: 0,
    n_chunks: job.n_chunks,
  });
  schedulePoll();
}

function paintConfirm(clusterId, data) {
  const panel = document.getElementById('cluster-album-panel');
  if (!panel) return;
  panel.hidden = false;
  panel.innerHTML = renderConfirmSync(
    clusterId,
    data.n_assets,
    data.saved_album_id
  );
}

function paintRunning(clusterId, data) {
  const panel = document.getElementById('cluster-album-panel');
  if (!panel) return;
  panel.hidden = false;
  panel.innerHTML = renderRunning(
    clusterId,
    data.n_added,
    data.n_assets,
    data.chunks_done,
    data.n_chunks
  );
  // Animate the bar to the current pct on the next frame so the CSS
  // transition fires from 0 → current.
  requestAnimationFrame(() => {
    const bar = panel.querySelector('.cluster-album-progress .bar');
    if (bar) {
      const pct = data.n_assets > 0
        ? Math.min(100, (data.n_added / data.n_assets) * 100)
        : 0;
      bar.style.width = `${pct}%`;
    }
  });
}

function paintError(clusterId, errMsg) {
  const panel = document.getElementById('cluster-album-panel');
  if (!panel) return;
  panel.hidden = false;
  panel.innerHTML = renderError(clusterId, errMsg);
}

function schedulePoll() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(pollOnce, POLL_MS);
}

async function pollOnce(clusterId) {
  if (!activeJobId) return;
  let job;
  try {
    job = await safeFetch(
      `/api/album/cluster/job/${encodeURIComponent(activeJobId)}`
    );
  } catch (e) {
    if (e && /HTTP 404/.test(e.message)) {
      finalize(null);
      return;
    }
    paintError(clusterId, e.message);
    finalize(null);
    return;
  }
  if (job.status === 'running') {
    const panel = document.getElementById('cluster-album-panel');
    if (panel && !panel.hidden) {
      const msg = panel.querySelector('.cluster-album-msg');
      if (msg) {
        msg.innerHTML =
          `Adding <strong>${(job.n_added || 0).toLocaleString()}</strong>` +
          ` / <strong>${job.n_assets.toLocaleString()}</strong> photos` +
          ` (chunk ${job.chunks_done}/${job.n_chunks})…`;
      }
      const bar = panel.querySelector('.cluster-album-progress .bar');
      if (bar) {
        const pct = job.n_assets > 0
          ? Math.min(100, (job.n_added / job.n_assets) * 100)
          : 0;
        bar.style.width = `${pct}%`;
      }
    }
    schedulePoll();
    return;
  }
  if (job.status === 'done') {
    finalize({ n_added: job.n_added });
    return;
  }
  if (job.status === 'error') {
    finalize({ error: job.error || 'unknown error' });
    return;
  }
}

async function finalize(result) {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  activeJobId = null;
  if (result === null) {
    hidePanel();
    toast('Cluster album job expired — check Immich.', 'info', 4000);
    await loadClusterAlbums();
    renderClusterAlbumControls();
    return;
  }
  if (result.error) {
    paintError(null, result.error);
    toast(`Cluster album failed: ${result.error}`, 'error', 5000);
    return;
  }
  const photosAdded = result.n_added || 0;
  toast(
    `Done — ${photosAdded.toLocaleString()} photos added`,
    'success',
    4000
  );
  // Refresh the mapping so the chip appears with the new album_id.
  await loadClusterAlbums();
  renderClusterAlbumControls();
}

// ---- delete flow -------------------------------------------------------

async function deleteClusterAlbum(clusterId, _btn) {
  const panel = document.getElementById('cluster-album-panel');
  if (!panel) return;
  panel.hidden = false;
  panel.innerHTML = renderDeleteConfirm(clusterId);
}

async function performDelete(clusterId, alsoDeleteInImmich) {
  const params = new URLSearchParams();
  if (alsoDeleteInImmich) params.set('delete_in_immich', 'true');
  const qs = params.toString();
  const url = `/api/album/cluster/${encodeURIComponent(clusterId)}${qs ? '?' + qs : ''}`;
  try {
    await safeFetch(url, { method: 'DELETE' });
    // Update local map and re-render.
    clusterAlbumsMap.delete(Number(clusterId));
    renderClusterAlbumControls();
    toast(
      alsoDeleteInImmich
        ? `Cluster ${clusterId} album deleted (mapping + Immich album).`
        : `Cluster ${clusterId} mapping dropped.`,
      'success',
      3500
    );
  } catch (e) {
    toast(`Delete failed: ${e.message}`, 'error', 5000);
  }
}

// ---- sync-all flow ------------------------------------------------------