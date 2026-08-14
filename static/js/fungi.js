// Fungi-album button. Mirrors the Plants-album UX (./plants.js) but
// pointed at /api/album/kingdom with kingdom="Fungi" so mushrooms get
// their own Immich album, separate from the Plantae one.
//
// Implementation is intentionally a thin duplicate of plants.js rather
// than a generic kingdom helper, because the existing plants.js is
// already wired against /api/album/plants endpoints and a refactor
// would touch every line. Both buttons run independently and can be
// triggered without conflict (each has its own job_id namespace).
import { safeFetch, escapeHtml, toast } from './utils.js';

const KINGDOM = 'Fungi';
const POLL_MS = 400;
let activeJobId = null;
let pollTimer = null;
let currentStatusEl = null;

export function wireFungiBtn() {
  const btn = document.getElementById('fungi-album-btn');
  if (!btn) return;
  ensurePanel();
  currentStatusEl = document.getElementById('fungi-album-status');
  btn.addEventListener('click', onButtonClick);
}

export async function refreshFungiStatus() {
  if (!currentStatusEl) return;
  try {
    const s = await safeFetch(
      `/api/kingdom-album-status?kingdom=${encodeURIComponent(KINGDOM)}`
    );
    if (s && s.exists) paintStatus(s);
    else currentStatusEl.textContent = 'No Fungi album yet';
  } catch (e) {
    currentStatusEl.textContent = '';
  }
}

// ---- DOM scaffolding ----------------------------------------------------

function ensurePanel() {
  if (document.getElementById('fungi-panel')) return;
  const btn = document.getElementById('fungi-album-btn');
  if (!btn) return;
  // Insert a status span next to the button (sibling) and the panel
  // after the button. Mirrors the Plants layout so the two buttons
  // read as a coherent pair in the toolbar.
  if (!document.getElementById('fungi-album-status')) {
    const status = document.createElement('span');
    status.id = 'fungi-album-status';
    status.style.cssText = 'color:#888; margin-right:12px';
    btn.insertAdjacentElement('beforebegin', status);
  }
  const panel = document.createElement('div');
  panel.id = 'fungi-panel';
  panel.className = 'plants-panel'; // reuse the same styling
  panel.hidden = true;
  btn.insertAdjacentElement('afterend', panel);
}

function paintPanel(state, data) {
  const panel = document.getElementById('fungi-panel');
  if (!panel) return;
  panel.hidden = false;
  switch (state) {
    case 'confirm':
      panel.innerHTML = `
        <div class="plants-msg">
          Tag <strong>${data.n_assets.toLocaleString()}</strong> photo${
            data.n_assets === 1 ? '' : 's'
          } as ${KINGDOM} album${
            data.saved_album_id ? ' (will reuse existing)' : ''
          }?
        </div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="cancel">Cancel</button>
          <button class="reclassify-btn primary" data-action="confirm">Confirm</button>
        </div>`;
      break;
    case 'running':
      panel.innerHTML = `
        <div class="plants-msg">
          Adding <strong>${(data.n_added || 0).toLocaleString()}</strong>
          / <strong>${data.n_assets.toLocaleString()}</strong> photos
          (chunk ${data.chunks_done}/${data.n_chunks})…
        </div>
        <div class="plants-progress"><div class="bar" style="width:0%"></div></div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="dismiss-running" title="Hide this panel — the job keeps running in the background">Hide</button>
        </div>`;
      requestAnimationFrame(() => {
        const bar = panel.querySelector('.plants-progress .bar');
        if (bar) {
          const pct = data.n_assets > 0
            ? Math.min(100, (data.n_added / data.n_assets) * 100)
            : 0;
          bar.style.width = `${pct}%`;
        }
      });
      break;
    case 'done':
      panel.innerHTML = `
        <div class="plants-msg success">
          ✓ Done — <strong>${(data.n_added || 0).toLocaleString()}</strong>
          photos added to
          <a href="${escapeHtml(data.album_url)}" target="_blank" rel="noopener">${escapeHtml(data.album_id.slice(0, 8))}…</a>
          ${data.reused_existing ? '<small>(reused existing album)</small>' : ''}
        </div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="rerun">Run again</button>
          <button class="reclassify-btn" data-action="dismiss">Dismiss</button>
        </div>`;
      refreshFungiStatus();
      break;
    case 'error':
      panel.innerHTML = `
        <div class="plants-msg error">✕ ${escapeHtml(data.error || 'unknown error')}</div>
        <div class="plants-actions">
          <button class="reclassify-btn" data-action="rerun">Retry</button>
          <button class="reclassify-btn" data-action="dismiss">Dismiss</button>
        </div>`;
      break;
    default:
      panel.hidden = true;
  }
}

function hidePanel() {
  const panel = document.getElementById('fungi-panel');
  if (panel) {
    panel.hidden = true;
    panel.innerHTML = '';
  }
}

// ---- click + delegation ------------------------------------------------

async function onButtonClick() {
  if (activeJobId) {
    toast(`${KINGDOM} album job already running — see the panel.`, 'info', 2500);
    return;
  }
  let preview;
  try {
    preview = await safeFetch(
      `/api/album/kingdom/preview?kingdom=${encodeURIComponent(KINGDOM)}`
    );
  } catch (e) {
    toast(`Could not preview ${KINGDOM} album: ${e.message}`, 'error', 4000);
    return;
  }
  if (!preview.n_assets) {
    toast(
      `No ${KINGDOM}-classified photos at the current threshold.`,
      'info',
      3500
    );
    return;
  }
  paintPanel('confirm', preview);
}

document.addEventListener('click', (e) => {
  const t = e.target.closest('#fungi-panel [data-action]');
  if (!t) return;
  const action = t.getAttribute('data-action');
  switch (action) {
    case 'confirm':
      startJob();
      break;
    case 'cancel':
    case 'dismiss':
      hidePanel();
      break;
    case 'dismiss-running':
      document.getElementById('fungi-panel').hidden = true;
      break;
    case 'rerun':
      hidePanel();
      onButtonClick();
      break;
  }
});

async function startJob() {
  const btn = document.getElementById('fungi-album-btn');
  if (btn) btn.disabled = true;
  let job;
  try {
    job = await safeFetch('/api/album/kingdom', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({ kingdom: KINGDOM }),
    });
  } catch (e) {
    paintPanel('error', { error: e.message });
    if (btn) btn.disabled = false;
    return;
  }
  activeJobId = job.job_id;
  paintPanel('running', {
    n_added: 0,
    n_assets: job.n_assets,
    chunks_done: 0,
    n_chunks: job.n_chunks,
  });
  schedulePoll();
  if (btn) btn.disabled = false;
}

function schedulePoll() {
  if (pollTimer) clearTimeout(pollTimer);
  pollTimer = setTimeout(pollOnce, POLL_MS);
}

async function pollOnce() {
  if (!activeJobId) return;
  let job;
  try {
    job = await safeFetch(
      `/api/album/kingdom/job/${encodeURIComponent(activeJobId)}`
    );
  } catch (e) {
    if (e && /HTTP 404/.test(e.message)) {
      finalize(null);
      return;
    }
    paintPanel('error', { error: e.message });
    finalize(null);
    return;
  }
  if (job.status === 'running') {
    const panel = document.getElementById('fungi-panel');
    if (panel && !panel.hidden) {
      const msg = panel.querySelector('.plants-msg');
      if (msg) {
        msg.innerHTML =
          `Adding <strong>${(job.n_added || 0).toLocaleString()}</strong>` +
          ` / <strong>${job.n_assets.toLocaleString()}</strong> photos` +
          ` (chunk ${job.chunks_done}/${job.n_chunks})…`;
      }
      const bar = panel.querySelector('.plants-progress .bar');
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
    finalize({
      n_added: job.n_added,
      album_id: job.album_id,
      album_url: job.album_url,
      reused_existing: job.reused_existing,
    });
    return;
  }
  if (job.status === 'error') {
    finalize({ error: job.error || 'unknown error' });
    return;
  }
}

function finalize(result) {
  if (pollTimer) {
    clearTimeout(pollTimer);
    pollTimer = null;
  }
  activeJobId = null;
  if (result === null) {
    hidePanel();
    toast(`${KINGDOM} album job expired — open Immich to see results.`, 'info', 4000);
    return;
  }
  if (result.error) {
    paintPanel('error', result);
    toast(`${KINGDOM} album failed: ${result.error}`, 'error', 5000);
    return;
  }
  paintPanel('done', result);
  toast(
    `Done — ${(result.n_added || 0).toLocaleString()} photos added to ${KINGDOM} album`,
    'success',
    4000
  );
}

// ---- toolbar saved-state indicator -------------------------------------

function paintStatus(s) {
  if (!currentStatusEl) return;
  currentStatusEl.innerHTML =
    `Last ${KINGDOM} album: <a href="${escapeHref(s.album_url)}" target="_blank" rel="noopener">` +
    `${escapeHtml(s.album_id.slice(0, 8))}…</a> (${s.n_assets} photos${s.min_score ? `, ≥ ${s.min_score.toFixed(2)}` : ''})`;
}

// URLs are safer to interpolate when only `"` is escaped (the dangerous
// char inside `href="..."`). escapeHtml() would also rewrite `&` and
// other URL-meaningful characters.
function escapeHref(s) {
  return String(s ?? '').replace(/"/g, '%22');
}