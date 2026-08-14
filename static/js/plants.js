// Plants-album button: one-click "create (or reuse) an Immich album
// containing every asset classified as Plantae". The actual work runs
// server-side in a background job (POST /api/album/plants kicks it off,
// GET /api/album/plants/job/{id} returns live progress). This module
// drives the inline two-step UI: a confirm panel first, then a live
// progress panel that polls the job.
//
// On init the toolbar paints the saved state via
// /api/plants-album-status so the user can see if they already have one
// without clicking.
import { safeFetch, escapeHtml, toast } from './utils.js';

const SAVED_LABEL = 'Tag as Plants album';
const POLL_MS = 400;

// In-flight job id. Non-null while a job is being polled. Used to
// ignore stale poll responses and to dedupe accidental double-clicks.
let activeJobId = null;
let pollTimer = null;

export function wirePlantsBtn() {
  const btn = document.getElementById('plants-album-btn');
  if (!btn) return;
  // Render the panel once so the toolbar layout is stable; the panel
  // itself is hidden by default and shown by transitions between states.
  ensurePanel();
  btn.addEventListener('click', onButtonClick);
}

export async function refreshPlantsStatus() {
  const slot = document.getElementById('plants-album-status');
  if (!slot) return;
  try {
    const s = await safeFetch('/api/plants-album-status');
    if (s && s.exists) paintStatus(s);
    else slot.textContent = 'No Plants album yet';
  } catch (e) {
    slot.textContent = '';
  }
}

// ---- DOM scaffolding ----------------------------------------------------

function ensurePanel() {
  if (document.getElementById('plants-panel')) return;
  const btn = document.getElementById('plants-album-btn');
  if (!btn) return;
  // Insert the panel as a sibling AFTER the button so the toolbar
  // doesn't reflow awkwardly. It's hidden by default (display:none
  // until a state paints it).
  const panel = document.createElement('div');
  panel.id = 'plants-panel';
  panel.className = 'plants-panel';
  panel.hidden = true;
  btn.insertAdjacentElement('afterend', panel);
}

function paintPanel(state, data) {
  const panel = document.getElementById('plants-panel');
  if (!panel) return;
  panel.hidden = false;
  switch (state) {
    case 'confirm':
      panel.innerHTML = `
        <div class="plants-msg">
          Tag <strong>${data.n_assets.toLocaleString()}</strong> photo${
            data.n_assets === 1 ? '' : 's'
          } as Plantae album${
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
          <button class="reclassify-btn" data-action="dismiss-running" title="Hide this panel — the job keeps running in the background and the result will appear on the saved-album status when it's done">Hide</button>
        </div>`;
      // Animate the bar to the current pct on the next frame so the
      // CSS transition fires from 0 → current.
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
      // Refresh the inline "Last Plants album" indicator in the
      // toolbar so it reflects this run's stats.
      refreshPlantsStatus();
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
  const panel = document.getElementById('plants-panel');
  if (panel) {
    panel.hidden = true;
    panel.innerHTML = '';
  }
}

// ---- click + delegation ------------------------------------------------

async function onButtonClick() {
  if (activeJobId) {
    // A job is already running. Re-clicking the button while the
    // panel is hidden is a no-op — the user should use the [Hide]
    // button or wait for completion. Surface a hint instead of doing
    // anything destructive.
    toast('Plants album job already running — see the panel.', 'info', 2500);
    return;
  }
  // Fetch the count up front so the user can see "tag N photos" before
  // committing. If there are zero, surface that and don't start a job.
  let preview;
  try {
    preview = await safeFetch(
      `/api/album/plants/preview?min_score=${encodeURIComponent(PLANTAE_GREEN)}`
    );
  } catch (e) {
    toast(`Could not preview Plants album: ${e.message}`, 'error', 4000);
    return;
  }
  if (!preview.n_assets) {
    toast(
      'No Plantae-classified photos at the current threshold.',
      'info',
      3500
    );
    return;
  }
  paintPanel('confirm', preview);
}

// We attach a single delegated handler on the panel because the inner
// buttons are re-rendered on every state change.
document.addEventListener('click', (e) => {
  const t = e.target.closest('#plants-panel [data-action]');
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
      // Hide the panel but keep polling so the final state shows up
      // when the user re-clicks the button.
      document.getElementById('plants-panel').hidden = true;
      break;
    case 'rerun':
      // Reset and let the user re-confirm with the current count.
      hidePanel();
      onButtonClick();
      break;
  }
});

// Hardcoded for the URL; the server uses the same default if missing.
const PLANTAE_GREEN = 0.60;

async function startJob() {
  const btn = document.getElementById('plants-album-btn');
  if (btn) btn.disabled = true;
  let job;
  try {
    job = await safeFetch('/api/album/plants', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({}),
    });
  } catch (e) {
    paintPanel('error', { error: e.message });
    if (btn) btn.disabled = false;
    return;
  }
  activeJobId = job.job_id;
  // First paint of the running state with the freshly-returned counts.
  paintPanel('running', {
    n_added: 0,
    n_assets: job.n_assets,
    chunks_done: 0,
    n_chunks: job.n_chunks,
  });
  schedulePoll();
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
      `/api/album/plants/job/${encodeURIComponent(activeJobId)}`
    );
  } catch (e) {
    // 404 = job aged out; assume done. Other errors: show & stop.
    if (e && /HTTP 404/.test(e.message)) {
      finalize(null);
      return;
    }
    paintPanel('error', { error: e.message });
    finalize(null);
    return;
  }
  if (job.status === 'running') {
    // Repaint the running state with new numbers. We DON'T re-paint
    // from scratch (that would reset the bar's CSS transition) — just
    // update the text and bar width.
    const panel = document.getElementById('plants-panel');
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
  const btn = document.getElementById('plants-album-btn');
  if (btn) btn.disabled = false;
  if (result === null) {
    // Server lost the job (aged out). Treat as best-effort done.
    hidePanel();
    toast('Plants album job expired — open Immich to see results.', 'info', 4000);
    return;
  }
  if (result.error) {
    paintPanel('error', result);
    toast(`Plants album failed: ${result.error}`, 'error', 5000);
    return;
  }
  paintPanel('done', result);
  toast(
    `Done — ${(result.n_added || 0).toLocaleString()} photos added`,
    'success',
    4000
  );
}

// ---- toolbar saved-state indicator -------------------------------------

function paintStatus(s) {
  const slot = document.getElementById('plants-album-status');
  if (!slot) return;
  if (!s.exists) {
    slot.textContent = 'No Plants album yet';
    return;
  }
  slot.innerHTML =
    `Last Plants album: <a href="${escapeHref(s.album_url)}" target="_blank" rel="noopener">` +
    `${s.album_id.slice(0, 8)}…</a> (${s.n_assets} photos${s.min_score ? `, ≥ ${s.min_score.toFixed(2)}` : ''})`;
}

function escapeHref(s) {
  return String(s ?? '').replace(/"/g, '%22');
}
