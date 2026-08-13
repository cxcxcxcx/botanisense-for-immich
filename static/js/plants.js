// Plants-album button: one-click "create (or reuse) an Immich album
// containing every asset classified as Plantae". Idempotent — the server
// persists the album id in `album_meta` so re-clicks reuse it.
//
// On init the toolbar paints the saved state via `/api/plants-album-status`
// so the user can see if they already have one without clicking.
import { safeFetch, toast } from './utils.js';

const SAVED_LABEL = 'Tag as Plants album';

export function wirePlantsBtn() {
  const btn = document.getElementById('plants-album-btn');
  if (!btn) return;
  btn.addEventListener('click', async () => {
    btn.disabled = true;
    btn.textContent = 'Creating album…';
    let minScore;
    try {
      const status = await safeFetch('/api/plants-album-status');
      minScore = (status && status.exists) ? status.min_score : undefined;
    } catch (_) { /* ignore — server uses default */ }

    const conf = confirm(
      minScore != null
        ? `Tag all assets classified as Plantae (BioCLIP kingdom score ≥ ${minScore.toFixed(2)}) as an Immich album?`
        : 'Tag all assets classified as Plantae (BioCLIP kingdom score ≥ the green-leg threshold) as an Immich album?'
    );
    if (!conf) {
      btn.disabled = false;
      btn.textContent = SAVED_LABEL;
      return;
    }

    try {
      const body = {};
      if (minScore != null) body.min_score = minScore;
      const data = await safeFetch('/api/album/plants', {
        method: 'POST',
        headers: {'Content-Type': 'application/json'},
        body: JSON.stringify(body),
      });
      toast(
        `Done — ${data.n_added_this_call} added to ${data.album_id.slice(0, 8)}… (reused: ${data.reused_existing})`,
        'success',
        5000
      );
      paintStatus(data);
      btn.textContent = `Done — ${data.n_added_this_call} added`;
      setTimeout(() => { btn.textContent = SAVED_LABEL; }, 3000);
    } catch (e) {
      toast(`Plants album failed: ${e.message}`, 'error', 5000);
      btn.textContent = SAVED_LABEL;
    } finally {
      btn.disabled = false;
    }
  });
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
  // The album_url is server-generated and trustworthy, but escape defensively.
  return String(s ?? '').replace(/"/g, '%22');
}
