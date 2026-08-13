// KNN reclassify — single asset (from the per-card button) or whole page.
import { state } from './state.js';
import { safeFetch, showError } from './utils.js';
import { loadResults, renderCurrent } from './results.js';
import { refreshModalKnn } from './modal.js';

export async function reclassifyOne(aid, btn, prevLabel) {
  try {
    const data = await safeFetch('/api/reclassify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({asset_ids: [aid]}),
    });
    const knn = (data.results || {})[aid];
    if (knn) {
      // Patch the cached payload + re-render so the KNN block fills in.
      if (state.lastResults && Array.isArray(state.lastResults.results)) {
        const row = state.lastResults.results.find(r => r.asset_id === aid);
        if (row) row.knn = knn;
      }
      renderCurrent();
      // Also patch the modal — openDetailModal stores the asset_id in the
      // title, so if the modal is currently showing THIS asset we refetch
      // and replace the body. This is no-op if the modal is closed or
      // showing a different asset.
      refreshModalKnn(aid).catch(() => { /* modal may be closed; ignore */ });
    }
  } catch (e) {
    showError('KNN reclassify failed.', e.message);
  } finally {
    btn.disabled = false;
    btn.textContent = prevLabel || 'Reclassify with KNN';
  }
}

export async function reclassifyVisible() {
  if (!state.lastResults || !Array.isArray(state.lastResults.results) || !state.lastResults.results.length) {
    showError('Nothing to reclassify.', 'Load a page of results first.');
    return;
  }
  const btn = document.getElementById('reclassify-visible-btn');
  if (btn) { btn.disabled = true; btn.textContent = 'Running KNN…'; }
  const ids = state.lastResults.results.map(r => r.asset_id);
  try {
    const data = await safeFetch('/api/reclassify', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify({asset_ids: ids}),
    });
    // Refetch /api/results so KNN blocks populate cleanly. (Cheap —
    // cache hit after the invalidate in /api/reclassify.)
    await loadResults({refreshPills: false});
  } catch (e) {
    showError('KNN reclassify failed.', e.message);
  } finally {
    if (btn) { btn.disabled = false; btn.textContent = 'Reclassify visible with KNN'; }
  }
}

export function wireReclassifyVisible() {
  const btn = document.getElementById('reclassify-visible-btn');
  if (btn) btn.addEventListener('click', reclassifyVisible);
}
