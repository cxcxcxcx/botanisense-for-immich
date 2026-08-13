// Params form → POST /api/run / GET /api/refresh.
import { state } from './state.js';
import { safeFetch, showError } from './utils.js';
import { loadResults } from './results.js';

function collectParams() {
  const params = {};
  const q = state.paramQuery.value.trim();
  const n = parseInt(state.paramN.value, 10);
  const k = parseInt(state.paramK.value, 10);
  const b = parseInt(state.paramBatch.value, 10);
  if (q) params.query = q;
  if (Number.isFinite(n) && n > 0) params.n_samples = n;
  if (Number.isFinite(k) && k > 0) params.top_k = k;
  if (Number.isFinite(b) && b > 0) params.batch_size = b;
  return params;
}

async function runWithParams() {
  const params = collectParams();
  state.runBtn.disabled = true;
  state.refreshBtn.disabled = true;
  state.runBtn.textContent = 'Running…';
  state.root.innerHTML = '<div class="empty">Running pipeline…</div>';
  try {
    const data = await safeFetch('/api/run', {
      method: 'POST',
      headers: {'Content-Type': 'application/json'},
      body: JSON.stringify(params),
    });
    await loadResults();
  } catch (e) {
    showError('Run failed.', e.message);
  } finally {
    state.runBtn.disabled = false;
    state.refreshBtn.disabled = false;
    state.runBtn.textContent = 'Run';
  }
}

async function refresh() {
  state.refreshBtn.disabled = true;
  state.refreshBtn.textContent = 'Refreshing…';
  state.root.innerHTML = '<div class="empty">Running pipeline…</div>';
  try {
    const data = await safeFetch('/api/refresh');
    await loadResults();
  } catch (e) {
    showError('Refresh failed.', e.message);
  } finally {
    state.refreshBtn.disabled = false;
    state.refreshBtn.textContent = 'Refresh';
  }
}

export function wireParams() {
  state.refreshBtn.addEventListener('click', refresh);
  state.runBtn.addEventListener('click', runWithParams);
}
