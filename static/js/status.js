// Status panel + per-stage timing bar.
import { state } from './state.js';

export function fmtDuration(s) {
  if (!s || s <= 0) return '';
  if (s < 60) return `${s.toFixed(0)}s`;
  const m = Math.floor(s / 60);
  const r = Math.floor(s % 60);
  if (m < 60) return `${m}m${r.toString().padStart(2, '0')}s`;
  const h = Math.floor(m / 60);
  return `${h}h${(m % 60).toString().padStart(2, '0')}m`;
}

export function renderStatus(s) {
  if (!s) return;
  const pct = s.chunk_total > 0
    ? Math.min(100, (s.chunk_idx / s.chunk_total) * 100)
    : (s.status === 'running' ? 5 : 100);
  state.statusBar.style.width = `${pct}%`;
  state.statusPanel.className = 'status-panel';  // reset
  if (s.status === 'running') {
    state.statusPanel.classList.add('running');
    const stage = s.stage || '?';
    const ci = s.chunk_idx || 0;
    const ct = s.chunk_total || 0;
    const eta = s.eta_seconds ? `  eta=${fmtDuration(s.eta_seconds)}` : '';
    const rate = s.rate ? `  ${s.rate.toFixed(1)}/s` : '';
    const q = s.query ? `  query=${s.query}` : '';
    state.statusText.textContent = `[${stage}] chunk ${ci}/${ct}${rate}${eta}${q}`;
    state.refreshBtn.disabled = true;
    state.runBtn.disabled = true;
    renderStageTimes(s);  // live: include the stage currently being timed
  } else if (s.status === 'done') {
    state.statusPanel.classList.add('done');
    state.statusText.textContent = `done  run #${s.current_run_id ?? '?'}  ` +
      `(${s.n_new} new, ${s.n_cached} cached)`;
    state.refreshBtn.disabled = false;
    state.runBtn.disabled = false;
    renderStageTimes(s);  // final: show all accumulated stages
  } else if (s.status === 'error') {
    state.statusPanel.classList.add('error');
    state.statusText.textContent = `error: ${s.last_error || 'unknown'}`;
    state.refreshBtn.disabled = false;
    state.runBtn.disabled = false;
    renderStageTimes(s);
  } else {
    state.statusPanel.classList.add('idle');
    state.statusText.textContent = 'idle';
    state.refreshBtn.disabled = false;
    state.runBtn.disabled = false;
    state.stageTimeBar.style.display = 'none';
    state.stageTimeLegend.innerHTML = '';
  }
}

// Per-stage timing bar — each stage = colored slice, width = pct of total.
const STAGE_COLORS = {
  search: '#4a9eff',       // blue
  species: '#50c878',      // green
  cluster: '#f5a623',      // orange
  kingdom: '#9b59b6',      // purple
  yolo: '#e74c3c',         // red
  'bbox-bioclip': '#1abc9c', // teal
};

export function renderStageTimes(s) {
  const times = s.stage_times || {};
  if (!Object.keys(times).length) {
    state.stageTimeBar.style.display = 'none';
    state.stageTimeLegend.innerHTML = '';
    return;
  }
  // While running, project the current stage's elapsed-so-far into the
  // bar so the slice is visibly growing in real time. We compute it from
  // (now - stage_t0) but only if the server's stage_t0 matches stage_times.
  const live = Object.assign({}, times);
  if (s.status === 'running' && s.stage && s.stage_t0) {
    const elapsed = (Date.now() / 1000) - s.stage_t0;
    if (elapsed > 0) live[s.stage] = (live[s.stage] || 0) + elapsed;
  }
  const total = Object.values(live).reduce((a, b) => a + b, 0) || 1;
  state.stageTimeBar.innerHTML = '';
  state.stageTimeLegend.innerHTML = '';
  let first = true;
  for (const [name, sec] of Object.entries(live)) {
    const pct = (sec / total) * 100;
    const color = STAGE_COLORS[name] || '#777';
    const slice = document.createElement('div');
    slice.className = 'slice';
    slice.style.background = color;
    slice.style.width = `${pct}%`;
    slice.title = `${name}: ${fmtDuration(sec)}`;
    state.stageTimeBar.appendChild(slice);
    const item = document.createElement('span');
    item.innerHTML = `<span class="swatch" style="background:${color}"></span>` +
      `<span class="stage-name">${name}</span>` +
      `<span class="stage-sec">${fmtDuration(sec)}</span>`;
    if (!first) item.style.marginLeft = '0';
    first = false;
    state.stageTimeLegend.appendChild(item);
  }
  state.stageTimeBar.style.display = 'flex';
}
