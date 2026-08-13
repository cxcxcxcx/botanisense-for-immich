// Background pollers for status + logs, plus the log-toggle button wiring.
import { state } from './state.js';
import { safeFetch } from './utils.js';
import { renderStatus } from './status.js';

export async function pollStatus() {
  try {
    const s = await safeFetch('/api/status');
    renderStatus(s);
  } catch (e) {
    // Transient — keep the last status visible.
    console.warn('status poll failed:', e);
  }
}

export async function pollLogs() {
  if (state.logPre.style.display === 'none') return;  // collapsed
  try {
    const data = await safeFetch('/api/logs?tail=200');
    const lines = data.lines || [];
    state.logPre.textContent = lines.join('\n');
    // Auto-scroll to bottom if user hasn't scrolled up.
    state.logPre.scrollTop = state.logPre.scrollHeight;
  } catch (e) {
    console.warn('logs poll failed:', e);
  }
}

export function wireLogToggle() {
  state.logToggle.addEventListener('click', () => {
    const showing = state.logPre.style.display !== 'none';
    state.logPre.style.display = showing ? 'none' : 'block';
    state.logToggle.textContent = showing ? '▸ logs (click to expand)' : '▾ logs';
    if (!showing) pollLogs();
  });
}
