// Tiny DOM helpers and fetch wrapper used by every module.
import { state } from './state.js';

export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

export async function safeFetch(url, opts) {
  const r = await fetch(url, opts);
  if (!r.ok) throw new Error(`${url} → HTTP ${r.status}`);
  return r.json();
}

export function showError(msg, detail) {
  console.error(msg, detail);
  state.root.innerHTML = `<div class="empty error">${escapeHtml(msg)}${
    detail ? `<br><small>${escapeHtml(String(detail))}</small>` : ''
  }</div>`;
}

// Lightweight toast used by action results (Plants album, etc.). Adds a
// fixed-position div that auto-removes after `duration` ms.
export function toast(message, kind = 'info', duration = 3000) {
  const div = document.createElement('div');
  div.className = `toast ${kind}`;
  div.textContent = message;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), duration);
}
