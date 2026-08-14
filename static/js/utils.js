// Tiny DOM helpers used by app.js for the toast.
export function escapeHtml(s) {
  return String(s ?? '').replace(/[&<>"']/g, c => ({
    '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;'
  })[c]);
}

// Lightweight toast. Adds a fixed-position div that auto-removes.
export function toast(message, kind = 'info', duration = 3000) {
  const div = document.createElement('div');
  div.className = `toast ${kind}`;
  div.textContent = message;
  document.body.appendChild(div);
  setTimeout(() => div.remove(), duration);
}
