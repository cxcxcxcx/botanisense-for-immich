// Numbered pagination bar: « first ‹ prev  1 2 ... 50  next › last »
// With ellipsis for runs that span many pages so the bar stays compact.
import { state } from './state.js';
import { escapeHtml } from './utils.js';
import { loadResults } from './results.js';

export function renderPagination(p) {
  const bar = document.getElementById('pagination-bar');
  if (!bar) return;
  if (!p || !p.total) {
    bar.innerHTML = '';
    return;
  }
  const cur = p.page;
  const total = p.total_pages;
  const totalItems = p.total;
  if (total <= 1) {
    bar.innerHTML = `<span class="summary">${totalItems} photo${totalItems === 1 ? '' : 's'}</span>`;
    return;
  }

  // Choose which page numbers to render. Always show first, last, current
  // and one on each side of current. Collapse the rest into ellipsis.
  const want = new Set([0, total - 1, cur, cur - 1, cur + 1]);
  const pages = [];
  for (let i = 0; i < total; i++) {
    if (want.has(i)) pages.push(i);
    else if (i > 0 && i < total - 1 && (want.has(i - 1) || want.has(i + 1))) {
      pages.push(-1); // ellipsis marker
    }
  }
  // de-dup adjacent ellipsis
  const compact = [];
  for (const x of pages) {
    if (x === -1 && compact.length && compact[compact.length - 1] === -1) continue;
    compact.push(x);
  }

  const btn = (label, target, opts = {}) => {
    const cls = opts.current ? ' class="current"' : '';
    const disabled = opts.disabled ? ' disabled' : '';
    return `<button${cls}${disabled} data-page="${target}">${escapeHtml(label)}</button>`;
  };

  const parts = [];
  parts.push(`<span class="summary">${totalItems} photos, page ${cur + 1}/${total}</span>`);
  parts.push(btn('« first', 0, {disabled: cur === 0}));
  parts.push(btn('‹ prev', cur - 1, {disabled: cur === 0}));
  for (const x of compact) {
    if (x === -1) {
      parts.push(`<span class="ellipsis">…</span>`);
    } else {
      parts.push(btn(String(x + 1), x, {current: x === cur}));
    }
  }
  parts.push(btn('next ›', cur + 1, {disabled: cur >= total - 1}));
  parts.push(btn('last »', total - 1, {disabled: cur >= total - 1}));

  bar.innerHTML = parts.join('');
  bar.querySelectorAll('button[data-page]').forEach(el => {
    el.addEventListener('click', () => {
      const target = Number(el.getAttribute('data-page'));
      if (target === state.currentPage) return;
      loadResults({page: target, refreshPills: false});
    });
  });
}
