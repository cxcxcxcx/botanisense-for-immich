// Delegated click handler for the photo grid. Bound once on `document`,
// so cards rendered after the first bind still work without rebinding.
//
// Handlers:
//   - card image / meta click → openDetailModal (also: any click on .card
//     outside an interactive child)
//   - .near-dup-toggle: show/hide the near-dup panel
//   - .near-dup-pick:   swap the card's primary thumbnail (cosmetic)
//   - [data-action="close-modal"]: close the photo detail modal
//   - [data-action="reclassify"]: KNN reclassify a single asset
import { reclassifyOne } from './reclassify.js';
import { openDetailModal, closeDetailModal } from './modal.js';

export function wireDelegatedHandlers() {
  document.addEventListener('click', (e) => {
    // Modal close triggers first (most specific).
    const closeBtn = e.target.closest('[data-action="close-modal"]');
    if (closeBtn) { closeDetailModal(); return; }

    // Click on a thumbnail inside the modal's "Closest photos" grid →
    // open THAT asset's modal (re-uses the same <dialog>; showModal() on
    // an already-open dialog is a no-op, so we just swap content).
    const nb = e.target.closest('.modal-neighbor[data-neighbor-aid]');
    if (nb) {
      const nbAid = nb.getAttribute('data-neighbor-aid');
      if (nbAid) { openDetailModal(nbAid); return; }
    }

    // Card click → open detail modal. Must come AFTER the close-button
    // check above and BEFORE the near-dup handlers (near-dup-pick lives
    // inside a .card). The check excludes clicks on interactive children
    // (select, button, .near-dup-*) so they fall through to their own
    // delegated handlers below.
    const interactive = e.target.closest('button, select, a, [data-action], .near-dup-toggle, .near-dup-pick');
    if (!interactive) {
      const card = e.target.closest('.card');
      if (card) {
        const img = card.querySelector('img[data-primary-asset]');
        const aid = img && img.getAttribute('data-primary-asset');
        if (aid) openDetailModal(aid);
        return;
      }
    }

    const t = e.target.closest('.near-dup-toggle');
    if (t) {
      const aid = t.getAttribute('data-aid');
      const panel = document.getElementById('ndp-' + aid);
      if (!panel) return;
      const showing = panel.style.display !== 'none';
      panel.style.display = showing ? 'none' : 'flex';
      t.textContent = showing
        ? `▸ ${t.textContent.replace(/^▸ |^▾ /, '')}`
        : `▾ ${t.textContent.replace(/^▸ |^▾ /, '')}`;
      return;
    }
    const pick = e.target.closest('.near-dup-pick');
    if (pick) {
      const item = pick.closest('.near-dup-item');
      const primaryAid = item.getAttribute('data-aid');
      const otherAid = item.getAttribute('data-other');
      const card = pick.closest('.card');
      const img = card.querySelector('.img-wrap img');
      const altName = item.querySelector('.near-dup-name').textContent;
      if (img) img.src = '/thumb/' + encodeURIComponent(otherAid);
      const nameEl = card.querySelector('.meta-card .name');
      if (nameEl) nameEl.textContent = altName + '  (was ' + primaryAid + ')';
      // Mark active in the panel.
      card.querySelectorAll('.near-dup-pick').forEach(b => b.classList.remove('active'));
      pick.classList.add('active');
      return;
    }
    // KNN reclassify handler — delegated so cards rendered after the
    // first bind still work. Disables the button + shows a spinner
    // label while the request is in flight.
    const rc = e.target.closest('[data-action="reclassify"]');
    if (rc) {
      const aid = rc.getAttribute('data-aid');
      if (!aid || rc.disabled) return;
      rc.disabled = true;
      const prev = rc.textContent;
      rc.textContent = 'Running…';
      reclassifyOne(aid, rc, prev);
      return;
    }
  });
}
