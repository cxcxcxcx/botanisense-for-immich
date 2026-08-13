// Photo detail modal — opens on card click, fetches /api/asset/{aid},
// renders the image + top-K + KNN + YOLO detections + bbox-BioCLIP.
//
// Reuses the project's existing coloring helpers (plantaeClass) and CSS
// for badges. The empty-state KNN reclassify button uses the delegated
// `[data-action="reclassify"]` listener that handlers.js wires up —
// so a click here flows through the same handler as the card-grid
// reclassify button and updates lastResults in-place.
import { state } from './state.js';
import { safeFetch, escapeHtml } from './utils.js';
import { plantaeClass } from './plantae.js';
import { renderCurrent } from './results.js';

let backdropWired = false;

function ensureBackdropHandler() {
  if (backdropWired) return;
  const dlg = document.getElementById('detail-dialog');
  if (!dlg) return;
  // Native <dialog> doesn't close on backdrop click by itself.
  // Close only when the click target is the dialog itself, not the
  // inner content.
  dlg.addEventListener('click', (e) => {
    if (e.target === dlg) closeDetailModal();
  });
  backdropWired = true;
}

export function closeDetailModal() {
  const dlg = document.getElementById('detail-dialog');
  if (!dlg) return;
  if (typeof dlg.close === 'function') dlg.close();
}

export async function openDetailModal(aid) {
  const dlg = document.getElementById('detail-dialog');
  const title = document.getElementById('modal-title');
  const body = document.getElementById('modal-body');
  if (!dlg || !title || !body) return;
  ensureBackdropHandler();
  title.textContent = aid;
  body.innerHTML = '<div class="modal-loading">Loading…</div>';
  if (typeof dlg.showModal === 'function') {
    dlg.showModal();
  } else {
    // Fallback for older browsers — just display.
    dlg.setAttribute('open', '');
  }

  let data;
  try {
    const r = await fetch(`/api/asset/${encodeURIComponent(aid)}`);
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    data = await r.json();
  } catch (e) {
    body.innerHTML = `<div class="modal-loading error">Failed to load: ${escapeHtml(e.message)}</div>`;
    return;
  }

  title.textContent = data.original_file_name || aid;
  body.innerHTML = renderDetailBody(data);
}

function renderDetailBody(d) {
  const green = 0.60;
  const yellow = 0.35;
  const top = d.top_predictions || [];
  const ts = d.taxon_scores || {};
  const detections = d.detections || [];
  const knn = d.knn;
  const neighbors = d.library_neighbors || [];

  // Badges: kingdom/family/genus + Plantae confidence.
  const top1 = top[0] || {};
  const plantaeScore = ts.Plantae;
  const plantaeBadge = plantaeScore == null ? '' :
    `<span class="plantae ${plantaeClass(plantaeScore, green, yellow)}">Plantae ${(plantaeScore * 100).toFixed(1)}%</span>`;
  const badges = `
    <div class="modal-badges">
      <span class="badge">${escapeHtml(top1.kingdom || '?')}</span>
      <span class="badge">${escapeHtml(top1.family || '?')}</span>
      <span class="badge">${escapeHtml(top1.genus || '?')}</span>
      ${plantaeBadge}
    </div>
  `;

  // Closest photos in the user's library — top-K by BioCLIP cosine
  // similarity, computed server-side by storage.find_library_neighbors.
  // Each thumbnail is wrapped in a `.modal-neighbor` so handlers.js can
  // wire delegated clicks to open that asset's modal.
  const neighborsHtml = neighbors.length ? `
    <div class="modal-section">
      <h3>Closest photos in your library (BioCLIP)</h3>
      <div class="modal-neighbor-grid">
        ${neighbors.map(n => {
          const simPct = (n.similarity * 100).toFixed(1);
          return `
            <div class="modal-neighbor" data-neighbor-aid="${escapeHtml(n.asset_id)}" title="open this asset">
              <img src="/thumb/${encodeURIComponent(n.asset_id)}?size=thumbnail" alt="${escapeHtml(n.original_file_name || '')}" loading="lazy">
              ${n.cluster_id !== null && n.cluster_id !== undefined
                ? `<div class="modal-neighbor-cluster">cluster ${escapeHtml(String(n.cluster_id))}</div>`
                : ''}
              <div class="modal-neighbor-name">${escapeHtml(n.original_file_name || n.asset_id.slice(0, 8))}</div>
              <div class="modal-neighbor-sim">${simPct}%</div>
            </div>
          `;
        }).join('')}
      </div>
    </div>
  ` : '';

  // Top-K full-image predictions.
  const topPredsHtml = top.length ? `
    <div class="modal-section">
      <h3>Top-K BioCLIP (image)</h3>
      ${top.slice(0, 5).map(p => `
        <div class="pred">
          <div class="sp">${escapeHtml(p.species)}<div class="cn">${escapeHtml(p.common_name || '')}</div></div>
          <div class="sc">${(p.score * 100).toFixed(2)}%</div>
        </div>
      `).join('')}
    </div>
  ` : '';

  // KNN neighbors.
  let knnHtml;
  if (knn && knn.top_species) {
    const topSpecies = escapeHtml(knn.top_species || '');
    const topCommon = escapeHtml(knn.common_name || '');
    const topScore = ((knn.score || 0) * 100).toFixed(1);
    const nbrs = (knn.neighbors || []).slice(0, 8).map(n => `
      <div class="pred">
        <div class="sp">${escapeHtml(n.species)}<div class="cn">${escapeHtml(n.common_name || '')}</div></div>
        <div class="sc">${(n.sim * 100).toFixed(2)}%</div>
      </div>
    `).join('');
    knnHtml = `
      <div class="modal-section">
        <h3>KNN (TreeOfLife-200M)</h3>
        <div style="margin-bottom:8px">
          Top-1 consensus: <strong>${topSpecies}</strong>${topCommon ? ` <span style="color:#888">(${topCommon})</span>` : ''} · ${topScore}%
          <div style="color:#888; font-size:11px">${escapeHtml(knn.kingdom || '')}${knn.family ? ' · ' + escapeHtml(knn.family) : ''}${knn.genus ? ' · ' + escapeHtml(knn.genus) : ''}</div>
        </div>
        <div style="color:#9cf; font:11px ui-monospace,monospace; margin-bottom:4px">Neighbors</div>
        ${nbrs}
      </div>
    `;
  } else {
    // Empty state: reclassify button. The delegated handler in handlers.js
    // will patch lastResults + rerender the grid; we also patch local data
    // so this modal's KNN section fills in.
    knnHtml = `
      <div class="modal-section">
        <h3>KNN (TreeOfLife-200M)</h3>
        <div class="knn-empty">No KNN result yet.</div>
        <button class="reclassify-btn" data-aid="${escapeHtml(d.asset_id)}" data-action="reclassify" data-reclassify-target="modal" style="margin-top:4px">Reclassify with KNN</button>
      </div>
    `;
  }

  // YOLO detections + bbox-BioCLIP predictions.
  let detsHtml;
  if (detections.length) {
    const rows = detections.map((det, di) => {
      const bbox = `x=${det.x.toFixed(2)} y=${det.y.toFixed(2)} w=${det.w.toFixed(2)} h=${det.h.toFixed(2)}`;
      const bp = (det.bbox_predictions || []).slice(0, 3);
      const bpHtml = bp.map(p => `
        <div class="pred">
          <div class="sp">${escapeHtml(p.species)}<div class="cn">${escapeHtml(p.common_name || '')}</div></div>
          <div class="sc">${(p.score * 100).toFixed(2)}%</div>
        </div>
      `).join('');
      return `
        <div class="modal-det-row">
          <div class="det-label">crop ${di + 1}: ${escapeHtml(det.label)} ${(det.score * 100).toFixed(0)}%</div>
          <div class="det-coords">${bbox}</div>
          ${bpHtml}
        </div>
      `;
    }).join('');
    detsHtml = `
      <div class="modal-section">
        <h3>YOLO detections (BioCLIP on each crop)</h3>
        ${rows}
      </div>
    `;
  } else {
    detsHtml = `
      <div class="modal-section">
        <h3>YOLO detections</h3>
        <div class="knn-empty">No YOLO detections (score ≥ 0.10).</div>
      </div>
    `;
  }

  // Asset metadata footer.
  const meta = `
    <div class="modal-section">
      <h3>Asset metadata</h3>
      <div class="modal-meta">
        filename: ${escapeHtml(d.original_file_name || '?')}<br>
        asset_id: ${escapeHtml(d.asset_id)}<br>
        ${d.cluster_id !== null && d.cluster_id !== undefined ? `cluster: ${escapeHtml(String(d.cluster_id))}<br>` : ''}
        <a href="${escapeHtml(d.original_url)}" target="_blank" rel="noopener">original →</a>
        &nbsp; <a href="${escapeHtml(d.thumbnail_url)}" target="_blank" rel="noopener">thumb →</a>
      </div>
    </div>
  `;

  return `
    <img class="modal-image" src="/thumb/${encodeURIComponent(d.asset_id)}?size=fullsize" alt="${escapeHtml(d.original_file_name || '')}">
    ${badges}
    ${neighborsHtml}
    ${topPredsHtml}
    ${knnHtml}
    ${detsHtml}
    ${meta}
  `;
}

// Called by reclassify.js after a /api/reclassify round-trip completes —
// patches `lastResults` (so the card-grid knn block refreshes too) and
// re-renders this modal's body. No-op if the modal isn't open.
export async function refreshModalKnn(aid) {
  const dlg = document.getElementById('detail-dialog');
  if (!dlg || !dlg.open) return;
  const data = await safeFetch(`/api/asset/${encodeURIComponent(aid)}`);
  const body = document.getElementById('modal-body');
  const title = document.getElementById('modal-title');
  if (title && data.original_file_name) title.textContent = data.original_file_name;
  if (body) body.innerHTML = renderDetailBody(data);
}
