// Main grid render: builds the photo cards from the /api/results payload.
import { state } from './state.js';
import { escapeHtml } from './utils.js';
import { plantaeClass, renderPlantaeLegend } from './plantae.js';

export function render(data) {
  if (data.error) {
    state.root.innerHTML = `<div class="empty">${escapeHtml(data.error)}</div>`;
    state.info.textContent = '';
    return;
  }
  state.info.textContent =
    `scope=${data.scope || 'run'}  ·  clusters=${data.n_clusters || 0}  ·  ` +
    `page ${data.pagination.page + 1}/${data.pagination.total_pages}`;
  renderPlantaeLegend(data);
  const green = data.plantae_green ?? 0.60;
  const yellow = data.plantae_yellow ?? 0.35;

  // Filter is applied SERVER-SIDE now (cluster_id / species / genus /
  // family passed in /api/results URL). data.results already contains
  // only matching assets, so no client-side filter needed here.
  let rows = data.results;

  // Sort by current mode. Search-rank is the identity transform — the
  // server already returns assets in (most-recent Immich rank ASC) order.
  if (state.currentSortMode === 'plantae') {
    rows = rows.slice().sort((a, b) =>
      (b.taxon_scores && b.taxon_scores.Plantae != null ? b.taxon_scores.Plantae : -1) -
      (a.taxon_scores && a.taxon_scores.Plantae != null ? a.taxon_scores.Plantae : -1));
  } else if (state.currentSortMode === 'score') {
    rows = rows.slice().sort((a, b) =>
      ((b.top_predictions && b.top_predictions[0] && b.top_predictions[0].score) || -1) -
      ((a.top_predictions && a.top_predictions[0] && a.top_predictions[0].score) || -1));
  } // 'rank' = leave as-is

  if (!rows.length) {
    state.root.innerHTML = '<div class="empty">No results in this view.</div>';
    return;
  }

  state.root.innerHTML = rows.map(r => {
    const preds = r.top_predictions.map(p => `
      <div class="pred">
        <div class="sp">${escapeHtml(p.species)}<div class="cn">${escapeHtml(p.common_name || '')}</div></div>
        <div class="sc">${(p.score * 100).toFixed(2)}%</div>
      </div>
    `).join('');
    const k = r.top_predictions[0] || {};
    const fullTop1 = k.species || '';
    const fullTop1Score = k.score || 0;
    const dets = (r.detections || []).filter(d => d.score >= 0.10);
    const bboxSvg = dets.map(d => {
      const x = (d.x * 100).toFixed(2);
      const y = (d.y * 100).toFixed(2);
      const w = (d.w * 100).toFixed(2);
      const h = (d.h * 100).toFixed(2);
      const lbl = `${escapeHtml(d.label)} ${(d.score * 100).toFixed(0)}%`;
      return `<rect x="${x}%" y="${y}%" width="${w}%" height="${h}%"/>` +
             `<text x="${x}%" y="${(Math.max(0, parseFloat(y) - 0.5)).toFixed(2)}%">${lbl}</text>`;
    }).join('');
    // For each detection, show top-3 species BioCLIP predicted on the
    // cropped bbox. Highlight any that beat the full-image top-1 score.
    const detBlocks = dets.map((d, di) => {
      const bp = (d.bbox_predictions || []).slice(0, 3);
      if (!bp.length) return '';
      const rowsHtml = bp.map(p => {
        const beats = p.species === fullTop1 && p.score > fullTop1Score;
        return `
          <div class="pred">
            <div class="sp">${escapeHtml(p.species)}<div class="cn">${escapeHtml(p.common_name || '')}</div></div>
            <div class="sc">${(p.score * 100).toFixed(2)}%${beats ? ' ↑' : ''}</div>
          </div>
        `;
      }).join('');
      return `
        <div class="det-block">
          <div class="det-header">crop ${di + 1}: ${escapeHtml(d.label)} ${(d.score * 100).toFixed(0)}%</div>
          ${rowsHtml}
        </div>
      `;
    }).join('');
    const nearDups = r.near_dups || [];
    const nearDupPanel = nearDups.length ? `
      <div class="near-dup">
        <button class="near-dup-toggle" data-aid="${escapeHtml(r.asset_id)}">
          ▸ ${nearDups.length} similar image${nearDups.length > 1 ? 's' : ''} (cosine ≥ 0.99)
        </button>
        <div class="near-dup-panel" id="ndp-${escapeHtml(r.asset_id)}" style="display:none">
          ${nearDups.map(d => `
            <div class="near-dup-item" data-aid="${escapeHtml(r.asset_id)}" data-other="${escapeHtml(d.asset_id)}">
              <img src="/thumb/${encodeURIComponent(d.asset_id)}" alt="${escapeHtml(d.original_file_name || '')}" loading="lazy" decoding="async">
              <div class="near-dup-name" title="${escapeHtml(d.original_file_name || '')}">${escapeHtml(d.original_file_name || d.asset_id)}</div>
              <button class="near-dup-pick" title="Use this image as the primary on this card">use as primary</button>
            </div>
          `).join('')}
        </div>
      </div>
    ` : '';
    return `
      <div class="card">
        <div class="img-wrap">
          <img src="/thumb/${encodeURIComponent(r.asset_id)}" alt="${escapeHtml(r.original_file_name)}" loading="lazy" decoding="async" data-primary-asset="${escapeHtml(r.asset_id)}">
          ${r.cluster_id !== null && r.cluster_id !== undefined ? `<span class="cluster-tag">cluster ${r.cluster_id}</span>` : ''}
          ${r.rank !== undefined && r.rank !== null ? `<span class="rank-tag" title="Immich smart-search rank (0-based)">#${r.rank + 1}</span>` : ''}
          ${(r.near_dups && r.near_dups.length) ? `<span class="dup-badge" title="${r.near_dups.length} similar photo${r.near_dups.length > 1 ? 's' : ''} (cosine ≥ 0.97)">${r.near_dups.length}×</span>` : ''}
          ${bboxSvg ? `<svg class="bbox" viewBox="0 0 100 100" preserveAspectRatio="none">${bboxSvg}</svg>` : ''}
        </div>
        <div class="meta-card">
          <div class="name">${escapeHtml(r.original_file_name)}</div>
          <div><span class="badge">${escapeHtml(k.kingdom || '?')}</span><span class="badge">${escapeHtml(k.family || '?')}</span><span class="badge">${escapeHtml(k.genus || '?')}</span>${(() => {
            const ts = r.taxon_scores || {};
            const plantae = ts.Plantae;
            if (plantae == null) return '';
            const cls = plantaeClass(plantae, green, yellow);
            const others = Object.entries(ts)
              .filter(([n]) => n !== 'Plantae')
              .sort((a, b) => b[1] - a[1])
              .slice(0, 3)
              .map(([n, s]) => `${escapeHtml(n)}=${(s*100).toFixed(0)}`)
              .join(' · ');
            return `<span class="plantae ${cls}" title="BioCLIP kingdom-rank score for Plantae${others ? ' — other kingdoms: '+others : ''}">Plantae ${(plantae*100).toFixed(1)}%</span>`;
          })()}</div>
          <div class="urls">
            <a href="${escapeHtml(r.original_url)}" target="_blank">original →</a>
            <a href="${escapeHtml(r.thumbnail_url)}" target="_blank">thumb →</a>
          </div>
        </div>
        <div class="preds">${preds}</div>
        ${detBlocks ? `<div class="dets">${detBlocks}</div>` : ''}
        ${nearDupPanel}
        ${renderKnnBlock(r)}
      </div>
    `;
  }).join('');
}

// KNN block: a collapsible panel under each card showing the top-1
// consensus species + top-K neighbors, plus a reclassify button when
// no row is cached yet. `render()` calls this for every card so the
// block is present in the DOM whether or not KNN results are cached.
function renderKnnBlock(r) {
  const knn = r.knn;
  if (knn && knn.top_species) {
    const sp = escapeHtml(knn.top_species || '');
    const cn = escapeHtml(knn.common_name || '');
    const sc = ((knn.score || 0) * 100).toFixed(1);
    const nbrs = (knn.neighbors || []).slice(0, 3).map(n =>
      `<div class="pred"><div class="sp">${escapeHtml(n.species)}<div class="cn">${escapeHtml(n.common_name || '')}</div></div><div class="sc">${(n.sim * 100).toFixed(1)}%</div></div>`
    ).join('');
    return `
      <div class="knn-block">
        <div class="knn-header">
          <span>KNN top-1: ${sp}${cn ? ' (' + cn + ')' : ''} · ${sc}%</span>
          <span style="color:#888">${escapeHtml(knn.kingdom || '')}${knn.family ? ' · ' + escapeHtml(knn.family) : ''}</span>
        </div>
        ${nbrs}
        <button class="reclassify-btn" data-aid="${escapeHtml(r.asset_id)}" data-action="reclassify" style="margin-top:6px">Re-run KNN</button>
      </div>
    `;
  }
  return `
    <div class="knn-block">
      <div class="knn-empty">No KNN result yet.</div>
      <button class="reclassify-btn" data-aid="${escapeHtml(r.asset_id)}" data-action="reclassify" style="margin-top:4px">Reclassify with KNN</button>
    </div>
  `;
}
