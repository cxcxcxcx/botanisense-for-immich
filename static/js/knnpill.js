// KNN status pill — poll once on load + after reclassify actions.
import { safeFetch } from './utils.js';

export async function refreshKnnPill() {
  const pill = document.getElementById('knn-pill');
  if (!pill) return;
  let s;
  try { s = await safeFetch('/api/knn-status'); }
  catch (e) { pill.textContent = 'KNN: error'; pill.className = 'knn-pill unavailable'; return; }
  if (!s.index_built) {
    pill.textContent = 'KNN: not built';
    pill.className = 'knn-pill unavailable';
    pill.title = `No FAISS index at ${s.inventory_dir}. Build with \`python -m knn --parquet-glob ... --index-dir ${s.inventory_dir} --tag ...\`.`;
    return;
  }
  const cached = s.cached_assets || 0;
  const nRows = s.n_rows ? `${(s.n_rows / 1e6).toFixed(0)}M` : '?';
  pill.textContent = `KNN: ${s.version || '?'} (${nRows} ref, ${cached} cached)`;
  pill.className = 'knn-pill ready';
  pill.title = `Inventory: ${s.inventory_dir}\nVersion: ${s.version}\nReference rows: ${s.n_rows}\nCached assets: ${cached}\nDefault k: ${s.k_default}`;
}
