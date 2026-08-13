// Fetch and render /api/results, plus track the active server-side filter.
import { state } from './state.js';
import { safeFetch, showError } from './utils.js';
import { render } from './grid.js';
import { renderPagination } from './pagination.js';
import { loadClusterBar } from './clusters.js';

export async function loadResults(opts = {}) {
  const page = opts.page ?? state.currentPage;
  const pageSize = opts.pageSize ?? state.currentPageSize;
  state.currentPage = page;
  state.currentPageSize = pageSize;
  // If a filter is passed, reset to page 0 (results are different now)
  // and use opts.filter directly. Otherwise use the current global filter.
  const filter = opts.filter !== undefined ? opts.filter : state.currentFilter;
  if (opts.filter !== undefined) state.currentFilter = filter;
  const params = new URLSearchParams({
    page: String(page),
    page_size: String(pageSize),
  });
  if (filter.cluster_id != null) params.set('cluster_id', String(filter.cluster_id));
  if (filter.species) params.set('species', filter.species);
  if (filter.genus) params.set('genus', filter.genus);
  if (filter.family) params.set('family', filter.family);
  const url = `/api/results?${params}`;
  try {
    const data = await safeFetch(url);
    state.lastResults = data;
    render(data);
    renderPagination(data.pagination);
    if (opts.refreshPills !== false) loadClusterBar();
  } catch (e) {
    showError('Could not load results.', `${url} — ${e.message}`);
  }
}

// Re-render the last-loaded results without re-fetching — used by the
// sort-mode dropdown so a sort change doesn't trigger a round-trip.
export function renderCurrent() {
  if (state.lastResults) render(state.lastResults);
}
