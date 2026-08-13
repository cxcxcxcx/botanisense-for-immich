// Cluster-bar / taxonomy-pill rendering, with the typeahead filter wired in.
import { state } from './state.js';
import { escapeHtml, safeFetch } from './utils.js';
import { loadResults } from './results.js';

// Compute taxonomy pill data from the current results, OR fall back
// to /api/clusters when grouping by cluster (HDBSCAN labels are global).
export async function loadClusterBar() {
  const pillsContainer = document.getElementById('cluster-pills');
  const searchInput = document.getElementById('cluster-search');
  if (state.currentGroupBy === 'cluster') {
    searchInput.style.display = 'none';
    try {
      const data = await safeFetch('/api/clusters');
      const pills = [
        `<span class="cluster-pill${state.currentGroupFilter === null ? ' on' : ''}" data-cluster="">all</span>`
      ];
      for (const c of data.clusters) {
        const cid = Number(c.cluster_id);
        // Prefer the human-readable name ("Saguaro", "Hydrangea & Pelargonium")
        // computed server-side from the cluster's top-1 prediction distribution.
        // Falls back to "cluster 173" if no name was generated (e.g. empty cluster).
        const label = c.name
          ? `${escapeHtml(c.name)} <small style="color:#888">(${cid}, ${c.n})</small>`
          : `cluster ${cid} (${c.n})`;
        const titleParts = [];
        if (c.top_species && c.top_species.length) {
          titleParts.push('top species: ' + c.top_species.map(s =>
            `${s.species} (${s.count}${s.common_name ? ', ' + s.common_name : ''})`
          ).join('; '));
        }
        if (c.top_genus && c.top_genus.length) {
          titleParts.push('top genera: ' + c.top_genus.map(g => `${g.genus} (${g.count})`).join('; '));
        }
        pills.push(
          `<span class="cluster-pill${state.currentGroupFilter === cid ? ' on' : ''}" data-cluster="${cid}" title="${escapeHtml(titleParts.join('\\n'))}">` +
          `${label}</span>`
        );
      }
      pillsContainer.innerHTML = pills.join('');
      pillsContainer.querySelectorAll('.cluster-pill').forEach(el => {
        el.addEventListener('click', () => {
          const v = el.getAttribute('data-cluster');
          state.currentGroupFilter = v === '' ? null : Number(v);
          loadClusterBar();
          loadResults({page: 0, refreshPills: false, filter: {cluster_id: state.currentGroupFilter}});
        });
      });
    } catch (e) {
      pillsContainer.innerHTML = '<span style="color:#888">cluster data unavailable</span>';
    }
    return;
  }
  // Taxonomy mode — counts come from the server-side summary, which
  // aggregates the entire database (so the count matches the photos
  // visible in the grid). Cache the entries for the typeahead.
  searchInput.style.display = '';
  let data;
  try {
    data = await safeFetch('/api/results?page=0&page_size=1');
  } catch (e) {
    pillsContainer.innerHTML = '<span style="color:#888">results unavailable</span>';
    return;
  }
  state.currentEntries = (data.taxonomy && data.taxonomy[state.currentGroupBy]) || [];
  state.activeQuery = '';
  searchInput.value = '';
  renderTaxonomyPills();
}

// Render the taxonomy pills filtered by `state.activeQuery`. Preserves the
// currently-selected filter even if it gets hidden by the search, in which
// case it shows up as a dashed "sticky" chip at the end.
export function renderTaxonomyPills() {
  const pillsContainer = document.getElementById('cluster-pills');
  const q = state.activeQuery.trim().toLowerCase();
  const visible = q
    ? state.currentEntries.filter(e => e.name.toLowerCase().includes(q))
    : state.currentEntries;
  const total = state.currentEntries.reduce((s, e) => s + e.n, 0);
  const pills = [
    `<span class="cluster-pill${state.currentGroupFilter === null ? ' on' : ''}" data-cluster="" title="${total} photos total">all (${total})</span>`
  ];
  for (const {name, n} of visible) {
    const esc = escapeHtml(name);
    pills.push(
      `<span class="cluster-pill${state.currentGroupFilter === name ? ' on' : ''}" data-cluster="${esc}">` +
      `${esc} (${n})</span>`
    );
  }
  // If a species/genus/family filter is active but the search has hidden
  // its pill, re-surface it as a sticky chip so the user can still see
  // (and deselect) the active filter.
  const filterStr = state.currentGroupFilter;
  if (filterStr !== null && !visible.some(e => e.name === filterStr)) {
    const full = state.currentEntries.find(e => e.name === filterStr);
    const n = full ? full.n : '?';
    const esc = escapeHtml(filterStr);
    pills.push(
      `<span class="cluster-pill on sticky" data-cluster="${esc}" title="current filter (hidden by search)">` +
      `… ${esc} (${n})</span>`
    );
  }
  if (q && visible.length === 0) {
    pills.push(`<span style="color:#888; padding: 4px 8px; font-size: 12px;">no matches</span>`);
  }
  pillsContainer.innerHTML = pills.join('');
  pillsContainer.querySelectorAll('.cluster-pill').forEach(el => {
    el.addEventListener('click', () => {
      const v = el.getAttribute('data-cluster');
      state.currentGroupFilter = v === '' ? null : v;
      renderTaxonomyPills();
      // Build a filter object for the chosen taxonomy dimension. Only
      // one of species/genus/family is active at a time (the group_by
      // selector enforces that).
      const filter = {};
      if (state.currentGroupFilter !== null) {
        filter[state.currentGroupBy] = state.currentGroupFilter;
      }
      loadResults({page: 0, refreshPills: false, filter});
    });
  });
}

// Wire the search input ONCE at module load. Each keystroke re-renders
// the pill list from the cached entries — no server round-trip.
export function wireClusterSearch() {
  document.getElementById('cluster-search').addEventListener('input', (e) => {
    state.activeQuery = e.target.value;
    renderTaxonomyPills();
  });
}
