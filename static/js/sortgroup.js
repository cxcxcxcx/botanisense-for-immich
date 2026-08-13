// Sort-mode + group-by dropdowns; their changes drive loadResults.
import { state } from './state.js';
import { loadResults } from './results.js';
import { loadClusterBar } from './clusters.js';

export function wireSortGroup() {
  const sortModeSel = document.getElementById('sort-mode');
  if (sortModeSel) {
    sortModeSel.value = state.currentSortMode;
    sortModeSel.addEventListener('change', () => {
      state.currentSortMode = sortModeSel.value;
      try { localStorage.setItem('sortMode', state.currentSortMode); } catch (_) {}
      // Re-render the latest data with the new sort.
      loadResults({refreshPills: false});
    });
  }
  const groupBySel = document.getElementById('group-by');
  if (groupBySel) {
    groupBySel.value = state.currentGroupBy;
    groupBySel.addEventListener('change', () => {
      state.currentGroupBy = groupBySel.value;
      state.currentGroupFilter = null;
      state.currentFilter = {};
      try { localStorage.setItem('groupBy', state.currentGroupBy); } catch (_) {}
      loadClusterBar();
      loadResults({page: 0, refreshPills: false, filter: {}});
    });
  }
}
