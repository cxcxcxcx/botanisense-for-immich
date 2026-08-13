// Shared mutable state across all modules. Single exported object so every
// module imports the same instance and changes propagate without needing
// top-level let bindings (which ES modules disallow across files anyway).

const root = document.getElementById('root');
const info = document.getElementById('run-info');
const refreshBtn = document.getElementById('refresh');
const runBtn = document.getElementById('run-btn');
const clustersBar = document.getElementById('clusters-bar');
const statusPanel = document.getElementById('status-panel');
const statusText = document.getElementById('status-text');
const statusBar = document.getElementById('status-bar');
const stageTimeBar = document.getElementById('stage-time-bar');
const stageTimeLegend = document.getElementById('stage-time-legend');
const logToggle = document.getElementById('log-toggle');
const logPre = document.getElementById('log-pre');
const paramQuery = document.getElementById('param-query');
const paramN = document.getElementById('param-n');
const paramK = document.getElementById('param-k');
const paramBatch = document.getElementById('param-batch');

// Surface the server URL we're actually talking to, so connection issues
// are obvious instead of mysterious "Loading…" forever.
document.getElementById('server-url').textContent = location.origin;

export const state = {
  // DOM refs.
  root, info, refreshBtn, runBtn, clustersBar, statusPanel, statusText,
  statusBar, stageTimeBar, stageTimeLegend, logToggle, logPre,
  paramQuery, paramN, paramK, paramBatch,

  // Persisted-ish UI choices.
  currentGroupBy: (typeof localStorage !== 'undefined' && localStorage.getItem('groupBy')) || 'species',
  currentGroupFilter: null,           // null = all; otherwise cluster_id (number) or taxon name (string)
  currentSortMode: (typeof localStorage !== 'undefined' && localStorage.getItem('sortMode')) || 'rank',

  // Taxonomy-mode cache for the typeahead (see clusters.js).
  currentEntries: [],
  activeQuery: '',

  // Pagination state.
  currentPage: 0,
  currentPageSize: 60,

  // Server-side filter (cluster_id / species / genus / family).
  currentFilter: {},

  // Cached /api/results payload (used by sort-mode + KNN reclassify patch).
  lastResults: null,

  // /api/logs polling delta.
  lastLogCount: 0,
};
