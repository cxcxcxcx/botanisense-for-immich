// Entry point. Imports every module and runs the boot sequence.
//
// Module imports each have a side effect: module-level `wireXxx()`
// functions attach the relevant DOM listeners. The init IIFE then
// fires the initial fetches.
import { refreshKnnPill } from './knnpill.js';
import { loadClusterBar, wireClusterSearch } from './clusters.js';
import { pollStatus, pollLogs, wireLogToggle } from './pollers.js';
import { loadResults } from './results.js';
import { wireParams } from './params.js';
import { wireSortGroup } from './sortgroup.js';
import { wireDelegatedHandlers } from './handlers.js';
import { wireReclassifyVisible } from './reclassify.js';
import { wirePlantsBtn, refreshPlantsStatus } from './plants.js';
import './modal.js';   // exports openDetailModal + closeDetailModal; no module-level wiring (handlers.js calls it)

wireClusterSearch();
wireLogToggle();
wireParams();
wireSortGroup();
wireReclassifyVisible();
wirePlantsBtn();
wireDelegatedHandlers();

(async () => {
  await refreshKnnPill();
  await loadClusterBar();
  await pollStatus();
  await loadResults();
  await refreshPlantsStatus();
})();

setInterval(pollStatus, 1000);
setInterval(pollLogs, 2000);
