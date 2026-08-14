// Immich × BioCLIP — HTMX + Alpine glue.
//
// Heavy lifting moved to Jinja templates. This file only contains the
// Alpine component definitions (statusPanel, jobPanel, knnPill,
// plantaeLegend) plus a tiny `toast` helper used by the @click handlers.

import { toast } from './utils.js';
window.toast = toast;

// ---- Alpine component: statusPanel --------------------------------------
// Polls /api/status every 1s; renders the badge, progress, and stacked
// stage-time bar. Clears accumulated stageTotals when a run ends so the
// next run doesn't show stale colors.

document.addEventListener('alpine:init', () => {
  Alpine.data('statusPanel', () => ({
    state: {},
    stageTotals: {},
    prevStatus: 'idle',
    timer: null,
    stageColors: {
      search: 'stage-search',
      species: 'stage-species',
      cluster: 'stage-cluster',
      kingdom: 'stage-kingdom',
      yolo: 'stage-yolo',
      'bbox-bioclip': 'stage-bbox-bioclip',
    },
    start() {
      this.tick();
      this.timer = setInterval(() => this.tick(), 1000);
    },
    async tick() {
      try {
        const r = await fetch('/api/status');
        if (!r.ok) return;
        const s = await r.json();
        // Clear stale totals on idle transition.
        if (this.prevStatus !== 'idle' && s.status !== 'running') {
          this.stageTotals = {};
        }
        this.state = s;
        this.prevStatus = s.status || 'idle';
        for (const [k, v] of Object.entries(s.stage_times || {})) {
          const v0 = this.stageTotals[k] || 0;
          this.stageTotals[k] = Math.max(v0, v);
        }
      } catch (e) { /* transient — keep last good state */ }
    },
    badge() {
      const st = this.state.status || 'idle';
      const sg = this.state.stage || '';
      return sg ? `${st} · ${sg}` : st;
    },
    badgeClass() {
      return `status-${this.state.status || 'idle'}`;
    },
    eta() {
      const parts = [];
      if (this.state.rate != null) parts.push(`${this.state.rate.toFixed(1)}/s`);
      if (this.state.eta_seconds != null) parts.push(`~${Math.ceil(this.state.eta_seconds)}s left`);
      return parts.join(' · ');
    },
    // Live projected stage slices (current stage grows with elapsed time).
    stageBars() {
      const projected = { ...this.stageTotals };
      const st = this.state.stage;
      const t0 = this.state.stage_t0;
      if (st && t0 && this.state.status === 'running') {
        projected[st] = (projected[st] || 0) +
          Math.max(0, (Date.now() / 1000) - t0);
      }
      const total = Object.values(projected).reduce((a, b) => a + b, 0);
      if (!total) return [];
      return Object.entries(projected).map(([k, dur]) => ({
        key: k, label: k, dur: dur.toFixed(1),
        pct: (dur / total) * 100,
        cls: this.stageColors[k] || '',
      }));
    },
    stageLegend() {
      return this.stageBars().map(b => ({
        key: b.key, label: b.label, dur: b.dur, cls: b.cls,
      }));
    },
    // Hooks for the Run form button state.
    isRunning() { return this.state.status === 'running'; },
    runLabel() {
      return this.state.status === 'running' ? 'Running…' : 'Run';
    },
  }));
});

// ---- Alpine component: knnPill -------------------------------------------
// One-shot fetch of /api/knn-status on load. Renders a colored pill.

document.addEventListener('alpine:init', () => {
  Alpine.data('knnPill', () => ({
    data: null,
    async refresh() {
      try {
        const r = await fetch('/api/knn-status');
        if (!r.ok) return;
        this.data = await r.json();
      } catch (e) { /* keep null */ }
    },
    label() {
      if (!this.data) return 'KNN: …';
      if (!this.data.index_built) return 'KNN: not built';
      return `KNN: ${this.data.version || 'v?'} · ${this.data.n_rows} ref · ${this.data.cached_assets} cached`;
    },
    cls() {
      if (!this.data) return 'knn-pill';
      if (!this.data.index_built) return 'knn-pill unavailable';
      return 'knn-pill ready';
    },
    titleText() {
      if (!this.data) return 'KNN status loading…';
      if (!this.data.index_built) {
        return `KNN index missing at ${this.data.inventory_dir}`;
      }
      return `inventory_dir: ${this.data.inventory_dir}\nversion: ${this.data.version}\nn_rows: ${this.data.n_rows}\ncached: ${this.data.cached_assets}`;
    },
  }));
});

// ---- Alpine component: plantaeLegend ------------------------------------

document.addEventListener('alpine:init', () => {
  Alpine.data('plantaeLegend', () => ({
    green: 0.60, yellow: 0.35,
    async refresh() {
      // Pull the unfiltered cache to read the active thresholds.
      try {
        const r = await fetch('/grid?page=0&page_size=1');
        if (!r.ok) return;
        const html = await r.text();
        // Cheap parse: extract plantae_green / plantae_yellow from a known span,
        // or just use defaults from the API by reading one results row.
        const probe = await fetch('/api/results?page=0&page_size=1');
        if (!probe.ok) return;
        const j = await probe.json();
        if (j.plantae_green != null) this.green = j.plantae_green;
        if (j.plantae_yellow != null) this.yellow = j.plantae_yellow;
      } catch (e) { /* keep defaults */ }
    },
    html() {
      return `Plantae <span class="swatch" style="background:#1a3a1a"></span>≥${this.green.toFixed(2)} ` +
             `≥<span class="swatch" style="background:#3a3a1a"></span>${this.yellow.toFixed(2)} ` +
             `<span class="swatch" style="background:#3a1a1a"></span>&lt;${this.yellow.toFixed(2)}`;
    },
  }));
});

// ---- Alpine component: jobPanel -----------------------------------------
// Generic confirm → running → done/error for all four album flows:
// Plants / Fungi (kind=kingdom, kingdom=…) / per-cluster (kind=cluster,
// clusterId=…) / sync-all (kind=sync-all).
//
// State transitions:
//   idle → (open-job-panel event) → preview check → confirm | preview-error
//   confirm → (click confirm) → start() → running
//   running → (poll) → done | error | (job expired → idle)
//   done → (click run-again) → confirm

document.addEventListener('alpine:init', () => {
  Alpine.data('jobPanel', () => ({
    state: 'idle',
    kind: null,
    kingdom: null,
    clusterId: null,
    progress: { added: 0, total: 0, album_id: null, album_url: null, reused: false },
    jobId: null,
    pollTimer: null,
    errorMsg: '',
    init() {
      // React to button clicks from anywhere on the page.
      this.$root.addEventListener('open-job-panel', (e) => {
        this.kind = e.detail?.kind || null;
        this.kingdom = e.detail?.kingdom || null;
        this.clusterId = e.detail?.clusterId || null;
        this.showConfirm();
      });
    },

    // ---- Per-kind config ------------------------------------------------

    previewUrl() {
      if (this.kind === 'kingdom') return `/api/album/kingdom/preview?kingdom=${encodeURIComponent(this.kingdom || 'Plantae')}`;
      if (this.kind === 'cluster') return `/api/album/cluster/${this.clusterId}/preview`;
      if (this.kind === 'sync-all') return null;  // no preview step
      return null;
    },
    endpoint() {
      if (this.kind === 'kingdom')  return '/api/album/kingdom';
      if (this.kind === 'cluster')  return `/api/album/cluster/${this.clusterId}/sync`;
      if (this.kind === 'sync-all') return '/api/album/cluster/sync-all';
      return '/api/album/kingdom';
    },
    body() {
      if (this.kind === 'kingdom') return { kingdom: this.kingdom };
      return {};
    },
    pollUrl(jid) {
      if (this.kind === 'kingdom' || this.kind === 'cluster') {
        return `/api/album/${this.kind}/job/${jid}`;
      }
      if (this.kind === 'sync-all') return `/api/album/cluster/job/${jid}`;
      return `/api/album/kingdom/job/${jid}`;
    },

    // ---- Display --------------------------------------------------------

    title() {
      if (this.kind === 'kingdom') return `Tag as ${this.kingdom} album`;
      if (this.kind === 'cluster') return `Sync cluster #${this.clusterId} as album`;
      if (this.kind === 'sync-all') return 'Sync all clusters as albums';
      return 'Job';
    },
    confirmText() {
      if (this.kind === 'kingdom')
        return `Tag every ${this.kingdom}-classified photo as an Immich album?`;
      if (this.kind === 'cluster')
        return `Sync every photo in cluster #${this.clusterId} as an Immich album?`;
      if (this.kind === 'sync-all')
        return 'Create one Immich album per cluster and add every photo?';
      return 'Run this job?';
    },
    runningText() {
      if (this.kind === 'sync-all')
        return `Syncing ${this.progress.added.toLocaleString()} / ${this.progress.total.toLocaleString()} photos across clusters…`;
      return `Adding ${this.progress.added.toLocaleString()} / ${this.progress.total.toLocaleString()} photos…`;
    },
    doneSubject() {
      if (this.kind === 'kingdom') return `photos added to ${this.kingdom} album`;
      if (this.kind === 'cluster') return 'photos added to cluster album';
      return 'photos added across all cluster albums';
    },

    // ---- State transitions ---------------------------------------------

    async showConfirm() {
      this.progress = { added: 0, total: 0, album_id: null, album_url: null, reused: false };
      this.errorMsg = '';
      // Sync-all has no preview; go straight to confirm.
      if (this.kind === 'sync-all') {
        this.state = 'confirm';
        return;
      }
      // For kingdom/cluster, fetch preview to populate the count.
      const url = this.previewUrl();
      if (!url) { this.state = 'confirm'; return; }
      try {
        const r = await fetch(url);
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const j = await r.json();
        if (!j.n_assets) {
          this.errorMsg = `No matching photos at the current threshold.`;
          this.state = 'preview-error';
          return;
        }
        this.progress.total = j.n_assets;
        this.state = 'confirm';
      } catch (e) {
        this.errorMsg = e.message;
        this.state = 'preview-error';
      }
    },

    hide() {
      if (this.pollTimer) { clearTimeout(this.pollTimer); this.pollTimer = null; }
      this.state = 'idle';
    },

    async start() {
      this.state = 'running';
      this.progress = { added: 0, total: this.progress.total || 0,
                        album_id: null, album_url: null, reused: false };
      let job;
      try {
        const r = await fetch(this.endpoint(), {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify(this.body()),
        });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        job = await r.json();
      } catch (e) {
        this.errorMsg = e.message;
        this.state = 'error';
        return;
      }
      this.jobId = job.job_id;
      if (job.n_assets) this.progress.total = job.n_assets;
      this.schedulePoll();
    },

    schedulePoll() {
      if (this.pollTimer) clearTimeout(this.pollTimer);
      this.pollTimer = setTimeout(() => this.pollOnce(), 400);
    },

    async pollOnce() {
      if (!this.jobId) return;
      let job;
      try {
        const r = await fetch(this.pollUrl(this.jobId));
        if (r.status === 404) { this.state = 'idle'; return; }
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        job = await r.json();
      } catch (e) {
        this.errorMsg = e.message;
        this.state = 'error';
        return;
      }
      if (job.status === 'running') {
        if (this.kind === 'sync-all') {
          this.progress.added = job.photos_added ?? job.clusters_done ?? 0;
          this.progress.total = job.n_assets_total ?? this.progress.total;
        } else {
          this.progress.added = job.n_added ?? 0;
          this.progress.total = job.n_assets ?? this.progress.total;
        }
        this.schedulePoll();
        return;
      }
      if (job.status === 'done') {
        this.progress.added = job.n_added ?? job.photos_added ?? 0;
        this.progress.album_id = job.album_id ?? null;
        this.progress.album_url = job.album_url ?? null;
        this.progress.reused = job.reused_existing ?? false;
        this.state = 'done';
        toast(`Done — ${this.progress.added.toLocaleString()} photos added`, 'success', 4000);
        return;
      }
      if (job.status === 'error') {
        this.errorMsg = job.error || 'unknown error';
        this.state = 'error';
      }
    },
  }));
});
