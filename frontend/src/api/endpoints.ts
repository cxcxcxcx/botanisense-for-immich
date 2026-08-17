// Typed wrappers, one per endpoint. Every function returns a typed promise.
// Use these from components instead of calling fetchJson directly so the
// shapes stay in one place.

import { fetchJson } from './client';
import type {
  AssetDetail,
  ClusterAlbumMapping,
  ClustersResponse,
  JobPreview,
  JobSnapshot,
  JobStart,
  KnnBuildArgs,
  KnnBuildJobSnapshot,
  KnnBuildLogsResponse,
  KnnStatus,
  PipelineStatus,
  PlantsAlbumStatus,
  PurgeMissingPreview,
  PurgeMissingResult,
  ResultsQuery,
  ResultsResponse,
  RunParams,
} from './types';

// ---- Reads ---------------------------------------------------------------

export const getResults   = (q: ResultsQuery) =>
  fetchJson<ResultsResponse>('/api/results', { query: q });

export const getAsset     = (id: string) =>
  fetchJson<AssetDetail>(`/api/asset/${id}`);

export const getClusters  = () =>
  fetchJson<ClustersResponse>('/api/clusters');

export const getKnnStatus = () =>
  fetchJson<KnnStatus>('/api/knn-status');

export const getStatus    = () =>
  fetchJson<PipelineStatus>('/api/status');

export const getLogs      = (tail = 200) =>
  fetchJson<{ lines: string[] }>('/api/logs', { query: { tail } });

export const listClusterAlbums = () =>
  fetchJson<{ albums: ClusterAlbumMapping[] }>('/api/album/cluster');

export const getPlantsAlbumStatus  = () =>
  fetchJson<PlantsAlbumStatus>('/api/plants-album-status');

export const getKingdomAlbumStatus = (kingdom: string) =>
  fetchJson<PlantsAlbumStatus>('/api/kingdom-album-status', { query: { kingdom } });

export const getClusteringStatus   = () =>
  fetchJson<{
    clusterer_loaded: boolean;
    last_refit_at: string | null;
    n_at_refit: number;
    n_current: number;
    growth_pct: number;
    pending_refit: boolean;
    latest_refit?: { status: string; n_clusters: number | null; n_assets: number | null; error: string | null };
    recent_remaps: { old_cluster_id: number; new_cluster_id: number; overlap_count: number; union_count: number; jaccard: number; kind: string; refit_at: string }[];
  }>('/api/clustering/status');

// ---- Pipeline run --------------------------------------------------------

export const runPipeline = (p: RunParams) =>
  fetchJson<unknown>('/api/run', { method: 'POST', body: p });

export const stopPipeline = () =>
  fetchJson<{ stopped: boolean; reason?: string }>(
    '/api/run/stop',
    { method: 'POST' },
  );

export const refreshPipeline = () =>
  fetchJson<unknown>('/api/refresh', { method: 'POST' });

// ---- KNN reclassify ------------------------------------------------------

export const reclassify = (asset_ids: string[], k?: number) =>
  fetchJson<{ stats: Record<string, unknown>; results: unknown[] }>(
    '/api/reclassify',
    { method: 'POST', body: { asset_ids, k } },
  );

export const reclassifyVisible = (asset_ids?: string[]) =>
  fetchJson<{ queued: number }>(
    '/api/reclassify-visible',
    { method: 'POST', body: { asset_ids } },
  );

export const reclassifyBbox = (params: { asset_id: string; bbox_index: number }) =>
  fetchJson<{ asset_id: string; bbox_index: number; note: string }>(
    '/api/reclassify-bbox',
    { method: 'POST', body: params },
  );

// ---- Plants / Kingdom album jobs ----------------------------------------

export const previewPlantsAlbum = (min_score?: number) =>
  fetchJson<JobPreview>('/api/album/plants/preview', { query: { min_score } });

export const startPlantsAlbum = (p: { min_score?: number; force_recreate?: boolean }) =>
  fetchJson<JobStart>('/api/album/plants', { method: 'POST', body: p });

export const pollPlantsAlbumJob = (job_id: string) =>
  fetchJson<JobSnapshot>(`/api/album/plants/job/${job_id}`);

export const previewKingdomAlbum = (kingdom: string, min_score?: number) =>
  fetchJson<JobPreview>('/api/album/kingdom/preview', { query: { kingdom, min_score } });

export const startKingdomAlbum = (
  kingdom: string,
  p: { min_score?: number; force_recreate?: boolean },
) =>
  fetchJson<JobStart>('/api/album/kingdom', { method: 'POST', body: { kingdom, ...p } });

export const pollKingdomAlbumJob = (job_id: string) =>
  fetchJson<JobSnapshot>(`/api/album/kingdom/job/${job_id}`);

// ---- Cluster album jobs (per-cluster + sync-all) -----------------------

export const previewClusterAlbum = (cluster_id: number) =>
  fetchJson<JobPreview>(`/api/album/cluster/${cluster_id}/preview`);

export const startClusterAlbumSync = (cluster_id: number, force_recreate = false) =>
  fetchJson<JobStart>(`/api/album/cluster/${cluster_id}/sync`, {
    method: 'POST',
    body: { force_recreate },
  });

export const pollClusterAlbumJob = (job_id: string) =>
  fetchJson<JobSnapshot>(`/api/album/cluster/job/${job_id}`);

export const startSyncAllClusterAlbums = (force_recreate = false) =>
  fetchJson<JobStart>('/api/album/cluster/sync-all', {
    method: 'POST',
    body: { force_recreate },
  });

export const deleteClusterAlbum = (cluster_id: number, delete_in_immich: boolean) =>
  fetchJson<{ deleted: boolean; remote_deleted: boolean }>(
    `/api/album/cluster/${cluster_id}`,
    { method: 'DELETE', query: { delete_in_immich: String(delete_in_immich) } },
  );

// ---- Clustering refit ----------------------------------------------------

export const startClusteringRefit = (threshold_pct?: number) =>
  fetchJson<{ job_id: string }>(
    '/api/clustering/refit',
    { method: 'POST', body: { threshold_pct } },
  );

// ---- Purge assets deleted from Immich -----------------------------------
// Two-phase flow: preview scans Immich and returns the count + sample;
// purge runs the scan again and DELETEs the missing rows from the local
// DB. Synchronous on the server (no job_id), so the UI uses JobPanel's
// 'isSync' branch to dispatch JOB_DONE directly on success.

export const previewPurgeMissing = async () => {
  const r = await fetchJson<PurgeMissingPreview>('/api/assets/purge-missing', {
    method: 'POST',
    query: { dry_run: 'true' },
  });
  // JobPanel's preview() expects {n_assets, sample?}; map the missing
  // count into the canonical field name so the existing UI works.
  return { n_assets: r.missing_count, sample: r.sample };
};

export const purgeMissing = () =>
  fetchJson<PurgeMissingResult>('/api/assets/purge-missing', {
    method: 'POST',
    query: { dry_run: 'false' },
  });

// ---- Backfill EXIF (time + geo) from Immich -------------------------
// One-shot pass that walks every DB row and pulls /api/assets/{id} to
// fill captured_at / latitude / longitude. Mirrors the KNN build
// flow — spawns a thread, returns a job_id, UI polls status + logs.
// ~30s for a 17k-row DB. The pipeline ingest also calls Immich for
// each new asset; this is for rows that existed before EXIF capture
// was added.

export interface BackfillExifJobSnapshot {
  job_id: string;
  status: 'running' | 'done' | 'error';
  started_at: string;
  finished_at: string | null;
  updated_count: number;
  skipped_count: number;
  no_exif_count: number;
  missing_count: number;
  error: string | null;
}

export interface BackfillExifLogsResponse {
  job_id: string;
  lines: string[];
}

export const startBackfillExif = () =>
  fetchJson<{ job_id: string }>('/api/assets/backfill-exif', {
    method: 'POST',
  });

export const pollBackfillExifJob = (job_id: string) =>
  fetchJson<BackfillExifJobSnapshot>(`/api/assets/backfill-exif/job/${job_id}`);

export const getBackfillExifLogs = (job_id: string, tail = 80) =>
  fetchJson<BackfillExifLogsResponse>('/api/assets/backfill-exif/logs', {
    query: { job_id, tail },
  });

// ---- Person-detection backfill -------------------------------------
// Standalone YOLO sweep that fills in `assets.has_person` for every
// asset where it's currently NULL. No BioCLIP — just YOLO-World with
// the 17-label model. Same job shape as backfill-exif so the UI can
// reuse the same modal pattern.

export interface PersonBackfillJobSnapshot {
  job_id: string;
  status: 'running' | 'done' | 'error';
  started_at: string;
  finished_at: string | null;
  // Total assets queued at start — drives the confirm-modal message.
  n_to_scan: number;
  // Live counters — updated as each outer page (500 assets) finishes.
  scanned_count: number;
  with_person_count: number;
  without_person_count: number;
  error: string | null;
}

export interface PersonBackfillLogsResponse {
  job_id: string;
  lines: string[];
}

export interface StartPersonBackfillResponse {
  job_id: string | null;
  n_to_scan: number;
  // true when the DB had zero unscanned assets — endpoint short-circuits
  // and returns this so the UI can show "already done" without
  // spawning a thread.
  noop: boolean;
}

export const startPersonBackfill = () =>
  fetchJson<StartPersonBackfillResponse>('/api/person/backfill', {
    method: 'POST',
  });

// Cheap count call so the confirm modal can show "N photos will be
// scanned". Same shape as the album-job previews (n_assets) so the
// JobPanel can treat it identically.
export const previewPersonBackfill = () =>
  fetchJson<{ n_assets: number }>('/api/person/backfill/preview');

export const pollPersonBackfillJob = (job_id: string) =>
  fetchJson<PersonBackfillJobSnapshot>(
    `/api/person/backfill/job/${job_id}`,
  );

export const getPersonBackfillLogs = (job_id: string, tail = 80) =>
  fetchJson<PersonBackfillLogsResponse>('/api/person/backfill/logs', {
    query: { job_id, tail },
  });

// Tell a running person-backfill to unwind at the next batch
// boundary. Returns 200 even if the job isn't running — the UI can
// show "already finished" without an error.
export const stopPersonBackfill = (job_id: string) =>
  fetchJson<{ stopped: boolean; job_id?: string; reason?: string }>(
    `/api/person/backfill/stop/${encodeURIComponent(job_id)}`,
    { method: 'POST' },
  );

// ---- Build KNN index (FAISS) ---------------------------------------
// Long-running job spawned by the Run view "Build KNN index" button.
// Three endpoints: start, poll-status, tail-logs.

export const startKnnBuild = (args: KnnBuildArgs) =>
  fetchJson<{ job_id: string }>('/api/knn/build', {
    method: 'POST',
    body: args,
  });

export const pollKnnBuildJob = (job_id: string) =>
  fetchJson<KnnBuildJobSnapshot>(`/api/knn/build/job/${job_id}`);

export const getKnnBuildLogs = (job_id: string, tail = 50) =>
  fetchJson<KnnBuildLogsResponse>('/api/knn/build/logs', {
    query: { job_id, tail },
  });