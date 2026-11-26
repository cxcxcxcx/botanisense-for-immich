// Typed wrappers, one per endpoint. Every function returns a typed promise.
// Use these from components instead of calling fetchJson directly so the
// shapes stay in one place.

import { fetchJson } from './client';
import type {
  BboxClusterItemsResponse,
  BboxClusteringStatus,
  BboxClustersResponse,
  BboxNeighborsResponse,
} from './types';
import type {
  AssetDetail,
  BulkClusterRemoveParams,
  BulkClusterRemoveResponse,
  ClusterAlbumMapping,
  ClusterGroupsResponse,
  ClustersResponse,
  Health,
  ImmichConnectionParams,
  JobPreview,
  JobSnapshot,
  JobStart,
  KnnBuildArgs,
  KnnBuildJobSnapshot,
  KnnBuildLogsResponse,
  KnnRow,
  KnnStatus,
  KingdomAlbumStatus,
  LocationGuessApplyResult,
  LocationGuessStats,
  MapClustersQuery,
  MapClustersResponse,
  PipelineStatus,
  PublicConfig,
  PurgeMissingPreview,
  PurgeMissingResult,
  RelatedClustersResponse,
  ResultsQuery,
  ResultsResponse,
  RunParams,
  SaveConnectionResponse,
  SyncDefaults,
  SyncDefaultsResponse,
} from './types';

// ---- Reads ---------------------------------------------------------------

export const getResults   = (q: ResultsQuery) =>
  fetchJson<ResultsResponse>('/api/results', { query: q });

export const getAsset     = (id: string) =>
  fetchJson<AssetDetail>(`/api/asset/${id}`);

export const getClusters  = (plantae?: 'plant' | 'non_plant' | 'uncertain') =>
  fetchJson<ClustersResponse>('/api/clusters', { query: { plantae } });

// Places on the map. Same filter surface as /api/results (minus
// pagination/sort) so a filtered browse can be viewed geographically.
// `min_photos` defaults to 2 server-side — pass 1 for the "show
// single-photo places" toggle.
export const getMapClusters = (q: MapClustersQuery) =>
  fetchJson<MapClustersResponse>('/api/map/clusters', { query: q });

// ---- Inferred locations (local only) ------------------------------------
// Coordinates guessed from GPS-tagged temporal neighbours. Stored in the
// local SQLite DB; never written back to Immich. The map reads them through
// COALESCE(real, guess), so a real EXIF fix always wins.

export const getLocationGuesses = () =>
  fetchJson<LocationGuessStats>('/api/location/guesses');

export const applyLocationGuesses = (
  p: { dry_run?: boolean; window_minutes?: number; max_span_m?: number } = {},
) =>
  fetchJson<LocationGuessApplyResult>('/api/location/guesses/apply', {
    method: 'POST',
    query: p,
  });

export const clearLocationGuesses = () =>
  fetchJson<{ cleared: number }>('/api/location/guesses', { method: 'DELETE' });

// Super-groups of clusters (the hierarchy level above clusters). Same
// cached pipeline as /api/clusters server-side, so counts reconcile.
export const getClusterGroups = () =>
  fetchJson<ClusterGroupsResponse>('/api/cluster-groups');

// Clusters nearest to `cluster_id` in embedding space (centroid cosine).
// Rendered as the chip strip under the back-bar when a cluster is open.
export const getRelatedClusters = (cluster_id: number, limit = 8) =>
  fetchJson<RelatedClustersResponse>('/api/clusters/related', { query: { cluster_id, limit } });

export const getKnnStatus = () =>
  fetchJson<KnnStatus>('/api/knn-status');

export const getStatus    = () =>
  fetchJson<PipelineStatus>('/api/status');

export const getHealth    = () =>
  fetchJson<Health>('/api/health');

export const getConfig    = () =>
  fetchJson<PublicConfig>('/api/config');

// POST /api/settings/immich — save the Immich connection (URL + API
// key) from the web UI. Blank api_key keeps the saved one. Returns the
// fresh health payload so the form can show connected/broken inline.
export const saveImmichConnection = (params: ImmichConnectionParams) =>
  fetchJson<SaveConnectionResponse>('/api/settings/immich', {
    method: 'POST',
    body: params,
  });

export const getLogs      = (tail = 200) =>
  fetchJson<{ lines: string[] }>('/api/logs', { query: { tail } });

export const listClusterAlbums = () =>
  fetchJson<{ albums: ClusterAlbumMapping[] }>('/api/album/cluster');

export const getKingdomAlbumStatus = (kingdom: string) =>
  fetchJson<KingdomAlbumStatus>('/api/kingdom-album-status', { query: { kingdom } });

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

export const reclassify = (asset_ids: string[], k?: number, force?: boolean) =>
  fetchJson<{ stats: Record<string, unknown>; results: Record<string, KnnRow> }>(
    '/api/reclassify',
    { method: 'POST', body: { asset_ids, k, force } },
  );

export const reclassifyVisible = (asset_ids?: string[]) =>
  fetchJson<{ queued: number }>(
    '/api/reclassify-visible',
    { method: 'POST', body: { asset_ids } },
  );

export const reclassifyBbox = (params: { asset_id: string; detection_id: number }) =>
  fetchJson<{
    detection_id: number;
    asset_id: string;
    index_version: string;
    k: number;
    result: {
      top_species: string | null;
      common_name: string;
      score: number;
      kingdom: string;
      family: string;
      genus: string;
      neighbors: { uuid: string; species: string; common_name: string; sim: number }[];
    };
    cached: boolean;
  }>(
    '/api/reclassify-bbox',
    { method: 'POST', body: params },
  );

// ---- Kingdom album jobs --------------------------------------------------
// One parameterized surface for "tag a kingdom as an album" (Plantae,
// Fungi, ...). The old /api/album/plants endpoints were the
// kingdom="Plantae" special case and were removed server-side.

export const previewKingdomAlbum = (
  kingdom: string,
  min_score?: number,
  people_filter?: 0 | 1 | null,
) =>
  fetchJson<JobPreview>('/api/album/kingdom/preview', {
    query: { kingdom, min_score, people_filter: people_filter ?? undefined },
  });

export const startKingdomAlbum = (
  kingdom: string,
  p: {
    min_score?: number;
    force_recreate?: boolean;
    people_filter?: 0 | 1 | null;
  },
) =>
  fetchJson<JobStart>('/api/album/kingdom', { method: 'POST', body: { kingdom, ...p } });

// ---- Cluster album jobs (per-cluster + sync-all) -----------------------

export const previewClusterAlbum = (
  cluster_id: number,
  people_filter?: 0 | 1 | null,
) =>
  fetchJson<JobPreview>(`/api/album/cluster/${cluster_id}/preview`, {
    query: { people_filter: people_filter ?? undefined },
  });

export const startClusterAlbumSync = (
  cluster_id: number,
  p: { force_recreate?: boolean; people_filter?: 0 | 1 | null } = {},
) =>
  fetchJson<JobStart>(`/api/album/cluster/${cluster_id}/sync`, {
    method: 'POST',
    body: p,
  });

export const startSyncAllClusterAlbums = (
  p: { force_recreate?: boolean; people_filter?: 0 | 1 | null } = {},
) =>
  fetchJson<JobStart>('/api/album/cluster/sync-all', {
    method: 'POST',
    body: p,
  });

// Always deletes the Immich album too (cluster sync is beta, policy
// is "never leave orphans"). If the Immich album is already gone
// (404), `remote_deleted=false` is returned and treated as success.
export const deleteClusterAlbum = (cluster_id: number) =>
  fetchJson<{ deleted: boolean; remote_deleted: boolean }>(
    `/api/album/cluster/${cluster_id}`,
    { method: 'DELETE' },
  );

// ---- Bulk-remove synced clusters --------------------------------------
// One round-trip replaces N single-cluster DELETE calls. The Run
// page's "Remove synced clusters" button uses this with
// cluster_ids=null (meaning "all currently synced clusters").
export const removeClusterAlbums = (p: BulkClusterRemoveParams) =>
  fetchJson<BulkClusterRemoveResponse>('/api/album/cluster/remove', {
    method: 'POST',
    body: p,
  });

// Undo the last bulk-remove by re-inserting the snapshot the frontend
// captured before the call. The Immich albums themselves are NOT
// recreated (those were DELETEd server-side); only the local
// mapping comes back. The user has to re-sync for actual albums.
export const undoBulkRemoveClusterAlbums = (rows: ClusterAlbumMapping[]) =>
  fetchJson<{ restored: number; skipped: number }>(
    '/api/album/cluster/undo-remove',
    { method: 'POST', body: { rows } },
  );

// Multi-select batch action (UX-17). Sets plantae_override on every
// asset_id in one round-trip; value=null clears the override.
export const batchSetPlantaeOverride = (asset_ids: string[], value: 0 | 1 | null) =>
  fetchJson<{ updated: number }>('/api/assets/batch/plantae-override', {
    method: 'POST',
    body: { asset_ids, value },
  });

// ---- Sync-to-IMMICH defaults (GET/PUT /api/settings/sync-defaults) ----
// Persisted on the server so they survive browser changes and are
// visible to non-browser callers. Seeded from env on first read; the
// `seeded` flag lets the UI show a one-time "Defaults initialised
// from environment" hint.
export const getSyncDefaults = () =>
  fetchJson<SyncDefaultsResponse>('/api/settings/sync-defaults');

export const putSyncDefaults = (defaults: Partial<SyncDefaults>) =>
  fetchJson<{ defaults: SyncDefaults }>('/api/settings/sync-defaults', {
    method: 'PUT',
    body: defaults,
  });

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

export interface BackfillExifJobSnapshot extends JobSnapshot {
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

export const getKnnBuildLogs = (job_id: string, tail = 50) =>
  fetchJson<KnnBuildLogsResponse>('/api/knn/build/logs', {
    query: { job_id, tail },
  });

// ---- Box-level similarity ------------------------------------------------
// One YOLO detection crop → similar source images + whole photos. The
// backend ranks both in the same teacher BioCLIP space, then collapses
// all crops from the same source image into one row (n_boxes) and
// excludes the query image itself.
export const getBboxNeighbors = (
  detectionId: number,
  k = 12,
  includePhotos = true,
) =>
  fetchJson<BboxNeighborsResponse>(`/api/bbox/${detectionId}/neighbors`, {
    query: { k, include_photos: includePhotos },
  });

// ---- Box-level clustering ------------------------------------------------

export const getBboxClusters = () =>
  fetchJson<BboxClustersResponse>('/api/bbox-clusters');

export const getBboxClusterItems = (
  clusterId: number,
  q: { page?: number; page_size?: number; sort?: 'score' | 'time' | 'area' | 'yolo'; dedupe_assets?: boolean } = {},
) =>
  fetchJson<BboxClusterItemsResponse>(`/api/bbox-clusters/${clusterId}`, {
    query: q,
  });

export const getClusterGroupItems = (
  groupId: number,
  q: { page?: number; page_size?: number; sort?: 'score' | 'time' | 'area' | 'yolo' } = {},
) =>
  fetchJson<BboxClusterItemsResponse>(`/api/cluster-groups/${groupId}/items`, {
    query: q,
  });

export const refitBboxClusters = () =>
  fetchJson<{ job_id: string }>('/api/bbox-clusters/refit', { method: 'POST' });

export const getBboxClusteringStatus = () =>
  fetchJson<BboxClusteringStatus>('/api/bbox-clusters/status');
