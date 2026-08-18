// TS interfaces matching the JSON shapes from app.py. Names + nesting match
// the backend 1:1 — when in doubt, copy the response of a curl against the
// endpoint, don't guess.

export interface Prediction {
  species: string;
  common_name: string | null;
  score: number;
  kingdom: string | null;
  family: string | null;
  genus: string | null;
}

export interface Detection {
  id: number;
  label: string;
  score: number;
  x: number;
  y: number;
  w: number;
  h: number;
  bbox_predictions?: Prediction[];
}

export interface NearDup {
  asset_id: string;
  original_file_name: string | null;
}

export interface KnnNeighbor {
  species: string;
  common_name: string | null;
  sim: number;
}

export interface KnnRow {
  top_species: string;
  common_name: string | null;
  score: number;
  kingdom: string | null;
  family: string | null;
  genus: string | null;
  neighbors: KnnNeighbor[];
  index_version: string;
  k: number;
  computed_at: string;
}

export interface ResultRow {
  asset_id: string;
  original_file_name: string;
  original_url: string;
  thumbnail_url: string;
  cluster_id: number | null;
  // first_seen_at is the wall-clock time we first inserted the asset
  // row (set by `DEFAULT (datetime('now'))` on the initial INSERT and
  // never overwritten by upsert). Drives the "added ↓" sort chip —
  // "what did we most recently ingest?" — distinguishing from
  // captured_at which is when the photo was *taken*.
  first_seen_at: string | null;
  // EXIF time + geo (from Immich's GET /api/assets/{id}). NULL until
  // the pipeline ingest captures them, or until the user runs the
  // one-shot /api/assets/backfill-exif.
  captured_at: string | null;
  latitude: number | null;
  longitude: number | null;
  rank: number | null;
  top_predictions: Prediction[];
  taxon_scores: Record<string, number>;
  detections: Detection[];
  // Person-detection flag from the pipeline's "person" stage. See
  // AssetDetail.has_person for the three-state semantics.
  has_person?: boolean | null;
  near_dups?: NearDup[];
  knn?: KnnRow | null;
}

export interface Pagination {
  page: number;
  page_size: number;
  total: number;
  total_pages: number;
}

export interface TaxonomyEntry {
  name: string;
  n: number;
  // Most recent captured_at across all assets in this taxon — drives the
  // "latest" sort chip in FilterCards so taxa with recent photos
  // bubble to the top.
  latest_captured_at: string | null;
  // Representative asset for this taxon — used as the thumbnail on the
  // filter card. The frontend ignores `sample_thumbnail_url` (Immich's
  // direct URL) and uses thumbUrl(sample_asset_id) so it goes through
  // our /thumb proxy and 24h cache.
  sample_asset_id: string | null;
  sample_thumbnail_url: string | null;
  sample_original_file_name: string | null;
}

export interface Taxonomy {
  species: TaxonomyEntry[];
  genus: TaxonomyEntry[];
  family: TaxonomyEntry[];
}

export interface ResultsResponse {
  scope: 'all';
  n_clusters: number;
  plantae_green: number;
  plantae_yellow: number;
  results: ResultRow[];
  pagination: Pagination;
  taxonomy: Taxonomy;
}

export interface LibraryNeighbor {
  asset_id: string;
  original_file_name: string | null;
  thumbnail_url: string;
  cluster_id: number | null;
  similarity: number;
}

export interface AssetDetail {
  asset_id: string;
  original_file_name: string;
  original_url: string;
  thumbnail_url: string;
  cluster_id: number | null;
  // EXIF time + geo. NULL until the pipeline ingest captures them,
  // or until /api/assets/backfill-exif populates existing rows.
  captured_at: string | null;
  latitude: number | null;
  longitude: number | null;
  // Person-detection flag from the pipeline's "person" stage. Three
  // states: true (YOLO detected person/face/human), false (scanned,
  // none found), null (not yet scanned — gate is `has_person IS NULL`
  // in storage.py). Drives the asset-detail "👤" chip and pairs with
  // the /api/results has_person=0/1 filter.
  has_person: boolean | null;
  top_predictions: Prediction[];
  taxon_scores: Record<string, number>;
  detections: Detection[];
  library_neighbors: LibraryNeighbor[];
  knn: KnnRow | null;
}

export interface ClusterSummary {
  cluster_id: number;
  n: number;
  name: string;
  top_species: { species: string; common_name: string | null; count: number }[];
  top_genus: { genus: string; count: number }[];
  // Most recent captured_at across all assets in this cluster — drives
  // the "latest" sort chip in FilterCards so clusters with recent
  // photos bubble to the top.
  latest_captured_at: string | null;
  // Representative asset for this cluster — used as the thumbnail on
  // the filter card.
  sample_asset_id: string | null;
  sample_thumbnail_url: string | null;
  sample_original_file_name: string | null;
}

export interface ClustersResponse {
  clusters: ClusterSummary[];
}

export interface KnnStatus {
  inventory_dir: string;
  index_built: boolean;
  version: string | null;
  n_rows: number;
  dim: number;
  by_kingdom: Record<string, number>;
  k_default: number;
  cached_assets: number;
  cached_versions?: { index_version: string; k: number; n: number }[];
}

// Build-KNN flow. POST /api/knn/build spawns a thread that imports
// knn.build_faiss_index() in-process, runs it, and writes its log to
// a per-job file. The frontend polls /job/{id} for status and /logs
// for the live tail.
export interface KnnBuildArgs {
  parquet_glob: string;
  tag: string;
  index_dir?: string | null;
  kingdoms?: string | null;
  index_type?: string | null;
}

export interface KnnBuildJobSnapshot extends JobSnapshot {
  job_id: string;
  status: 'running' | 'done' | 'error';
  started_at: string;
  finished_at: string | null;
  error: string | null;
}

export interface KnnBuildLogsResponse {
  job_id: string;
  lines: string[];
}

export interface PipelineStatus {
  status: 'idle' | 'running' | 'done' | 'error' | 'stopped';
  stage: string | null;
  chunk_idx: number | null;
  chunk_total: number | null;
  eta_seconds: number | null;
  rate: number | null;
  stage_times: Record<string, number>;
  stage_t0: number | null;
  last_error: string | null;
}

export interface JobPreview {
  n_assets: number;
  n_chunks?: number;
  saved_album_id?: string | null;
  cluster_id?: number;
  kingdom?: string;
  min_score?: number;
}

export interface JobStart {
  job_id: string;
  n_assets?: number;
  n_chunks?: number;
  saved_album_id?: string | null;
  cluster_id?: number;
  kingdom?: string;
  min_score?: number;
}

export interface JobSnapshot {
  status: 'running' | 'done' | 'error';
  // Per-flow shape — concrete fields are read in JobPanel configs.
  [k: string]: unknown;
}

export interface ClusterAlbumMapping {
  cluster_id: number;
  cluster_name: string;
  album_id: string;
  album_url: string;
  n_assets: number;
  reused_existing: boolean;
  last_synced_at: string;
}

// Purge-missing-assets flow. Matches the JSON shape of
// POST /api/assets/purge-missing on the backend (storage.py +
// immich_client.py + app.py).
export interface PurgeMissingSample {
  asset_id: string;
  original_file_name: string;
  cluster_id: number | null;
}

export interface PurgeMissingPreview {
  // JobPanel's preview expects `n_assets` as the canonical count.
  // We map missing_count → n_assets at the call site.
  missing_count: number;
  // Up to 5 sample rows so the confirm modal can show filenames.
  sample: PurgeMissingSample[];
}

export interface PurgeMissingResult {
  missing_count: number;
  deleted_count: number;
  sample?: PurgeMissingSample[];
}

export interface PlantsAlbumStatus {
  exists: boolean;
  kingdom?: string;
  album_id?: string;
  album_url?: string;
  n_assets?: number;
  min_score?: number;
}

export interface ResultsQuery {
  page?: number;
  page_size?: number;
  cluster_id?: number | undefined;
  species?: string | undefined;
  genus?: string | undefined;
  family?: string | undefined;
  [k: string]: string | number | undefined;
}

export interface RunParams {
  query?: string | null;
  n_samples?: number | null;
  top_k?: number | null;
  batch_size?: number | null;
}

// SSE snapshot pushed by /api/events. One object per `data:` line,
// emitted whenever ANY of pipeline status, a job registry, or the
// log file tail changes. Replaces the 5 polling loops the UI used to
// run (StatusPanel 1Hz, LogTail 2Hz, JobPanel 400ms, KnnBuildModal
// 1Hz, BackfillExifModal 1Hz).
export interface EventSnapshot {
  pipeline: PipelineStatus;
  plants_jobs: Record<string, JobSnapshot>;
  cluster_album_jobs: Record<string, JobSnapshot>;
  kingdom_jobs: Record<string, JobSnapshot>;
  clustering_jobs: Record<string, JobSnapshot>;
  knn_build_jobs: Record<string, JobSnapshot>;
  backfill_exif_jobs: Record<string, JobSnapshot>;
  // Log tail is included in the snapshot so LogTail can drop its own
  // 2-second poll. `mtime` lets the client dedupe / know freshness.
  logs: { lines: string[]; mtime: number };
}