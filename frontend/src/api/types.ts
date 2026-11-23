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

// "Plants in this image" attribution summary. One block per asset that
// combines whole-image BioCLIP + YOLO bbox-bioclip into a single
// deduped species list. Computed server-side by
// storage.compute_plant_attribution and reused on the grid (whole-image
// only) and the asset-detail modal (whole-image + YOLO fallback).
export interface PlantAttributionRow {
  species: string;
  common_name: string | null;
  score: number;
  kingdom: string | null;
  family: string | null;
  genus: string | null;
  source: 'whole_image' | 'yolo' | 'mixed';
  // Set when source ∈ {yolo, mixed} — the detection row whose top-1
  // crop produced this row. Drawn on the canvas as a highlighted box.
  detection_id: number | null;
  yolo_score: number | null;
}

export interface PlantAttribution {
  source: 'whole_image' | 'yolo' | 'mixed' | 'none';
  rows: PlantAttributionRow[];
  highlight_detection_ids: number[];
  plantae_score: number | null;
}

export interface NearDup {
  asset_id: string;
  original_file_name: string | null;
}

export interface KnnNeighbor {
  species: string;
  common_name: string | null;
  sim: number;
  // TreeOfLife-200M record id for this reference occurrence.
  uuid?: string | null;
  // Source image URL of the reference photo. Present for gbif + eol
  // rows (~99.9% of the index), null for bioscan. Render it through
  // refThumbUrl() — never hotlink (leaks the user's IP, and many
  // hosts send no CORS headers).
  identifier?: string | null;
  source_dataset?: string | null;
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
  // 1 when the effective position came from a local inference
  // (location_guesses) rather than EXIF/Immich. Per-photo provenance;
  // the map marks these individual thumbnails, not whole clusters.
  location_guessed?: number | null;
  rank: number | null;
  top_predictions: Prediction[];
  taxon_scores: Record<string, number>;
  detections: Detection[];
  // "Plants in this image" summary — whole-image seed only for the
  // grid (YOLO bbox-bioclip is too expensive to ship with every row).
  // The asset-detail endpoint carries the full version with YOLO
  // fallback. Add backwards-compat: cached rows pre-feature may
  // expose this as undefined.
  plant_attribution?: PlantAttribution;
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
  // Present only when ?cluster_id= filters the grid: the cluster's
  // card name ("Camellia"), for the grid title. Absent/empty for
  // unnamed clusters and other filters.
  cluster_name?: string;
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
  // "Plants in this image" summary — full version with YOLO bbox
  // fallback. Renders as the top block of the asset-detail modal.
  plant_attribution: PlantAttribution;
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

// One entry of /api/clusters/related — a cluster whose BioCLIP centroid
// sits close to the one being viewed. `similarity` is centroid cosine
// similarity in [−1, 1]; anything the UI shows is typically > 0.7.
export interface RelatedCluster {
  cluster_id: number;
  similarity: number;
  name: string;
  n: number;
  sample_asset_id: string | null;
  sample_thumbnail_url: string | null;
  top_genus: { genus: string; count: number }[];
}

export interface RelatedClustersResponse {
  cluster_id: number;
  related: RelatedCluster[];
}

// One entry of /api/cluster-groups — a super-group of HDBSCAN clusters
// (complete-linkage over cluster centroids, cosine-distance cut). The
// Ungrouped bucket (clusters with no close siblings) uses group_id 0.
export interface ClusterGroup {
  group_id: number;
  name: string;
  // Photos across all member clusters.
  n: number;
  n_clusters: number;
  sample_asset_id: string | null;
  sample_thumbnail_url: string | null;
  cluster_ids: number[];
  // Display rows for the member strip — biggest clusters first, capped
  // at 12 server-side (the Ungrouped bucket has hundreds of members).
  members: { cluster_id: number; name: string; n: number }[];
}

export interface ClusterGroupsResponse {
  groups: ClusterGroup[];
  ungrouped: ClusterGroup | null;
}

export interface KnnStatus {
  inventory_dir: string;
  index_built: boolean;
  // True iff /api/knn/build has an active child process. UI suppresses
  // the "missing index" warning while this is set (the JobPanel shows
  // progress separately, and stacking both is confusing).
  building?: boolean;
  version: string | null;
  n_rows: number;
  dim: number;
  by_kingdom: Record<string, number>;
  k_default: number;
  cached_assets: number;
  cached_versions?: { index_version: string; k: number; n: number }[];
  // Result of knn.validate_index() at load time. false = the stored
  // vectors aren't L2-normalized, so neighbours are unreliable and the
  // index needs rebuilding. null = not loaded / probe inconclusive.
  healthy?: boolean | null;
  health_reason?: string;
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

// /api/health — pre-flight check surfaced by HealthBanner.
// Every field is optional at the type level because the endpoint is
// best-effort and may degrade gracefully; the banner falls back to
// "loading…" if the whole request fails.
// /api/config — non-secret config surfaced to the SPA.
// `immich_public_url` drives AssetDetail's "Open in Immich" button.
// `base_url` / `api_key_set` / `source` drive the connection form
// (Welcome screen + "Fix connection" modal): the URL is not a secret
// and is returned so the form can pre-fill it; the API key is only
// ever surfaced as a boolean — its value never leaves the server.
// `env_pinned` = env vars are set, so a UI-saved connection is
// shadowed until they're removed (the form says so).
export interface PublicConfig {
  immich_public_url: string;
  base_url: string;
  api_key_set: boolean;
  source: 'env' | 'db' | 'none';
  env_pinned: boolean;
  /** Number of box embeddings in this DB; 0 hides the Similar UI. */
  bbox_embeddings?: number;
}

// POST /api/settings/immich — save the Immich connection from the web
// UI. Blank api_key = keep the saved one. The response carries the
// fresh health payload so the form can show connected/broken without
// a second round trip.
export interface ImmichConnectionParams {
  base_url: string;
  api_key?: string;
}
export interface SaveConnectionResponse {
  saved: boolean;
  env_pinned: boolean;
  health: Health;
}

export interface HealthImmich {
  ok?: boolean;
  version?: string | null;
  error?: string | null;
}
export interface HealthApiKey {
  ok?: boolean;
  scope?: 'ok' | 'none' | string;
  error?: string | null;
}
export interface HealthKnn {
  built?: boolean;
  version?: string | null;
  path?: string;
  configured?: boolean;
}
export interface HealthBioclip {
  weights_present?: boolean;
  cache_path?: string;
}
export interface HealthDb {
  ok?: boolean;
  path?: string;
  size_bytes?: number;
}
// Which inference engine this instance runs (pipeline ENGINE env).
// 'teacher' = BioCLIP-2 (production default); 'mobile-*' = BioCLIP-Mobile
// package (compose.s0.yml A/B instance). Only non-teacher engines get a
// header badge so two side-by-side tabs are distinguishable.
export interface HealthEngine {
  id?: string;
  yolo_backend?: string;
  package?: string | null;
  code_rev?: string | null;
  error?: string | null;
}
export interface Health {
  immich?: HealthImmich;
  api_key?: HealthApiKey;
  knn?: HealthKnn;
  bioclip?: HealthBioclip;
  engine?: HealthEngine;
  db?: HealthDb;
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
  // ISO timestamp of the next scheduled run (UX-19b). null when
  // PIPELINE_CRON is unset on the server.
  next_run_at?: string | null;
}

export interface JobPreview {
  n_assets: number;
  n_chunks?: number;
  saved_album_id?: string | null;
  cluster_id?: number;
  kingdom?: string;
  min_score?: number;
  // Membership filter echoed by the preview so the UI can render the
  // count chip without re-deriving the value the server used.
  people_filter?: 0 | 1 | null;
}

export interface JobStart {
  job_id: string;
  n_assets?: number;
  n_chunks?: number;
  saved_album_id?: string | null;
  cluster_id?: number;
  kingdom?: string;
  min_score?: number;
  people_filter?: 0 | 1 | null;
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

export interface KingdomAlbumStatus {
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
  // /api/cluster-groups id — server resolves it to the member cluster
  // list (keeps group membership consistent with the cards).
  cluster_group?: number | undefined;
  species?: string | undefined;
  genus?: string | undefined;
  family?: string | undefined;
  has_person?: 0 | 1 | undefined;
  plantae?: 'plant' | 'non_plant' | 'uncertain' | undefined;
  // Manual opt-in to surface non-plant photos in the unfiltered
  // all-photo view. Default false (server-side). When true the grid
  // returns every photo regardless of Plantae score; when false the
  // grid scopes to Plantae >= plantae_yellow (same boundary as the
  // HDBSCAN cluster policy).
  include_non_plants?: boolean | undefined;
  // Map click-through: keep only assets within `near_radius` metres of
  // (near_lat, near_lng). All three must be present for the filter to
  // apply; the map view passes a cluster's centroid + its `radius_m`.
  near_lat?: number | undefined;
  near_lng?: number | undefined;
  near_radius?: number | undefined;
  [k: string]: string | number | boolean | undefined;
}

// --- Map view ------------------------------------------------------------
// One place from /api/map/clusters: geotagged photos grouped by distance
// alone (see map_clusters.py). `radius_m` is the farthest member from the
// centroid (+5 m), and is what the UI sends back as `near_radius` so a
// marker click opens exactly that place's photos.
export interface MapCluster {
  id: string;
  latitude: number;
  longitude: number;
  count: number;
  radius_m: number;
  sample_asset_id: string | null;
  // ISO capture timestamps (min/max among members), null when no member
  // has EXIF time.
  start: string | null;
  end: string | null;
  // Temporal runs (gaps > 7 days) inside this place — "3 visits".
  visits: number;
  // Photos in this place whose coordinate was inferred locally from
  // temporal neighbours (location_guesses), not read from EXIF.
  guessed_count: number;
}

export interface MapClustersResponse {
  granularity: 'fine' | 'wide' | 'region';
  radius_m: number;
  // Photos currently plotted (places >= min_photos), and every photo with
  // GPS before that cut (single-photo places are a toggle in the UI).
  photo_count: number;
  located_photos: number;
  // Photos whose plotted coordinate came from location_guesses.
  guessed_photos: number;
  cluster_count: number;
  singleton_places: number;
  unlocated: number;
  total_assets: number;
  clusters: MapCluster[];
}

export interface MapClustersQuery {
  granularity?: 'fine' | 'wide' | 'region';
  min_photos?: number;
  cluster_id?: number | undefined;
  cluster_group?: number | undefined;
  species?: string | undefined;
  genus?: string | undefined;
  family?: string | undefined;
  has_person?: 0 | 1 | undefined;
  plantae?: 'plant' | 'non_plant' | 'uncertain' | undefined;
  include_non_plants?: boolean | undefined;
  // Locally inferred coordinates: include (default), exclude, or only.
  guessed?: 'include' | 'exclude' | 'only';
  [k: string]: string | number | boolean | undefined;
}

export interface LocationGuessStats {
  total: number;
  high: number;
  medium: number;
  plants: number;
  last_applied: string | null;
}

export interface LocationGuessApplyResult extends LocationGuessStats {
  dry_run: boolean;
  computed: number;
  computed_high: number;
  computed_medium: number;
  window_minutes: number;
  max_span_m: number;
}

export interface RunParams {
  query?: string | null;
  n_samples?: number | null;
  top_k?: number | null;
  batch_size?: number | null;
  // Stop-criteria threshold expressed as a percent (0-100) of a batch
  // that must look like plants-or-fungi for the search to keep going.
  // Lower = stricter (the search stops sooner, gathering fewer photos).
  // null/missing → server default (currently 30%).
  stop_criteria_pct?: number | null;
}

// Sync-to-IMMICH defaults — persisted on the server (album_meta
// key="sync_defaults"). `people_filter` semantics: null = doesn't
// matter, 0 = without people, 1 = with people. The same shape is
// used by GET /api/settings/sync-defaults and PUT body.
export interface SyncDefaultsSlot {
  people_filter: 0 | 1 | null;
  min_score?: number;
}
export interface SyncDefaultsClustersSlot {
  people_filter: 0 | 1 | null;
}
export interface SyncDefaults {
  plants: SyncDefaultsSlot;
  fungi: SyncDefaultsSlot;
  clusters: SyncDefaultsClustersSlot;
  sync_all: SyncDefaultsClustersSlot;
}
export interface SyncDefaultsResponse {
  defaults: SyncDefaults;
  // True when the server seeded the row on first read (post-upgrade);
  // lets the UI show a one-time "Defaults initialised from
  // environment" hint instead of pretending the user set them.
  seeded: boolean;
}

// Bulk-remove cluster album mappings. Mirrors POST
// /api/album/cluster/remove. `cluster_ids=null` means "all
// currently synced clusters" — the default Run-page action.
//
// `delete_in_immich` was removed: cluster sync is beta, and the
// policy is "never leave orphans" — the Immich album always goes
// with the local mapping.
export interface BulkClusterRemoveParams {
  cluster_ids: number[] | null;
}
export interface BulkClusterRemoveError {
  cluster_id: number;
  error: string;
}
export interface BulkClusterRemoveResponse {
  removed: number;
  remote_deleted: number;
  errors: BulkClusterRemoveError[];
}

// SSE snapshot pushed by /api/events. One object per `data:` line,
// emitted whenever ANY of pipeline status, a job registry, or the
// log file tail changes. Replaces the 5 polling loops the UI used to
// run (StatusPanel 1Hz, LogTail 2Hz, JobPanel 400ms, KnnBuildModal
// 1Hz, BackfillExifModal 1Hz).
export interface EventSnapshot {
  pipeline: PipelineStatus;
  cluster_album_jobs: Record<string, JobSnapshot>;
  kingdom_jobs: Record<string, JobSnapshot>;
  clustering_jobs: Record<string, JobSnapshot>;
  knn_build_jobs: Record<string, JobSnapshot>;
  backfill_exif_jobs: Record<string, JobSnapshot>;
  // Log tail is included in the snapshot so LogTail can drop its own
  // 2-second poll. `mtime` lets the client dedupe / know freshness.
  logs: { lines: string[]; mtime: number };
}
// ---- Box-level similarity ------------------------------------------------

export interface BboxNeighbor {
  kind: 'box' | 'photo';
  detection_id?: number;
  asset_id: string;
  label?: string | null;
  yolo_score?: number | null;
  similarity: number;
  species?: string | null;
  common_name?: string | null;
  species_score?: number | null;
  family?: string | null;
  genus?: string | null;
  original_file_name?: string | null;
  thumb_url: string;
  // Number of crop embeddings for this source image. The backend
  // collapses all of them into this one row; >1 means "N crops seen,
  // best one shown". Photo rows can also carry the count.
  n_boxes?: number;
}

export interface BboxNeighborsResponse {
  query: {
    detection_id: number;
    asset_id: string;
    label: string;
    score: number;
    x: number;
    y: number;
    w: number;
    h: number;
    original_file_name: string;
    species?: string | null;
    common_name?: string | null;
    species_score?: number | null;
    thumb_url: string;
  };
  neighbors: BboxNeighbor[];
  model_version: string;
  include_photos: boolean;
}
