"""SQLite-backed cache for the Immich × BioCLIP pipeline.

Storage model
-------------
Most tables are pooled across runs — keyed by `asset_id` (Immich UUID)
with no `run_id` column. This is intentional and is what makes cached
re-runs near-instant: BioCLIP forward + YOLO-World inference are
skipped for any asset that already has predictions / detections / an
embedding stored, regardless of which run originally produced them.

Per-run tables (the only ones keyed by `run_id`):
  - `runs`             one row per pipeline invocation (query, params,
                       counts, timestamps, rank).
  - `search_results`   which assets appeared in which run, and in what
                       order (rank_position) — this is what the UI uses
                       to display results in Immich's smart-search order.
  - `text_scores`      optional per-run text→image scores (currently
                       populated only when explicitly requested).

To list results for a run, join `search_results` against the pooled
tables (see `get_run_results`). Deleting a run does not delete its
asset / prediction / embedding rows — those outlive the run.

Tables
------
assets           Immich asset metadata (id, filename, urls).
embeddings       One BioCLIP image embedding per asset (BLOB of float32).
predictions      Top-k predictions per (asset, rank, position).
clusters         Each asset's cluster id (assigned by k-means on the
                 embeddings).
detections       YOLO-World bounding boxes per asset (label, score, xywh).
runs             One row per pipeline invocation (query, started/finished,
                 n_new, n_cached).
search_results   Many-to-many: which assets appeared in which run, in what
                 order.

The cache is NOT versioned by model. There is one prediction set per
(asset, rank), and one embedding per asset. If the model changes, those
rows are overwritten on the next pass for that asset.
"""

from __future__ import annotations

import logging
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

_log = logging.getLogger(__name__)

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  asset_id           TEXT PRIMARY KEY,
  original_file_name TEXT NOT NULL,
  original_url       TEXT NOT NULL,
  thumbnail_url      TEXT NOT NULL,
  cluster_id         INTEGER,
  first_seen_at      TEXT NOT NULL DEFAULT (datetime('now')),
  -- EXIF capture time (dateTimeOriginal from Immich), used by the
  -- "newest first" sort chip. ISO 8601 string, nullable since not every
  -- photo has EXIF.
  captured_at        TEXT,
  -- GPS coordinates from EXIF (latitude, longitude). Both NULL when
  -- the asset has no GPS tag; treated as a single location when
  -- exactly one is set, so the UI should check both.
  latitude           REAL,
  longitude          REAL
);
CREATE INDEX IF NOT EXISTS idx_assets_cluster ON assets(cluster_id);

CREATE TABLE IF NOT EXISTS embeddings (
  asset_id    TEXT PRIMARY KEY REFERENCES assets(asset_id) ON DELETE CASCADE,
  dim         INTEGER NOT NULL,
  vector      BLOB NOT NULL,
  computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);

CREATE TABLE IF NOT EXISTS predictions (
  asset_id    TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
  rank        TEXT NOT NULL,
  position    INTEGER NOT NULL,
  species     TEXT NOT NULL,
  common_name TEXT NOT NULL DEFAULT '',
  score       REAL NOT NULL,
  kingdom     TEXT NOT NULL DEFAULT '',
  family      TEXT NOT NULL DEFAULT '',
  genus       TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (asset_id, rank, position)
);

CREATE TABLE IF NOT EXISTS detections (
  id         INTEGER PRIMARY KEY AUTOINCREMENT,
  asset_id   TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
  label      TEXT NOT NULL,
  score      REAL NOT NULL,
  x          REAL NOT NULL,    -- top-left x, normalized 0..1
  y          REAL NOT NULL,    -- top-left y, normalized 0..1
  w          REAL NOT NULL,    -- width,  normalized 0..1
  h          REAL NOT NULL,    -- height, normalized 0..1
  computed_at TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_detections_asset ON detections(asset_id, score DESC);

-- One row per (detection, rank, position): BioCLIP top-K run on the
-- cropped bbox (vs. the full-image top-K in `predictions`). Cascade-
-- deletes with the parent detection so re-running YOLO invalidates
-- the cache.
CREATE TABLE IF NOT EXISTS bbox_predictions (
  detection_id INTEGER NOT NULL REFERENCES detections(id) ON DELETE CASCADE,
  rank         TEXT NOT NULL,
  position     INTEGER NOT NULL,
  species      TEXT NOT NULL,
  common_name  TEXT NOT NULL DEFAULT '',
  score        REAL NOT NULL,
  kingdom      TEXT NOT NULL DEFAULT '',
  family       TEXT NOT NULL DEFAULT '',
  genus        TEXT NOT NULL DEFAULT '',
  PRIMARY KEY (detection_id, rank, position)
);
CREATE INDEX IF NOT EXISTS idx_bbox_preds_detection
  ON bbox_predictions(detection_id, score DESC);

-- One row per (asset, rank, taxon_name) — a slice of the BioCLIP
-- predict() output at a higher taxonomic level (e.g. KINGDOM). We use
-- this to surface, e.g., the Plantae vs Fungi confidence per image
-- without re-running inference.
CREATE TABLE IF NOT EXISTS taxon_scores (
  asset_id    TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
  rank        TEXT NOT NULL,
  taxon_name  TEXT NOT NULL,
  score       REAL NOT NULL,
  computed_at TEXT NOT NULL DEFAULT (datetime('now')),
  PRIMARY KEY (asset_id, rank, taxon_name)
);
CREATE INDEX IF NOT EXISTS idx_taxon_scores_asset
  ON taxon_scores(asset_id, rank);

CREATE TABLE IF NOT EXISTS runs (
  id          INTEGER PRIMARY KEY AUTOINCREMENT,
  query       TEXT NOT NULL,
  n_requested INTEGER NOT NULL,
  rank        TEXT NOT NULL DEFAULT '',
  top_k       INTEGER NOT NULL DEFAULT 0,
  n_clusters  INTEGER NOT NULL DEFAULT 0,
  started_at  TEXT NOT NULL DEFAULT (datetime('now')),
  finished_at TEXT,
  n_new       INTEGER NOT NULL DEFAULT 0,
  n_cached    INTEGER NOT NULL DEFAULT 0
);

CREATE TABLE IF NOT EXISTS search_results (
  run_id        INTEGER NOT NULL REFERENCES runs(id) ON DELETE CASCADE,
  asset_id      TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
  rank_position INTEGER NOT NULL,
  PRIMARY KEY (run_id, asset_id)
);
CREATE INDEX IF NOT EXISTS idx_search_results_run
  ON search_results(run_id, rank_position);
-- The correlated subquery in get_all_results filters by asset_id
-- and orders by run_id DESC; the PK is (run_id, asset_id) so this
-- query is O(N) per asset without a matching index. 19k × N
-- subqueries is the cause of a 60s first-hit on /api/results.
CREATE INDEX IF NOT EXISTS idx_search_results_asset
  ON search_results(asset_id, run_id DESC);

-- KNN reclassification results from a labeled reference inventory
-- (currently TreeOfLife-200M-Embeddings via knn.py). Stored SEPARATELY
-- from the zero-shot `predictions` so a re-run of the regular pipeline
-- never invalidates these rows. Cache key is (asset_id, index_version, k)
-- so a new inventory build (or a different K) gets its own row.
CREATE TABLE IF NOT EXISTS knn_predictions (
  asset_id      TEXT NOT NULL REFERENCES assets(asset_id) ON DELETE CASCADE,
  index_version TEXT NOT NULL,
  k             INTEGER NOT NULL,
  computed_at   TEXT NOT NULL DEFAULT (datetime('now')),
  top_species   TEXT NOT NULL,
  common_name   TEXT NOT NULL DEFAULT '',
  score         REAL NOT NULL,
  kingdom       TEXT NOT NULL DEFAULT '',
  family        TEXT NOT NULL DEFAULT '',
  genus         TEXT NOT NULL DEFAULT '',
  neighbors     TEXT NOT NULL,
  PRIMARY KEY (asset_id, index_version, k)
);
CREATE INDEX IF NOT EXISTS idx_knn_asset ON knn_predictions(asset_id);
CREATE INDEX IF NOT EXISTS idx_knn_version ON knn_predictions(index_version);

-- Human-readable cluster summaries, computed by compute_cluster_names()
-- from the top-1 prediction per asset (rank='species', position=0). Stored
-- once per cluster_id so the API can serve cluster pills ("Saguaro (17)"
-- instead of "cluster 173 (17)") without re-aggregating on every request.
-- Recomputed when the asset→cluster mapping shifts (after a pipeline run
-- invalidates this table — see invalidate_cluster_names()).
CREATE TABLE IF NOT EXISTS cluster_names (
  cluster_id      INTEGER PRIMARY KEY,
  name            TEXT NOT NULL,        -- human-readable label
  n_total         INTEGER NOT NULL,     -- assets in cluster when computed
  top_species_json TEXT NOT NULL DEFAULT '[]',  -- top-3 species + counts
  top_genus_json  TEXT NOT NULL DEFAULT '[]',   -- top-3 genera + counts
  computed_at     TEXT NOT NULL DEFAULT (datetime('now'))
);
-- Idempotency for album-management features (e.g., "tag Plants album").
-- The Plants-album button reads `plants_album_id` to know whether to reuse
-- or create an album, so a re-run of the button doesn't create duplicates.
CREATE TABLE IF NOT EXISTS album_meta (
  key          TEXT PRIMARY KEY,
  value_json   TEXT NOT NULL,
  updated_at   TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Pickled HDBSCAN clusterer, persisted across server restarts so the
-- incremental-assign path can call approximate_predict() without
-- retraining. Single row enforced by CHECK (id = 1). `n_assets_at_fit`
-- is what `should_refit()` compares against the current row count to
-- decide whether the corpus has grown enough to warrant a full refit.
CREATE TABLE IF NOT EXISTS clusterer_state (
  id                INTEGER PRIMARY KEY CHECK (id = 1),
  clusterer_blob    BLOB NOT NULL,
  min_cluster_size  INTEGER NOT NULL,
  n_assets_at_fit   INTEGER NOT NULL,
  n_clusters        INTEGER NOT NULL,
  metric            TEXT NOT NULL DEFAULT 'euclidean',
  selection_method  TEXT NOT NULL DEFAULT 'eom',
  fitted_at         TEXT NOT NULL DEFAULT (datetime('now'))
);

-- Audit trail of Hungarian decisions from refit_with_remap(). One row
-- per (old_cluster_id, new_cluster_id) pair, plus extras for `new` /
-- `gone` clusters and the noise row. Lets the UI / debugging answer
-- "what changed at the last refit?" without re-running the algorithm.
CREATE TABLE IF NOT EXISTS cluster_remap (
  id              INTEGER PRIMARY KEY AUTOINCREMENT,
  old_cluster_id  INTEGER NOT NULL,
  new_cluster_id  INTEGER NOT NULL,
  overlap_count   INTEGER NOT NULL,
  union_count     INTEGER NOT NULL,
  jaccard         REAL NOT NULL,
  kind            TEXT NOT NULL CHECK (kind IN ('match','merge','split','new','gone','noise')),
  refit_at        TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS idx_cluster_remap_at   ON cluster_remap(refit_at DESC);
CREATE INDEX IF NOT EXISTS idx_cluster_remap_old  ON cluster_remap(old_cluster_id);
CREATE INDEX IF NOT EXISTS idx_cluster_remap_new  ON cluster_remap(new_cluster_id);

-- Server-owned mapping cluster_id -> Immich album. PRIMARY KEY enforces
-- 1:1 (one cluster, at most one Immich album). No FK to assets.cluster_id
-- because cluster_ids are renumbered across refits; the reference is
-- logical, not enforced.
CREATE TABLE IF NOT EXISTS cluster_albums (
  cluster_id        INTEGER PRIMARY KEY,
  album_id          TEXT NOT NULL,
  album_url         TEXT NOT NULL,
  n_assets          INTEGER NOT NULL DEFAULT 0,
  reused_existing   INTEGER NOT NULL DEFAULT 0,
  created_at        TEXT NOT NULL DEFAULT (datetime('now')),
  last_synced_at    TEXT NOT NULL DEFAULT (datetime('now'))
);
"""


def init_db(path: str | Path) -> None:
    """Create the DB file (and parent dir) and apply the schema."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with connect(p) as conn:
        conn.executescript(SCHEMA)
        # Idempotent column migrations — `CREATE TABLE IF NOT EXISTS`
        # doesn't add columns to existing tables, so each new column
        # needs a separate ALTER that no-ops on re-run.
        for stmt, column in [
            ("ALTER TABLE assets ADD COLUMN captured_at TEXT", "captured_at"),
            ("ALTER TABLE assets ADD COLUMN latitude REAL", "latitude"),
            ("ALTER TABLE assets ADD COLUMN longitude REAL", "longitude"),
            # Person-detection flag: 0 = scanned, no person; 1 = scanned,
            # person found; NULL = not yet scanned (gate for the
            # pipeline's person stage). See pipeline.py:_stage_person.
            ("ALTER TABLE assets ADD COLUMN has_person INTEGER", "has_person"),
        ]:
            try:
                conn.execute(stmt)
            except sqlite3.OperationalError as e:
                # "duplicate column name: captured_at" is fine — the
                # migration already ran on a previous init.
                if "duplicate column" not in str(e):
                    raise
        # Partial index — only scanned rows are interesting for the
        # filter, NULL rows are the pipeline's "todo" set. Partial index
        # keeps it small even when most assets are NULL.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_assets_has_person "
            "ON assets(has_person) WHERE has_person IS NOT NULL"
        )
        conn.commit()


@contextmanager
def connect(path: str | Path):
    conn = sqlite3.connect(str(path), timeout=30.0)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        # 30 s busy-timeout: if another process holds the writer lock
        # (e.g. the web server mid-refresh), wait instead of failing
        # immediately with "database is locked".
        conn.execute("PRAGMA busy_timeout=30000")
        conn.row_factory = sqlite3.Row
        yield conn
        conn.commit()  # auto-commit on clean exit; rolls back on exception
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------- assets --------------------------------------------------------


def upsert_asset(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    original_file_name: str,
    original_url: str,
    thumbnail_url: str,
    captured_at: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> None:
    """Insert or refresh the asset row. EXIF fields (captured_at,
    latitude, longitude) are nullable; the pipeline passes them when
    Immich returns the asset detail and the standalone backfill pass
    updates them after-the-fact via update_asset_exif()."""
    conn.execute(
        """
        INSERT INTO assets (
          asset_id, original_file_name, original_url, thumbnail_url,
          captured_at, latitude, longitude
        )
        VALUES (?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
          original_file_name = excluded.original_file_name,
          original_url       = excluded.original_url,
          thumbnail_url      = excluded.thumbnail_url,
          captured_at        = excluded.captured_at,
          latitude           = excluded.latitude,
          longitude          = excluded.longitude
        """,
        (asset_id, original_file_name, original_url, thumbnail_url,
         captured_at, latitude, longitude),
    )


def asset_row_matches(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    original_file_name: str,
    original_url: str,
    thumbnail_url: str,
) -> bool:
    """Cheap pre-check for the no-op UPDATE case. Returns True when the
    asset row already exists AND every URL/filename field matches what
    the pipeline is about to write — so the caller can skip the
    `upsert_asset` round-trip entirely.

    The pipeline emits these fields with deterministic templates
    (`/api/assets/{id}/...`) so matches are the common case on a stable
    Immich installation. Mismatches still trigger the normal
    `upsert_asset` to refresh name/URLs.
    """
    row = conn.execute(
        """
        SELECT 1
          FROM assets
         WHERE asset_id = ?
           AND original_file_name = ?
           AND original_url = ?
           AND thumbnail_url = ?
         LIMIT 1
        """,
        (asset_id, original_file_name, original_url, thumbnail_url),
    ).fetchone()
    return row is not None


def bulk_asset_rows(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
) -> dict[str, dict]:
    """Bulk: returns {asset_id: {original_file_name, original_url,
    thumbnail_url, captured_at}} for the given asset_ids, in a single
    SELECT. Replaces the per-asset `asset_row_matches` check which
    was the second-largest cost in the search loop.
    """
    if not asset_ids:
        return {}
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"SELECT asset_id, original_file_name, original_url, "
        f"       thumbnail_url, captured_at "
        f"  FROM assets WHERE asset_id IN ({placeholders})",
        asset_ids,
    ).fetchall()
    return {
        r["asset_id"]: dict(r)
        for r in rows
    }


def bulk_assets_missing_captured_at(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
) -> set[str]:
    """Returns the subset of `asset_ids` whose `assets.captured_at` is
    NULL — i.e. the rows that need an EXIF fetch. Replaces the
    per-asset `asset_has_captured_at` check.
    """
    if not asset_ids:
        return set()
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"SELECT asset_id FROM assets "
        f"WHERE captured_at IS NULL AND asset_id IN ({placeholders})",
        asset_ids,
    ).fetchall()
    return {r["asset_id"] for r in rows}


def refresh_asset_urls(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    original_file_name: str,
    original_url: str,
    thumbnail_url: str,
) -> None:
    """Update only the URL/filename fields of an existing asset row.
    Leaves captured_at/latitude/longitude untouched. Used by the
    pipeline when Immich reuses an asset id with a different name or
    the base_url has changed since the first ingest. Contrast with
    `upsert_asset`, which writes the EXIF columns and would clobber
    them with NULL when called without EXIF kwargs."""
    conn.execute(
        """
        UPDATE assets
           SET original_file_name = ?,
               original_url       = ?,
               thumbnail_url      = ?
         WHERE asset_id = ?
        """,
        (original_file_name, original_url, thumbnail_url, asset_id),
    )


def asset_has_captured_at(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
) -> bool:
    """True iff the asset row exists with a non-null captured_at. The
    pipeline uses this to skip the `client.get_asset_metadata` HTTP
    round-trip on cache misses where EXIF was already captured by an
    earlier run (EXIF backfill, smart-search ingest, etc.).
    """
    row = conn.execute(
        "SELECT 1 FROM assets WHERE asset_id = ? AND captured_at IS NOT NULL LIMIT 1",
        (asset_id,),
    ).fetchone()
    return row is not None


def update_asset_exif(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    captured_at: str | None = None,
    latitude: float | None = None,
    longitude: float | None = None,
) -> None:
    """Backfill EXIF fields on an existing asset row. Used by the
    /api/assets/backfill-exif endpoint to populate columns for assets
    inserted before the EXIF pipeline was wired up. Only the provided
    fields are touched; pass None to leave them unchanged (SQLite keeps
    the existing value via the COALESCE-style ? assignment)."""
    conn.execute(
        """
        UPDATE assets
           SET captured_at = ?,
               latitude    = ?,
               longitude   = ?
         WHERE asset_id = ?
        """,
        (captured_at, latitude, longitude, asset_id),
    )


def all_assets(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT asset_id, cluster_id FROM assets ORDER BY asset_id"
    ).fetchall()
    return [dict(r) for r in rows]


def delete_assets_by_ids(
    conn: sqlite3.Connection, asset_ids: list[str]
) -> int:
    """Bulk-delete the listed asset_ids from `assets`. FK CASCADE handles
    the seven child tables (embeddings, predictions, detections,
    bbox_predictions, taxon_scores, knn_predictions, search_results) so
    a single statement cleans everything keyed by the asset.

    `PRAGMA foreign_keys=ON` is set on every connection via connect(),
    so the cascades fire. Returns the rowcount from the parent delete.
    """
    if not asset_ids:
        return 0
    placeholders = ",".join("?" * len(asset_ids))
    cur = conn.execute(
        f"DELETE FROM assets WHERE asset_id IN ({placeholders})",
        list(asset_ids),
    )
    conn.commit()
    return cur.rowcount


def all_embeddings(conn: sqlite3.Connection) -> tuple[list[str], np.ndarray]:
    """Load every (asset_id, vector) pair, returning aligned lists."""
    rows = conn.execute(
        "SELECT asset_id, vector FROM embeddings ORDER BY asset_id"
    ).fetchall()
    if not rows:
        return [], np.empty((0, 0), dtype="<f4")
    ids = [r["asset_id"] for r in rows]
    matrix = np.stack(
        [np.frombuffer(r["vector"], dtype="<f4").copy() for r in rows]
    )
    return ids, matrix


def find_library_neighbors(
    conn: sqlite3.Connection, *, asset_id: str, k: int = 12
) -> list[dict]:
    """Top-K most-similar OTHER assets in the local library by BioCLIP
    embedding. Returns [{asset_id, original_file_name, thumbnail_url,
    cluster_id, similarity}, ...] sorted by similarity desc.

    Vectors are L2-normalized by pybioclip, so cosine similarity equals
    the dot product. We materialize all embeddings (small N — ~12k) and
    use a single matrix multiply to score every candidate, then sort and
    pull top-K. Cosine similarity is rank-equivalent to euclidean in
    this regime, so either metric would pick the same neighbors.
    """
    ids, matrix = all_embeddings(conn)
    if matrix.size == 0 or asset_id not in ids:
        return []
    self_idx = ids.index(asset_id)
    # One matrix multiply gives sim to every asset at once.
    sims = matrix @ matrix[self_idx]  # shape (N,), all in [-1, 1]
    sims[self_idx] = -2.0  # mask self out of the top-K
    n = len(sims)
    if n <= 1:
        return []
    take = min(k, n - 1)
    # argpartition is O(N), partial-sort just the K we want.
    top = np.argpartition(-sims, take)[:take]
    top = top[np.argsort(-sims[top])]
    # Bulk-fetch metadata for the K neighbors (one IN-clause query).
    result_ids = [ids[i] for i in top]
    placeholders = ",".join("?" for _ in result_ids)
    rows = conn.execute(
        f"""
        SELECT asset_id, original_file_name, thumbnail_url, cluster_id
        FROM assets
        WHERE asset_id IN ({placeholders})
        """,
        result_ids,
    ).fetchall()
    by_aid = {r["asset_id"]: dict(r) for r in rows}
    out: list[dict] = []
    for i in top:
        aid = ids[i]
        meta = by_aid.get(aid)
        if meta is None:
            continue
        out.append({
            "asset_id": aid,
            "original_file_name": meta.get("original_file_name") or "",
            "thumbnail_url": meta.get("thumbnail_url") or "",
            "cluster_id": meta.get("cluster_id"),
            "similarity": float(sims[i]),
        })
        if len(out) >= k:
            break
    return out


def set_cluster_ids(
    conn: sqlite3.Connection, *, asset_to_cluster: dict[str, int]
) -> None:
    if not asset_to_cluster:
        return
    conn.executemany(
        "UPDATE assets SET cluster_id = ? WHERE asset_id = ?",
        [(cid, aid) for aid, cid in asset_to_cluster.items()],
    )


def cluster_summary(
    conn: sqlite3.Connection,
    *,
    plantae: str | None = None,
    plantae_green: float = 0.60,
    plantae_yellow: float = 0.35,
) -> list[dict]:
    """One SQL aggregate per cluster: count + representative sample + the
    most recent capture time in the cluster (lets the FilterCards UI
    sort by "latest" so clusters with recent photos bubble to the top).

    The representative is the asset in the cluster with the highest top-1
    BioCLIP score (ties broken by captured_at DESC then filename ASC).
    The representative pool is tier-scoped — when `plantae` is set, the
    sample comes from among the same assets counted in `n`, so a
    non-plant thumbnail doesn't bleed into a plants-only cluster card.

    `plantae` narrows both the count and the representative so card
    counts reconcile with the grid breadcrumb for the same facet.
    """
    # The query has 1 outer tier predicate + 3 inner tier predicates (one
    # per sample subquery). Each `?` placeholder needs its own bind,
    # even when the bound value is the same threshold. Build the SQL
    # first so we know how many placeholders; then build a params list
    # of matching length.
    tier_outer = _plantae_score_expr("a")
    tier_inner = _plantae_score_expr("a2")
    if plantae == "plant":
        outer_pred = f"{tier_outer} >= ?"
        inner_pred = f"{tier_inner} >= ?"
        # 1 outer + 3 inner = 4 placeholders, all bound to plantae_green
        bind_values = [plantae_green]
    elif plantae == "uncertain":
        outer_pred = f"{tier_outer} >= ? AND {tier_outer} < ?"
        inner_pred = f"{tier_inner} >= ? AND {tier_inner} < ?"
        # 2 outer + 6 inner = 8 placeholders
        bind_values = [plantae_yellow, plantae_green]
    elif plantae == "non_plant":
        outer_pred = f"{tier_outer} < ?"
        inner_pred = f"{tier_inner} < ?"
        # 1 outer + 3 inner = 4 placeholders
        bind_values = [plantae_yellow]
    else:
        outer_pred = "1=1"
        inner_pred = "1=1"
        bind_values = []
    # How many times does each tier predicate appear? 1× outer + 3× inner.
    # `bind_values` already contains one value per `?` in the predicate
    # (it was built to match the predicate), so just extend it once per
    # occurrence — do NOT multiply by `n_outer`/`n_inner`.
    params: list = list(bind_values) + list(bind_values) * 3

    # The sample subquery picks a single best asset over the same tier
    # using a LEFT JOIN on predictions so non-plant clusters (no
    # predictions) still get a thumbnail. `p.score IS NULL ASC` puts
    # prediction-less assets last; secondary sort by captured_at DESC
    # ranks recent photos first so a freshly-ingested non-plant surfaces
    # instead of an old one with no species label.
    rows = conn.execute(
        f"""
        SELECT
          a.cluster_id,
          COUNT(*) AS n,
          MAX(a.captured_at) AS latest_captured_at,
          (
            SELECT a2.asset_id
            FROM assets a2
            LEFT JOIN predictions p
              ON p.asset_id = a2.asset_id
             AND p.rank = 'species' AND p.position = 0
            WHERE a2.cluster_id = a.cluster_id
              AND {inner_pred}
            ORDER BY p.score IS NULL ASC, p.score DESC,
                     a2.captured_at DESC, a2.original_file_name ASC
            LIMIT 1
          ) AS sample_asset_id,
          (
            SELECT a2.thumbnail_url
            FROM assets a2
            LEFT JOIN predictions p
              ON p.asset_id = a2.asset_id
             AND p.rank = 'species' AND p.position = 0
            WHERE a2.cluster_id = a.cluster_id
              AND {inner_pred}
            ORDER BY p.score IS NULL ASC, p.score DESC,
                     a2.captured_at DESC, a2.original_file_name ASC
            LIMIT 1
          ) AS sample_thumbnail_url,
          (
            SELECT a2.original_file_name
            FROM assets a2
            LEFT JOIN predictions p
              ON p.asset_id = a2.asset_id
             AND p.rank = 'species' AND p.position = 0
            WHERE a2.cluster_id = a.cluster_id
              AND {inner_pred}
            ORDER BY p.score IS NULL ASC, p.score DESC,
                     a2.captured_at DESC, a2.original_file_name ASC
            LIMIT 1
          ) AS sample_original_file_name
        FROM assets a
        WHERE a.cluster_id IS NOT NULL AND {outer_pred}
        GROUP BY a.cluster_id
        ORDER BY a.cluster_id ASC
        """,
        params,
    ).fetchall()
    return [dict(r) for r in rows]


# ---------- cluster_names ------------------------------------------------


def compute_cluster_names(conn: sqlite3.Connection) -> dict[int, dict]:
    """Aggregate the top-1 species prediction per asset (rank='species',
    position=0) for every cluster and produce a short human-readable label.

    Returns {cluster_id: {"name", "n_total", "top_species": [...],
                           "top_genus": [...]}}.

    Heuristic (rules tried in priority order):
      1. If a 2nd genus has ≥ 25% share (and a 3rd has ≥ 20%) → join the
         top genera: "{g1} & {g2}" or "{g1}, {g2} & {g3}". Matches
         "Hydrangea & Pelargonium" and "Echeveria, Agave & Aloe".
      2. Else if the top genus has < 50% of cluster assets → "Mixed"
         (the cluster is noise / too diverse to label).
      3. Else if a single species covers ≥ 65% of the cluster → use its
         common_name (fallback to species). Matches "Saguaro",
         "Giant-rhubarb", "Camellia".
      4. Otherwise (single genus dominant but species diverse) → use the
         top genus name. Matches "Wisteria".

    Multi-genus (rule 1) is checked BEFORE the noise threshold (rule 2)
    so a 3-way succulent mix (Echeveria/Agave/Aloe at 0.33/0.26/0.23)
    gets a real label rather than being lumped into "Mixed".

    Only the top-1 prediction per asset is used — the bbox-cropped species
    predictions are not considered (they're often more specific and would
    disagree on the same photo).
    """
    rows = conn.execute(
        """
        SELECT a.cluster_id AS cid, p.species, p.common_name, p.genus
        FROM predictions p
        JOIN assets a ON a.asset_id = p.asset_id
        WHERE p.rank = 'species' AND p.position = 0
          AND a.cluster_id IS NOT NULL
        """,
    ).fetchall()

    # Group rows by cluster for single-pass aggregation.
    by_cluster: dict[int, list] = {}
    for r in rows:
        by_cluster.setdefault(r["cid"], []).append(r)

    out: dict[int, dict] = {}
    for cid, items in by_cluster.items():
        total = len(items)
        if total == 0:
            continue

        # Species counter — keep first-seen common_name for each species.
        species: dict[str, dict] = {}
        genus: dict[str, int] = {}
        for r in items:
            sp = r["species"]
            if sp:
                entry = species.setdefault(
                    sp, {"species": sp, "common_name": r["common_name"] or "", "count": 0}
                )
                entry["count"] += 1
            ge = r["genus"]
            if ge:
                genus[ge] = genus.get(ge, 0) + 1

        sp_sorted = sorted(species.values(), key=lambda d: -d["count"])
        ge_sorted = sorted(genus.items(), key=lambda kv: -kv[1])

        g1, g1_n = ge_sorted[0] if ge_sorted else ("", 0)
        g2, g2_n = ge_sorted[1] if len(ge_sorted) >= 2 else ("", 0)
        g3, g3_n = ge_sorted[2] if len(ge_sorted) >= 3 else ("", 0)

        g1_share = g1_n / total
        g2_share = g2_n / total
        g3_share = g3_n / total

        top_sp = sp_sorted[0] if sp_sorted else None
        sp_share = (top_sp["count"] / total) if top_sp else 0.0

        # Rule 1: multi-genus cluster — list top-2 (or top-3 if 3rd
        # matters). Checked BEFORE the noise threshold so a balanced
        # 3-way mix still gets a real label.
        if g2_share >= 0.25 and g3_share >= 0.20:
            name = f"{g1}, {g2} & {g3}"
        elif g2_share >= 0.25:
            name = f"{g1} & {g2}"
        # Rule 2: noise cluster — top genus doesn't even cover half.
        elif g1_share < 0.50:
            name = "Mixed"
        # Rule 3: single species dominant — use its common_name.
        elif sp_share >= 0.65 and top_sp:
            cn = top_sp["common_name"]
            name = cn if cn else top_sp["species"]
        # Rule 4: single genus dominant — use the genus name.
        else:
            name = g1

        out[cid] = {
            "name": name,
            "n_total": total,
            "top_species": sp_sorted[:3],
            "top_genus": [{"genus": g, "count": c} for g, c in ge_sorted[:3]],
        }
    return out


def save_cluster_names(conn: sqlite3.Connection, names: dict[int, dict]) -> None:
    """Persist the computed names to the cluster_names table.

    Replaces the entire table — cluster_ids are stable across re-runs of
    compute_cluster_names() but the cluster→asset mapping may shift, so
    a full refresh is simpler than a diff.

    Commits before returning so the next /api/clusters call sees the rows
    (the connect() contextmanager doesn't auto-commit on close).
    """
    import json as _json
    conn.execute("DELETE FROM cluster_names")
    for cid, info in names.items():
        conn.execute(
            """
            INSERT INTO cluster_names
              (cluster_id, name, n_total, top_species_json, top_genus_json)
            VALUES (?, ?, ?, ?, ?)
            """,
            (
                cid,
                info["name"],
                int(info["n_total"]),
                _json.dumps(info.get("top_species", [])),
                _json.dumps(info.get("top_genus", [])),
            ),
        )
    conn.commit()


def load_cluster_names(conn: sqlite3.Connection) -> dict[int, dict]:
    """Read every cluster_names row. Empty dict if none stored yet."""
    import json as _json
    rows = conn.execute(
        """
        SELECT cluster_id, name, n_total, top_species_json, top_genus_json,
               computed_at
        FROM cluster_names
        """,
    ).fetchall()
    out: dict[int, dict] = {}
    for r in rows:
        try:
            top_sp = _json.loads(r["top_species_json"])
        except Exception as e:
            _log.warning(
                "cluster %s: bad top_species_json (%s); defaulting to []",
                r["cluster_id"], e,
            )
            top_sp = []
        try:
            top_ge = _json.loads(r["top_genus_json"])
        except Exception as e:
            _log.warning(
                "cluster %s: bad top_genus_json (%s); defaulting to []",
                r["cluster_id"], e,
            )
            top_ge = []
        out[int(r["cluster_id"])] = {
            "name": r["name"],
            "n_total": int(r["n_total"]),
            "top_species": top_sp,
            "top_genus": top_ge,
            "computed_at": r["computed_at"],
        }
    return out


def invalidate_cluster_names(conn: sqlite3.Connection) -> None:
    """Drop cached cluster names so the next ensure_cluster_names() call
    recomputes from scratch. Called after every pipeline run — the
    asset→cluster mapping may have shifted even if cluster_ids are
    numerically stable."""
    conn.execute("DELETE FROM cluster_names")
    conn.commit()


# ---------- album_meta ---------------------------------------------------
# Generic key/value store for album-management state (e.g., the saved
# Plants-album id so subsequent clicks reuse the existing album instead
# of creating a duplicate). Values are JSON-encoded dicts.


def get_album_meta(conn: sqlite3.Connection, key: str) -> dict | None:
    """Read a value from album_meta by key. Returns the parsed dict, or
    None if the key isn't set yet.
    """
    import json as _json
    row = conn.execute(
        "SELECT value_json FROM album_meta WHERE key = ?", (key,)
    ).fetchone()
    if row is None:
        return None
    try:
        return _json.loads(row["value_json"])
    except Exception as e:
        _log.warning("album_meta key %r: bad value_json (%s)", key, e)
        return None


def set_album_meta(conn: sqlite3.Connection, key: str, value: dict) -> None:
    """Upsert a value into album_meta. Stored as JSON."""
    import json as _json
    conn.execute(
        """
        INSERT INTO album_meta (key, value_json, updated_at)
            VALUES (?, ?, datetime('now'))
        ON CONFLICT(key) DO UPDATE SET
            value_json = excluded.value_json,
            updated_at = excluded.updated_at
        """,
        (key, _json.dumps(value)),
    )
    conn.commit()


def ensure_cluster_names(conn: sqlite3.Connection) -> dict[int, dict]:
    """Return the cluster_names dict, recomputing if it's missing or stale.

    Stale = the current number of distinct cluster_ids in assets exceeds
    what's in cluster_names. This catches the case where a new cluster
    appears mid-pipeline (HDBSCAN often produces 1-2 more clusters per
    pass) but the existing rows are still valid for the old clusters.
    On any divergence we recompute the whole table (cheap — one SQL
    pass over predictions).
    """
    cur_clusters = conn.execute(
        "SELECT COUNT(DISTINCT cluster_id) FROM assets WHERE cluster_id IS NOT NULL"
    ).fetchone()[0]
    saved_clusters = conn.execute("SELECT COUNT(*) FROM cluster_names").fetchone()[0]
    if cur_clusters > saved_clusters:
        names = compute_cluster_names(conn)
        save_cluster_names(conn, names)
        return names
    return load_cluster_names(conn)


# ---------- clusterer_state ---------------------------------------------
# Pickled HDBSCAN clusterer persisted across server restarts so the
# incremental-assign path can call approximate_predict() without a full
# refit. The blob is written by clusters.save_clusterer(); the loader
# is in clusters.py so it can clear state on unpickle failure (HDBSCAN
# version mismatch) and fall back to a full refit.


def get_clusterer_state(conn: sqlite3.Connection) -> dict | None:
    """Read the singleton clusterer_state row. Returns None when no
    clusterer has ever been fit (fresh DB)."""
    row = conn.execute(
        """
        SELECT clusterer_blob, min_cluster_size, n_assets_at_fit, n_clusters,
               metric, selection_method, fitted_at
        FROM clusterer_state WHERE id = 1
        """
    ).fetchone()
    if row is None:
        return None
    return {
        "clusterer_blob": bytes(row["clusterer_blob"]),
        "min_cluster_size": int(row["min_cluster_size"]),
        "n_assets_at_fit": int(row["n_assets_at_fit"]),
        "n_clusters": int(row["n_clusters"]),
        "metric": row["metric"],
        "selection_method": row["selection_method"],
        "fitted_at": row["fitted_at"],
    }


def save_clusterer_state(
    conn: sqlite3.Connection,
    *,
    clusterer_blob: bytes,
    min_cluster_size: int,
    n_assets_at_fit: int,
    n_clusters: int,
    metric: str = "euclidean",
    selection_method: str = "eom",
) -> None:
    """UPSERT the singleton row. Caller passes the already-pickled blob
    (clusters.save_clusterer does the pickling so the storage layer
    doesn't need to import hdbscan)."""
    conn.execute(
        """
        INSERT INTO clusterer_state
          (id, clusterer_blob, min_cluster_size, n_assets_at_fit, n_clusters,
           metric, selection_method, fitted_at)
        VALUES (1, ?, ?, ?, ?, ?, ?, datetime('now'))
        ON CONFLICT(id) DO UPDATE SET
          clusterer_blob   = excluded.clusterer_blob,
          min_cluster_size = excluded.min_cluster_size,
          n_assets_at_fit  = excluded.n_assets_at_fit,
          n_clusters       = excluded.n_clusters,
          metric           = excluded.metric,
          selection_method = excluded.selection_method,
          fitted_at        = excluded.fitted_at
        """,
        (
            clusterer_blob,
            int(min_cluster_size),
            int(n_assets_at_fit),
            int(n_clusters),
            metric,
            selection_method,
        ),
    )
    conn.commit()


def clear_clusterer_state(conn: sqlite3.Connection) -> None:
    """Drop the clusterer singleton. Used by clusters.load_clusterer when
    the pickled blob fails to unpickle (HDBSCAN version mismatch); the
    caller then falls back to a full refit. Also exposed for tests."""
    conn.execute("DELETE FROM clusterer_state WHERE id = 1")
    conn.commit()


# ---------- cluster_remap ------------------------------------------------
# Audit trail of Hungarian decisions from refit_with_remap(). Each
# refit event appends rows; the UI / debugging surfaces them via
# recent_cluster_remap().


def insert_cluster_remap(
    conn: sqlite3.Connection, rows: Iterable[dict]
) -> None:
    """Bulk insert audit rows for one refit event. Each row must have
    keys: old_cluster_id (int), new_cluster_id (int), overlap_count (int),
    union_count (int), jaccard (float), kind (str in the CHECK set).
    refit_at is set by SQLite's default (datetime('now'))."""
    rows = list(rows)
    if not rows:
        return
    conn.executemany(
        """
        INSERT INTO cluster_remap
          (old_cluster_id, new_cluster_id, overlap_count, union_count,
           jaccard, kind)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        [
            (
                int(r["old_cluster_id"]),
                int(r["new_cluster_id"]),
                int(r["overlap_count"]),
                int(r["union_count"]),
                float(r["jaccard"]),
                str(r["kind"]),
            )
            for r in rows
        ],
    )
    conn.commit()


def recent_cluster_remap(
    conn: sqlite3.Connection, *, limit: int = 20
) -> list[dict]:
    """Most recent remap rows, ordered by id DESC (== refit_at DESC,
    since id is autoincrement and each refit writes a contiguous batch).
    Includes the `refit_at` so callers can group by event."""
    rows = conn.execute(
        """
        SELECT id, old_cluster_id, new_cluster_id, overlap_count,
               union_count, jaccard, kind, refit_at
        FROM cluster_remap
        ORDER BY id DESC
        LIMIT ?
        """,
        (int(limit),),
    ).fetchall()
    return [dict(r) for r in rows]


# ---------- cluster_albums -----------------------------------------------
# Server-owned cluster_id -> Immich album mapping. PRIMARY KEY on
# cluster_id enforces 1:1 (one cluster, at most one Immich album). The
# mapping is a logical reference to assets.cluster_id, not a foreign key,
# because cluster_ids get renumbered across refits.


def get_cluster_album(
    conn: sqlite3.Connection, cluster_id: int
) -> dict | None:
    """Read one mapping. None if no album is synced for this cluster."""
    row = conn.execute(
        """
        SELECT cluster_id, album_id, album_url, n_assets, reused_existing,
               created_at, last_synced_at
        FROM cluster_albums WHERE cluster_id = ?
        """,
        (int(cluster_id),),
    ).fetchone()
    return dict(row) if row else None


def set_cluster_album(
    conn: sqlite3.Connection,
    *,
    cluster_id: int,
    album_id: str,
    album_url: str,
    n_assets: int,
    reused: bool,
) -> None:
    """UPSERT one mapping. `reused` is a bool (stored as 0/1) so the UI
    can show "(reused existing album)" after the second click."""
    conn.execute(
        """
        INSERT INTO cluster_albums
          (cluster_id, album_id, album_url, n_assets, reused_existing,
           created_at, last_synced_at)
        VALUES (?, ?, ?, ?, ?, datetime('now'), datetime('now'))
        ON CONFLICT(cluster_id) DO UPDATE SET
          album_id       = excluded.album_id,
          album_url      = excluded.album_url,
          n_assets       = excluded.n_assets,
          reused_existing = excluded.reused_existing,
          last_synced_at = datetime('now')
        """,
        (int(cluster_id), album_id, album_url, int(n_assets), 1 if reused else 0),
    )
    conn.commit()


def delete_cluster_album(
    conn: sqlite3.Connection, cluster_id: int
) -> bool:
    """Drop the local mapping. Returns whether a row was removed. The
    Immich-side album is left alone; the caller decides whether to also
    call ImmichClient.delete_album() (delete_in_immich=true)."""
    cur = conn.execute(
        "DELETE FROM cluster_albums WHERE cluster_id = ?",
        (int(cluster_id),),
    )
    conn.commit()
    return cur.rowcount > 0


def list_cluster_albums(conn: sqlite3.Connection) -> list[dict]:
    """Every mapping, joined with cluster_names so the UI can render
    "Album for Saguaro (17)" without a second round-trip."""
    rows = conn.execute(
        """
        SELECT ca.cluster_id, ca.album_id, ca.album_url, ca.n_assets,
               ca.reused_existing, ca.created_at, ca.last_synced_at,
               cn.name AS cluster_name
        FROM cluster_albums ca
        LEFT JOIN cluster_names cn ON cn.cluster_id = ca.cluster_id
        ORDER BY ca.cluster_id ASC
        """
    ).fetchall()
    return [dict(r) for r in rows]


# People-filter SQL fragment. `None` means "no filter" (default).
# `0` means "without people" — treats unscanned (NULL) assets as
# without-people so the gate stays inclusive while a YOLO person
# backfill is still in progress (otherwise an unscanned asset would
# be silently dropped from "without people" → double-counting risk
# with the has_person=0 chip on the toolbar).
# `1` means "with people" — strict equality, since unscanned assets
# have no person evidence yet and shouldn't be inferred as "has".
#
# The fragment references `has_person` unqualified so it works whether
# the surrounding query uses `FROM assets a` (cluster pickers, joins)
# or `FROM assets` (cluster-kingdom variant). Same column either way.
_PEOPLE_FILTER_SQL = {
    None: "",
    0: " AND (has_person = 0 OR has_person IS NULL)",
    1: " AND has_person = 1",
}


def get_assets_in_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    *,
    people_filter: int | None = None,
) -> list[str]:
    """Asset_ids belonging to a cluster, in alphabetical order (stable
    for bulk-add idempotency). Used by the per-cluster album sync
    worker to drive PUT /api/albums/{id}/assets.

    `people_filter`:
      - None (default): no filter — every asset in the cluster
      - 0: only assets with no person detected (or unscanned)
      - 1: only assets with a detected person
    The unscanned-NULL case is treated as "without people" so the
    gate doesn't silently drop photos whose person stage hasn't run
    yet.
    """
    extra = _PEOPLE_FILTER_SQL[people_filter]
    rows = conn.execute(
        f"""
        SELECT asset_id FROM assets
        WHERE cluster_id = ?{extra}
        ORDER BY asset_id ASC
        """,
        (int(cluster_id),),
    ).fetchall()
    return [r["asset_id"] for r in rows]


def count_assets_in_cluster(
    conn: sqlite3.Connection,
    cluster_id: int,
    *,
    people_filter: int | None = None,
) -> int:
    """SQL COUNT(*) variant. Cheap; used by the per-cluster sync
    preview endpoint. Same `people_filter` semantics as
    `get_assets_in_cluster`."""
    extra = _PEOPLE_FILTER_SQL[people_filter]
    row = conn.execute(
        f"SELECT COUNT(*) AS n FROM assets WHERE cluster_id = ?{extra}",
        (int(cluster_id),),
    ).fetchone()
    return int(row["n"]) if row else 0


# ---------- kingdom helpers ----------------------------------------------
# Kingdom-parameterized versions of the Plantae-specific helpers that
# already exist (find_plantae_assets, get_album_meta). The kingdom
# parameter is bound to taxon_name in taxon_scores. The old Plantae
# functions become thin shims that delegate here.


def find_kingdom_assets(
    conn: sqlite3.Connection,
    *,
    kingdom: str,
    min_score: float,
    people_filter: int | None = None,
) -> list[str]:
    """Distinct asset_ids whose kingdom-rank score for `kingdom` is at
    least `min_score`. Mirrors the green-leg threshold on the card grid
    but parameterized by kingdom (Plantae / Fungi / Animalia / ...).

    `people_filter`:
      - None (default): no filter — every kingdom-qualified asset
      - 0: only assets with no person detected (or unscanned)
      - 1: only assets with a detected person
    `has_person` lives on `assets`, so a people-filter INNER JOINs to
    `assets`; assets absent from `assets` are still excluded by the
    outer WHERE on taxon_scores (FK CASCADE).
    """
    extra = _PEOPLE_FILTER_SQL[people_filter]
    join_assets = " JOIN assets a ON a.asset_id = t.asset_id" if extra else ""
    rows = conn.execute(
        f"""
        SELECT DISTINCT t.asset_id
        FROM taxon_scores t{join_assets}
        WHERE t.rank = 'kingdom' AND t.taxon_name = ? AND t.score >= ?
        {extra}
        """,
        (kingdom, float(min_score)),
    ).fetchall()
    return [r["asset_id"] for r in rows]


def count_kingdom_assets(
    conn: sqlite3.Connection,
    *,
    kingdom: str,
    min_score: float,
    people_filter: int | None = None,
) -> int:
    """Count-only variant of find_kingdom_assets for the preview
    endpoint (avoids materializing the asset_id list). Same
    `people_filter` semantics as `find_kingdom_assets`."""
    extra = _PEOPLE_FILTER_SQL[people_filter]
    join_assets = " JOIN assets a ON a.asset_id = t.asset_id" if extra else ""
    row = conn.execute(
        f"""
        SELECT COUNT(DISTINCT t.asset_id) AS n
        FROM taxon_scores t{join_assets}
        WHERE t.rank = 'kingdom' AND t.taxon_name = ? AND t.score >= ?
        {extra}
        """,
        (kingdom, float(min_score)),
    ).fetchone()
    return int(row["n"]) if row else 0


def get_kingdom_album_meta(
    conn: sqlite3.Connection, kingdom: str
) -> dict | None:
    """Read album_meta by kingdom name. Backed by the existing
    album_meta table (key = kingdom name)."""
    return get_album_meta(conn, kingdom)


def set_kingdom_album_meta(
    conn: sqlite3.Connection, kingdom: str, value: dict
) -> None:
    """Write album_meta by kingdom name."""
    set_album_meta(conn, kingdom, value)


# ---------- embeddings ---------------------------------------------------


def upsert_embedding(
    conn: sqlite3.Connection, *, asset_id: str, vector: np.ndarray
) -> None:
    arr = np.asarray(vector, dtype="<f4").reshape(-1)
    conn.execute(
        """
        INSERT INTO embeddings (asset_id, dim, vector)
        VALUES (?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
          vector      = excluded.vector,
          dim         = excluded.dim,
          computed_at = datetime('now')
        """,
        (asset_id, arr.size, arr.tobytes()),
    )


def fetch_embedding(
    conn: sqlite3.Connection, *, asset_id: str
) -> np.ndarray | None:
    row = conn.execute(
        "SELECT dim, vector FROM embeddings WHERE asset_id=?",
        (asset_id,),
    ).fetchone()
    if row is None:
        return None
    return np.frombuffer(row["vector"], dtype="<f4").copy()


def embeddings_have_asset(conn: sqlite3.Connection, asset_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM embeddings WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    return row is not None


# ---------- predictions --------------------------------------------------


def replace_predictions(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    rank: str,
    rows: Iterable[dict],
) -> None:
    """Delete any cached predictions for (asset, rank) and insert the new
    rows in `position` order. Overwrites unconditionally — there is no
    model versioning in this cache."""
    conn.execute(
        "DELETE FROM predictions WHERE asset_id=? AND rank=?",
        (asset_id, rank),
    )
    conn.executemany(
        """
        INSERT INTO predictions
          (asset_id, rank, position, species, common_name, score,
           kingdom, family, genus)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                asset_id,
                rank,
                i,
                r.get("species", "?"),
                r.get("common_name", ""),
                float(r.get("score", 0.0)),
                r.get("kingdom", ""),
                r.get("family", ""),
                r.get("genus", ""),
            )
            for i, r in enumerate(rows)
        ],
    )


def predictions_have_asset(
    conn: sqlite3.Connection, *, asset_id: str, rank: str, need: int
) -> bool:
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM predictions WHERE asset_id=? AND rank=?",
        (asset_id, rank),
    ).fetchone()
    return bool(row) and row["n"] >= need


def fetch_predictions(
    conn: sqlite3.Connection, *, asset_id: str, rank: str
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT position, species, common_name, score, kingdom, family, genus
        FROM predictions
        WHERE asset_id=? AND rank=?
        ORDER BY position ASC
        """,
        (asset_id, rank),
    ).fetchall()
    return [dict(r) for r in rows]


def bulk_fetch_prediction_counts(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
    rank: str,
) -> dict[str, int]:
    """Returns {asset_id: count} for the given asset_ids at the given
    rank. Used by the pipeline's stitch loop to decide which assets
    still need a BioCLIP predict call, replacing an N+1 of
    `fetch_predictions` + `len()`. Asset ids that don't appear in the
    result set are simply absent — callers should treat absent as 0.
    """
    if not asset_ids:
        return {}
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"SELECT asset_id, COUNT(*) AS n "
        f"  FROM predictions "
        f" WHERE rank = ? AND asset_id IN ({placeholders}) "
        f" GROUP BY asset_id",
        [rank] + asset_ids,
    ).fetchall()
    return {r["asset_id"]: r["n"] for r in rows}


# ---------- detections ---------------------------------------------------


def replace_detections(
    conn: sqlite3.Connection, *, asset_id: str, detections: Iterable[dict]
) -> list[int]:
    """`detections` is an iterable of dicts with keys
    `label, score, x, y, w, h` where x/y/w/h are normalized 0..1 and
    describe the top-left corner + size. Returns the new detection row
    IDs in insertion order so the caller can chain a per-detection
    follow-up (e.g. BioCLIP on each bbox crop) without re-querying."""
    conn.execute("DELETE FROM detections WHERE asset_id=?", (asset_id,))
    rows = [
        (
            asset_id,
            d["label"],
            float(d["score"]),
            float(d["x"]),
            float(d["y"]),
            float(d["w"]),
            float(d["h"]),
        )
        for d in detections
    ]
    if not rows:
        return []
    conn.executemany(
        """
        INSERT INTO detections (asset_id, label, score, x, y, w, h)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    # Re-SELECT in insertion order so we can pair detection_id <-> bbox
    # dict for the BioCLIP follow-up. rowid preserves insertion order
    # within a single connection.
    sel = conn.execute(
        """
        SELECT id, label, score, x, y, w, h
        FROM detections
        WHERE asset_id = ?
        ORDER BY rowid DESC
        LIMIT ?
        """,
        (asset_id, len(rows)),
    ).fetchall()
    # Re-order DESC -> ASC and match by structural equality.
    sel.reverse()
    out: list[int] = []
    used = set()
    for r_det in rows:
        for sel_row in sel:
            if sel_row["id"] in used:
                continue
            if (
                sel_row["label"] == r_det[1]
                and abs(sel_row["score"] - r_det[2]) < 1e-6
                and abs(sel_row["x"] - r_det[3]) < 1e-6
                and abs(sel_row["y"] - r_det[4]) < 1e-6
                and abs(sel_row["w"] - r_det[5]) < 1e-6
                and abs(sel_row["h"] - r_det[6]) < 1e-6
            ):
                out.append(int(sel_row["id"]))
                used.add(sel_row["id"])
                break
    return out


def detections_have_asset(conn: sqlite3.Connection, asset_id: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM detections WHERE asset_id = ?", (asset_id,)
    ).fetchone()
    return row is not None


# YOLO labels that count as "this image has a person in it". Kept in
# sync with the trailing entries of yolo.PLANT_LABELS — used by the
# pipeline's person stage to derive `assets.has_person` and by the
# frontend filter chip "has person". "hand" is deliberately NOT in this
# set — a hand reaching into frame shouldn't flip has_person=1.
PERSON_LABELS: frozenset[str] = frozenset({"person", "face", "human"})

# Hand label. Used by the suppression rule below — if a photo contains
# a hand detection, the person flag is suppressed UNLESS the photo also
# contains a person detection with substantially more body evidence
# (multiple person boxes, or face + person).
HAND_LABEL: str = "hand"

# Minimum YOLO confidence for a hand detection to suppress has_person.
# Set to YOLO's own conf floor (0.05) — any emission at all is enough
# evidence to consider the photo a hand-only photo. YOLO rarely emits
# `hand` on non-hand content, so this is safe; but the suppression is
# only applied when there's no face or 2+ person boxes (the existence
# of strong body evidence wins).
HAND_MIN_CONF: float = 0.05

# Minimum hand-box area (w*h, in proportion of image 0..1) for the
# hand-suppression rule to fire. Real-person photos have small hand
# boxes (~0.2% of image area) — only the wrist/hand pokes out of the
# body silhouette. Hand-only photos have large hand boxes (~4-12% of
# image area) — the whole arm/forearm is the silhouette. Threshold
# sits at 0.02 (2%) so the gap between 0.2% (incidental) and 4.4%
# (hand silhouette) is well-separated.
HAND_MIN_AREA: float = 0.02

# Minimum containment (fraction of hand-box area inside person-box
# area) for the hand to count as a suppression signal. 0.85 = "the
# hand box is essentially a sub-region of the person box" — the
# signature of the hand-silhouette false positive (YOLO labeled the
# same arm as both). Below 0.85 = "incidental hand near a person"
# (e.g. someone pointing or holding something — the hand sticks out
# of the body silhouette and is NOT contained in it).
HAND_CONTAINMENT: float = 0.85

# Minimum YOLO confidence for a person/face/human detection to flip
# `has_person=1`. YOLO's own conf floor is 0.05 (fine for plant crops
# because BioCLIP re-scores each box), but for the binary person flag
# there is no second-stage filter — sub-50% boxes are overwhelmingly
# noise (tiny edge slivers, textured foliage, etc.) and produce false
# positives like a 6.77%-conf "person" pinned to the top of a flower
# photo, or a 38% "person" label on a shrub corner. Matches
# BboxOverlay's display cutoff so every has_person=1 asset has at
# least one visible box in the asset detail modal. A person who's
# actually in frame almost always scores ≥0.5; the false-positive
# rate at this floor is near zero in practice.
PERSON_MIN_CONF: float = 0.50


def _box_iou(a: dict, b: dict) -> float:
    """Intersection-over-Union of two bbox dicts (x, y, w, h in 0..1).
    Used to detect when a `hand` box overlaps a `person` box — that's
    the signature of the hand-silhouette false positive (YOLO labeled
    the same region as both).
    """
    ax1, ay1, ax2, ay2 = a["x"], a["y"], a["x"] + a["w"], a["y"] + a["h"]
    bx1, by1, bx2, by2 = b["x"], b["y"], b["x"] + b["w"], b["y"] + b["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = a["w"] * a["h"] + b["w"] * b["h"] - inter
    return inter / union if union > 0 else 0.0


def _box_containment(small: dict, big: dict) -> float:
    """Fraction of `small`'s area that lies inside `big`. Returns 0 if
    they don't intersect. Used to detect hand-silhouette false
    positives where the hand box is essentially a sub-region of the
    person box (YOLO labeled the same arm as both). Real-person hand
    boxes typically stick out of the body, so containment is low.
    """
    ax1, ay1 = small["x"], small["y"]
    ax2, ay2 = ax1 + small["w"], ay1 + small["h"]
    bx1, by1 = big["x"], big["y"]
    bx2, by2 = bx1 + big["w"], by1 + big["h"]
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    small_area = small["w"] * small["h"]
    return inter / small_area if small_area > 0 else 0.0


def has_real_person(detections: list[dict]) -> bool:
    """Decide if YOLO's detections on a single photo constitute "has
    person" given the per-photo rules:

    1. Must have at least one person/face/human box ≥ PERSON_MIN_CONF
       (the existence-of-person test).
    2. Suppression: a `hand` box that is MOSTLY CONTAINED in a
       `person` box (containment ≥ HAND_CONTAINMENT) is the signature
       of a hand-silhouette false positive — YOLO labeled the same
       arm region as both, producing a tight hand-shaped box that
       fits entirely inside the looser person silhouette. In that
       case, the photo is hand-only and does NOT count as has_person
       UNLESS there's a separate face box ≥ PERSON_MIN_CONF OR a
       person box that DOESN'T have a hand contained in it.

    Why containment-based:
      - Real person photos may have small incidental hand boxes
        (someone pointing, holding a tool) where the hand sticks out
        of the body. Containment of hand in person is < 0.5 → no
        suppression.
      - Hand-only photos have YOLO emit a tight hand box that's a
        sub-region of the looser person silhouette for the same arm.
        Containment is ~1.0 → suppression.
      - IoU alone misses the rose-hand case where the person bbox is
        4× the hand bbox (IoU ~0.14 but containment ~1.0).
      - Tuned on the 18-label YOLO-World model (yolov8s-world.pt).
    """
    person_boxes = [
        d for d in detections
        if d["label"] in PERSON_LABELS
        and d["score"] >= PERSON_MIN_CONF
    ]
    if not person_boxes:
        return False
    hand_boxes = [
        d for d in detections
        if d["label"] == HAND_LABEL
        and d["score"] >= HAND_MIN_CONF
    ]
    if not hand_boxes:
        return True
    # Hand detected. The hand is a suppression signal only if it's
    # mostly contained in some person box — otherwise it's incidental
    # (a hand holding a price tag, pointing at a leaf, etc.).
    contained_pairs = [
        (h, p) for h in hand_boxes for p in person_boxes
        if _box_containment(h, p) >= HAND_CONTAINMENT
    ]
    if not contained_pairs:
        return True
    # Some hand is mostly contained in a person box. Look for body
    # evidence that argues the person is real and distinct from the
    # hand silhouette.
    has_face = any(
        d["label"] == "face" and d["score"] >= PERSON_MIN_CONF
        for d in detections
    )
    if has_face:
        return True
    # A person box that DOESN'T fully contain any hand box is a real
    # person distinct from the hand silhouette.
    has_separate_person = any(
        all(_box_containment(h, p) < HAND_CONTAINMENT for h in hand_boxes)
        for p in person_boxes
    )
    if has_separate_person:
        return True
    # No face, no separate person box → hand-only silhouette case.
    return False


def append_detections(
    conn: sqlite3.Connection, *, asset_id: str, detections: Iterable[dict]
) -> list[int]:
    """Insert detections WITHOUT deleting existing rows for the asset.

    Distinct from `replace_detections` (which wipes + reinserts all
    detections). Use this when you want to ADD new detections to an
    asset that may already have rows — for example, the person stage
    adds person/face/human boxes on top of existing plant boxes
    without re-running the (potentially expensive) plant YOLO pass.
    Crucially, leaving the existing rows alone keeps their FK
    CASCADE children (`bbox_predictions`) intact — re-running plant
    YOLO would wipe the bbox cache and force the bbox-bioclip stage
    to re-process every crop.

    Detection dict shape: `{label, score, x, y, w, h}` where x/y/w/h
    are normalized 0..1.

    Returns the new detection row IDs in insertion order so the caller
    can chain per-detection follow-ups. Empty list when no detections
    are provided.
    """
    rows = [
        (
            asset_id,
            d["label"],
            float(d["score"]),
            float(d["x"]),
            float(d["y"]),
            float(d["w"]),
            float(d["h"]),
        )
        for d in detections
    ]
    if not rows:
        return []
    conn.executemany(
        """
        INSERT INTO detections (asset_id, label, score, x, y, w, h)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    # Pair rowids (insertion order) with their assigned ids.
    sel = conn.execute(
        """
        SELECT id FROM detections
        WHERE asset_id = ? AND rowid >= last_insert_rowid() - ?
        ORDER BY rowid ASC
        """,
        (asset_id, len(rows) - 1),
    ).fetchall()
    return [int(r["id"]) for r in sel]


def get_assets_without_person_scan(
    conn: sqlite3.Connection,
    *,
    limit: int = 1000,
    after_first_seen_at: str | None = None,
    after_asset_id: str | None = None,
) -> list[tuple[str, str | None, str]]:
    """Return `(asset_id, original_file_name, first_seen_at)` for assets
    that haven't been scanned for person yet.

    Uses **cursor pagination**, not LIMIT/OFFSET — the previous
    offset-based version had a subtle bug: the same process was
    marking rows NULL→0/1 while the outer loop incremented the offset,
    so the offset would skip entries that had moved up in the
    shrinking NULL set. With ~19k assets and 500-row pages, this
    caused the loop to terminate early at ~9000 scans even though
    ~8800 unscanned rows remained. Cursor pagination uses a compound
    (first_seen_at, asset_id) key — pass back the last row's values
    on each call and the next call picks up right where we left off,
    regardless of how many rows the current call marked.

    Returns rows in `ORDER BY first_seen_at ASC, asset_id ASC` so
    re-running after an interruption always resumes deterministically
    — the same asset never gets scanned twice even if the cursor
    pointed at it before the previous run died.
    """
    if after_first_seen_at is None:
        rows = conn.execute(
            """
            SELECT asset_id, original_file_name, first_seen_at
            FROM assets
            WHERE has_person IS NULL
            ORDER BY first_seen_at ASC, asset_id ASC
            LIMIT ?
            """,
            (limit,),
        ).fetchall()
    else:
        # Compound key: skip past (after_ts, after_id). Both columns
        # are part of ORDER BY so this is a clean continuation even when
        # multiple rows share the same first_seen_at.
        rows = conn.execute(
            """
            SELECT asset_id, original_file_name, first_seen_at
            FROM assets
            WHERE has_person IS NULL
              AND (first_seen_at > ?
                   OR (first_seen_at = ? AND asset_id > ?))
            ORDER BY first_seen_at ASC, asset_id ASC
            LIMIT ?
            """,
            (after_first_seen_at, after_first_seen_at, after_asset_id, limit),
        ).fetchall()
    return [
        (r["asset_id"], r["original_file_name"], r["first_seen_at"])
        for r in rows
    ]


def count_assets_without_person_scan(conn: sqlite3.Connection) -> int:
    """How many assets still need to be scanned. Used by the pipeline
    log line at stage entry."""
    return conn.execute(
        "SELECT COUNT(*) FROM assets WHERE has_person IS NULL"
    ).fetchone()[0]


def mark_person_scanned(
    conn: sqlite3.Connection, *, asset_id: str, has_person: bool
) -> None:
    """Atomic flag-set after YOLO has scanned this asset for person.
    Setting `has_person` to 0 or 1 removes the row from the pipeline's
    "needs scan" set (the `IS NULL` gate), so a successful mark is
    idempotent against re-runs."""
    conn.execute(
        "UPDATE assets SET has_person = ? WHERE asset_id = ?",
        (1 if has_person else 0, asset_id),
    )


def fetch_detections(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    min_score: float = 0.0,
    rank: str = "species",
) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, label, score, x, y, w, h
        FROM detections
        WHERE asset_id = ? AND score >= ?
        ORDER BY score DESC
        """,
        (asset_id, min_score),
    ).fetchall()
    if not rows:
        return []
    # Attach bbox_predictions (BioCLIP on the crop) in one IN-clause
    # query so callers don't have to do the join themselves.
    det_ids = [r["id"] for r in rows]
    bbox = fetch_bbox_predictions(conn, detection_ids=det_ids, rank=rank)
    out = []
    for r in rows:
        d = dict(r)
        d["bbox_predictions"] = bbox.get(r["id"], [])
        out.append(d)
    return out


# ---------- plant attribution --------------------------------------------
# A small pure function that combines the whole-image BioCLIP top-K with
# the YOLO bbox / BioCLIP-on-crop predictions into a single "plants in
# this image" answer. The rule is described in the plan at
# ~/.claude/plans/sparkling-strolling-sparkle.md — short version:
#   - When the whole-image Plantae kingdom score is at or above
#     PLANTAE_FLOOR (default 0.10), the whole-image top-K species are
#     the attribution. The YOLO bbox is shown only when a YOLO crop
#     added a species the whole image missed.
#   - When the whole image is below PLANTAE_FLOOR, fall back to YOLO
#     crops whose top-1 BioCLIP species is Plantae (any score) OR any
#     BioCLIP species with score >= CROP_SCORE_FLOOR. The bbox is then
#     "justified" because there is no whole-image signal to anchor it.
#   - YOLO candidates below YOLO_BBOX_FLOOR (default 0.20) are dropped
#     entirely — they're usually spurious.
#
# The function is deliberately a pure transformer over already-fetched
# rows so it works on both the per-asset detail endpoint (which has the
# full detections payload) and the grid endpoint (which has only the
# whole-image top-K). Both call sites build the same dict shape so the
# frontend renders one consistent component.

# Default constants. Override per-call from app.py if needed.
_PLANTAE_FLOOR_DEFAULT = 0.10
_YOLO_BBOX_FLOOR_DEFAULT = 0.20
_CROP_SCORE_FLOOR_DEFAULT = 0.50
# Per-row minimum score for whole-image species to appear in the
# summary. BioCLIP's species-rank top-K is exhaustive (typically 5
# rows descending from a confident top-1 down to ~1%) — listing the
# tail as "plants in this image" is noise. 5% is a reasonable cutoff:
# a real second candidate on a 6k-species tree sits well above this,
# while plausible-but-wrong confusion barely reaches it.
_WHOLE_IMAGE_SPECIES_FLOOR_DEFAULT = 0.05
# Per-row minimum score for a YOLO crop to contribute. Looser than the
# whole-image floor because the bbox label already restricts the
# candidate to a plant-sized object, but still needs SOMETHING — a
# 0.03 crop means BioCLIP saw a uniform blob.
_YOLO_CROP_FLOOR_DEFAULT = 0.05


def _attribution_seed_whole_image(
    *,
    taxon_scores: dict,
    top_predictions: list[dict],
    plantae_floor: float,
    species_floor: float,
) -> list[dict]:
    """Return the whole-image rows if Plantae >= plantae_floor, else [].

    Each row is shaped like a PlantAttributionRow with source="whole_image"
    and no detection_id / yolo_score. The species order from the backend
    (BioCLIP top-K descending) is preserved. Two filters apply:
      - kingdom must be Plantae (the summary is plant-focused), and
      - score must be >= species_floor (drops the long tail of the
        top-K — typically 1-5% mass on candidates the model wasn't
        really considering).
    """
    plantae = (taxon_scores or {}).get("Plantae")
    if plantae is None or plantae < plantae_floor:
        return []
    rows: list[dict] = []
    for p in top_predictions or []:
        kingdom = (p.get("kingdom") or "").strip()
        if kingdom.lower() != "plantae":
            continue
        score = float(p.get("score", 0.0))
        if score < species_floor:
            continue
        rows.append(
            {
                "species": p.get("species", ""),
                "common_name": p.get("common_name") or None,
                "score": score,
                "kingdom": p.get("kingdom") or None,
                "family": p.get("family") or None,
                "genus": p.get("genus") or None,
                "source": "whole_image",
                "detection_id": None,
                "yolo_score": None,
            }
        )
    return rows


def _attribution_seed_yolo(
    *,
    taxon_scores: dict,
    detections: list[dict],
    yolo_bbox_floor: float,
    crop_score_floor: float,
    crop_floor: float,
) -> list[dict]:
    """Return YOLO+bbox-bioclip rows whose crop justifies attribution.

    A YOLO detection contributes its top-1 crop species when:
      - the YOLO box score is >= yolo_bbox_floor (drops noise), AND
      - the crop's top-1 BioCLIP species is Plantae (any score above
        crop_floor) OR has score >= crop_score_floor.

    Plantae-on-crop is more permissive than the whole-image floor (it
    uses crop_floor, default 0.05) because the bbox label already
    restricts the candidate to a plant-sized object — a 0.03 Plantae
    crop means BioCLIP saw a uniform blob, not a real plant, so we
    still drop it. The non-Plantae branch is gated on crop_score_floor
    (default 0.50) so generic YOLO boxes only contribute when the
    whole image is also strong plantae.
    """
    plantae = (taxon_scores or {}).get("Plantae")
    has_plantae = plantae is not None and plantae >= 0.10
    rows: list[dict] = []
    for d in detections or []:
        yolo_score = float(d.get("score", 0.0))
        if yolo_score < yolo_bbox_floor:
            continue
        bbox_preds = d.get("bbox_predictions") or []
        if not bbox_preds:
            continue
        top = bbox_preds[0]
        species = (top.get("species") or "").strip()
        if not species:
            continue
        crop_score = float(top.get("score", 0.0))
        kingdom = (top.get("kingdom") or "").strip()
        is_plantae_crop = kingdom.lower() == "plantae"
        # A crop is "justified" (i.e. the bbox is a real plant candidate)
        # when either:
        #   - the crop is Plantae AND its own score >= crop_floor (a
        #     real plant, even a weak one), OR
        #   - the whole image has Plantae context AND the crop itself
        #     is high-confidence (>= crop_score_floor). Whole-image
        #     Plantae alone isn't enough — without the bbox label being
        #     agreed to be a plant, it's just a generic YOLO box.
        # The crop_score >= crop_score_floor alone branch is
        # deliberately NOT included: a non-plant photo (e.g. person /
        # car) has no business contributing rows to a "plants in this
        # image" summary.
        justified = (
            (is_plantae_crop and crop_score >= crop_floor)
            or (has_plantae and crop_score >= crop_score_floor)
        )
        if not justified:
            continue
        rows.append(
            {
                "species": species,
                "common_name": top.get("common_name") or None,
                "score": crop_score,
                "kingdom": top.get("kingdom") or None,
                "family": top.get("family") or None,
                "genus": top.get("genus") or None,
                "source": "yolo",
                "detection_id": d.get("id"),
                "yolo_score": yolo_score,
            }
        )
    return rows


def _merge_attribution_rows(
    whole_rows: list[dict], yolo_rows: list[dict]
) -> list[dict]:
    """Merge whole-image + YOLO rows, deduping by species.

    Match key is case-insensitive species. Score is the max of the two
    contributing scores; source is the higher-priority of the two
    sources (`whole_image > mixed > yolo`). When the species came from
    both sources, the merged row keeps the detection_id / yolo_score
    from the YOLO row (so the bbox can be drawn) and is tagged
    source="mixed".
    """
    # Source priority — lower number wins when picking the tag.
    priority = {"whole_image": 0, "mixed": 1, "yolo": 2}

    by_key: dict[str, dict] = {}
    by_key_meta: dict[str, dict] = {}
    for r in whole_rows:
        key = (r.get("species") or "").strip().lower()
        if not key:
            continue
        by_key[key] = r
        by_key_meta[key] = {"sources": {"whole_image"}}
    for r in yolo_rows:
        key = (r.get("species") or "").strip().lower()
        if not key:
            continue
        existing = by_key.get(key)
        if existing is None:
            by_key[key] = r
            by_key_meta[key] = {"sources": {"yolo"}}
            continue
        # Both sources contributed. Keep the higher score for display
        # but tag as "mixed" and retain the YOLO detection_id so the
        # bbox is drawn.
        if r["score"] > existing["score"]:
            display_score = r["score"]
        else:
            display_score = existing["score"]
        merged = dict(existing)
        merged["score"] = display_score
        merged["source"] = "mixed"
        merged["detection_id"] = r.get("detection_id")
        merged["yolo_score"] = r.get("yolo_score")
        # Prefer non-empty taxonomy fields from the row that's contributing
        # to the display score.
        donor = r if r["score"] >= existing["score"] else existing
        for f in ("common_name", "kingdom", "family", "genus"):
            if not merged.get(f) and donor.get(f):
                merged[f] = donor[f]
        by_key[key] = merged
        by_key_meta[key]["sources"].add("yolo")

    # Stable order: highest score first; source priority breaks ties so
    # whole-image rows beat YOLO-only rows of the same score.
    merged_rows = list(by_key.values())
    merged_rows.sort(
        key=lambda x: (-float(x.get("score") or 0.0), priority.get(x.get("source", "yolo"), 99))
    )
    return merged_rows


def compute_plant_attribution(
    *,
    taxon_scores: dict | None,
    top_predictions: list[dict] | None,
    detections: list[dict] | None = None,
    plantae_floor: float = _PLANTAE_FLOOR_DEFAULT,
    yolo_bbox_floor: float = _YOLO_BBOX_FLOOR_DEFAULT,
    crop_score_floor: float = _CROP_SCORE_FLOOR_DEFAULT,
    whole_image_species_floor: float = _WHOLE_IMAGE_SPECIES_FLOOR_DEFAULT,
    yolo_crop_floor: float = _YOLO_CROP_FLOOR_DEFAULT,
) -> dict:
    """Combine whole-image + YOLO evidence into a single attribution.

    Pure function — no DB access. The DB-bound helpers (`fetch_taxon_scores`,
    `fetch_predictions`, `fetch_detections`) need to be called by the
    caller; this just joins their results.

    Returns a dict shaped like the backend `plant_attribution` field:
        {
          "source": "whole_image" | "yolo" | "mixed" | "none",
          "rows": [PlantAttributionRow, ...],
          "highlight_detection_ids": [int, ...],
          "plantae_score": float | None,
        }
    """
    taxon_scores = taxon_scores or {}
    top_predictions = top_predictions or []
    detections = detections or []
    plantae = taxon_scores.get("Plantae")

    whole_rows = _attribution_seed_whole_image(
        taxon_scores=taxon_scores,
        top_predictions=top_predictions,
        plantae_floor=plantae_floor,
        species_floor=whole_image_species_floor,
    )
    yolo_rows = _attribution_seed_yolo(
        taxon_scores=taxon_scores,
        detections=detections,
        yolo_bbox_floor=yolo_bbox_floor,
        crop_score_floor=crop_score_floor,
        crop_floor=yolo_crop_floor,
    )
    rows = _merge_attribution_rows(whole_rows, yolo_rows)

    # What the canvas should draw. A detection_id is highlighted when
    # ANY of its YOLO crops contributed to the rows (i.e. source was
    # yolo or mixed). Whole-image-only rows leave the canvas untouched.
    highlight_ids = sorted({
        r["detection_id"]
        for r in rows
        if r.get("detection_id") is not None
    })

    if whole_rows and yolo_rows:
        source = "mixed"
    elif whole_rows:
        source = "whole_image"
    elif yolo_rows:
        source = "yolo"
    else:
        source = "none"

    return {
        "source": source,
        "rows": rows,
        "highlight_detection_ids": highlight_ids,
        "plantae_score": float(plantae) if plantae is not None else None,
    }


# ---------- bbox_predictions --------------------------------------------


def replace_bbox_predictions(
    conn: sqlite3.Connection,
    *,
    detection_id: int,
    rank: str,
    rows: Iterable[dict],
) -> None:
    """Delete any cached bbox predictions for (detection, rank) and insert
    the new top-k rows in `position` order. No model versioning."""
    conn.execute(
        "DELETE FROM bbox_predictions WHERE detection_id=? AND rank=?",
        (detection_id, rank),
    )
    conn.executemany(
        """
        INSERT INTO bbox_predictions
          (detection_id, rank, position, species, common_name, score,
           kingdom, family, genus)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        [
            (
                detection_id,
                rank,
                i,
                r.get("species", "?"),
                r.get("common_name", ""),
                float(r.get("score", 0.0)),
                r.get("kingdom", ""),
                r.get("family", ""),
                r.get("genus", ""),
            )
            for i, r in enumerate(rows)
        ],
    )


def bbox_predictions_have_detection(
    conn: sqlite3.Connection, *, detection_id: int, rank: str, need: int
) -> bool:
    row = conn.execute(
        """
        SELECT COUNT(*) AS n FROM bbox_predictions
        WHERE detection_id=? AND rank=?
        """,
        (detection_id, rank),
    ).fetchone()
    return bool(row) and row["n"] >= need


def fetch_bbox_predictions(
    conn: sqlite3.Connection, *, detection_ids: list[int], rank: str
) -> dict[int, list[dict]]:
    """Bulk fetch: returns {detection_id: [top-k rows in position order]}."""
    if not detection_ids:
        return {}
    placeholders = ",".join("?" for _ in detection_ids)
    rows = conn.execute(
        f"""
        SELECT detection_id, position, species, common_name, score,
               kingdom, family, genus
        FROM bbox_predictions
        WHERE detection_id IN ({placeholders}) AND rank = ?
        ORDER BY detection_id ASC, position ASC
        """,
        [*detection_ids, rank],
    ).fetchall()
    out: dict[int, list[dict]] = {did: [] for did in detection_ids}
    for row in rows:
        out[row["detection_id"]].append(
            {
                "position": row["position"],
                "species": row["species"],
                "common_name": row["common_name"] or "",
                "score": row["score"],
                "kingdom": row["kingdom"] or "",
                "family": row["family"] or "",
                "genus": row["genus"] or "",
            }
        )
    return out


# ---------- taxon_scores -------------------------------------------------


def upsert_taxon_scores(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    rank: str,
    scores: dict[str, float],
) -> None:
    """scores: {taxon_name: score}. Replaces any prior rows for the
    (asset, rank) pair."""
    conn.execute(
        "DELETE FROM taxon_scores WHERE asset_id=? AND rank=?",
        (asset_id, rank),
    )
    if not scores:
        return
    conn.executemany(
        """
        INSERT INTO taxon_scores (asset_id, rank, taxon_name, score)
        VALUES (?, ?, ?, ?)
        """,
        [(asset_id, rank, name, float(s)) for name, s in scores.items()],
    )


def taxon_scores_have_asset(
    conn: sqlite3.Connection, *, asset_id: str, rank: str
) -> bool:
    row = conn.execute(
        "SELECT 1 FROM taxon_scores WHERE asset_id=? AND rank=? LIMIT 1",
        (asset_id, rank),
    ).fetchone()
    return row is not None


def bulk_taxon_scores_have_assets(
    conn: sqlite3.Connection, *, asset_ids: list[str], rank: str
) -> set[str]:
    """Returns the subset of `asset_ids` that have at least one row
    in `taxon_scores` for `rank`. One SELECT for the whole batch —
    replaces per-asset `taxon_scores_have_asset` which was the
    dominant cost (~30 s for 10k cached photos)."""
    if not asset_ids:
        return set()
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"SELECT DISTINCT asset_id FROM taxon_scores "
        f"WHERE rank=? AND asset_id IN ({placeholders})",
        [rank, *asset_ids],
    ).fetchall()
    return {r["asset_id"] for r in rows}


def bulk_fetch_taxon_scores(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
    rank: str,
) -> dict[str, dict[str, float]]:
    """Bulk: returns {asset_id: {taxon_name: score}} for all the
    given asset_ids in a single SELECT. Replaces per-asset
    `fetch_taxon_scores` calls. The earlier bulk version of the
    pipeline search loop confused `bulk_asset_rows` (assets table)
    with `fetch_taxon_scores` (taxon_scores table) — the assets
    table has no Plantae/Fungi columns, so every cached asset was
    scored as 0.0 and the search boundary logic stopped after the
    first batch. This helper fixes that.
    """
    if not asset_ids:
        return {}
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"""
        SELECT asset_id, taxon_name, score
          FROM taxon_scores
         WHERE rank = ? AND asset_id IN ({placeholders})
        """,
        [rank, *asset_ids],
    ).fetchall()
    out: dict[str, dict[str, float]] = {}
    for r in rows:
        out.setdefault(r["asset_id"], {})[r["taxon_name"]] = float(r["score"])
    return out


def fetch_taxon_scores(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
    rank: str,
) -> dict[str, dict[str, float]]:
    """Bulk: returns {asset_id: {taxon_name: score}}."""
    if not asset_ids:
        return {}
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"""
        SELECT asset_id, taxon_name, score
        FROM taxon_scores
        WHERE rank = ? AND asset_id IN ({placeholders})
        """,
        [rank, *asset_ids],
    ).fetchall()
    out: dict[str, dict[str, float]] = {aid: {} for aid in asset_ids}
    for r in rows:
        out[r["asset_id"]][r["taxon_name"]] = float(r["score"])
    return out


def find_plantae_assets(
    conn: sqlite3.Connection,
    *,
    min_score: float,
    people_filter: int | None = None,
) -> list[str]:
    """Backward-compat shim — delegates to find_kingdom_assets with
    kingdom='Plantae'. Used by the existing Plants-album feature;
    new code should call find_kingdom_assets directly so it works for
    any kingdom (Fungi / Animalia / ...). Accepts `people_filter` for
    the new sync-defaults UX; passed through unchanged."""
    return find_kingdom_assets(
        conn,
        kingdom="Plantae",
        min_score=min_score,
        people_filter=people_filter,
    )


# ---------- runs --------------------------------------------------------


def create_run(
    conn: sqlite3.Connection,
    *,
    query: str,
    n_requested: int,
    rank: str = "",
    top_k: int = 0,
) -> int:
    cur = conn.execute(
        """
        INSERT INTO runs (query, n_requested, rank, top_k)
        VALUES (?, ?, ?, ?)
        """,
        (query, n_requested, rank, top_k),
    )
    return int(cur.lastrowid)


def finish_run(
    conn: sqlite3.Connection, *, run_id: int, n_new: int, n_cached: int
) -> None:
    conn.execute(
        """
        UPDATE runs
        SET finished_at = datetime('now'),
            n_new       = ?,
            n_cached    = ?
        WHERE id = ?
        """,
        (n_new, n_cached, run_id),
    )


def list_runs(conn: sqlite3.Connection, *, limit: int = 50) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, query, n_requested, rank, top_k, n_clusters,
               started_at, finished_at, n_new, n_cached
        FROM runs
        ORDER BY id DESC
        LIMIT ?
        """,
        (limit,),
    ).fetchall()
    return [dict(r) for r in rows]


def get_run(conn: sqlite3.Connection, run_id: int) -> dict | None:
    row = conn.execute(
        """
        SELECT id, query, n_requested, rank, top_k, n_clusters,
               started_at, finished_at, n_new, n_cached
        FROM runs WHERE id = ?
        """,
        (run_id,),
    ).fetchone()
    return dict(row) if row else None


def latest_run(conn: sqlite3.Connection) -> dict | None:
    rows = list_runs(conn, limit=1)
    return rows[0] if rows else None


# ---------- knn_predictions ---------------------------------------------


def replace_knn_predictions(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    index_version: str,
    k: int,
    rows: Iterable[dict],
) -> None:
    """Delete any cached KNN rows for (asset, version, k) and insert the
    new ones. Caller passes a list with exactly one dict (the KNN result
    for this asset) — same shape regardless of K (top-1 consensus is
    stored as the row; the full neighbor list is in `neighbors` JSON).
    """
    conn.execute(
        "DELETE FROM knn_predictions WHERE asset_id=? AND index_version=? AND k=?",
        (asset_id, index_version, k),
    )
    for r in rows:
        # `neighbors` is a list of {uuid, species, common_name, sim} dicts;
        # serialize once for the JSON column.
        import json as _json
        neighbors_json = r.get("neighbors_json")
        if neighbors_json is None:
            neighbors_json = _json.dumps(r.get("neighbors", []))
        conn.execute(
            """
            INSERT INTO knn_predictions
              (asset_id, index_version, k, top_species, common_name,
               score, kingdom, family, genus, neighbors)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                asset_id,
                index_version,
                k,
                r.get("top_species") or "",
                r.get("common_name") or "",
                float(r.get("score", 0.0)),
                r.get("kingdom") or "",
                r.get("family") or "",
                r.get("genus") or "",
                neighbors_json,
            ),
        )


def fetch_knn_predictions(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
    index_version: str,
    k: int,
) -> dict[str, dict]:
    """Bulk fetch KNN results for a list of asset_ids. Returns
    {asset_id: {top_species, common_name, score, kingdom, family,
                genus, neighbors, computed_at}}. Missing rows are
    omitted from the result dict (caller should default-handle).
    """
    if not asset_ids:
        return {}
    import json as _json
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"""
        SELECT asset_id, top_species, common_name, score, kingdom,
               family, genus, neighbors, computed_at
        FROM knn_predictions
        WHERE index_version=? AND k=? AND asset_id IN ({placeholders})
        """,
        [index_version, k, *asset_ids],
    ).fetchall()
    out: dict[str, dict] = {}
    for r in rows:
        try:
            neighbors = _json.loads(r["neighbors"])
        except Exception as e:
            _log.warning(
                "knn %s: bad neighbors json (%s); defaulting to []",
                r["asset_id"], e,
            )
            neighbors = []
        out[r["asset_id"]] = {
            "top_species": r["top_species"] or None,
            "common_name": r["common_name"] or "",
            "score": float(r["score"]),
            "kingdom": r["kingdom"] or "",
            "family": r["family"] or "",
            "genus": r["genus"] or "",
            "neighbors": neighbors,
            "computed_at": r["computed_at"],
        }
    return out


def knn_predictions_have_asset(
    conn: sqlite3.Connection,
    *,
    asset_id: str,
    index_version: str,
    k: int,
) -> bool:
    """Cache check: is there already a KNN row for (asset, version, k)?"""
    row = conn.execute(
        """
        SELECT 1 FROM knn_predictions
        WHERE asset_id=? AND index_version=? AND k=? LIMIT 1
        """,
        (asset_id, index_version, k),
    ).fetchone()
    return row is not None


def list_knn_versions(conn: sqlite3.Connection) -> list[dict]:
    """Distinct (index_version, k) pairs in the cache, with row counts.
    Used by /api/knn-status to surface what's available."""
    rows = conn.execute(
        """
        SELECT index_version, k, COUNT(*) AS n,
               MIN(computed_at) AS first_at,
               MAX(computed_at) AS last_at
        FROM knn_predictions
        GROUP BY index_version, k
        ORDER BY last_at DESC
        """
    ).fetchall()
    return [dict(r) for r in rows]


def enrich_with_knn(
    conn: sqlite3.Connection,
    buckets: list[dict],
    *,
    index_version: str | None = None,
    k: int | None = None,
) -> None:
    """Attach a `knn` field to each bucket in-place if a KNN row exists
    for (asset, version, k). If `index_version` / `k` are None, the
    most-populated pair is auto-detected from the knn_predictions table
    (cheap — one GROUP BY query). Missing rows are left with no `knn`
    key, which the UI renders as the "Reclassify with KNN" empty state."""
    if not buckets:
        return
    if index_version is None or k is None:
        row = conn.execute(
            """
            SELECT index_version, k, COUNT(*) AS n
            FROM knn_predictions
            GROUP BY index_version, k
            ORDER BY n DESC, MAX(computed_at) DESC
            LIMIT 1
            """
        ).fetchone()
        if row is None:
            return
        if index_version is None:
            index_version = row["index_version"]
        if k is None:
            k = row["k"]
    asset_ids = [b["asset_id"] for b in buckets]
    rows = fetch_knn_predictions(
        conn, asset_ids=asset_ids, index_version=index_version, k=k
    )
    for b in buckets:
        if b["asset_id"] in rows:
            b["knn"] = rows[b["asset_id"]]


# ---------- near-dup groups ---------------------------------------------


# Cosine similarity >= this between two BioCLIP embeddings means the
# assets are visually (near-)identical — different bytes, same picture.
# 0.99 is a strict threshold tuned for BioCLIP-2 768-d L2-normalized
# vectors: identical-looking JPEGs round to ≥0.99 even when the bytes
# differ slightly (re-encoding, EXIF, etc.).
NEAR_DUP_THRESHOLD = 0.97  # cosine: 0.99 = exact dup, 0.97 ≈ same plant/slightly different angle


def compute_near_dup_groups(
    conn: sqlite3.Connection,
    *,
    asset_ids: list[str],
    threshold: float = NEAR_DUP_THRESHOLD,
) -> dict[str, list[str]]:
    """Return {asset_id: [other_asset_ids in same near-dup group]} for the
    given asset_ids. Groups are transitively closed: A~B and B~C put all
    three in one group. The returned lists do NOT include the asset
    itself; only its near-dup siblings.

    Memory-aware: computes the upper-triangle of `mat @ mat.T` in row
    chunks instead of materializing the full N×N matrix. The dense
    matmul was 1.4 GB at N=19k; the chunked path peaks at ~40 MB
    (chunk=512 × N × 4 bytes), and the `np.where` + bulk-union
    replaces the old Python double loop (~180M iterations at N=19k)
    with a single vectorized pass per chunk.
    """
    # Chunk size for the row-wise matmul. 512 rows × 19k cols × 4 B
    # ≈ 39 MB; tweak if your machine's tighter or looser on RAM.
    _CHUNK = 512

    if not asset_ids:
        return {}
    placeholders = ",".join("?" for _ in asset_ids)
    rows = conn.execute(
        f"SELECT asset_id, vector FROM embeddings "
        f"WHERE asset_id IN ({placeholders})",
        asset_ids,
    ).fetchall()
    if not rows:
        return {aid: [] for aid in asset_ids}
    ids = [r["asset_id"] for r in rows]
    mat = np.stack(
        [np.frombuffer(r["vector"], dtype="<f4").copy() for r in rows]
    )
    n = len(ids)
    if n < 2:
        return {aid: [] for aid in ids}

    # Union-Find with path compression.
    parent = {aid: aid for aid in ids}

    def find(x: str) -> str:
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:
            parent[x], x = root, parent[x]
        return root

    def union(x: str, y: str) -> None:
        px, py = find(x), find(y)
        if px != py:
            parent[px] = py

    # Chunked row matmul. Each block is (chunk, N) float32 — bounded
    # by _CHUNK. np.where surfaces all (i_local, j) pairs above the
    # threshold in one C-level pass; the upper-triangle mask drops
    # half of them (the other half was captured in an earlier chunk).
    for i in range(0, n, _CHUNK):
        sim_block = mat[i:i + _CHUNK] @ mat.T  # (chunk, N) float32
        rows_idx, cols_idx = np.where(sim_block >= threshold)
        i_global = i + rows_idx
        # Only union pairs where j > i_global (upper triangle).
        mask = cols_idx > i_global
        for ii, jj in zip(i_global[mask].tolist(), cols_idx[mask].tolist()):
            union(ids[ii], ids[jj])

    # Group by root.
    members: dict[str, list[str]] = {}
    for aid in ids:
        members.setdefault(find(aid), []).append(aid)

    # Each asset's near-dup list = everyone else in its group.
    out: dict[str, list[str]] = {aid: [] for aid in asset_ids}
    for group in members.values():
        if len(group) < 2:
            continue
        for aid in group:
            out[aid] = [other for other in group if other != aid]
    return out



# Module-level cache for the global near-dup map. Computed once across
# every embedded asset (~10k matmul O(N²) dot products). Invalidate by
# calling invalidate_global_near_dup_cache() after a pipeline run.
_GLOBAL_NEAR_DUP_CACHE: dict[str, list[str]] | None = None


def invalidate_global_near_dup_cache() -> None:
    global _GLOBAL_NEAR_DUP_CACHE
    _GLOBAL_NEAR_DUP_CACHE = None


def save_near_dup_cache(path: str | Path) -> None:
    """Pickle the current near-dup map to `path`. ~1 MB for 19k assets.
    Safe to call when the cache is populated; no-op if it's None (the
    pipeline-end warm should populate then save)."""
    if _GLOBAL_NEAR_DUP_CACHE is None:
        return
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    # Write to a temp file then rename — atomic, so a crash mid-write
    # doesn't leave a half-baked pickle that the loader would reject.
    tmp = p.with_suffix(p.suffix + ".tmp")
    import pickle
    with open(tmp, "wb") as f:
        pickle.dump(_GLOBAL_NEAR_DUP_CACHE, f, protocol=pickle.HIGHEST_PROTOCOL)
    tmp.replace(p)


def load_near_dup_cache(path: str | Path) -> dict[str, list[str]] | None:
    """Load a previously-saved near-dup map. Returns None if the file
    doesn't exist, is missing/wrong-version, or fails to unpickle (e.g.
    because the embeddings table changed)."""
    p = Path(path)
    if not p.exists():
        return None
    try:
        import pickle
        with open(p, "rb") as f:
            cache = pickle.load(f)
        # Sanity: dict[str, list[str]]
        if not isinstance(cache, dict):
            return None
        if any(not isinstance(v, list) for v in cache.values()):
            return None
        return cache
    except Exception:
        # Corrupt or mismatched pickle — treat as missing and let the
        # caller recompute.
        return None


def delete_near_dup_cache(path: str | Path) -> None:
    """Remove the on-disk cache. Called when the embeddings change so
    the next load falls back to a fresh compute."""
    p = Path(path)
    try:
        p.unlink()
    except FileNotFoundError:
        pass


def get_global_near_dup_groups(
    conn: sqlite3.Connection,
    *,
    threshold: float = NEAR_DUP_THRESHOLD,
) -> dict[str, list[str]]:
    """Compute (and cache) the corpus-wide near-dup map.

    Returns {asset_id: [sibling_asset_ids]} for every embedded asset.
    An asset never seen by compute_near_dup_groups gets an empty list
    (via the union-find result). This is the data the per-page results
    slice from.
    """
    global _GLOBAL_NEAR_DUP_CACHE
    if _GLOBAL_NEAR_DUP_CACHE is not None:
        return _GLOBAL_NEAR_DUP_CACHE

    # Build the corpus: every asset with an embedding.
    rows = conn.execute("SELECT asset_id FROM embeddings").fetchall()
    if not rows:
        _GLOBAL_NEAR_DUP_CACHE = {}
        return _GLOBAL_NEAR_DUP_CACHE

    asset_ids = [r["asset_id"] for r in rows]
    _GLOBAL_NEAR_DUP_CACHE = compute_near_dup_groups(
        conn, asset_ids=asset_ids, threshold=threshold
    )
    return _GLOBAL_NEAR_DUP_CACHE


# ---------- search_results ----------------------------------------------


def insert_search_results(
    conn: sqlite3.Connection,
    *,
    run_id: int,
    ordered_asset_ids: Iterable[str],
) -> None:
    conn.executemany(
        "INSERT INTO search_results (run_id, asset_id, rank_position) VALUES (?, ?, ?)",
        [(run_id, aid, i) for i, aid in enumerate(ordered_asset_ids)],
    )


def get_run_results(conn: sqlite3.Connection, *, run_id: int) -> list[dict]:
    """Join assets × predictions for the given run in a single SQL
    round-trip, ordered by the run's original search rank. Detections are
    fetched in a second query (joined to assets) and stitched in Python.
    The model_id/rank/top_k are read from the run row itself."""
    run = get_run(conn, run_id)
    if run is None:
        return []
    rank = run["rank"]
    top_k = run["top_k"] or 0

    rows = conn.execute(
        """
        SELECT sr.rank_position,
               a.asset_id,
               a.original_file_name,
               a.original_url,
               a.thumbnail_url,
               a.cluster_id,
               p.position  AS pred_pos,
               p.species   AS pred_species,
               p.common_name AS pred_common,
               p.score     AS pred_score,
               p.kingdom   AS pred_kingdom,
               p.family    AS pred_family,
               p.genus     AS pred_genus
        FROM search_results sr
        JOIN assets a ON a.asset_id = sr.asset_id
        LEFT JOIN predictions p
          ON p.asset_id = a.asset_id AND p.rank = ? AND p.position < ?
        WHERE sr.run_id = ?
        ORDER BY sr.rank_position ASC, p.position ASC
        """,
        (rank, top_k, run_id),
    ).fetchall()

    # Stitch the rows: one bucket per asset, preserving search order.
    by_asset: dict[str, dict] = {}
    order: list[str] = []
    for row in rows:
        aid = row["asset_id"]
        bucket = by_asset.get(aid)
        if bucket is None:
            bucket = {
                "asset_id": aid,
                "rank": row["rank_position"],   # 0-based Immich search rank
                "original_file_name": row["original_file_name"],
                "original_url": row["original_url"],
                "thumbnail_url": row["thumbnail_url"],
                "cluster_id": row["cluster_id"],
                "top_predictions": [],
                "taxon_scores": {},
            }
            by_asset[aid] = bucket
            order.append(aid)
        if row["pred_pos"] is not None:
            bucket["top_predictions"].append(
                {
                    "position": row["pred_pos"],
                    "species": row["pred_species"],
                    "common_name": row["pred_common"] or "",
                    "score": row["pred_score"],
                    "kingdom": row["pred_kingdom"] or "",
                    "family": row["pred_family"] or "",
                    "genus": row["pred_genus"] or "",
                }
            )

    # Detections in one query, indexed by asset. We also pull `id` so we
    # can attach bbox_predictions to each detection below.
    if by_asset:
        placeholders = ",".join("?" for _ in by_asset)
        det_rows = conn.execute(
            f"""
            SELECT id, asset_id, label, score, x, y, w, h
            FROM detections
            WHERE asset_id IN ({placeholders})
            ORDER BY asset_id, score DESC
            """,
            list(by_asset.keys()),
        ).fetchall()
        dets_by_asset: dict[str, list[dict]] = {aid: [] for aid in by_asset}
        all_det_ids: list[int] = []
        for row in det_rows:
            dets_by_asset[row["asset_id"]].append(
                {
                    "id": row["id"],
                    "label": row["label"],
                    "score": row["score"],
                    "x": row["x"],
                    "y": row["y"],
                    "w": row["w"],
                    "h": row["h"],
                    "bbox_predictions": [],
                }
            )
            all_det_ids.append(row["id"])
    else:
        dets_by_asset = {}
        all_det_ids = []

    # Bbox top-K, in one query, attached by detection_id.
    bbox_preds_by_det: dict[int, list[dict]] = {}
    if all_det_ids:
        bbox_preds_by_det = fetch_bbox_predictions(
            conn, detection_ids=all_det_ids, rank=rank
        )
    for aid in order:
        for d in dets_by_asset.get(aid, []):
            d["bbox_predictions"] = bbox_preds_by_det.get(d["id"], [])

    for aid in order:
        by_asset[aid]["detections"] = dets_by_asset.get(aid, [])

    # Near-duplicate groups: assets whose BioCLIP embeddings have
    # cosine similarity >= NEAR_DUP_THRESHOLD. Computed once per run
    # load (cheap — N=634 is ~200K dot products) and attached to each
    # result so the UI can fold the alternatives under one card.
    near_dup_groups = compute_near_dup_groups(
        conn, asset_ids=list(by_asset)
    )
    for aid, others in near_dup_groups.items():
        if aid in by_asset:
            by_asset[aid]["near_dups"] = [
                {
                    "asset_id": oid,
                    "original_file_name": by_asset[oid]["original_file_name"]
                    if oid in by_asset else "",
                    "thumbnail_url": by_asset[oid]["thumbnail_url"]
                    if oid in by_asset else "",
                }
                for oid in others
            ]
        else:
            by_asset[aid]["near_dups"] = []

    # Make sure every result has the key (even if no near-dups).
    for aid in by_asset:
        by_asset[aid].setdefault("near_dups", [])

    # Per-asset taxon-level scores. The kingdom-rank slice is the only
    # taxon rank we currently populate in the pipeline; the badge UI
    # reads `rank=kingdom` regardless of the run's primary rank.
    if by_asset:
        scores = fetch_taxon_scores(
            conn, asset_ids=list(by_asset), rank="kingdom"
        )
        for aid, bucket in by_asset.items():
            bucket["taxon_scores"] = scores.get(aid, {})

    # Attach cached KNN results (if any). enrich_with_knn mutates in place.
    enrich_with_knn(conn, list(by_asset.values()))

    return [by_asset[aid] for aid in order]


def get_run_taxonomy_summary(
    conn: sqlite3.Connection, *, run_id: int, rank_name: str | None = None
) -> dict[str, list[dict]]:
    """Aggregate per-asset taxonomy labels across an entire run.

    Returns {rank: [{name, n}, ...]} where rank is "species" / "genus" /
    "family". Counts reflect unique assets (one row per (asset, position 0)
    prediction). Used by the UI to render pill counts that don't change
    with the current pagination slice.

    `rank_name` is the run's primary rank (e.g. "species") — we filter
    predictions to that rank so genus/family values come from the SAME
    predictions used by get_run_results, not from a different BioCLIP
    rank.
    """
    run = get_run(conn, run_id)
    if run is None:
        return {"species": [], "genus": [], "family": []}
    pred_rank = rank_name or run["rank"]

    rows = conn.execute(
        """
        SELECT p.species, p.genus, p.family, COUNT(*) AS n
        FROM search_results sr
        JOIN assets a ON a.asset_id = sr.asset_id
        JOIN predictions p
          ON p.asset_id = a.asset_id AND p.rank = ? AND p.position = 0
        WHERE sr.run_id = ?
        GROUP BY p.species, p.genus, p.family
        """,
        (pred_rank, run_id),
    ).fetchall()

    # Pivot the rows into per-rank buckets. A row contributes to all three
    # ranks (species/genus/family) — the same asset has all three labels.
    species: dict[str, int] = {}
    genus: dict[str, int] = {}
    family: dict[str, int] = {}
    for r in rows:
        if r["species"]:
            species[r["species"]] = species.get(r["species"], 0) + r["n"]
        if r["genus"]:
            genus[r["genus"]] = genus.get(r["genus"], 0) + r["n"]
        if r["family"]:
            family[r["family"]] = family.get(r["family"], 0) + r["n"]

    def sorted_dict(d: dict[str, int]) -> list[dict]:
        return [
            {"name": k, "n": v}
            for k, v in sorted(d.items(), key=lambda kv: (-kv[1], kv[0]))
        ]

    return {
        "species": sorted_dict(species),
        "genus": sorted_dict(genus),
        "family": sorted_dict(family),
    }


# Plantae-kingdom scored expression. The correlated subquery returns at
# most one row per asset (PRIMARY KEY (asset_id, rank, taxon_name)),
# so it adds zero rows to the outer query. COALESCE catches the rare
# asset that has no Plantae score (verified: exactly 1 out of 19,304)
# and treats it as non_plant.
def _plantae_score_expr(asset_alias: str = "a") -> str:
    """Return the SQL expression that yields the Plantae kingdom score
    for the given asset table alias. Use `a` in the outer query and
    `a2` inside correlated subqueries that JOIN a different alias."""
    return (
        f"COALESCE((SELECT ts.score FROM taxon_scores ts "
        f"WHERE ts.asset_id = {asset_alias}.asset_id AND ts.rank = 'kingdom' "
        f"AND ts.taxon_name = 'Plantae'), 0)"
    )


_PLANTAE_SCORE = _plantae_score_expr("a")


def get_all_results(
    conn: sqlite3.Connection, *,
    page: int = 0,
    page_size: int = 60,
    cluster_id: int | None = None,
    species: str | None = None,
    genus: str | None = None,
    family: str | None = None,
    has_person: int | None = None,
    plantae: str | None = None,
    plantae_green: float = 0.60,
    plantae_yellow: float = 0.35,
) -> tuple[list[dict], int, int]:
    """Return (page_of_buckets, total_reps, total_assets).

    `total_assets` is the count of every file matching the filters
    (pre-dedup). `total_reps` is the count after collapsing near-dup
    siblings. The grid uses `total_assets` for the breadcrumb + pagination
    "total" so every photo is reflected, and `total_reps` for page-button
    count (pages of representative cards). The "all" tile in FilterCards
    reads `total_assets` so counts reconcile across cluster / genus / the
    photo grid.

    The base set is every asset in the DB. Previously this was restricted
    to assets with predictions, which silently dropped 4,830 non-plants
    (and 7 plants) from the grid even though they were counted on cluster
    cards. Species/genus/family filters still INNER JOIN predictions, so
    selecting a taxon still implies predictions.

    Sort order: each asset's `rank_position` from its MOST RECENT run
    (so the user sees photos in the order Immich last returned them),
    then cluster_id, then filename for ties / assets never in a run.

    Optional filters narrow both the page slice and the `total` count:
    - cluster_id: assets.cluster_id == cluster_id
    - species/genus/family: predictions.position=0 (rank='species') has
      matching label. These are applied via JOIN so the filter is on the
      same prediction row used by the taxonomy pills.
    - has_person: 0 (no person) or 1 (has person). Plain WHERE clause on
      the indexed `assets.has_person` column. NULL rows (not yet
      scanned) are excluded by both 0 and 1 — correct UX: don't claim
      "no person" until you've actually checked.
    - plantae: "plant" (≥plantae_green), "uncertain" (≥plantae_yellow
      and <plantae_green), "non_plant" (<plantae_yellow). Cached on
      every asset via taxon_scores(rank='kingdom', taxon_name='Plantae').
    """
    # Build the WHERE / JOIN clauses dynamically. The taxonomy filters
    # need an INNER JOIN on predictions (position=0); the cluster_id
    # filter is a plain WHERE on assets.
    joins = ""
    where = ["1=1"]
    params: list = []
    if cluster_id is not None:
        where.append("a.cluster_id = ?")
        params.append(cluster_id)
    if species is not None:
        joins += " JOIN predictions p_species ON p_species.asset_id = a.asset_id AND p_species.rank = 'species' AND p_species.position = 0"
        where.append("p_species.species = ?")
        params.append(species)
    if genus is not None:
        joins += " JOIN predictions p_genus ON p_genus.asset_id = a.asset_id AND p_genus.rank = 'species' AND p_genus.position = 0"
        where.append("p_genus.genus = ?")
        params.append(genus)
    if family is not None:
        joins += " JOIN predictions p_family ON p_family.asset_id = a.asset_id AND p_family.rank = 'species' AND p_family.position = 0"
        where.append("p_family.family = ?")
        params.append(family)
    if has_person is not None:
        # 0 or 1 — excludes unscanned (NULL) rows in both cases.
        where.append("a.has_person = ?")
        params.append(has_person)
    if plantae == "plant":
        where.append(f"{_PLANTAE_SCORE} >= ?")
        params.append(plantae_green)
    elif plantae == "uncertain":
        where.append(f"{_PLANTAE_SCORE} >= ? AND {_PLANTAE_SCORE} < ?")
        params.append(plantae_yellow)
        params.append(plantae_green)
    elif plantae == "non_plant":
        where.append(f"{_PLANTAE_SCORE} < ?")
        params.append(plantae_yellow)

    where_sql = " AND ".join(where)

    # Fetch ALL matching assets (no LIMIT/OFFSET) so we can apply
    # near-dup dedup BEFORE pagination. Pagination is done in Python
    # below on the deduped set.
    all_filtered = conn.execute(
        f"""
        SELECT a.asset_id, a.original_file_name, a.original_url,
               a.thumbnail_url, a.cluster_id,
               a.first_seen_at,
               a.captured_at, a.latitude, a.longitude,
               a.has_person,
               (SELECT sr.rank_position
                  FROM search_results sr
                  WHERE sr.asset_id = a.asset_id
                  ORDER BY sr.run_id DESC
                  LIMIT 1) AS latest_rank
        FROM assets a {joins}
        WHERE {where_sql}
        ORDER BY latest_rank IS NULL ASC,
                 latest_rank ASC,
                 a.cluster_id ASC,
                 a.original_file_name ASC
        """,
        params,
    ).fetchall()

    if not all_filtered:
        return [], 0, 0

    # Near-dup dedup: pick ONE representative per group of cosine>=0.97
    # siblings within this filtered set. Hidden members are excluded
    # from total + the page slice. Group detection is scoped to the
    # filtered set so a photo filtered OUT by cluster_id/species doesn't
    # "hide" a representative that IS in the filter.
    near_dup_map = get_global_near_dup_groups(conn)
    rank_lookup = {r["asset_id"]: r["latest_rank"] for r in all_filtered}
    fn_lookup = {r["asset_id"]: r["original_file_name"] for r in all_filtered}
    filtered_ids = set(rank_lookup.keys())
    seen: set[str] = set()
    hidden: set[str] = set()
    for aid in filtered_ids:
        if aid in seen:
            continue
        siblings = near_dup_map.get(aid, [])
        if not siblings:
            seen.add(aid)
            continue
        group = {aid, *siblings} & filtered_ids
        seen.update(group)
        if len(group) < 2:
            continue  # no real group in the filter
        # Pick the lowest-rank member, tiebreak by filename. None ranks
        # sort last so a never-seen-by-Immich photo doesn't win ties.
        rep = min(group, key=lambda a: (
            rank_lookup[a] if rank_lookup[a] is not None else 1_000_000,
            fn_lookup[a],
        ))
        for m in group:
            if m != rep:
                hidden.add(m)

    # Apply dedup before pagination. Two totals: total_assets is the
    # raw file count (the user-facing "all" number), total_reps is the
    # number of representative cards we'll paginate through.
    deduped_rows = [r for r in all_filtered if r["asset_id"] not in hidden]
    total_assets = len(all_filtered)
    total_reps = len(deduped_rows)
    start = page * page_size
    end = start + page_size
    rows = deduped_rows[start:end]
    asset_ids = [r["asset_id"] for r in rows]
    placeholders = ",".join("?" for _ in asset_ids)

    # Fetch predictions for this page only (top-K per asset).
    pred_rows = conn.execute(
        f"""
        SELECT asset_id, position, species, common_name, score,
               kingdom, family, genus
        FROM predictions
        WHERE asset_id IN ({placeholders})
        ORDER BY asset_id ASC, position ASC
        """,
        asset_ids,
    ).fetchall()

    preds_by_asset: dict[str, list[dict]] = {}
    for p in pred_rows:
        preds_by_asset.setdefault(p["asset_id"], []).append({
            "position": p["position"],
            "species": p["species"],
            "common_name": p["common_name"],
            "score": p["score"],
            "kingdom": p["kingdom"],
            "family": p["family"],
            "genus": p["genus"],
        })

    # Kingdom-rank taxon scores (Plantae / Fungi) for the page's assets.
    score_rows = conn.execute(
        f"""
        SELECT asset_id, taxon_name, score
        FROM taxon_scores
        WHERE rank = 'kingdom' AND asset_id IN ({placeholders})
        """,
        asset_ids,
    ).fetchall()
    taxon_by_asset: dict[str, dict[str, float]] = {}
    for s in score_rows:
        taxon_by_asset.setdefault(s["asset_id"], {})[s["taxon_name"]] = float(s["score"])

    # Build sibling meta for every asset that passed the SQL filter
    # (so we can populate near_dups with hidden siblings too).
    sib_meta: dict[str, tuple[str, str]] = {}
    for r in all_filtered:
        sib_meta[r["asset_id"]] = (r["original_file_name"], r["thumbnail_url"])

    buckets: list[dict] = []
    for r in rows:  # page slice = representatives only
        aid = r["asset_id"]
        siblings = near_dup_map.get(aid, [])
        near_dups = []
        for sib in siblings:
            # Only show siblings that passed the SQL filter AND were
            # dedup-hidden. Siblings filtered out by cluster/species
            # are skipped (they're not in sib_meta anyway).
            if sib in hidden and sib in sib_meta:
                fn, thumb = sib_meta[sib]
                near_dups.append({
                    "asset_id": sib,
                    "original_file_name": fn,
                    "thumbnail_url": thumb,
                })
        buckets.append({
            "asset_id": aid,
            "rank": r["latest_rank"],  # Immich rank from most recent run
            "original_file_name": r["original_file_name"],
            "original_url": r["original_url"],
            "thumbnail_url": r["thumbnail_url"],
            "cluster_id": r["cluster_id"],
            # first_seen_at is the wall-clock time we first inserted the
            # asset row (set by `DEFAULT (datetime('now'))` and never
            # overwritten by upsert). Drives the "added ↓" sort on the
            # grid so users can see what was recently ingested.
            "first_seen_at": r["first_seen_at"],
            "captured_at": r["captured_at"],
            "latitude": r["latitude"],
            "longitude": r["longitude"],
            # has_person: 0/1/NULL — drives the "has person" filter
            # chip. NULL means the person stage hasn't scanned this
            # asset yet, so it appears in no filter and is hidden by
            # both has_person=0 and has_person=1.
            "has_person": r["has_person"],
            "top_predictions": preds_by_asset.get(aid, []),
            "taxon_scores": taxon_by_asset.get(aid, {}),
            # Slim attribution for the grid: whole-image seed only.
            # The grid payload doesn't join YOLO detections (the
            # bbox-bioclip query is expensive), so the grid card can't
            # show a YOLO source chip — it just falls back to the
            # whole-image top-K. The asset-detail endpoint carries the
            # full version with YOLO fallback.
            "plant_attribution": compute_plant_attribution(
                taxon_scores=taxon_by_asset.get(aid, {}),
                top_predictions=preds_by_asset.get(aid, []),
                detections=None,
            ),
            "near_dups": near_dups,
        })

    # Attach cached KNN results (if any) so the UI can render the KNN
    # block without a separate fetch. Auto-detects (version, k).
    enrich_with_knn(conn, buckets)

    return buckets, total_reps, total_assets


def get_global_taxonomy_summary(
    conn: sqlite3.Connection,
    *,
    plantae: str | None = None,
    plantae_green: float = 0.60,
    plantae_yellow: float = 0.35,
) -> dict[str, list[dict]]:
    """Aggregate species/genus/family counts across assets matching the
    given `plantae` tier (or all assets if None). Database-wide — does
    NOT depend on any run. Used by the filter cards in the global view
    so counts match the photos in the grid. Also picks a representative
    asset per (rank, name) using the same heuristic as cluster_summary
    (highest top-1 score, filename tie-break) so the cards have a
    thumbnail. Also tracks the most recent captured_at per (rank, name)
    so the FilterCards UI can sort by "latest" — taxa with recent photos
    bubble to the top.

    Counts include `p.rank = 'species'` (all present rows are species
    today — defensive against future ranks). The plantae tier is
    applied via a correlated subquery on `taxon_scores`. Non-plants
    have few/no predictions, so non-plant taxonomy cards are sparse —
    correct.
    """
    # Build the tier predicate. The query aliases `a` for assets.
    tier = _plantae_score_expr("a")
    params: list = []
    if plantae == "plant":
        tier_pred = f"{tier} >= ?"
        tier_params = [plantae_green]
    elif plantae == "uncertain":
        tier_pred = f"{tier} >= ? AND {tier} < ?"
        tier_params = [plantae_yellow, plantae_green]
    elif plantae == "non_plant":
        tier_pred = f"{tier} < ?"
        tier_params = [plantae_yellow]
    else:
        tier_pred = "1=1"
        tier_params = []
    params.extend(tier_params)

    rows = conn.execute(
        f"""
        SELECT p.species, p.genus, p.family, COUNT(*) AS n,
               MAX(a.captured_at) AS latest_captured_at
        FROM predictions p
        JOIN assets a ON a.asset_id = p.asset_id
        WHERE p.rank = 'species' AND p.position = 0 AND {tier_pred}
        GROUP BY p.species, p.genus, p.family
        """,
        params,
    ).fetchall()

    species: dict[str, dict] = {}
    genus: dict[str, dict] = {}
    family: dict[str, dict] = {}
    for r in rows:
        if r["species"]:
            species[r["species"]] = {"n": (species.get(r["species"]) or {"n": 0})["n"] + r["n"],
                                     "latest": _max_iso(species.get(r["species"], {}).get("latest"), r["latest_captured_at"])}
        if r["genus"]:
            genus[r["genus"]] = {"n": (genus.get(r["genus"]) or {"n": 0})["n"] + r["n"],
                                 "latest": _max_iso(genus.get(r["genus"], {}).get("latest"), r["latest_captured_at"])}
        if r["family"]:
            family[r["family"]] = {"n": (family.get(r["family"]) or {"n": 0})["n"] + r["n"],
                                   "latest": _max_iso(family.get(r["family"], {}).get("latest"), r["latest_captured_at"])}

    # Per (rank, name): the asset with the best (score DESC, filename ASC)
    # top-1 prediction — used as the card thumbnail.
    sample_rows = conn.execute(
        f"""
        SELECT p.species, p.genus, p.family,
               p.asset_id, p.score,
               a.original_file_name, a.thumbnail_url
        FROM predictions p
        JOIN assets a ON a.asset_id = p.asset_id
        WHERE p.rank = 'species' AND p.position = 0 AND {tier_pred}
        """,
        params,
    ).fetchall()

    # best[rank][name] = (score, filename, asset_id, thumbnail_url).
    # Pick highest score; tie-break by filename ASC for stability.
    best: dict[str, dict[str, tuple]] = {"species": {}, "genus": {}, "family": {}}
    for r in sample_rows:
        entry = (r["score"], r["original_file_name"] or "",
                 r["asset_id"], r["thumbnail_url"])
        for rank, col in (("species", "species"), ("genus", "genus"), ("family", "family")):
            name = r[col]
            if not name:
                continue
            prev = best[rank].get(name)
            if prev is None or entry > prev:
                best[rank][name] = entry

    def to_list(d: dict[str, dict], rank: str) -> list[dict]:
        return [
            {
                "name": k,
                "n": v["n"],
                "latest_captured_at": v["latest"],
                "sample_asset_id": (best[rank].get(k) or (None, None, None, None))[2],
                "sample_thumbnail_url": (best[rank].get(k) or (None, None, None, None))[3],
                "sample_original_file_name": (best[rank].get(k) or (None, None, None, None))[1],
            }
            for k, v in sorted(d.items(), key=lambda kv: (-kv[1]["n"], kv[0]))
        ]

    return {
        "species": to_list(species, "species"),
        "genus": to_list(genus, "genus"),
        "family": to_list(family, "family"),
    }


def _max_iso(a: str | None, b: str | None) -> str | None:
    """Return the lexicographically larger of two ISO 8601 strings
    (chronologically larger for UTC timestamps), or the non-None one if
    the other is None. Used to compute MAX(captured_at) across groups
    where SQLite's MAX skips NULLs but we also want to preserve the
    NULL-vs-non-NULL distinction."""
    if a is None:
        return b
    if b is None:
        return a
    return a if a > b else b