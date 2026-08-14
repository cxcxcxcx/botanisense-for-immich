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

import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import numpy as np

SCHEMA = """
CREATE TABLE IF NOT EXISTS assets (
  asset_id           TEXT PRIMARY KEY,
  original_file_name TEXT NOT NULL,
  original_url       TEXT NOT NULL,
  thumbnail_url      TEXT NOT NULL,
  cluster_id         INTEGER,
  first_seen_at      TEXT NOT NULL DEFAULT (datetime('now'))
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
) -> None:
    conn.execute(
        """
        INSERT INTO assets (asset_id, original_file_name, original_url, thumbnail_url)
        VALUES (?, ?, ?, ?)
        ON CONFLICT(asset_id) DO UPDATE SET
          original_file_name = excluded.original_file_name,
          original_url       = excluded.original_url,
          thumbnail_url      = excluded.thumbnail_url
        """,
        (asset_id, original_file_name, original_url, thumbnail_url),
    )


def all_assets(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        "SELECT asset_id, cluster_id FROM assets ORDER BY asset_id"
    ).fetchall()
    return [dict(r) for r in rows]


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


def cluster_summary(conn: sqlite3.Connection) -> list[dict]:
    rows = conn.execute(
        """
        SELECT cluster_id, COUNT(*) AS n
        FROM assets
        WHERE cluster_id IS NOT NULL
        GROUP BY cluster_id
        ORDER BY cluster_id ASC
        """,
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
        except Exception:
            top_sp = []
        try:
            top_ge = _json.loads(r["top_genus_json"])
        except Exception:
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
    except Exception:
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


def get_assets_in_cluster(
    conn: sqlite3.Connection, cluster_id: int
) -> list[str]:
    """Asset_ids belonging to a cluster, in alphabetical order (stable
    for bulk-add idempotency). Used by the per-cluster album sync
    worker to drive PUT /api/albums/{id}/assets."""
    rows = conn.execute(
        """
        SELECT asset_id FROM assets
        WHERE cluster_id = ?
        ORDER BY asset_id ASC
        """,
        (int(cluster_id),),
    ).fetchall()
    return [r["asset_id"] for r in rows]


def count_assets_in_cluster(
    conn: sqlite3.Connection, cluster_id: int
) -> int:
    """SQL COUNT(*) variant. Cheap; used by the per-cluster sync
    preview endpoint."""
    row = conn.execute(
        "SELECT COUNT(*) AS n FROM assets WHERE cluster_id = ?",
        (int(cluster_id),),
    ).fetchone()
    return int(row["n"]) if row else 0


# ---------- kingdom helpers ----------------------------------------------
# Kingdom-parameterized versions of the Plantae-specific helpers that
# already exist (find_plantae_assets, get_album_meta). The kingdom
# parameter is bound to taxon_name in taxon_scores. The old Plantae
# functions become thin shims that delegate here.


def find_kingdom_assets(
    conn: sqlite3.Connection, *, kingdom: str, min_score: float
) -> list[str]:
    """Distinct asset_ids whose kingdom-rank score for `kingdom` is at
    least `min_score`. Mirrors the green-leg threshold on the card grid
    but parameterized by kingdom (Plantae / Fungi / Animalia / ...)."""
    rows = conn.execute(
        """
        SELECT DISTINCT asset_id
        FROM taxon_scores
        WHERE rank = 'kingdom' AND taxon_name = ? AND score >= ?
        """,
        (kingdom, float(min_score)),
    ).fetchall()
    return [r["asset_id"] for r in rows]


def count_kingdom_assets(
    conn: sqlite3.Connection, *, kingdom: str, min_score: float
) -> int:
    """Count-only variant of find_kingdom_assets for the preview
    endpoint (avoids materializing the asset_id list)."""
    row = conn.execute(
        """
        SELECT COUNT(DISTINCT asset_id) AS n
        FROM taxon_scores
        WHERE rank = 'kingdom' AND taxon_name = ? AND score >= ?
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


def find_plantae_assets(conn: sqlite3.Connection, *, min_score: float) -> list[str]:
    """Backward-compat shim — delegates to find_kingdom_assets with
    kingdom='Plantae'. Used by the existing Plants-album feature;
    new code should call find_kingdom_assets directly so it works for
    any kingdom (Fungi / Animalia / ...)."""
    return find_kingdom_assets(conn, kingdom="Plantae", min_score=min_score)


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
        except Exception:
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

    Cheap: for N=634 assets this is ~200K dot products in a single
    numpy matmul, well under a second.
    """
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
    # Embeddings are L2-normalized so cosine == dot product.
    sim = mat @ mat.T

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

    n = len(ids)
    for i in range(n):
        for j in range(i + 1, n):
            if sim[i, j] >= threshold:
                union(ids[i], ids[j])

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


def get_all_results(
    conn: sqlite3.Connection, *,
    page: int = 0,
    page_size: int = 60,
    cluster_id: int | None = None,
    species: str | None = None,
    genus: str | None = None,
    family: str | None = None,
) -> tuple[list[dict], int]:
    """Return (page_of_buckets, total) for every asset with predictions.

    The "all assets" view replaces the per-run view: an asset is included
    if it has any row in `predictions`, regardless of which runs it
    appeared in. Deduplication is automatic since we key by asset_id.

    Sort order: each asset's `rank_position` from its MOST RECENT run
    (so the user sees photos in the order Immich last returned them),
    then cluster_id, then filename for ties / assets never in a run.

    Optional filters narrow both the page slice and the `total` count:
    - cluster_id: assets.cluster_id == cluster_id
    - species/genus/family: predictions.position=0 (rank='species') has
      matching label. These are applied via JOIN so the filter is on the
      same prediction row used by the taxonomy pills.
    """
    # Build the WHERE / JOIN clauses dynamically. The taxonomy filters
    # need an INNER JOIN on predictions (position=0); the cluster_id
    # filter is a plain WHERE on assets.
    joins = ""
    where = ["a.asset_id IN (SELECT DISTINCT asset_id FROM predictions)"]
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

    where_sql = " AND ".join(where)

    count_sql = f"SELECT COUNT(*) FROM assets a {joins} WHERE {where_sql}"
    total = conn.execute(count_sql, params).fetchone()[0]
    if total == 0:
        return [], 0

    # Fetch ALL matching assets (no LIMIT/OFFSET) so we can apply
    # near-dup dedup BEFORE pagination. Pagination is done in Python
    # below on the deduped set.
    all_filtered = conn.execute(
        f"""
        SELECT a.asset_id, a.original_file_name, a.original_url,
               a.thumbnail_url, a.cluster_id,
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
        return [], 0

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

    # Apply dedup before pagination
    deduped_rows = [r for r in all_filtered if r["asset_id"] not in hidden]
    total = len(deduped_rows)
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
            "top_predictions": preds_by_asset.get(aid, []),
            "taxon_scores": taxon_by_asset.get(aid, {}),
            "near_dups": near_dups,
        })

    # Attach cached KNN results (if any) so the UI can render the KNN
    # block without a separate fetch. Auto-detects (version, k).
    enrich_with_knn(conn, buckets)

    return buckets, total


def get_global_taxonomy_summary(
    conn: sqlite3.Connection,
) -> dict[str, list[dict]]:
    """Aggregate species/genus/family counts across ALL assets that have
    predictions. Database-wide — does NOT depend on any run. Used by the
    pills in the global view so counts match the photos in the grid.
    """
    rows = conn.execute(
        """
        SELECT p.species, p.genus, p.family, COUNT(*) AS n
        FROM predictions p
        WHERE p.position = 0
        GROUP BY p.species, p.genus, p.family
        """,
    ).fetchall()

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