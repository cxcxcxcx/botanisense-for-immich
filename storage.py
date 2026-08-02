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


# ---------- near-dup groups ---------------------------------------------


# Cosine similarity >= this between two BioCLIP embeddings means the
# assets are visually (near-)identical — different bytes, same picture.
# 0.99 is a strict threshold tuned for BioCLIP-2 768-d L2-normalized
# vectors: identical-looking JPEGs round to ≥0.99 even when the bytes
# differ slightly (re-encoding, EXIF, etc.).
NEAR_DUP_THRESHOLD = 0.99


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

    return [by_asset[aid] for aid in order]