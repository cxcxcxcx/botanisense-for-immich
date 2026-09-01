#!/usr/bin/env python3
"""Analyze the main pipeline cache (data/state.db) and any sibling caches.

Run: .venv/bin/python scripts/analyze_cache.py

Why:  state.db is currently the only durable cache (508 MB on disk), but we
have no read-only surface that lets a human or admin browse what's in it
without running the pipeline. This script dumps:

- Per-table row count, on-disk size (incl. indexes), columns, existing
  indexes/foreign-keys.
- Distinct-value coverage that actually matters for browsing (species,
  families, kingdoms, runs, clusters, neighbors string length etc.).
- Freshness / coverage of last-N runs.
- Sibling caches that DON'T live in state.db: near_dup_cache.pkl, JSON
  status files in data/.
- Tells you which tables look "browsable as a UI" today vs. which are pure
  cache plumbing you'd want to filter out of any browse view.

Read-only. Never writes to state.db.
"""
from __future__ import annotations

import json
import os
import pickle
import sys
from pathlib import Path

# Make `import storage` work when invoked as `scripts/analyze_cache.py`.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import storage as storage_mod  # noqa: E402

DB = ROOT / "data" / "state.db"
DATA_DIR = ROOT / "data"


def humanize_bytes(n: float) -> str:
    units = ("B", "KB", "MB", "GB", "TB")
    for unit in units:
        if n < 1024 or unit == units[-1]:
            return f"{n:6.1f} {unit}"
        n /= 1024
    return f"{n:6.1f} TB"


def file_size(p: Path) -> int:
    return p.stat().st_size if p.exists() else 0


# ----- 1. The on-disk picture (before we even open SQLite) ----------------


def report_sibling_caches() -> None:
    print("=" * 78)
    print("1. Sibling caches on disk (NOT inside state.db)")
    print("=" * 78)

    rows = []
    for p in sorted(DATA_DIR.iterdir()):
        if p.name in {"state.db", "app.db", "immich_bioclip.db", "biobrick.db"}:
            continue
        if p.name.endswith((".pkl", ".json", ".log", ".status.json")) or p.is_dir():
            kind = "DIR" if p.is_dir() else p.suffix.lstrip(".") or "file"
            n = sum(1 for _ in p.iterdir()) if p.is_dir() else None
            rows.append((p.name, kind, file_size(p), n))
    for name, kind, size, n in rows[:60]:
        extra = f" ({n} entries)" if n is not None else ""
        print(f"  {name:54s}  {kind:6s}  {humanize_bytes(size)}{extra}")
    if len(rows) > 60:
        print(f"  … +{len(rows) - 60} more sibling files (logs/status/dirs)")


# ----- 2. The SQLite tables ----------------------------------------------


def list_tables(conn) -> list[str]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%' ORDER BY name"
    ).fetchall()
    return [r[0] for r in rows]


def table_columns(conn, name: str) -> list[tuple[str, str]]:
    return [
        (r[1], r[2]) for r in conn.execute(f"PRAGMA table_info({name})").fetchall()
    ]


def table_indexes(conn, name: str) -> list[str]:
    rows = conn.execute(f"PRAGMA index_list({name})").fetchall()
    out = []
    for r in rows:
        idx_name = r[1]
        cols = [c[2] for c in conn.execute(f"PRAGMA index_info({idx_name})").fetchall()]
        out.append(f"{idx_name}({','.join(cols)})")
    return out


def table_fks(conn, name: str) -> list[str]:
    return [r[0] for r in conn.execute(f"PRAGMA foreign_key_list({name})").fetchall()]


def count_or_none(conn, sql: str, *args) -> int | None:
    try:
        return conn.execute(sql, args).fetchone()[0]
    except Exception as e:  # pragma: no cover - any SQL issue, report it
        return f"ERR {e!r}"


def report_db_overview(conn) -> None:
    print()
    print("=" * 78)
    print("2. state.db overview")
    print("=" * 78)
    total_pages = conn.execute("PRAGMA page_count").fetchone()[0]
    page_size = conn.execute("PRAGMA page_size").fetchone()[0]
    freelist = conn.execute("PRAGMA freelist_count").fetchone()[0]
    size = file_size(DB)
    print(f"  File size        : {humanize_bytes(size)}")
    print(f"  Page size        : {page_size} B × {total_pages} pages → "
          f"{humanize_bytes(total_pages * page_size)} allocated")
    print(f"  Free pages       : {freelist}  (WAL checkpoint will reclaim)")
    journal = conn.execute("PRAGMA journal_mode").fetchone()[0]
    print(f"  Journal mode     : {journal}")


def report_each_table(conn) -> None:
    print()
    print("=" * 78)
    print("3. Tables (row count, columns, indices, FKs, distributional peek)")
    print("=" * 78)
    tables = list_tables(conn)
    for name in tables:
        cols = table_columns(conn, name)
        idx = table_indexes(conn, name)
        fks = table_fks(conn, name)
        n = count_or_none(conn, f"SELECT COUNT(*) FROM {name}")
        print(f"\n--- {name}  ({n} rows)")
        for c, t in cols:
            print(f"     {c:24s} {t}")
        if idx:
            for i in idx:
                print(f"   INDEX  {i}")
        if fks:
            for fk in fks:
                print(f"   FK     {fk}")


def report_distributions(conn) -> None:
    print()
    print("=" * 78)
    print("4. Coverage / freshness — what would a 'Browse' UI actually list?")
    print("=" * 78)

    def section(title):
        print(f"\n### {title}")

    section("assets")
    n = conn.execute("SELECT COUNT(*) FROM assets").fetchone()[0]
    have_exif = conn.execute("SELECT COUNT(*) FROM assets WHERE captured_at IS NOT NULL").fetchone()[0]
    have_gps = conn.execute("SELECT COUNT(*) FROM assets WHERE latitude IS NOT NULL AND longitude IS NOT NULL").fetchone()[0]
    have_person = conn.execute("SELECT COUNT(*) FROM assets WHERE has_person = 1").fetchone()[0]
    person_todo = conn.execute("SELECT COUNT(*) FROM assets WHERE has_person IS NULL").fetchone()[0]
    have_cluster = conn.execute("SELECT COUNT(*) FROM assets WHERE cluster_id IS NOT NULL").fetchone()[0]
    n_clusters = conn.execute("SELECT COUNT(DISTINCT cluster_id) FROM assets WHERE cluster_id IS NOT NULL").fetchone()[0]
    print(f"  total assets           : {n}")
    print(f"  with EXIF captured_at  : {have_exif}")
    print(f"  with GPS               : {have_gps}")
    print(f"  with cluster_id        : {have_cluster}  (clusters: {n_clusters})")
    print(f"  has_person=1           : {have_person}")
    print(f"  has_person NULL (todo) : {person_todo}")

    section("predictions (species-rank top-1)")
    row = conn.execute(
        """
        SELECT rank, COUNT(*) AS rows,
               COUNT(DISTINCT asset_id) AS assets,
               COUNT(DISTINCT species) AS species,
               COUNT(DISTINCT family)  AS families,
               COUNT(DISTINCT genus)   AS genera,
               COUNT(DISTINCT kingdom) AS kingdoms
        FROM predictions GROUP BY rank ORDER BY rank
        """
    ).fetchall()
    for r in row:
        print(f"  rank={r[0]:8s} rows={r[1]:>8}  assets={r[2]:>6}  species={r[3]:>6}  families={r[4]:>5}  genera={r[5]:>5}  kingdoms={r[6]}")

    section("top-10 species by prediction count (rank=species, position=0)")
    for r in conn.execute(
        """
        SELECT species, common_name, COUNT(*) AS hits,
               ROUND(AVG(score)*100,2) AS avg_pct
        FROM predictions WHERE rank='species' AND position=0
        GROUP BY species, common_name ORDER BY hits DESC LIMIT 10
        """
    ).fetchall():
        print(f"  {r[0]:32s}  '{r[1]}'  hits={r[2]:>5}  avg={r[3]:5.2f}%")

    section("taxon_scores (kingdom coverage)")
    for r in conn.execute(
        """
        SELECT rank, taxon_name, COUNT(*) AS hits,
               ROUND(AVG(score)*100,2) AS avg_pct
        FROM taxon_scores GROUP BY rank, taxon_name ORDER BY rank, hits DESC
        """
    ).fetchall()[:30]:
        print(f"  rank={r[0]:8s}  {r[1]:14s}  hits={r[2]:>5}  avg={r[3]:5.2f}%")

    section("runs (lifetime audit trail of pipeline invocations)")
    n_runs = conn.execute("SELECT COUNT(*) FROM runs").fetchone()[0]
    newest = conn.execute("SELECT id, query, started_at, finished_at, n_new, n_cached FROM runs ORDER BY id DESC LIMIT 5").fetchall()
    print(f"  total runs              : {n_runs}")
    print("  most recent 5 (id DESC):")
    for r in newest:
        print(f"    run={r[0]:>4}  q={r[1]!r:30s}  start={r[2]}  end={r[3]}  new={r[4]}  cached={r[5]}")

    section("cluster_names")
    n = conn.execute("SELECT COUNT(*) FROM cluster_names").fetchone()[0]
    recent = conn.execute("SELECT cluster_id, name, n_total, computed_at FROM cluster_names ORDER BY computed_at DESC LIMIT 8").fetchall()
    print(f"  total named clusters : {n}")
    for r in recent:
        print(f"    cluster {r[0]:>5}  n_total={r[2]:>5}  name={r[1]!r:60s}  at {r[3]}")

    section("knn_predictions")
    n = conn.execute("SELECT COUNT(*) FROM knn_predictions").fetchone()[0]
    versions = conn.execute("SELECT index_version, k, COUNT(*) c FROM knn_predictions GROUP BY index_version, k ORDER BY index_version, k").fetchall()
    distinct_species = conn.execute("SELECT COUNT(DISTINCT top_species) FROM knn_predictions").fetchone()[0]
    print(f"  total KNN rows       : {n}")
    print(f"  distinct top_species : {distinct_species}")
    for v, k, c in versions:
        print(f"    version={v}  k={k}  rows={c}")

    section("detections (YOLO-World bboxes)")
    n = conn.execute("SELECT COUNT(*) FROM detections").fetchone()[0]
    top_labels = conn.execute("SELECT label, COUNT(*) c FROM detections GROUP BY label ORDER BY c DESC LIMIT 10").fetchall()
    print(f"  total detections: {n}")
    for lbl, c in top_labels:
        print(f"    {lbl:30s}  {c}")

    section("bbox_predictions (BioCLIP on YOLO crops)")
    n = conn.execute("SELECT COUNT(*) FROM bbox_predictions").fetchone()[0]
    print(f"  total bbox predictions: {n}")

    section("album_meta, cluster_albums, clusterer_state, cluster_remap")
    for tbl in ("album_meta", "cluster_albums", "clusterer_state", "cluster_remap"):
        n = conn.execute(f"SELECT COUNT(*) FROM {tbl}").fetchone()[0]
        print(f"  {tbl:20s}: {n}")

    section("emb_size estimate (BLOB size for embeddings)")
    sample = conn.execute("SELECT dim, LENGTH(vector) FROM embeddings ORDER BY rowid LIMIT 5").fetchall()
    n_emb = conn.execute("SELECT COUNT(*) FROM embeddings").fetchone()[0]
    print(f"  total embeddings rows: {n_emb}")
    for dim, nbytes in sample:
        print(f"    dim={dim:>5}  blob={nbytes} B   ({humanize_bytes(nbytes)} per row)")
    # If every row uses the same dim, multiply out:
    if sample:
        nbytes_avg = sum(r[1] for r in sample) / len(sample)
        approx = nbytes_avg * n_emb
        print(f"  ≈ total embeddings bytes: {humanize_bytes(approx)}")

    section("stale data — rows whose computed_at is oldest / newest in each main table")
    for tbl in ("predictions", "bbox_predictions", "taxon_scores", "knn_predictions", "cluster_names"):
        try:
            oldest = conn.execute(f"SELECT MIN(computed_at) FROM {tbl}").fetchone()[0]
            newest = conn.execute(f"SELECT MAX(computed_at) FROM {tbl}").fetchone()[0]
            print(f"  {tbl:22s}  oldest={oldest}  newest={newest}")
        except Exception as e:
            print(f"  {tbl:22s}  (no computed_at: {e!r})")


def report_near_dup_pickle() -> None:
    print()
    print("=" * 78)
    print("5. Near-dup pickle (the only NON-SQLite cache)")
    print("=" * 78)
    p = DATA_DIR / "near_dup_cache.pkl"
    if not p.exists():
        print("  no near_dup_cache.pkl found")
        return
    print(f"  file: {p.name}  size: {humanize_bytes(file_size(p))}")
    try:
        with p.open("rb") as f:
            d = pickle.load(f)
        if not isinstance(d, dict):
            print(f"  unexpected top-level type: {type(d).__name__}")
            return
        print(f"  top-level keys (asset_id -> list[asset_id])")
        print(f"  groups: {len(d)}")
        sizes = [len(v) for v in d.values()]
        if sizes:
            print(f"  group sizes: min={min(sizes)}  max={max(sizes)}  "
                  f"avg={sum(sizes)/len(sizes):.2f}")
            big = [(k, len(v)) for k, v in d.items() if len(v) > 5]
            print(f"  groups with >5 members: {len(big)}")
            for k, n in big[:5]:
                print(f"    {k}  → {n} neighbors")
    except Exception as e:
        print(f"  could not unpickle: {e!r}")


def main() -> None:
    if not DB.exists():
        print(f"!! state.db not found at {DB}")
        sys.exit(1)
    report_sibling_caches()
    with storage_mod.connect(DB) as conn:
        report_db_overview(conn)
        report_each_table(conn)
        report_distributions(conn)
    report_near_dup_pickle()
    print()
    print("=" * 78)
    print("DONE")
    print("=" * 78)


if __name__ == "__main__":
    main()
