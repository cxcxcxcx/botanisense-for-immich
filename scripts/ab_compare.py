#!/usr/bin/env python3
"""Numeric A/B between two BotaniSense state DBs (read-only).

Companion to docs/mobile_engine_ab.md. Both DBs are opened read-only; this
script never writes to either one (the shadow DB is throwaway but the
production DB is not).

    python scripts/ab_compare.py --a data/ab/state.db --b data/ab/state_v2.db

What it reports:
  * engine / model_version markers of each side (identity, not a guess)
  * row counts per model-side table
  * on the SHARED asset set: top-1 species agreement, genus agreement,
    label-vocabulary sizes, and the biggest movers

INTERPRETATION WARNING (learned the hard way, 2026-09-22): comparing a
mobile engine against the *production teacher* DB is meaningless — they
carry different label spaces (mobile package table vs the full teacher
table), so agreement is near zero by construction. The meaningful
comparison is mobile-vN vs mobile-vM: same label table family, so the
number reflects the model change and nothing else.
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path

TABLES = ("assets", "embeddings", "predictions", "taxon_scores",
          "detections", "bbox_predictions", "search_results", "cluster_names")


def meta(db: Path) -> dict[str, str]:
    """engine / model_version markers (album_meta is key -> value_json)."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = c.execute("select key, value_json from album_meta").fetchall()
    finally:
        c.close()
    return {k: v.strip('"') for k, v in rows}


def counts(db: Path) -> dict[str, int]:
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    out = {}
    try:
        have = {r[0] for r in c.execute(
            "select name from sqlite_master where type='table'")}
        for t in TABLES:
            if t in have:
                out[t] = c.execute(f'select count(*) from "{t}"').fetchone()[0]
    finally:
        c.close()
    return out


def top1(db: Path, rank: str = "species") -> dict[str, tuple[str, str, float]]:
    """asset_id -> (taxon, genus, score) for position 0 of `rank`."""
    c = sqlite3.connect(f"file:{db}?mode=ro", uri=True)
    try:
        rows = c.execute(
            "select asset_id, species, coalesce(genus,''), score from predictions "
            "where rank=? and position=0", (rank,)).fetchall()
    finally:
        c.close()
    return {a: (s, g, sc) for a, s, g, sc in rows}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--a", type=Path, required=True, help="baseline DB (e.g. mobile-v0)")
    ap.add_argument("--b", type=Path, required=True, help="challenger DB (e.g. mobile-v2)")
    ap.add_argument("--movers", type=int, default=15,
                    help="how many changed assets to list (0 = none)")
    args = ap.parse_args()

    for p in (args.a, args.b):
        if not p.exists():
            raise SystemExit(f"DB not found: {p}")

    ma, mb = meta(args.a), meta(args.b)
    warn = ""
    if ma.get("engine", "teacher") != mb.get("engine", "teacher"):
        if "teacher" in (ma.get("engine"), mb.get("engine")):
            warn = ("\n  !! one side is the TEACHER engine — label spaces differ, "
                    "agreement numbers below are NOT a quality signal\n")
    print(f"A = {args.a}\n    engine={ma.get('engine', 'teacher')} "
          f"version={ma.get('model_version', '-')}")
    print(f"B = {args.b}\n    engine={mb.get('engine', 'teacher')} "
          f"version={mb.get('model_version', '-')}{warn}")

    ca, cb = counts(args.a), counts(args.b)
    print(f"\n{'table':<20}{'A':>10}{'B':>10}")
    for t in TABLES:
        if t in ca or t in cb:
            print(f"{t:<20}{ca.get(t, 0):>10,}{cb.get(t, 0):>10,}")

    ta, tb = top1(args.a), top1(args.b)
    shared = set(ta) & set(tb)
    if not shared:
        print("\nno shared assets with a top-1 species prediction — nothing to compare")
        return

    same_sp = sum(1 for a in shared if ta[a][0] == tb[a][0])
    same_gn = sum(1 for a in shared if ta[a][1] and ta[a][1] == tb[a][1])
    va = Counter(v[0] for v in ta.values())
    vb = Counter(v[0] for v in tb.values())
    n = len(shared)
    print(f"\nshared assets with predictions: {n:,}")
    print(f"  top-1 species identical : {same_sp:,} ({same_sp / n * 100:.1f}%)")
    print(f"  genus identical         : {same_gn:,} ({same_gn / n * 100:.1f}%)")
    print(f"  species vocabulary      : A {len(va):,}  B {len(vb):,}")
    both = set(va) & set(vb)
    print(f"  vocabulary shared       : {len(both):,} "
          f"(A-only {len(set(va) - both):,}, B-only {len(set(vb) - both):,})")

    if args.movers:
        moved = [(a, ta[a], tb[a]) for a in shared if ta[a][0] != tb[a][0]]
        moved.sort(key=lambda r: (r[1][2] + r[2][2]), reverse=True)
        print(f"\nchanged assets: {len(moved):,}; top {min(args.movers, len(moved))} "
              f"by combined confidence:")
        print(f"  {'asset':<38}{'A (score)':<34}{'B (score)'}")
        for a, x, y in moved[:args.movers]:
            print(f"  {a:<38}{x[0][:28] + f' ({x[2]:.3f})':<34}"
                  f"{y[0][:28]} ({y[2]:.3f})")


if __name__ == "__main__":
    main()
