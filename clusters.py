"""Cluster cached BioCLIP embeddings with HDBSCAN.

Embeddings are L2-normalized already (pybioclip's
`create_image_features(..., normalize=True)`), so cosine similarity reduces
to a dot product. HDBSCAN handles non-spherical clusters and varying
densities better than k-means on small N (10s–1000s), and doesn't require
specifying K up front.

Incremental clustering (this module)
------------------------------------
HDBSCAN is fundamentally batch — its mutual-reachability graph +
single-linkage dendrogram depend on every point at once, so the library
exposes no incremental API. We work around it with two paths:

  - **Incremental assign** (`assign_new_points`): load the trained
    clusterer we pickled last time and call
    `clusterer.approximate_predict(new_vectors)`. This uses the existing
    condensed tree to figure out which existing cluster (or noise) a new
    point would land in, without rebuilding K-NN. ~1 ms per new point.

  - **Full refit + Hungarian remap** (`refit_with_remap`): every time the
    corpus grows by `growth_threshold_pct` (default 10%), or when the
    user hits "Force refit", we re-fit HDBSCAN on every embedded asset.
    To keep cluster IDs stable across refits, we build a membership-
    overlap matrix between OLD and NEW clusters, run scipy's Hungarian
    algorithm to find the optimal pairing (max overlap), and reassign
    the new raw labels so matched pairs keep their old IDs. Unmatched
    new clusters get fresh IDs appended after the highest old ID; the
    audit trail lands in the `cluster_remap` table.

Noise label: HDBSCAN assigns -1 to points it considers outliers. In the
DB we remap -1 to the highest positive cluster_id so it sorts last in
the UI (existing convention). `assign_new_points` does the same remap
via the `label_map` persisted alongside the clusterer.

Public surface (used by pipeline.py / app.py):
  - `should_refit(conn, *, growth_threshold_pct=0.10) -> bool`
  - `assign_new_points(conn, *, new_asset_ids, new_vectors) -> dict | None`
  - `refit_with_remap(conn, *, growth_threshold_pct=0.10) -> tuple`
  - `save_clusterer(conn, clusterer, *, ...) -> None`
  - `load_clusterer(conn) -> tuple[hdbscan.HDBSCAN, dict] | None`
  - `get_min_cluster_size(n) -> int`  (utility, also exposed)
  - `cluster_centroids(conn) -> tuple[list[int], np.ndarray]`
  - `nearest_clusters(cluster_ids, centroids, *, cluster_id, k=8) -> list[dict]`
  - `build_cluster_groups(cluster_ids, centroids, *, max_distance) -> tuple`
"""

from __future__ import annotations

import logging
import os
import pickle
import sqlite3
from typing import Iterable

import hdbscan
import numpy as np
from hdbscan import approximate_predict
from scipy.optimize import linear_sum_assignment

import storage

_log = logging.getLogger("clusters")
if not _log.handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter("%(asctime)s %(message)s", datefmt="%H:%M:%S"))
    _log.addHandler(_h)
    _log.setLevel(logging.INFO)
    _log.propagate = False

# Plantae kingdom score floor for cluster inclusion. Assets with
# Plantae score < CLUSTER_PLANTAE_MIN are excluded from the HDBSCAN fit
# (the "cluster computation" itself) AND any existing cluster_id on
# them is cleared on each refit, so the cluster view only shows plant
# images. Matches the existing `plantae_yellow` floor (the boundary
# between the "uncertain" and "non_plant" tiers in the UI), so the
# cutoff aligns with what the rest of the app treats as a plant.
# Override via CLUSTER_PLANTAE_MIN_SCORE env var without code changes.
CLUSTER_PLANTAE_MIN = float(os.environ.get("CLUSTER_PLANTAE_MIN_SCORE", "0.35"))


def get_min_cluster_size(n: int) -> int:
    """Pick min_cluster_size as a small fraction of N, with sensible bounds.

    For small N (1–9) we force min_cluster_size = 2 — HDBSCAN requires
    min_cluster_size >= 2 (it would otherwise raise on the `> 1` check
    inside HDBSCAN itself), and a small floor still lets the algorithm
    collapse to all-noise which the caller's remap surface catches. For
    larger N, `n // 6` clamped to [2, 5] gives a small fraction that
    scales without exploding on corpus growth.
    """
    if n <= 1:
        return 2
    return max(2, min(5, max(2, n // 6)))


# Backward-compat alias. Kept so any caller that imported the private name
# still resolves; new code should call get_min_cluster_size.
_min_cluster_size = get_min_cluster_size


# ---------- clusterer persistence ---------------------------------------


def save_clusterer(
    conn: sqlite3.Connection,
    clusterer: "hdbscan.HDBSCAN",
    *,
    min_cluster_size: int,
    n_assets_at_fit: int,
    n_clusters: int,
    label_map: dict[int, int],
) -> None:
    """Persist the trained clusterer + the raw-label -> stable-id map.

    The blob is pickled as a dict {"clusterer": ..., "label_map": ...}
    so subsequent `load_clusterer` calls can recover both. The label_map
    is what `assign_new_points` uses to translate approximate_predict's
    raw HDBSCAN labels back into the user-facing stable integer space.
    """
    payload = {"clusterer": clusterer, "label_map": dict(label_map)}
    blob = pickle.dumps(payload, protocol=pickle.HIGHEST_PROTOCOL)
    storage.save_clusterer_state(
        conn,
        clusterer_blob=blob,
        min_cluster_size=int(min_cluster_size),
        n_assets_at_fit=int(n_assets_at_fit),
        n_clusters=int(n_clusters),
    )


def load_clusterer(
    conn: sqlite3.Connection,
) -> tuple["hdbscan.HDBSCAN", dict[int, int]] | None:
    """Return (clusterer, label_map) or None if no clusterer has been fit
    yet (fresh DB) or if the pickled blob fails to unpickle (e.g., HDBSCAN
    version mismatch).

    On unpickle failure we clear the state row so the caller can fall
    back to a full refit without infinite retry.
    """
    state = storage.get_clusterer_state(conn)
    if state is None:
        return None
    try:
        payload = pickle.loads(state["clusterer_blob"])
        # Backward compat: legacy blobs (pre-label_map) only carry the
        # raw clusterer; treat them as "no incremental path available".
        if not isinstance(payload, dict) or "clusterer" not in payload:
            _log.warning("[clusters] legacy clusterer blob has no metadata; clearing")
            storage.clear_clusterer_state(conn)
            return None
        return payload["clusterer"], dict(payload.get("label_map") or {})
    except Exception as e:
        _log.warning(
            f"[clusters] unpickle failed ({type(e).__name__}: {e}); "
            "clearing state so the next call falls back to a full refit"
        )
        storage.clear_clusterer_state(conn)
        return None


# ---------- refit trigger -------------------------------------------------


def should_refit(
    conn: sqlite3.Connection, *, growth_threshold_pct: float = 0.10
) -> bool:
    """Return True when the corpus has grown enough since the last fit
    to warrant a full refit (default: 10%). Always True on a fresh DB
    (no clusterer_state row) so the first refit happens unconditionally.

    Counts are scoped to plant assets (Plantae score >= CLUSTER_PLANTAE_MIN)
    because that's the population HDBSCAN actually fits — counting
    every embedding would over-trigger refits as new non-plant photos
    flow in without affecting the cluster topology.
    """
    state = storage.get_clusterer_state(conn)
    if state is None:
        return True  # First fit; no clusterer persisted yet.
    n_at_fit = state["n_assets_at_fit"]
    if n_at_fit <= 0:
        return True
    # "Current N" = number of plant assets with an embedding right now.
    # We don't want to force a refit just because someone re-uploaded the
    # same image, OR because non-plant photos were added since the last
    # fit (they don't change the cluster topology).
    plant_ids = storage.plantae_asset_ids(conn, min_score=CLUSTER_PLANTAE_MIN)
    if not plant_ids:
        # No plant assets at all — the clusterer has nothing to do, no
        # point refitting. Caller will short-circuit anyway.
        return False
    plant_params = list(plant_ids)
    placeholders = ",".join("?" for _ in plant_params)
    row = conn.execute(
        f"SELECT COUNT(DISTINCT e.asset_id) AS n "
        f"FROM embeddings e "
        f"WHERE e.asset_id IN ({placeholders})",
        plant_params,
    ).fetchone()
    n_current = int(row["n"]) if row else 0
    growth = (n_current - n_at_fit) / n_at_fit
    return growth >= growth_threshold_pct


# ---------- incremental assign -------------------------------------------


def assign_new_points(
    conn: sqlite3.Connection,
    *,
    new_asset_ids: list[str],
    new_vectors: np.ndarray,
) -> dict[str, int] | None:
    """Assign cluster labels to new points using the persisted clusterer.

    Returns None when no clusterer is persisted yet (caller should refit
    first). Otherwise returns {asset_id: stable_cluster_id} for the
    PLANT subset of new_asset_ids — assets with Plantae score <
    CLUSTER_PLANTAE_MIN are silently dropped (they keep cluster_id =
    NULL so they don't pollute the cluster view). New clusters are NOT
    discovered here — approximate_predict only emits labels from the
    training set, so a genuinely-novel group of new photos will be
    marked noise (-1) until the next full refit promotes them.

    `new_vectors` must be (N, EMB_DIM) float32, L2-normalized, aligned
    with `new_asset_ids`. Caller is responsible for normalization
    (BioCLIP-2 already normalizes upstream).
    """
    loaded = load_clusterer(conn)
    if loaded is None:
        return None
    clusterer, label_map = loaded
    if not new_asset_ids:
        return {}
    if new_vectors.shape[0] != len(new_asset_ids):
        raise ValueError(
            f"new_vectors has {new_vectors.shape[0]} rows but "
            f"new_asset_ids has {len(new_asset_ids)} entries"
        )
    if not label_map:
        _log.warning(
            "[clusters] persisted clusterer has no label_map; "
            "falling back to full refit"
        )
        return None

    # Plant-only filter: load every plant asset_id once, then slice
    # `new_asset_ids` and `new_vectors` together so approximate_predict
    # sees only embeddings that were part of the (plant-only) training
    # population. Doing the filter here keeps the caller's contract
    # unchanged — they still hand us every newly-embedded asset — but
    # ensures non-plants never appear in the returned mapping.
    plant_ids = storage.plantae_asset_ids(conn, min_score=CLUSTER_PLANTAE_MIN)
    if not plant_ids:
        # No plant assets in the DB at all — the persisted clusterer
        # was fit against plants so it has no useful mapping for the
        # incoming points either. Bail with an empty mapping; callers
        # typically treat empty as "nothing to persist".
        return {}
    keep_idx = [i for i, aid in enumerate(new_asset_ids) if aid in plant_ids]
    if not keep_idx:
        return {}
    if len(keep_idx) == len(new_asset_ids):
        # Fast path: every new asset is a plant, no filtering needed.
        ids_for_predict = new_asset_ids
        vecs_for_predict = new_vectors
    else:
        ids_for_predict = [new_asset_ids[i] for i in keep_idx]
        vecs_for_predict = new_vectors[keep_idx]

    raw_labels, _strengths = approximate_predict(clusterer, vecs_for_predict)
    out: dict[str, int] = {}
    for aid, raw in zip(ids_for_predict, raw_labels):
        raw_int = int(raw)
        # If approximate_predict emits a raw label we don't recognize
        # (shouldn't happen — it's bounded by the training set), fall
        # back to noise via the persisted label_map[-1].
        out[aid] = label_map.get(raw_int, label_map.get(-1, -1))
    return out


# ---------- full refit + Hungarian remap ---------------------------------


def refit_with_remap(
    conn: sqlite3.Connection, *, growth_threshold_pct: float = 0.10
) -> tuple[dict[str, int], int, dict[int, int]]:
    """Full HDBSCAN refit, then Hungarian-match new cluster IDs onto the
    existing IDs to preserve stability.

    Returns:
      (final_asset_to_cluster, n_real_clusters, remap_new_to_stable)
      where `remap_new_to_stable[raw_hdbscan_label] = stable_cluster_id`.

    Plant-only scoping: every embedding, the Hungarian snapshot, and
    the final cluster_id write are restricted to assets with Plantae
    score >= CLUSTER_PLANTAE_MIN. Non-plant assets that previously had
    a cluster_id have it cleared at the end of each refit so the
    cluster view only shows plant images.

    Steps:
      0. Load the set of plant asset_ids (Plantae score >=
         CLUSTER_PLANTAE_MIN).
      1. Snapshot the current asset->cluster mapping from the DB,
         restricted to plants.
      2. Load every embedding, filter to plants, and run HDBSCAN.
      3. Build a membership-overlap matrix between OLD clusters (rows)
         and NEW clusters (columns); -1 is included in both if present.
      4. `scipy.optimize.linear_sum_assignment(-M)` to maximize overlap.
      5. Build a remap so matched pairs keep their old IDs (and the
         matched new cluster "becomes" the matched old one); unmatched
         new clusters get fresh dense IDs starting at max(old) + 1.
         Noise is forced to the highest ID (existing convention).
      6. Persist the new mapping via storage.set_cluster_ids.
      7. Clear cluster_id on every asset NOT in final_mapping — drops
         any stale cluster_id that was lingering on non-plant assets.
      8. Save the clusterer + label_map via save_clusterer.
      9. Append audit rows to cluster_remap.
     10. Invalidate cluster_names so the next /api/clusters call
         recomputes labels for any newly-renumbered clusters.

    `growth_threshold_pct` is currently unused inside this function but
    accepted for API symmetry with `should_refit`; the caller decides
    whether to call refit at all.
    """
    del growth_threshold_pct  # API symmetry; see docstring.

    # Step 0: load plant asset_ids once. Empty set → no plants fit the
    # criteria, nothing to cluster; clear state + cluster_ids and bail.
    plant_ids = storage.plantae_asset_ids(conn, min_score=CLUSTER_PLANTAE_MIN)

    ids_all, matrix_all = storage.all_embeddings(conn)
    if matrix_all.size == 0 or not plant_ids:
        # No embeddings, OR no plant assets at all. Wipe clusterer state
        # and any lingering cluster_ids so the cluster view starts
        # clean, then return empty.
        storage.clear_clusterer_state(conn)
        storage.clear_cluster_ids_for_non_plants(conn, keep_asset_ids=set())
        return {}, 0, {}

    # Filter to plant embeddings only (vectorized boolean mask; set
    # lookup is O(1) per row).
    keep_mask = np.array(
        [aid in plant_ids for aid in ids_all], dtype=bool
    )
    ids = [aid for aid, k in zip(ids_all, keep_mask) if k]
    matrix = matrix_all[keep_mask]
    n = len(ids)
    mcs = get_min_cluster_size(n)

    # Step 1: snapshot old mapping (plants only — non-plants should
    # never have carried a cluster_id under the new policy, but we
    # filter defensively in case a stale row slipped through).
    old_assets = storage.all_assets(conn)
    old_mapping: dict[str, int] = {
        r["asset_id"]: int(r["cluster_id"])
        for r in old_assets
        if r["cluster_id"] is not None and r["asset_id"] in plant_ids
    }
    old_clusters: dict[int, set[str]] = {}
    for aid, cid in old_mapping.items():
        old_clusters.setdefault(cid, set()).add(aid)

    # Step 2: fit HDBSCAN.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=mcs,
        min_samples=max(1, mcs // 2),
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=1,
        # prediction_data=True caches the minimum-spanning tree and
        # condensed cluster tree alongside the fit, so a later
        # approximate_predict() call can reuse them without retraining.
        # ~10-20% slower fit, ~30% larger pickled blob — acceptable
        # trade-off since we pickle once per refit and predict many times.
        prediction_data=True,
    )
    raw_labels = clusterer.fit_predict(matrix)
    raw_labels = np.asarray(raw_labels, dtype=np.int64)

    # Step 3: build new_clusters.
    new_clusters: dict[int, set[str]] = {}
    for aid, raw in zip(ids, raw_labels):
        new_clusters.setdefault(int(raw), set()).add(aid)

    # Include noise in both views so the matrix is well-defined when
    # one side has outliers and the other doesn't.
    if -1 not in new_clusters:
        new_clusters[-1] = set()

    # Step 4: Hungarian on the overlap matrix.
    old_ids_list = sorted(old_clusters.keys())  # sorted for determinism
    new_ids_list = sorted(new_clusters.keys())
    n_old = len(old_ids_list)
    n_new = len(new_ids_list)
    if n_old == 0 or n_new == 0:
        # No old (first refit) or no new (degenerate) — skip matching,
        # every new cluster gets a fresh dense ID.
        row_ind = np.empty((0,), dtype=np.int64)
        col_ind = np.empty((0,), dtype=np.int64)
    else:
        M = np.zeros((n_old, n_new), dtype=np.int64)
        for i, oc in enumerate(old_ids_list):
            oc_set = old_clusters[oc]
            for j, nc in enumerate(new_ids_list):
                M[i, j] = len(oc_set & new_clusters[nc])
        row_ind, col_ind = linear_sum_assignment(-M)

    # Step 5: build remap (new_id -> stable_id).
    # Initially: matched pairs reuse old IDs. Unmatched new clusters get
    # fresh IDs after max(old). Noise gets the highest dense ID.
    matched_new_ids: set[int] = set()
    matched_old_ids: set[int] = set()
    new_to_matched_old: dict[int, list[int]] = {}
    audit_rows: list[dict] = []

    # Pre-compute fresh-ID counter.
    max_old = max(old_clusters.keys()) if old_clusters else -1
    next_id = max_old + 1

    for r, c in zip(row_ind, col_ind):
        oc = old_ids_list[int(r)]
        nc = new_ids_list[int(c)]
        ov = int(
            len(old_clusters[oc] & new_clusters[nc])
        )
        if ov == 0:
            # Hungarian sometimes pairs a 0-overlap pair just to fill the
            # rectangular matrix; ignore those.
            continue
        matched_new_ids.add(nc)
        matched_old_ids.add(oc)
        new_to_matched_old.setdefault(nc, []).append(oc)
        union = len(old_clusters[oc] | new_clusters[nc]) or 1
        audit_rows.append(
            {
                "old_cluster_id": int(oc),
                "new_cluster_id": int(nc),
                "overlap_count": ov,
                "union_count": int(union),
                "jaccard": float(ov / union),
                "kind": "match",  # possibly upgraded to 'merge' below
            }
        )

    # Remap dict: new_id -> stable_id. Matched new clusters adopt the
    # OLD id of their single-matched old cluster (no merging for now —
    # we treat each old cluster's assets as belonging to the matched new
    # cluster; if two old clusters land on the same new cluster, the
    # smaller-overlap old is marked 'merge' in the audit and its assets
    # are reassigned to the dominant new cluster's stable id below).
    remap: dict[int, int] = {}
    for nc, olds in new_to_matched_old.items():
        if len(olds) == 1:
            remap[nc] = int(olds[0])
        else:
            # Multiple old clusters matched to one new cluster: the
            # largest-overlap old "wins" the stable id, the others are
            # marked 'merge' in the audit (their assets will be moved
            # into the winner's stable id by `final_mapping` below).
            ranked = sorted(
                olds,
                key=lambda o: -len(old_clusters[o] & new_clusters[nc]),
            )
            winner = int(ranked[0])
            remap[nc] = winner
            for loser in ranked[1:]:
                loser = int(loser)
                for row in audit_rows:
                    if (
                        row["old_cluster_id"] == loser
                        and row["new_cluster_id"] == nc
                    ):
                        row["kind"] = "merge"
                        break

    # Unmatched old clusters -> 'gone'. Their assets were either
    # absorbed into another cluster (merge) or pushed to noise in the
    # refit; we record them either way.
    for oc in old_clusters.keys():
        if oc not in matched_old_ids:
            audit_rows.append(
                {
                    "old_cluster_id": int(oc),
                    "new_cluster_id": int(oc),  # self-ref for visibility
                    "overlap_count": 0,
                    "union_count": len(old_clusters[oc]),
                    "jaccard": 0.0,
                    "kind": "gone",
                }
            )

    # Unmatched new clusters -> 'new'. Assign fresh dense IDs. Noise
    # always becomes the highest ID (existing convention).
    for nc in new_clusters.keys():
        if nc in matched_new_ids:
            continue
        if nc == -1:
            # Noise gets the highest stable ID so it sorts last in the UI.
            remap[nc] = next_id
            next_id += 1
        else:
            remap[nc] = next_id
            next_id += 1
        audit_rows.append(
            {
                "old_cluster_id": -1,
                "new_cluster_id": int(remap[nc]),
                "overlap_count": 0,
                "union_count": len(new_clusters[nc]),
                "jaccard": 0.0,
                "kind": "new",
            }
        )

    # Step 6: build the final asset->stable_cluster_id mapping.
    final_mapping: dict[str, int] = {}
    for aid, raw in zip(ids, raw_labels):
        raw_int = int(raw)
        if raw_int in remap:
            final_mapping[aid] = remap[raw_int]
        else:
            # Shouldn't happen — every raw label is in new_clusters.
            # Fall back to noise (the highest stable ID, if remap has it).
            final_mapping[aid] = remap.get(-1, max(remap.values(), default=-1))

    n_real_clusters = len(
        {v for v in final_mapping.values() if v != remap.get(-1, -1)}
    )

    # Persist mapping (plants only — every asset in final_mapping IS a
    # plant, so the set_cluster_ids UPSERT leaves non-plant assets
    # untouched at this point).
    storage.set_cluster_ids(conn, asset_to_cluster=final_mapping)

    # Step 7: clear cluster_id on every asset NOT in final_mapping.
    # This drops cluster_id on any non-plant asset (and on any plant
    # asset that HDBSCAN emitted as "noise" / unassigned — keep_asset_ids
    # excludes those, so the assignment also gets wiped for noise-only
    # plant assets). Cheap (one indexed UPDATE).
    cleared = storage.clear_cluster_ids_for_non_plants(
        conn, keep_asset_ids=set(final_mapping.keys()),
    )

    # Step 8: save clusterer + label_map.
    save_clusterer(
        conn,
        clusterer,
        min_cluster_size=mcs,
        n_assets_at_fit=n,
        n_clusters=n_real_clusters,
        label_map=remap,
    )

    # Step 9: audit trail.
    storage.insert_cluster_remap(conn, audit_rows)

    # Step 10: invalidate cached cluster names — the asset->cluster
    # mapping shifted (even if some IDs survived, the membership inside
    # any cluster may have changed).
    storage.invalidate_cluster_names(conn)

    _log.info(
        f"[clusters] refit  n_plants={n}/{len(ids_all)}  "
        f"n_clusters={n_real_clusters}  matched={len(matched_new_ids)}  "
        f"new={n_new - len(matched_new_ids)}  "
        f"gone={len(old_clusters) - len(matched_old_ids)}  "
        f"cleared={cleared}"
    )

    return final_mapping, n_real_clusters, remap


# ---------- cluster-to-cluster similarity --------------------------------
#
# HDBSCAN gives us membership but nothing about how clusters relate to
# each other — two clusters can be siblings in embedding space even
# though every point landed firmly in one or the other. For the browse
# UI ("open a cluster → show related clusters") we score each cluster by
# its centroid: the mean of its member embeddings, re-normalized. With
# unit-length embeddings the centroid mean is a decent direction
# estimate, and cosine(centroid_i, centroid_j) is a single dot product.


def cluster_centroids(conn: sqlite3.Connection) -> tuple[list[int], "np.ndarray"]:
    """Compute one L2-normalized centroid vector per cluster.

    Returns (cluster_ids_asc, centroids) where centroids is (C, D)
    float32 with rows aligned to cluster_ids (sorted ascending for
    determinism). Clusters with no embedded assets are skipped — a
    centroid of nothing is undefined, and the row simply can't be
    compared against anything.

    O(N) over every embedding in the DB (same load the refit uses);
    callers should cache the result and invalidate it whenever
    cluster_ids or embeddings change.
    """
    ids_all, matrix_all = storage.all_embeddings(conn)
    cluster_map = storage.asset_cluster_map(conn)
    if matrix_all.size == 0 or not cluster_map:
        return [], np.empty((0, 0), dtype=np.float32)

    keep_mask = np.array([aid in cluster_map for aid in ids_all], dtype=bool)
    ids = [aid for aid, k in zip(ids_all, keep_mask) if k]
    matrix = matrix_all[keep_mask]
    if not ids:
        return [], np.empty((0, 0), dtype=np.float32)

    labels = np.array([cluster_map[aid] for aid in ids], dtype=np.int64)
    cluster_ids = sorted(set(labels.tolist()))
    id_to_row = {cid: i for i, cid in enumerate(cluster_ids)}
    dim = matrix.shape[1]
    centroids = np.zeros((len(cluster_ids), dim), dtype=np.float32)
    # np.add.at accumulates rows into their cluster's slot in one pass
    # (a plain `centroids[idx] += ...` would drop duplicate indices).
    np.add.at(centroids, np.array([id_to_row[int(l)] for l in labels]), matrix)

    norms = np.linalg.norm(centroids, axis=1, keepdims=True)
    # A zero-norm centroid (all member vectors exactly cancelled) can't
    # be normalized — leave it at zero so every similarity vs it is 0
    # rather than NaN.
    safe = np.where(norms > 0, norms, 1.0)
    centroids = centroids / safe.astype(np.float32)
    return cluster_ids, centroids


def nearest_clusters(
    cluster_ids: list[int],
    centroids: "np.ndarray",
    *,
    cluster_id: int,
    k: int = 8,
) -> list[dict]:
    """Top-K most-similar OTHER clusters to `cluster_id`, by centroid
    cosine similarity.

    Returns [{cluster_id, similarity}, ...] sorted similarity-desc
    (ties broken by cluster_id ASC so the order is stable across
    refits). The queried cluster itself is always excluded. Clusters
    missing from `cluster_ids` (no embedded assets) can't appear.
    """
    if cluster_id not in cluster_ids or centroids.size == 0:
        return []
    self_idx = cluster_ids.index(cluster_id)
    sims = centroids @ centroids[self_idx]  # (C,), cosine — rows are unit
    sims[self_idx] = -2.0  # mask self out of the top-K
    n = sims.shape[0]
    take = min(k, n - 1)
    if take <= 0:
        return []
    # Same argpartition trick as storage.find_library_neighbors.
    top = np.argpartition(-sims, take - 1)[:take]
    top = top[np.argsort(-sims[top], kind="stable")]
    out = [
        {"cluster_id": int(cluster_ids[i]), "similarity": round(float(sims[i]), 4)}
        for i in top
    ]
    # argpartition+sort preserves the -2.0 mask only for self; drop any
    # other cluster that lost the tie-break ordering guarantee by
    # re-sorting on (similarity desc, id asc).
    out.sort(key=lambda d: (-d["similarity"], d["cluster_id"]))
    return out[:k]


# ---------- hierarchy over clusters --------------------------------------
#
# The related-clusters strip answers "what's NEAR this cluster"; the
# grouping below answers "which clusters belong to the same FAMILY" —
# super-groups that make Level 1 browsable when the corpus has hundreds
# of clusters. We run agglomerative clustering over the centroids (not
# the assets — 655 centroids vs tens of thousands of photos) and cut the
# dendrogram at a fixed cosine-DISTANCE threshold.
#
# Why a distance threshold instead of "K groups": measured on the real
# library, K-way cuts (maxclust) always fuse hundreds of loose clusters
# into a couple of "everything" mega-groups, because much of the
# embedding blob is genuinely continuous. A threshold only merges
# clusters that are actually close (sim >= 1 - max_distance) and leaves
# the long tail solo — the solo bucket is the honest answer for clusters
# with no family. Complete linkage (not average/single) avoids chaining,
# which is what keeps "Rosa, Rosa, China rose, China rose" from picking
# up unrelated strays.
#
# Tuned on the production library (655 clusters): 0.15 yields 164
# families covering 441 clusters + 214 solo, with top families reading
# as Prunus / cherry / Narcissus / Rosa / Acer / Dahlia. Override via
# CLUSTER_GROUP_DISTANCE env var.
CLUSTER_GROUP_DISTANCE = float(os.environ.get("CLUSTER_GROUP_DISTANCE", "0.15"))


def build_cluster_groups(
    cluster_ids: list[int],
    centroids: "np.ndarray",
    *,
    max_distance: float = CLUSTER_GROUP_DISTANCE,
) -> tuple[list[list[int]], list[int]]:
    """Group clusters whose centroids are mutually close.

    Returns (families, solo): `families` is a list of member-cluster-id
    lists (each ≥ 2 members), sorted size-desc (member ids ascending
    within a family) so the browse UI's default size-sort is stable;
    `solo` holds clusters that belong to no family. Every input cluster
    appears exactly once across the two.

    Deterministic given (cluster_ids, centroids): scipy's linkage on a
    fixed matrix is deterministic, and the family ordering is a total
    sort. A refit changes centroids, so callers must invalidate any
    cached result alongside the centroid index.
    """
    from scipy.cluster.hierarchy import fcluster, linkage
    from scipy.spatial.distance import squareform

    n = len(cluster_ids)
    if n == 0:
        return [], []
    if n == 1 or centroids.size == 0:
        return [], list(cluster_ids)

    dist = 1.0 - centroids @ centroids.T  # cosine distance
    # Zero the diagonal BEFORE symmetrizing: (1 - x@x.T) is already
    # symmetric up to float error, but squareform(checks=False) trusts us.
    np.fill_diagonal(dist, 0.0)
    np.clip(dist, 0.0, 2.0, out=dist)
    Z = linkage(squareform(dist, checks=False), method="complete")
    labels = fcluster(Z, t=max_distance, criterion="distance")

    groups: dict[int, list[int]] = {}
    for cid, lab in zip(cluster_ids, labels):
        groups.setdefault(int(lab), []).append(int(cid))

    families = [sorted(m) for m in groups.values() if len(m) >= 2]
    families.sort(key=lambda m: (-len(m), m[0]))
    solo = sorted(m[0] for m in groups.values() if len(m) == 1)
    return families, solo