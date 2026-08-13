"""Cluster cached BioCLIP embeddings with HDBSCAN.

Embeddings are L2-normalized already (pybioclip's
`create_image_features(..., normalize=True)`), so cosine similarity reduces
to a dot product. HDBSCAN handles non-spherical clusters and varying
densities better than k-means on small N (10s–1000s), and doesn't require
specifying K up front.

Noise label: HDBSCAN assigns label -1 to points it considers outliers (not
in any cluster). We preserve that as cluster_id = -1 so the UI can show
"outliers (N)" as a distinct pill.

`compute_clusters()` returns `{asset_id: cluster_id}` and the number of
clusters actually produced (excluding noise). The caller persists
cluster_id onto each asset and the run row.
"""

from __future__ import annotations

import sqlite3

import hdbscan
import numpy as np

import storage


def _min_cluster_size(n: int) -> int:
    """Pick min_cluster_size as a small fraction of N, with sensible bounds.

    For small N (< 10) we fall back to k-means style behaviour (each point
    its own cluster) by forcing min_cluster_size = 2 — HDBSCAN still gives
    meaningful groupings, but if it can't, returns all-noise which we
    surface as a single cluster_id = 0 group via the caller's fallback.
    """
    if n <= 1:
        return 1
    return max(2, min(5, max(2, n // 6)))


def compute_clusters(
    conn: sqlite3.Connection, *, min_cluster_size: int | None = None
) -> tuple[dict[str, int], int]:
    """Re-cluster every embedded asset. Returns `{asset_id: cluster_id}`
    and the number of clusters actually produced (excluding noise label -1).
    """
    ids, matrix = storage.all_embeddings(conn)
    if matrix.size == 0:
        return {}, 0
    if min_cluster_size is None:
        min_cluster_size = _min_cluster_size(len(ids))

    # HDBSCAN with cosine-friendly settings:
    #   metric='euclidean' is fine because vectors are L2-normalized
    #   (cosine sim = euclidean distance up to monotonic transform).
    #   cluster_selection_method='eom' merges sibling leaves when the
    #   parent branch is stable, giving species-level groupings instead
    #   of sub-variant leaves. Switch to 'leaf' for finer-grained clusters.
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=max(1, min_cluster_size // 2),
        metric="euclidean",
        cluster_selection_method="eom",
        core_dist_n_jobs=1,
    )
    raw_labels = clusterer.fit_predict(matrix)

    # Remap noise (-1) and any negative labels into a 0-based dense range so
    # the UI's cluster_id filter stays intuitive (no negative numbers).
    unique = sorted(set(int(x) for x in raw_labels))
    remap: dict[int, int] = {}
    next_id = 0
    for u in unique:
        if u == -1:
            # Push noise to the END so it's the last cluster_id in the UI.
            continue
        remap[u] = next_id
        next_id += 1
    if -1 in unique:
        remap[-1] = next_id  # noise becomes the highest cluster_id
    n_clusters = next((
        len([k for k in remap if k != -1]) for _ in [None]
    ), 0)

    mapping = {aid: remap[int(c)] for aid, c in zip(ids, raw_labels)}
    return mapping, len({v for k, v in remap.items() if k != -1})
