"""Cluster cached BioCLIP embeddings with k-means.

Embeddings are L2-normalized already (pybioclip's
`create_image_features(..., normalize=True)`), so cosine similarity reduces
to a dot product. We run MiniBatchKMeans on the raw vectors — the cosine
geometry is preserved well enough for visualization, and MiniBatch scales to
thousands of vectors on CPU.

`compute_clusters()` returns the (asset_id → cluster_id) mapping and the
k used; the caller persists cluster_id onto each asset and the run row.
"""

from __future__ import annotations

import sqlite3
from typing import Iterable

import numpy as np
from sklearn.cluster import MiniBatchKMeans

import storage


def _choose_k(n: int) -> int:
    if n <= 1:
        return 1
    # ~sqrt(n/2), clamped to [2, 12]. A heuristic that works for the small
    # gallery sizes we expect (10s–100s of images).
    return int(min(12, max(2, round((n / 2) ** 0.5))))


def compute_clusters(
    conn: sqlite3.Connection, *, k: int | None = None
) -> tuple[dict[str, int], int]:
    """Re-cluster every embedded asset. Returns
    `{asset_id: cluster_id}` and the k actually used."""
    ids, matrix = storage.all_embeddings(conn)
    if matrix.size == 0:
        return {}, 0
    if k is None:
        k = _choose_k(len(ids))
    if k >= len(ids):
        # Degenerate: one cluster per image.
        mapping = {aid: i for i, aid in enumerate(ids)}
        return mapping, len(ids)

    km = MiniBatchKMeans(
        n_clusters=k,
        random_state=0,
        n_init=3,
        batch_size=min(64, max(8, len(ids) // 4)),
    )
    labels = km.fit_predict(matrix)
    mapping = {aid: int(c) for aid, c in zip(ids, labels)}
    return mapping, k