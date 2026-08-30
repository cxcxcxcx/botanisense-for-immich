"""Tests for knn.validate_index(), the index health sentinel.

The v1 index shipped trained-on-unnormalized / added-normalized, so
every entry decoded to roughly its coarse centroid and queries returned
near-identical scores (~20.7) for unrelated species. The ranking was
arbitrary but the output looked plausible, so it went unnoticed.

`validate_index` catches that class of corruption with the one
invariant that cannot be argued with: for an inner-product index over
L2-normalized vectors, a unit query cannot score above 1.0.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

faiss = pytest.importorskip("faiss")

DIM = 64


def _unit_rows(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    m = rng.standard_normal((n, DIM)).astype(np.float32)
    m /= np.linalg.norm(m, axis=1, keepdims=True)
    return m


def test_normalized_index_passes():
    import knn
    idx = faiss.IndexFlatIP(DIM)
    idx.add(_unit_rows(500, seed=1))
    v = knn.validate_index(idx)
    assert v["ok"] is True, v
    assert v["max_sim"] <= 1.01


def test_unnormalized_index_fails():
    """The actual v1 failure: reference vectors left at their published
    scale (TreeOfLife-200M norms run ~40-57)."""
    import knn
    idx = faiss.IndexFlatIP(DIM)
    idx.add(_unit_rows(500, seed=2) * 45.0)
    v = knn.validate_index(idx)
    assert v["ok"] is False, v
    assert v["max_sim"] > 1.01
    assert "rebuild" in v["reason"]


def test_mildly_over_unit_still_passes():
    """PQ reconstruction error can nudge a score fractionally past 1.0.
    The sentinel must not cry wolf at 1.005 — the real failure mode
    overshoots by ~20x."""
    import knn
    idx = faiss.IndexFlatIP(DIM)
    idx.add(_unit_rows(500, seed=3) * 1.005)
    v = knn.validate_index(idx)
    assert v["ok"] is True, v


def test_validation_never_raises():
    """A broken index must degrade to a warning, not take the server
    down at startup."""
    import knn

    class Exploding:
        d = DIM

        def search(self, *a, **kw):
            raise RuntimeError("boom")

    v = knn.validate_index(Exploding())
    assert v["ok"] is None
    assert "boom" in v["reason"]
