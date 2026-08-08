"""Smoke tests for pipeline.py that don't need an Immich server.

We monkey-patch the heavy bits (ImmichClient, TreeOfLifeClassifier, YOLO)
with fakes so we can verify the chunked write path actually persists
predictions and detections for every asset — i.e. the regression we just
fixed where the BioCLIP loop was iterating a stale slice and writing empty
prediction rows.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

import numpy as np
from PIL import Image

# Ensure repo root is on sys.path so `import pipeline` works.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402
import storage  # noqa: E402


class _FakeAsset:
    def __init__(self, asset_id: str, name: str) -> None:
        self.id = asset_id
        self.original_file_name = name
        self.type = "IMAGE"


class _FakeImmichClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.base_url = base_url
        self.api_key = api_key

    def search_text(self, query: str, *, size: int, **kw) -> dict:
        return {
            "assets": {
                "items": [
                    {
                        "id": f"id-{i}",
                        "originalFileName": f"img_{i}.jpg",
                        "type": "IMAGE",
                    }
                    for i in range(size)
                ]
            }
        }

    def list_search_assets(self, payload):
        items = payload["assets"]["items"]
        return [_FakeAsset(it["id"], it["originalFileName"]) for it in items]

    def download_bytes(self, path: str) -> bytes:
        # 1×1 red PNG.
        from PIL import Image
        import io
        im = Image.new("RGB", (4, 4), (200, 80, 80))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def close(self):
        pass


class _FakeClassifier:
    """Returns a fixed-shape embedding per image and predictable top-k rows."""
    model_str = "fake-bioclip"

    def create_image_features(self, images, *, normalize: bool = True):
        n = len(images)
        arr = np.zeros((n, 8), dtype="<f4") + 0.1
        arr[:, 0] = np.linspace(0, 1, n)
        import torch
        return torch.from_numpy(arr)

    def predict(self, images, *, rank, k):
        from bioclip import Rank
        # Match the real API: returns a flat list of dicts with file_name.
        # Plantae scores are kept above the boundary threshold
        # (PLANT_MIN_PROB=0.90) so discover_plantae_assets accepts them.
        out = []
        for path in images:
            name = Path(path).name
            for pos in range(k):
                out.append(
                    {
                        "file_name": path,
                        "kingdom": "Plantae",
                        "family": f"Fam-{name}-{pos}",
                        "genus": f"Gen-{name}-{pos}",
                        "species": f"Species-{name}-{pos}",
                        "common_name": f"common-{pos}",
                        "score": 0.95 - pos * 0.1,
                    }
                )
        return out


def _fake_yolo(images, *, conf):
    # One detection per image at a fixed location.
    return [
        [{"label": "plant", "score": 0.42, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}]
        for _ in images
    ]


def test_chunked_pipeline_writes_predictions_for_every_asset():
    """With 5 assets and batch_size=2, predictions must land for all 5."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
             mock.patch.object(pipeline, "TreeOfLifeClassifier", _FakeClassifier), \
             mock.patch.object(pipeline.yolo, "detect", _fake_yolo), \
             mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
             mock.patch.object(pipeline.yolo, "unload_model", lambda: None):
            out = pipeline.run_pipeline(
                base_url="http://fake",
                api_key="fake",
                query="q",
                n_samples=5,
                top_k=3,
                db_path=db,
                batch_size=2,
            )

        assert out.n_new == 5, out
        with storage.connect(db) as conn:
            # Every asset must have predictions.
            pred_assets = {
                row["asset_id"]
                for row in conn.execute(
                    "SELECT DISTINCT asset_id FROM predictions"
                ).fetchall()
            }
            assert len(pred_assets) == 5, pred_assets
            # Every asset must have detections.
            det_assets = {
                row["asset_id"]
                for row in conn.execute(
                    "SELECT DISTINCT asset_id FROM detections"
                ).fetchall()
            }
            assert len(det_assets) == 5, det_assets
            # Each asset must have exactly top_k prediction rows.
            for asset_id in pred_assets:
                n = conn.execute(
                    "SELECT COUNT(*) AS n FROM predictions WHERE asset_id=?",
                    (asset_id,),
                ).fetchone()["n"]
                assert n == 3, (asset_id, n)
            # Each result must have non-empty top_predictions in the
            # returned RunResult (this is what the regression silently
            # produced zeros for).
            for r in out.results:
                assert r.top_predictions, r.asset_id
                assert len(r.top_predictions) == 3, r.asset_id
                assert r.detections, r.asset_id


def test_incremental_re_run_is_pure_cache():
    """Second run on the same query should report n_new=0, n_cached=5."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
             mock.patch.object(pipeline, "TreeOfLifeClassifier", _FakeClassifier), \
             mock.patch.object(pipeline.yolo, "detect", _fake_yolo), \
             mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
             mock.patch.object(pipeline.yolo, "unload_model", lambda: None):
            first = pipeline.run_pipeline(
                base_url="http://fake",
                api_key="fake",
                query="q",
                n_samples=5,
                top_k=3,
                db_path=db,
                batch_size=2,
            )
            second = pipeline.run_pipeline(
                base_url="http://fake",
                api_key="fake",
                query="q",
                n_samples=5,
                top_k=3,
                db_path=db,
                batch_size=2,
            )

        assert first.n_new == 5
        assert second.n_new == 0, second
        assert second.n_cached == 5, second


def test_knn_reclassify_writes_row_and_cache_hits():
    """End-to-end smoke: build a tiny synthetic FAISS index, insert a
    fake asset + embedding, run_knn_reclassify, assert the row is
    persisted and that a second call is fully cached."""
    import tempfile, os
    from pathlib import Path
    import numpy as np
    import storage, knn
    from pipeline import run_knn_reclassify

    with tempfile.TemporaryDirectory() as d:
        dbp = os.path.join(d, "t.db")
        idx_dir = Path(d) / "idx"
        storage.init_db(dbp)

        # Build a tiny 5-row index by hand so we don't depend on HF. Use the
# production EMB_DIM=768 so the test catches dimension-mismatch bugs.
        # All 5 reference rows share the SAME species label so a self-
# query (asset embedding == reference row 0) returns that species
# unanimously — the score ends up ~1.0 from the weighted vote.
        idx_dir.mkdir()
        dim = 768
        rng = np.random.default_rng(42)
        vecs = rng.normal(size=(5, dim)).astype("f4")
        vecs /= np.linalg.norm(vecs, axis=1, keepdims=True)
        faiss_idx = faiss_mod.IndexFlatIP(dim)
        faiss_idx.add(vecs)
        faiss_mod.write_index(faiss_idx, str(idx_dir / "tol200m_bioclip2.faiss"))
        # Labels aligned with the FAISS row order.
        import pyarrow as pa, pyarrow.parquet as pq
        labels = pa.Table.from_pylist([
            {"uuid": f"u{i}", "kingdom": "Animalia", "phylum": None, "class": None,
             "order": None, "family": None, "genus": None, "species": "sp-shared",
             "scientific_name": None, "common_name": "shared-species",
             "identifier": None, "source_dataset": None}
            for i in range(5)
        ])
        pq.write_table(labels, str(idx_dir / "labels.parquet"))
        (idx_dir / "INDEX_VERSION.txt").write_text("synthetic-v1\nn_rows=5\n")

        # Insert one asset whose embedding = the first reference vector
        # (so the KNN should find itself as top-1 with sim=1.0).
        with storage.connect(dbp) as conn:
            storage.upsert_asset(conn, asset_id="a1",
                                 original_file_name="x.jpg",
                                 original_url="u", thumbnail_url="t")
            storage.upsert_embedding(conn, asset_id="a1", vector=vecs[0].astype("<f4"))
            conn.commit()

        # 1st call — should classify.
        with storage.connect(dbp) as conn:
            stats = run_knn_reclassify(conn, asset_ids=["a1"], index_dir=idx_dir, k=3)
            assert stats["scanned"] == 1 and stats["cached"] == 0
            assert storage.knn_predictions_have_asset(
                conn, asset_id="a1", index_version="synthetic-v1", k=3
            )
            fetched = storage.fetch_knn_predictions(
                conn, asset_ids=["a1"], index_version="synthetic-v1", k=3
            )
            assert fetched["a1"]["top_species"] == "sp-shared"
            # Score is the weighted vote mass for the top species as a
            # fraction of total vote mass. With all neighbors sharing
            # one species label, the consensus is unanimous.
            assert fetched["a1"]["score"] > 0.95
            assert len(fetched["a1"]["neighbors"]) == 3

        # 2nd call — should be entirely cache-hit.
        with storage.connect(dbp) as conn:
            stats2 = run_knn_reclassify(conn, asset_ids=["a1"], index_dir=idx_dir, k=3)
            assert stats2["scanned"] == 0 and stats2["cached"] == 1

        # enrich_with_knn attaches the row to a result bucket.
        with storage.connect(dbp) as conn:
            buckets = [{"asset_id": "a1"}]
            storage.enrich_with_knn(conn, buckets)
            assert "knn" in buckets[0]
            assert buckets[0]["knn"]["top_species"] == "sp-shared"


# Local import so the test file works without pytest.
try:
    import faiss as faiss_mod
except ImportError:
    faiss_mod = None  # noqa: F841 — only used by the KNN test