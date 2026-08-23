"""Regression tests for the search-loop cache-hit path.

Background: in commit 069ec9a I bulked the per-asset SELECTs that
the search loop did for cached scores. The new code read from
`bulk_asset_rows` — which queries the `assets` table — for the
Plantae/Fungi values, but those columns live in `taxon_scores`.
Result: every cached asset got Plantae=0.0 / Fungi=0.0, the
boundary logic saw 0 of 32 plants, accepted the first batch as a
"non-plant boundary", and the run stopped after 32 photos
regardless of the requested n_samples.

These tests pin that behavior:
- The cache-hit path must read cached scores from taxon_scores,
  not from the assets row.
- A run on a fully-cached collection with high Plantae scores
  should fill up to n_samples, not stop at 32.

We don't need BioCLIP, Immich, or YOLO for these tests — the
search loop returns after finding candidates and stops scoring.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path
from unittest import mock

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
    """Returns N pages of M fake assets. The pipeline will request
    pages until `n_samples` is reached or the search boundary logic
    decides to stop."""

    def __init__(self, base_url: str, api_key: str, *, total_assets: int,
                 page_size: int = 500) -> None:
        self.base_url = base_url
        self.api_key = api_key
        self.total_assets = total_assets
        self.page_size = page_size

    def search_text(self, query: str, *, size: int, **kw) -> dict:
        n = min(size, self.total_assets)
        return {
            "assets": {
                "items": [
                    {
                        "id": f"id-{i:06d}",
                        "originalFileName": f"img_{i:06d}.jpg",
                        "type": "IMAGE",
                    }
                    for i in range(n)
                ]
            }
        }

    def list_search_assets(self, payload):
        items = payload["assets"]["items"]
        return [_FakeAsset(it["id"], it["originalFileName"]) for it in items]

    def download_bytes(self, path: str) -> bytes:
        # 1x1 red PNG.
        from PIL import Image
        import io
        im = Image.new("RGB", (4, 4), (200, 80, 80))
        buf = io.BytesIO()
        im.save(buf, format="PNG")
        return buf.getvalue()

    def get_asset_metadata(self, asset_id):
        # The species stage re-fetches EXIF for every asset (a
        # separate cache-skip gap from the search loop one). The fake
        # returns an empty metadata dict — upsert_asset() handles
        # captured_at=None gracefully.
        return {}

    @staticmethod
    def extract_exif(metadata_response):
        # The cache-hit path skips this (assets already have captured_at).
        # If a cache miss triggers it, return empty defaults.
        return {"captured_at": None, "latitude": None, "longitude": None}

    def close(self):
        pass


def _make_fake_client_class(client_instance):
    """Wrap a _FakeImmichClient instance in a class with the
    `extract_exif` staticmethod so monkey-patched `pipeline.ImmichClient`
    exposes the right attributes. The pipeline calls both
    `ImmichClient(base_url, api_key)` (instantiation) and
    `ImmichClient.extract_exif(response)` (static)."""

    class _Wrapper:
        extract_exif = staticmethod(
            lambda resp: {"captured_at": None, "latitude": None,
                          "longitude": None}
        )

        def __new__(cls, base_url, api_key):
            client_instance.base_url = base_url
            client_instance.api_key = api_key
            return client_instance

    return _Wrapper


class _FakeClassifier:
    """We don't need BioCLIP — the cache-hit path should never
    instantiate it. If the search loop regresses and starts calling
    predict(), the fake will be called and the test can assert so."""

    def __init__(self):
        self.predict_calls = 0
        self.create_image_features_calls = 0

    def predict(self, images, *, rank, k):
        from bioclip import Rank
        self.predict_calls += 1
        out = []
        for path in images:
            for pos in range(k):
                out.append({
                    "file_name": path,
                    "kingdom": "Plantae",
                    "family": "Fam",
                    "genus": "Gen",
                    "species": "sp",
                    "common_name": "common",
                    "score": 0.95,
                })
        return out

    def create_image_features(self, pil_images, *, normalize=True):
        """Returns a tensor of zeros — the species embedding isn't
        used in this test because all assets are pre-cached. We just
        need the call to not raise."""
        import torch
        self.create_image_features_calls += 1
        return torch.zeros(len(pil_images), 8)


def _seed_cached_scores(db_path: str, asset_ids: list[str],
                         plantae: float = 0.95, fungi: float = 0.02) -> None:
    """Pre-populate `taxon_scores` with high-Plantae rows so the
    cache-hit path returns real (non-zero) scores. Without this seed,
    the buggy `bulk_asset_rows` codepath would silently report 0.0
    and the boundary logic would stop after the first batch.

    Also populates `assets.captured_at` so the EXIF-skip fast path
    triggers (the test's _FakeImmichClient doesn't implement
    get_asset_metadata), and seeds `predictions` rows so the species
    stage is a full cache hit too (no BioCLIP predict needed).
    """
    storage.init_db(db_path)
    with storage.connect(db_path) as conn:
        for aid in asset_ids:
            storage.upsert_asset(
                conn, asset_id=aid,
                original_file_name=f"{aid}.jpg",
                original_url=f"u/{aid}",
                thumbnail_url=f"t/{aid}",
                captured_at="2025-01-01T00:00:00.000+00:00",
                latitude=None,
                longitude=None,
            )
            storage.upsert_taxon_scores(
                conn,
                asset_id=aid,
                rank="kingdom",
                scores={"Plantae": plantae, "Fungi": fungi},
            )
            # Seed species-rank predictions directly via SQL — no
            # public upsert helper. top_k=3 rows per asset so the
            # species stage treats it as a complete cache hit.
            for pos in range(3):
                conn.execute(
                    "INSERT OR REPLACE INTO predictions "
                    "(asset_id, rank, position, species, common_name, "
                    " score, kingdom, family, genus) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (aid, "species", pos, f"sp-{aid}-{pos}", "",
                     0.95 - pos * 0.1, "Plantae", "Fam", "Gen"),
                )
        conn.commit()


class _PatchedRun:
    """Context manager that patches the heavy collaborators so
    discover_plantae_assets runs end-to-end on a temp SQLite DB."""

    def __init__(self, client, clf):
        self.client = client
        self.clf = clf
        self._cm = None

    def __enter__(self):
        self._cm = mock.patch.multiple(
            pipeline,
            ImmichClient=_make_fake_client_class(self.client),
            _resolve_classifier=lambda: self.clf,
        )
        self._cm.__enter__()
        # Patch YOLO globally so the yolo stage doesn't crash.
        self._yolo_cm = mock.patch.multiple(
            pipeline.yolo,
            detect=lambda *a, **kw: [],
            get_model=lambda: None,
            unload_model=lambda: None,
        )
        self._yolo_cm.__enter__()
        # Patch the clusters module that's lazily imported in
        # run_pipeline. The lazy import binds `_clusters` to the
        # `clusters` module, so we patch the module itself.
        import clusters as _clusters_module
        self._clusters_cm = mock.patch.object(
            _clusters_module, "assign_new_points",
            lambda **kw: ({}, 0),
        )
        self._clusters_cm.__enter__()
        return self

    def __exit__(self, *a):
        self._clusters_cm.__exit__(*a)
        self._yolo_cm.__exit__(*a)
        self._cm.__exit__(*a)


def test_search_loop_reads_cached_scores_from_taxon_scores_not_assets():
    """Regression: bulk_asset_rows returns file/url/captured_at only.
    Reading Plantae/Fungi from there returned None and produced 0.0
    for every cached asset, which made the boundary logic stop after
    32 photos. Fix: bulk_fetch_taxon_scores from taxon_scores.

    This test runs the pipeline against a fully-cached collection
    with HIGH Plantae scores. With the bug, the run returns
    n_cached=32. After the fix, n_cached equals n_samples.
    """
    n_samples = 200
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        all_ids = [f"id-{i:06d}" for i in range(n_samples)]
        _seed_cached_scores(db, all_ids)

        client = _FakeImmichClient("http://fake", "fake",
                                    total_assets=n_samples)
        clf = _FakeClassifier()

        with _PatchedRun(client, clf):
            out = pipeline.run_pipeline(
                base_url="http://fake",
                api_key="fake",
                query="q",
                n_samples=n_samples,
                top_k=3,
                db_path=db,
                batch_size=16,
            )

        # The regression: with the bug, the search loop saw every
        # cached asset as Plantae=0.0, hit the 0/32 < 30% branch,
        # added 32, and returned. After the fix, all n_samples
        # cached assets should pass through.
        assert out.n_cached == n_samples, (
            f"expected n_cached={n_samples}, got n_cached={out.n_cached} "
            f"(n_new={out.n_new}). "
            "If this is 32, the search loop is reading cached scores "
            "from the wrong table."
        )
        assert out.n_new == 0, out

        # And confirm BioCLIP predict was never called — every
        # asset was a cache hit, so no predict work.
        assert clf.predict_calls == 0, clf.predict_calls


def test_search_loop_zero_score_cached_assets_get_boundary_still():
    """Sanity: when cached Plantae scores ARE legitimately 0.0 (i.e.
    the photos are not plants), the boundary logic should still kick
    in and stop the search after the first batch of non-plants.

    This pins the *intended* behavior so the bug fix doesn't
    accidentally remove the boundary check.
    """
    n_samples = 200
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        all_ids = [f"id-{i:06d}" for i in range(n_samples)]
        # Real scores of 0.0 — these photos truly are not plants.
        _seed_cached_scores(db, all_ids, plantae=0.0, fungi=0.0)

        client = _FakeImmichClient("http://fake", "fake",
                                    total_assets=n_samples)
        clf = _FakeClassifier()

        with _PatchedRun(client, clf):
            out = pipeline.run_pipeline(
                base_url="http://fake",
                api_key="fake",
                query="q",
                n_samples=n_samples,
                top_k=3,
                db_path=db,
                batch_size=16,
            )

        # Photos are genuinely not plants. Boundary kicks in
        # after the first batch of 32 non-plant assets.
        # We don't pin the exact number (the boundary logic uses
        # `len(scored) >= 24` and there's some minor noise from
        # dedup), but it should be FAR less than n_samples.
        assert out.n_cached < n_samples // 2, out
        # And BioCLIP predict was still never called — cache hit
        # is a hit, even when the score is 0.
        assert clf.predict_calls == 0


def test_storage_bulk_fetch_taxon_scores_matches_per_asset_path():
    """Direct storage-layer test: bulk_fetch_taxon_scores must
    return the same {asset_id: {taxon_name: score}} as the per-asset
    fetch_taxon_scores loop. If these drift apart, the search loop
    bug returns."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        ids = [f"id-{i}" for i in range(5)]
        _seed_cached_scores(db, ids, plantae=0.9, fungi=0.05)

        # Per-asset ground truth
        per_asset = {}
        with storage.connect(db) as conn:
            for aid in ids:
                per_asset[aid] = storage.fetch_taxon_scores(
                    conn, asset_ids=[aid], rank="kingdom"
                )[aid]

        # Bulk path
        with storage.connect(db) as conn:
            bulk = storage.bulk_fetch_taxon_scores(
                conn, asset_ids=ids, rank="kingdom"
            )

        assert bulk == per_asset, (bulk, per_asset)


# Tiny test runner so failures actually propagate exit code. Without
# this, exceptions in test functions just bubble up and Python exits
# 0 — same as success.
if __name__ == "__main__":
    import inspect
    import traceback

    funcs = [
        (name, obj)
        for name, obj in globals().items()
        if name.startswith("test_") and callable(obj)
    ]
    failures = 0
    for name, fn in funcs:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            print(f"FAIL  {name}")
            traceback.print_exc()
            failures += 1
    if failures:
        print(f"\n{failures} test(s) failed")
        sys.exit(1)
    print(f"\n{len(funcs)} test(s) passed")
    sys.exit(0)
