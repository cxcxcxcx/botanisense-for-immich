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

    # Pipeline calls ImmichClient.extract_exif(client.get_asset_metadata(id))
    # — return None so the fixture yields {"captured_at": None, ...} and
    # the test stays focused on the prediction-write path, not EXIF.
    def get_asset_metadata(self, asset_id: str) -> dict | None:
        return None

    # Mirror of the real static method on ImmichClient (which the test
    # monkey-patches out). Tests don't care about EXIF, just that the
    # call returns a dict with the expected keys.
    @staticmethod
    def extract_exif(asset_payload: dict | None) -> dict:
        return {"captured_at": None, "latitude": None, "longitude": None}

    def close(self):
        pass


class _FakeClassifier:
    """Returns a fixed-shape embedding per image and predictable top-k rows.

    Mirrors the real TreeOfLifeClassifier surface the pipeline uses:
    create_image_features + get_txt_embeddings + create_probabilities +
    format_*_probs (the encode-once path), plus predict() for the bbox
    stage which still classifies from file paths.
    """
    model_str = "fake-bioclip"
    device = "cpu"

    def __init__(self):
        self.predict_calls = 0
        # Total crop paths handed to predict() across all calls —
        # predict() is only used by the bbox stage, so this counts
        # bbox crops BioCLIP actually classified.
        self.predict_images = 0
        self.create_image_features_calls = 0
        # Total number of IMAGES passed through encode_image across all
        # calls — the encode-once regression test asserts this equals the
        # asset count, not 2-3x it.
        self.images_encoded = 0

    def get_txt_embeddings(self):
        import torch
        # (embed_dim, n_classes) — matches the real text-embedding layout.
        return torch.zeros(8, 5)

    def get_current_txt_names(self):
        # Per-row name_ary = [[kingdom, ...ranks], common] —
        # _kingdom_bins reads name_ary[0][0] as the kingdom. 4 of the
        # 5 label rows Plantae so the uniform fake softmax (1/5 each)
        # sums to 0.8 > PLANT_MIN_PROB and the search boundary accepts.
        names = []
        for i in range(5):
            kingdom = "Plantae" if i < 4 else "Fungi"
            names.append([[kingdom, "ph", "cl", "or", "fam", "gen", "sp"], "common"])
        return names

    def create_image_features(self, images, *, normalize: bool = True):
        n = len(images)
        self.create_image_features_calls += 1
        self.images_encoded += n
        arr = np.zeros((n, 8), dtype="<f4") + 0.1
        arr[:, 0] = np.linspace(0, 1, n)
        import torch
        return torch.from_numpy(arr)

    def create_probabilities(self, feats, txt_embeddings):
        import torch
        # Uniform-ish probs; the formatting fakes ignore the values.
        n = feats.shape[0]
        return torch.softmax(
            torch.zeros(n, txt_embeddings.shape[1]) + 0.2, dim=1
        )

    def format_species_probs(self, image_key, probs, k=5):
        name = Path(image_key).name
        return [
            {
                "file_name": image_key,
                "kingdom": "Plantae",
                "family": f"Fam-{name}-{pos}",
                "genus": f"Gen-{name}-{pos}",
                "species": f"Species-{name}-{pos}",
                "common_name": f"common-{pos}",
                "score": 0.95 - pos * 0.1,
            }
            for pos in range(k)
        ]

    def format_grouped_probs(self, image_key, probs, rank, min_prob=1e-9, k=5):
        # Kingdom-rank rows: Plantae on top so the search boundary
        # (PLANT_MIN_PROB) accepts and kingdom scores persist.
        rows = [
            {"file_name": image_key, "kingdom": "Plantae", "score": 0.95},
            {"file_name": image_key, "kingdom": "Fungi", "score": 0.02},
        ]
        return rows[:k]

    def predict(self, images, *, rank, k):
        self.predict_calls += 1
        self.predict_images += len(images)
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


class _CountingYolo:
    """Wraps a detect() callable with a call/image counter so tests can
    assert how many images actually went through the detection scan."""

    def __init__(self, inner=_fake_yolo):
        self.calls = 0
        self.images = 0
        self._inner = inner

    def __call__(self, images, *, conf):
        self.calls += 1
        self.images += len(images)
        return self._inner(images, conf=conf)


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


def test_species_stage_reuses_search_embeddings():
    """Regression for the double-encode slowdown: the species stage used
    to call create_image_features() AND clf.predict(), with predict()
    re-encoding every image internally (BioCLIP ViT-L/14 on CPU — ~half
    the stage's wall time), while the search stage had already encoded
    the same images for kingdom scoring. Now search encodes once and
    persists the embedding; the species stage classifies from stored
    features and must encode ZERO images itself.

    The fake search client returns 500 results; with decision batches of
    4 (batch_size=2 × 2), search scores two batches (8 images) before
    reaching n_samples=5. So exactly 8 encodes — all in search — is the
    expected total; a species-stage regression would make it 13.
    """
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        # _resolve_classifier caches in a module global; clear it so our
        # patched instance (with counters) is actually constructed.
        pipeline._clf_cache = None
        clf = _FakeClassifier()
        try:
            with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                 mock.patch.object(pipeline, "TreeOfLifeClassifier", lambda: clf), \
                 mock.patch.object(pipeline.yolo, "detect", _fake_yolo), \
                 mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
                 mock.patch.object(pipeline.yolo, "unload_model", lambda: None):
                pipeline.run_pipeline(
                    base_url="http://fake",
                    api_key="fake",
                    query="q",
                    n_samples=5,
                    top_k=3,
                    db_path=db,
                    batch_size=2,
                )
                assert clf.images_encoded == 8, (
                    f"expected 8 encodes (all in search batches of 4), "
                    f"got {clf.images_encoded} — the species stage is "
                    f"re-encoding images it should reuse"
                )
                # Search persisted one vector per scored image — that's
                # what the species stage reused.
                with storage.connect(db) as conn:
                    n_emb = conn.execute(
                        "SELECT COUNT(*) AS n FROM embeddings"
                    ).fetchone()["n"]
                    assert n_emb == 8, n_emb

                # Fully cached re-run: no image touches the vision tower.
                clf.images_encoded = 0
                pipeline.run_pipeline(
                    base_url="http://fake",
                    api_key="fake",
                    query="q",
                    n_samples=5,
                    top_k=3,
                    db_path=db,
                    batch_size=2,
                )
                assert clf.images_encoded == 0, (
                    f"cached re-run encoded {clf.images_encoded} images"
                )
        finally:
            pipeline._clf_cache = None


def test_yolo_backend_dispatch(monkeypatch):
    """RTMDet is the default backend; obj6 (and its pre-rename id) is
    opt-in; the removed YOLO-World id fails loudly instead of silently
    falling back."""
    import pytest

    monkeypatch.delenv("YOLO_BACKEND", raising=False)
    assert pipeline._yolo_backend_module() == "yolo_rtmdet"
    for value, module in (("rtmdet", "yolo_rtmdet"), ("obj6", "yolo_obj6"),
                          ("gemini1", "yolo_obj6")):
        monkeypatch.setenv("YOLO_BACKEND", value)
        assert pipeline._yolo_backend_module() == module
    monkeypatch.setenv("YOLO_BACKEND", "world")
    with pytest.raises(ValueError, match="removed"):
        pipeline._yolo_backend_module()


def test_yolo_resumes_after_species_completed():
    """Regression for the yolo checkpoint gap: yolo candidacy used to be
    gated on missing species predictions, so a run that completed the
    species stage and then died/stopped/errored in the yolo stage would
    NEVER detection-scan those assets on re-run (they were full species
    cache hits → excluded from needs_yolo). Now the yolo worklist is
    gated on the independent assets.yolo_scanned_at marker: a run that
    finds species predictions cached but the marker NULL scans the
    assets anyway, setting the marker (and detections)."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        pipeline._clf_cache = None
        yolo_calls = _CountingYolo()
        try:
            with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                 mock.patch.object(pipeline, "TreeOfLifeClassifier", _FakeClassifier), \
                 mock.patch.object(pipeline.yolo, "detect", yolo_calls), \
                 mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
                 mock.patch.object(pipeline.yolo, "unload_model", lambda: None):
                pipeline.run_pipeline(
                    base_url="http://fake", api_key="fake", query="q",
                    n_samples=5, top_k=3, db_path=db, batch_size=2,
                )
                assert yolo_calls.images > 0  # run 1 scanned

                # Simulate a crash between species and yolo: keep every
                # species prediction but clear the yolo marker (and any
                # detections).
                with storage.connect(db) as conn:
                    conn.execute("UPDATE assets SET yolo_scanned_at = NULL")
                    conn.execute("DELETE FROM detections")
                    conn.commit()

                yolo_calls.images = 0
                out2 = pipeline.run_pipeline(
                    base_url="http://fake", api_key="fake", query="q",
                    n_samples=5, top_k=3, db_path=db, batch_size=2,
                )
                # The 5 run assets must have been re-scanned even though
                # every species prediction was cached.
                assert yolo_calls.images >= 5, yolo_calls.images
                run_ids = [r.asset_id for r in out2.results]
                with storage.connect(db) as conn:
                    n_unscanned = conn.execute(
                        f"SELECT COUNT(*) AS n FROM assets "
                        f"WHERE yolo_scanned_at IS NULL "
                        f"AND asset_id IN ({','.join('?' for _ in run_ids)})",
                        run_ids,
                    ).fetchone()["n"]
                    assert n_unscanned == 0, n_unscanned
                    det_assets = {
                        r["asset_id"]
                        for r in conn.execute(
                            f"SELECT DISTINCT asset_id FROM detections "
                            f"WHERE label = 'plant' "
                            f"AND asset_id IN ({','.join('?' for _ in run_ids)})",
                            run_ids,
                        ).fetchall()
                    }
                    assert set(run_ids) <= det_assets, det_assets
        finally:
            pipeline._clf_cache = None


def test_yolo_scanned_marker_skips_rescan():
    """The flip side: once yolo_scanned_at is set, a fully-cached re-run
    must not call the detection model at all."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        pipeline._clf_cache = None
        yolo_calls = _CountingYolo()
        try:
            with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                 mock.patch.object(pipeline, "TreeOfLifeClassifier", _FakeClassifier), \
                 mock.patch.object(pipeline.yolo, "detect", yolo_calls), \
                 mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
                 mock.patch.object(pipeline.yolo, "unload_model", lambda: None):
                pipeline.run_pipeline(
                    base_url="http://fake", api_key="fake", query="q",
                    n_samples=5, top_k=3, db_path=db, batch_size=2,
                )
                yolo_calls.images = 0
                pipeline.run_pipeline(
                    base_url="http://fake", api_key="fake", query="q",
                    n_samples=5, top_k=3, db_path=db, batch_size=2,
                )
                assert yolo_calls.images == 0, (
                    f"cached re-run re-scanned {yolo_calls.images} images"
                )
        finally:
            pipeline._clf_cache = None


def test_replace_detections_deletes_all_labels():
    """The yolo stage relies on unscoped replace: it derives plant,
    person AND hand boxes from the same forward pass, so any earlier
    rows (e.g. person rows appended by the standalone backfill) are
    stale and must be wiped, not preserved."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        storage.init_db(db)
        with storage.connect(db) as conn:
            storage.upsert_asset(conn, asset_id="a1",
                                 original_file_name="x.jpg",
                                 original_url="u", thumbnail_url="t")
            storage.append_detections(conn, asset_id="a1", detections=[
                {"label": "person", "score": 0.8, "x": 0, "y": 0, "w": 1, "h": 1},
                {"label": "hand", "score": 0.6, "x": 0, "y": 0, "w": .2, "h": .2},
            ])
            storage.replace_detections(
                conn, asset_id="a1",
                detections=[
                    {"label": "plant", "score": 0.42,
                     "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4}
                ],
            )
            rows = [
                r["label"]
                for r in conn.execute(
                    "SELECT label FROM detections WHERE asset_id='a1'"
                ).fetchall()
            ]
            assert rows == ["plant"], rows


def _person_yolo(images, *, conf):
    """Alternating real-person / hand-silhouette photos (both with a
    plant box, since these are plantae candidates):

    - even positions (0, 2, 4): person box + hand box that STICKS OUT
      of the body (containment 0) → has_person=1.
    - odd positions (1, 3): person silhouette with a tight hand box
      fully INSIDE it → has_real_person() suppresses → has_person=0.
    """
    out = []
    for pos, _img in enumerate(images):
        if pos % 2 == 0:
            person = {"label": "person", "score": 0.8, "x": 0.0, "y": 0.0, "w": 0.4, "h": 1.0}
            hand = {"label": "hand", "score": 0.6, "x": 0.7, "y": 0.4, "w": 0.2, "h": 0.2}
        else:
            person = {"label": "person", "score": 0.7, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0}
            hand = {"label": "hand", "score": 0.6, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2}
        out.append([
            {"label": "plant", "score": 0.42, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            person,
            hand,
        ])
    return out


def test_yolo_stage_persists_person_boxes_and_has_person():
    """The yolo model's label table already includes person/face/human/
    hand, so ONE forward pass must persist plant + person + hand boxes
    AND stamp has_person — leaving nothing for the person stage to
    re-scan. Regression: the yolo stage used to throw person boxes away
    and the person stage re-ran the identical model over the same
    images (~21 min of duplicated YOLO per 10k fresh assets), and hand
    boxes landed in BOTH stages' writes (duplicated rows)."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        pipeline._clf_cache = None
        clf = _FakeClassifier()
        yolo_calls = _CountingYolo(inner=_person_yolo)
        try:
            with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                 mock.patch.object(pipeline, "TreeOfLifeClassifier", lambda: clf), \
                 mock.patch.object(pipeline.yolo, "detect", yolo_calls), \
                 mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
                 mock.patch.object(pipeline.yolo, "unload_model", lambda: None), \
                 mock.patch.object(pipeline.face, "detect_faces",
                                   lambda img, **k: []):
                pipeline.run_pipeline(
                    base_url="http://fake", api_key="fake", query="q",
                    n_samples=5, top_k=3, db_path=db, batch_size=2,
                )

                # ONE pass per asset. 8 = the 5 run candidates in the
                # yolo stage + the 3 evaluated-but-not-selected assets
                # (in `assets` with has_person NULL) that the person
                # stage sweeps as residue. The old code scanned each of
                # the 5 candidates TWICE (yolo stage + person stage) —
                # 13 images here.
                assert yolo_calls.images == 8, yolo_calls.images

                with storage.connect(db) as conn:
                    run_ids = [row["asset_id"] for row in conn.execute(
                        "SELECT asset_id FROM assets WHERE yolo_scanned_at "
                        "IS NOT NULL"
                    ).fetchall()]
                    assert len(run_ids) == 5, run_ids
                    for aid in run_ids:
                        counts = {
                            row["label"]: row["n"]
                            for row in conn.execute(
                                "SELECT label, COUNT(*) AS n FROM detections "
                                "WHERE asset_id=? GROUP BY label",
                                (aid,),
                            ).fetchall()
                        }
                        # Exactly one box per label — no duplicated hand
                        # rows from the person stage re-appending.
                        assert counts == {
                            "plant": 1, "person": 1, "hand": 1,
                        }, (aid, counts)
                    # has_person must be set (never NULL — NULL would
                    # re-enter the person sweep) and reflect the
                    # suppression rule, not raw person-box presence.
                    flags = {
                        row["has_person"]: row["n"]
                        for row in conn.execute(
                            "SELECT has_person, COUNT(*) AS n FROM assets "
                            "WHERE yolo_scanned_at IS NOT NULL "
                            "GROUP BY has_person"
                        ).fetchall()
                    }
                    assert flags == {1: 3, 0: 2}, flags

                # The bbox stage must classify ONLY the 5 plant crops —
                # person (0.8) and hand (0.6) crops all clear
                # BBOX_BIOCLIP_MIN=0.15 but are skipped (BioCLIP is
                # plant-focused). Pre-fix this was 15.
                assert clf.predict_images == 5, clf.predict_images

                # And a cached re-run scans nothing at all — neither
                # stage re-opens an image.
                yolo_calls.images = 0
                with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                     mock.patch.object(pipeline, "TreeOfLifeClassifier", _FakeClassifier), \
                     mock.patch.object(pipeline.yolo, "detect", yolo_calls), \
                     mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
                     mock.patch.object(pipeline.yolo, "unload_model", lambda: None):
                    second = pipeline.run_pipeline(
                        base_url="http://fake", api_key="fake", query="q",
                        n_samples=5, top_k=3, db_path=db, batch_size=2,
                    )
                assert second.n_cached == 5, second
                assert yolo_calls.images == 0, yolo_calls.images
        finally:
            pipeline._clf_cache = None


def _person_yolo_big_contained_hand(images, *, conf):
    """Every photo: person + hand-silhouette (4% of frame, fully
    contained) — has_real_person() suppresses, so every asset is a
    YuNet-rescue candidate."""
    return [
        [
            {"label": "plant", "score": 0.42, "x": 0.1, "y": 0.2, "w": 0.3, "h": 0.4},
            {"label": "person", "score": 0.8, "x": 0.0, "y": 0.0, "w": 1.0, "h": 1.0},
            {"label": "hand", "score": 0.6, "x": 0.1, "y": 0.1, "w": 0.2, "h": 0.2},
        ]
        for _img in images
    ]


def _face_inside(img, **k):
    return [{"label": "face", "score": 0.91,
             "x": 0.4, "y": 0.4, "w": 0.2, "h": 0.2}]


def test_yolo_stage_yunet_rescue_flips_suppressed_photo():
    """A photo whose hand suppression fired (person box exists, hand
    fully contained, no YOLO face) must flip to has_person=1 when YuNet
    finds a face inside the person box — and the rescued `face` row must
    be persisted alongside the other tiers, while the bbox stage still
    classifies only the plant crop."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        pipeline._clf_cache = None
        clf = _FakeClassifier()
        yolo_calls = _CountingYolo(inner=_person_yolo_big_contained_hand)
        try:
            with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                 mock.patch.object(pipeline, "TreeOfLifeClassifier", lambda: clf), \
                 mock.patch.object(pipeline.yolo, "detect", yolo_calls), \
                 mock.patch.object(pipeline.yolo, "get_model", lambda: None), \
                 mock.patch.object(pipeline.yolo, "unload_model", lambda: None), \
                 mock.patch.object(pipeline.face, "detect_faces", _face_inside):
                pipeline.run_pipeline(
                    base_url="http://fake", api_key="fake", query="q",
                    n_samples=5, top_k=3, db_path=db, batch_size=2,
                )

                with storage.connect(db) as conn:
                    rows = conn.execute(
                        "SELECT has_person, COUNT(*) AS n FROM assets "
                        "WHERE yolo_scanned_at IS NOT NULL "
                        "GROUP BY has_person"
                    ).fetchall()
                    assert {r["has_person"]: r["n"] for r in rows} == {1: 5}, rows

                    for aid in ("id-0", "id-1", "id-4"):
                        counts = {
                            row["label"]: row["n"]
                            for row in conn.execute(
                                "SELECT label, COUNT(*) AS n FROM detections "
                                "WHERE asset_id=? GROUP BY label",
                                (aid,),
                            ).fetchall()
                        }
                        # Exactly one face row per asset — the face tier
                        # must not double-persist via the person tier
                        # ("face" is a PERSON_LABELS member).
                        assert counts == {
                            "plant": 1, "person": 1, "hand": 1, "face": 1,
                        }, (aid, counts)

                # bbox stage classifies only the plant crop — face is a
                # PERSON_LABELS member and is excluded like person/hand.
                assert clf.predict_images == 5, clf.predict_images
        finally:
            pipeline._clf_cache = None


def test_species_classifies_assets_with_stored_embedding_only():
    """An asset that has kingdom scores + an embedding but NO species
    predictions (e.g. vector persisted by an earlier version's search
    stage) must get species-classified from the STORED vector — zero
    image encodes in the species stage."""
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        storage.init_db(db)
        # Seed the first two search batches (ids id-0..id-7) as fully
        # kingdom-cached with embeddings but no species predictions.
        with storage.connect(db) as conn:
            for i in range(8):
                aid = f"id-{i}"
                storage.upsert_asset(
                    conn, asset_id=aid, original_file_name=f"{aid}.jpg",
                    original_url="u", thumbnail_url="t",
                    captured_at="2025-01-01T00:00:00.000+00:00",
                    latitude=None, longitude=None,
                )
                storage.upsert_taxon_scores(
                    conn, asset_id=aid, rank="kingdom",
                    scores={"Plantae": 0.95},
                )
                storage.upsert_embedding(
                    conn, asset_id=aid,
                    vector=(np.zeros(8, dtype="<f4") + 0.1),
                )
            conn.commit()

        pipeline._clf_cache = None
        clf = _FakeClassifier()
        try:
            with mock.patch.object(pipeline, "ImmichClient", _FakeImmichClient), \
                 mock.patch.object(pipeline, "TreeOfLifeClassifier", lambda: clf), \
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
            assert clf.images_encoded == 0, (
                f"species stage encoded {clf.images_encoded} images "
                f"despite stored embeddings"
            )
            with storage.connect(db) as conn:
                for r in out.results:
                    n = conn.execute(
                        "SELECT COUNT(*) AS n FROM predictions "
                        "WHERE asset_id=? AND rank='species'",
                        (r.asset_id,),
                    ).fetchone()["n"]
                    assert n == 3, (r.asset_id, n)
        finally:
            pipeline._clf_cache = None


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