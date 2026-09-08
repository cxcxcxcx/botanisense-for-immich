"""The search stage must COMMIT per batch, not hold one run-long
transaction.

storage.connect() only commits when its with-block exits — which for
run_pipeline is the END of the entire multi-hour run. Every batch's
asset rows + kingdom scores used to sit in that one open transaction,
so (a) /api/results served an empty DB for the whole run and (b) any
stop/crash rolled the run back to nothing (the "ran for hours, every
page is empty" report).

This test drives discover_plantae_assets with fakes and, from a
SECOND connection (the uvicorn-reader position), checks that page 1's
rows are already visible when page 2 is requested.
"""

import contextlib
import sqlite3
import sys
from io import BytesIO
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline
import storage


# --- fakes -----------------------------------------------------------------


def _jpeg_bytes() -> bytes:
    from PIL import Image

    buf = BytesIO()
    Image.new("RGB", (4, 4), (20, 120, 50)).save(buf, format="JPEG")
    return buf.getvalue()


class _FakeClient:
    """Two pages: the second one snapshots what another connection can
    see mid-run, then returns empty so the search loop ends."""

    def __init__(self, assets, db_path):
        self.assets = assets
        self.db_path = db_path
        self.observed_at_page2: dict | None = None

    def search_text(self, query, size, page):
        if page == 2:
            conn = sqlite3.connect(str(self.db_path))
            try:
                self.observed_at_page2 = {
                    "assets": conn.execute(
                        "SELECT COUNT(*) FROM assets").fetchone()[0],
                    "taxon_scores": conn.execute(
                        "SELECT COUNT(*) FROM taxon_scores").fetchone()[0],
                }
            finally:
                conn.close()
            return {"empty": True}
        return {"page": page}

    def list_search_assets(self, payload):
        if payload.get("empty"):
            return []
        return self.assets

    def download_bytes(self, url):
        return _jpeg_bytes()

    def get_asset_metadata(self, asset_id):
        return None  # no EXIF — extract_exif handles the empty payload


class _FakeClassifier:
    """Search-stage fake: encode-once path (create_image_features +
    create_probabilities + format_grouped_probs). Kingdom rows come back
    Plantae=0.9 so every batch clears the plantae boundary threshold."""

    device = "cpu"

    def get_txt_embeddings(self):
        import torch
        return torch.zeros(8, 5)

    def get_current_txt_names(self):
        # name_ary[0][0] = kingdom for _kingdom_bins; 4 of 5 rows
        # Plantae so uniform softmax sums to 0.8 > PLANT_MIN_PROB.
        names = []
        for i in range(5):
            kingdom = "Plantae" if i < 4 else "Fungi"
            names.append([[kingdom, "ph", "cl", "or", "fam", "gen", "sp"], "common"])
        return names

    def create_image_features(self, pil_images, *, normalize=True):
        import torch
        return torch.zeros(len(pil_images), 8)

    def create_probabilities(self, feats, txt_embeddings):
        import torch
        return torch.softmax(
            torch.zeros(feats.shape[0], txt_embeddings.shape[1]) + 0.2, dim=1
        )

    def format_grouped_probs(self, image_key, probs, rank, min_prob=1e-9, k=5):
        return [
            {"file_name": str(image_key), "kingdom": "Plantae", "score": 0.9},
        ]

    def predict(self, images, rank, k):
        return [
            {"file_name": str(p), "kingdom": "Plantae", "score": 0.9}
            for p in images
        ]


# --- test ------------------------------------------------------------------


def test_search_batches_are_visible_to_other_readers(tmp_path):
    db_path = tmp_path / "state.db"
    storage.init_db(db_path)

    assets = [
        pipeline.ImmichAsset(id=f"a{i}", original_file_name=f"p{i}.jpg",
                             type="IMAGE")
        for i in range(8)
    ]
    client = _FakeClient(assets, db_path)

    # Open the connection via the context manager exactly like
    # run_pipeline does (one connection spanning the whole search).
    with storage.connect(str(db_path)) as conn:
        accepted = pipeline.discover_plantae_assets(
            conn,
            client,
            _FakeClassifier(),
            query="test",
            n_samples=100,
            base_url="http://x",
            api_key="k",
            decision_batch_size=2,
        )

    assert [a.id for a in accepted] == [a.id for a in assets]
    # The mid-run snapshot (taken when page 2 was requested, i.e. after
    # every page-1 batch) must already show the rows. Before the
    # per-batch commits this was 0/0 — the reader saw nothing until
    # the whole run ended.
    assert client.observed_at_page2 is not None, (
        "fake client never observed mid-run state")
    assert client.observed_at_page2["assets"] == 8
    # 2 rows per asset since _score_batch_for_kingdoms persists both
    # the Plantae and the Fungi kingdom probability.
    assert client.observed_at_page2["taxon_scores"] == 16
