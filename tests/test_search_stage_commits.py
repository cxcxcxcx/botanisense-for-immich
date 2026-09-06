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
    thumb_dir = tmp_path / "thumbs"
    thumb_dir.mkdir(parents=True)  # run_pipeline creates it before calling
    with storage.connect(str(db_path)) as conn:
        accepted = pipeline.discover_plantae_assets(
            conn,
            client,
            _FakeClassifier(),
            query="test",
            n_samples=100,
            base_url="http://x",
            api_key="k",
            thumb_dir=thumb_dir,
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
    assert client.observed_at_page2["taxon_scores"] == 8
