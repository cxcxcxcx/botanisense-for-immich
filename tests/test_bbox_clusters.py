"""Tests for box-level plant clustering (clusters.refit_bbox_clusters + storage + API)."""

from __future__ import annotations

import numpy as np
import pytest

import clusters
import storage


def _unit(vec: list[float], dim: int = 16) -> bytes:
    arr = np.zeros(dim, dtype=np.float32)
    arr[: len(vec)] = vec
    n = float(np.linalg.norm(arr))
    if n > 0:
        arr /= n
    return arr.astype("<f4").tobytes()


def test_refit_bbox_clusters_and_summary(tmp_path):
    db = tmp_path / "test.db"
    storage.init_db(db)

    with storage.connect(db) as conn:
        # Create 6 assets: 4 with whole-image cluster_id, 2 without (newly discovered!)
        for i in range(6):
            storage.upsert_asset(
                conn,
                asset_id=f"asset-{i}",
                original_file_name=f"img_{i}.jpg",
                original_url=f"/a/{i}",
                thumbnail_url=f"/t/{i}",
                captured_at=f"2026-05-0{i + 1}T12:00:00Z",
            )
        storage.set_cluster_ids(
            conn,
            asset_to_cluster={"asset-0": 0, "asset-1": 0, "asset-2": 1, "asset-3": 1},
        )

        # Group A: 4 tight Plantae flower boxes across 3 assets (incl. asset-4 which has no whole cluster)
        rng = np.random.default_rng(42)
        base_a = np.array([1.0, 0.0, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)
        base_b = np.array([0.0, 1.0, 0.0, 0.0] + [0.0] * 12, dtype=np.float32)

        def add_box(asset_id: str, label: str, det_score: float, w: float, h: float,
                    vec: np.ndarray, kingdom: str, species: str, common: str, genus: str, sp_score: float):
            cur = conn.execute(
                "INSERT INTO detections (asset_id, label, score, x, y, w, h) VALUES (?, ?, ?, 0.1, 0.1, ?, ?)",
                (asset_id, label, det_score, w, h),
            )
            did = int(cur.lastrowid)
            v = vec / np.linalg.norm(vec)
            conn.execute(
                "INSERT INTO bbox_embeddings (detection_id, model_version, dim, vector) VALUES (?, 'teacher:bioclip-2', ?, ?)",
                (did, len(v), v.astype("<f4").tobytes()),
            )
            conn.execute(
                """
                INSERT INTO bbox_predictions
                  (detection_id, rank, position, species, common_name, score, kingdom, family, genus)
                VALUES (?, 'species', 0, ?, ?, ?, ?, 'Fam', ?)
                """,
                (did, species, common, sp_score, kingdom, genus),
            )
            return did

        # 5 boxes in cluster A across asset-0, asset-1, asset-4
        for idx, aid in enumerate(["asset-0", "asset-0", "asset-1", "asset-4", "asset-4"]):
            v = base_a + rng.normal(0, 0.02, size=16).astype(np.float32)
            add_box(aid, "flower", 0.85, 0.25, 0.25, v, "Plantae", "Rosa chinensis", "China rose", "Rosa", 0.91)

        # 5 boxes in cluster B across asset-2, asset-3, asset-5
        for idx, aid in enumerate(["asset-2", "asset-3", "asset-3", "asset-5", "asset-5"]):
            v = base_b + rng.normal(0, 0.02, size=16).astype(np.float32)
            add_box(aid, "tree", 0.80, 0.30, 0.30, v, "Plantae", "Acer palmatum", "Japanese maple", "Acer", 0.88)

        # Non-plant box (should be excluded)
        add_box("asset-0", "plant", 0.90, 0.25, 0.25, base_a, "Animalia", "Canis lupus", "Dog", "Canis", 0.95)
        # Tiny sliver box (area 0.0025 < 0.01, should be excluded)
        add_box("asset-0", "flower", 0.90, 0.05, 0.05, base_a, "Plantae", "Rosa chinensis", "China rose", "Rosa", 0.91)

        # Also add a whole-photo fallback asset (Plantae >= 0.35, NO box) close to cluster A
        storage.upsert_asset(
            conn,
            asset_id="asset-fallback",
            original_file_name="macro_rose.jpg",
            original_url="/a/fb",
            thumbnail_url="/t/fb",
            captured_at="2026-05-09T12:00:00Z",
        )
        v_fb = base_a + rng.normal(0, 0.015, size=16).astype(np.float32)
        v_fb = v_fb / np.linalg.norm(v_fb)
        storage.upsert_embedding(conn, asset_id="asset-fallback", vector=v_fb)
        conn.execute(
            "INSERT INTO taxon_scores (asset_id, rank, taxon_name, score) VALUES (?, 'kingdom', 'Plantae', 0.92)",
            ("asset-fallback",),
        )
        conn.execute(
            """
            INSERT INTO predictions
              (asset_id, rank, position, species, common_name, score, kingdom, family, genus)
            VALUES (?, 'species', 0, 'Rosa chinensis', 'China rose', 0.93, 'Plantae', 'Rosaceae', 'Rosa')
            """,
            ("asset-fallback",),
        )

        # Also add an isolated single-shot photo of the same genus slightly further from base_a
        # (sim ~0.92) to verify Stage 3 noise rescue picks it up cleanly.
        storage.upsert_asset(
            conn,
            asset_id="asset-rescued",
            original_file_name="single_rose.jpg",
            original_url="/a/res",
            thumbnail_url="/t/res",
            captured_at="2026-05-10T12:00:00Z",
        )
        v_res = np.array([0.92, 0.0, 0.39, 0.0] + [0.0] * 12, dtype=np.float32)
        add_box("asset-rescued", "flower", 0.85, 0.20, 0.20, v_res, "Plantae", "Rosa chinensis", "China rose", "Rosa", 0.90)

        res = clusters.refit_bbox_clusters(
            conn,
            min_cluster_size=3,
            min_samples=2,
            min_score=0.30,
            min_area=0.01,
            min_assets_per_cluster=2,
        )
        assert res["n_box_inputs"] == 11
        assert res["n_photo_inputs"] == 1
        assert res["n_boxes_at_fit"] == 12
        assert res["n_clusters"] == 2
        assert res["n_clustered"] == 12
        assert res["n_photo_clustered"] == 1

        summary = storage.bbox_cluster_summary(conn)
        assert summary["stats"]["n_clusters"] == 2
        assert summary["stats"]["n_boxes"] == 11
        assert summary["stats"]["n_photo_fallbacks"] == 1
        assert summary["stats"]["n_assets"] == 8
        assert summary["stats"]["n_new_assets"] == 4  # asset-4, asset-5, asset-fallback, asset-rescued

        by_name = {c["name"]: c for c in summary["clusters"]}
        assert set(by_name.keys()) == {"China rose", "Japanese maple"}
        assert by_name["China rose"]["n_assets"] == 5
        assert by_name["China rose"]["n_photo_fallbacks"] == 1
        assert by_name["Japanese maple"]["n_assets"] == 3

        # Check item pagination + default per-asset merge with boxes[] + fallback photo
        rose_cid = by_name["China rose"]["cluster_id"]
        merged = storage.get_bbox_cluster_items(conn, rose_cid)
        assert merged["pagination"]["total"] == 5
        assert any(it["kind"] == "photo" and it["asset_id"] == "asset-fallback" for it in merged["items"])
        assert any(it["asset_id"] == "asset-rescued" for it in merged["items"])
        assert any(it["kind"] == "box" and it["n_boxes"] == 2 for it in merged["items"])
        assert any(it["is_new_asset"] for it in merged["items"])

        # Virtual "Unclustered" bucket (cluster_id -1): an eligible box that
        # arrived after the refit shows up; a sub-threshold one doesn't.
        assert summary["unclustered"] is None
        for aid in ("asset-late", "asset-weak"):
            storage.upsert_asset(
                conn, asset_id=aid, original_file_name=f"{aid}.jpg",
                original_url=f"/a/{aid}", thumbnail_url=f"/t/{aid}",
                captured_at="2026-05-11T12:00:00Z",
            )
        add_box("asset-late", "flower", 0.85, 0.25, 0.25, base_b, "Plantae", "Acer palmatum", "Japanese maple", "Acer", 0.88)
        add_box("asset-weak", "flower", clusters.BBOX_CLUSTER_MIN_SCORE - 0.05, 0.25, 0.25, base_b,
                "Plantae", "Acer palmatum", "Japanese maple", "Acer", 0.88)
        summary = storage.bbox_cluster_summary(conn)
        assert summary["unclustered"]["n_assets"] == 1
        uncl = storage.get_bbox_cluster_items(conn, -1)
        assert [it["asset_id"] for it in uncl["items"]] == ["asset-late"]
