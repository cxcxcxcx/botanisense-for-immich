"""Tests for the person-flag classifier veto (pipeline.PERSON_VETO_P).

The veto flips person-box evidence to has_person=0 when the BioCLIP
whole-image classifier is CONFIDENTLY negative (p < 0.05) — the
hand-holding-X class YOLO-World misreads as a big person box.
Missing embedding / missing model file never veto.
"""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

# Ensure repo root is on sys.path so `import pipeline` works.
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import pipeline  # noqa: E402
import storage  # noqa: E402


@pytest.fixture(autouse=True)
def _reset_clf_cache():
    """The model blob is cached at module level; every test starts clean
    and points PERSON_CLF_PATH at a per-test artifact."""
    pipeline._PERSON_CLF = None
    yield
    pipeline._PERSON_CLF = None


def _write_model(path: Path, intercept: float, dim: int = 4) -> None:
    """A zero-coef logreg: logit == intercept, so p is fully controlled."""
    import joblib

    joblib.dump({"coef": np.zeros(dim, dtype=np.float32),
                 "intercept": intercept}, path)


def _seed_asset(db: Path, asset_id: str, dim: int = 4) -> None:
    with storage.connect(db) as conn:
        storage.upsert_asset(conn, asset_id=asset_id,
                             original_file_name="x.jpg",
                             original_url="u", thumbnail_url="t")
        conn.execute(
            "INSERT INTO embeddings (asset_id, dim, vector, computed_at) "
            "VALUES (?, ?, ?, '2026-01-01')",
            (asset_id, dim, np.zeros(dim, dtype="<f4").tobytes()),
        )


# --- pure decision function ----------------------------------------------


def test_veto_flips_confident_negative():
    assert pipeline.apply_person_veto(True, 0.014) is False
    assert pipeline.apply_person_veto(True, 0.049) is False


def test_veto_keeps_positive_and_borderline():
    assert pipeline.apply_person_veto(True, 0.982) is True
    assert pipeline.apply_person_veto(True, pipeline.PERSON_VETO_P) is True
    assert pipeline.apply_person_veto(True, 0.055) is True


def test_veto_never_promotes_and_never_vetoes_without_p():
    assert pipeline.apply_person_veto(False, 0.0) is False
    assert pipeline.apply_person_veto(False, 0.99) is False
    assert pipeline.apply_person_veto(True, None) is True


# --- model loading + p computation ---------------------------------------


def test_missing_model_file_disables_veto(tmp_path, monkeypatch):
    monkeypatch.setenv("PERSON_CLF_PATH", str(tmp_path / "nope.joblib"))
    assert pipeline._load_person_clf() == {}
    assert pipeline._person_clf_p(None, "whatever") is None
    assert pipeline.apply_person_veto(True, None) is True


def test_p_computed_from_persisted_embedding(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        storage.init_db(db)
        _seed_asset(Path(db), "a1")
        model = tmp_path / "clf.joblib"
        _write_model(model, intercept=-10.0)  # zero vector -> p ~ 4.5e-5
        monkeypatch.setenv("PERSON_CLF_PATH", str(model))
        with storage.connect(db) as conn:
            p = pipeline._person_clf_p(conn, "a1")
        assert p is not None and p < pipeline.PERSON_VETO_P
        assert pipeline.apply_person_veto(True, p) is False


def test_no_embedding_returns_none(tmp_path, monkeypatch):
    with tempfile.TemporaryDirectory() as td:
        db = os.path.join(td, "state.db")
        storage.init_db(db)
        model = tmp_path / "clf.joblib"
        _write_model(model, intercept=0.0)
        monkeypatch.setenv("PERSON_CLF_PATH", str(model))
        with storage.connect(db) as conn:
            assert pipeline._person_clf_p(conn, "missing") is None
