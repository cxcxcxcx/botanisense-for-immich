"""Shared test fixtures.

The person-flag classifier veto (pipeline._load_person_clf) loads
data/person_classifier.joblib from the CWD — a gitignored artifact that
may or may not exist on any given checkout. Point it at a path that
never exists so pipeline tests are hermetic; tests that exercise the
veto (test_person_veto.py) set PERSON_CLF_PATH to their own fixture
model inside the test body, which overrides this.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


@pytest.fixture(autouse=True)
def _no_person_clf(monkeypatch):
    import pipeline

    monkeypatch.setenv("PERSON_CLF_PATH",
                       str(ROOT / "data" / "__no_such_person_clf.joblib"))
    monkeypatch.setattr(pipeline, "_PERSON_CLF", None)
    yield
