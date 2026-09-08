"""Regression tests for the /api/results cache TTL while a run is
active.

The cache used to pin its first mid-run payload (usually the empty
pre-run grid) for the entire multi-hour pipeline run, because entries
only expired when the run-end path invalidated them AFTER the run
finished. Now entries expire after _RESULTS_CACHE_RUNNING_TTL seconds
while a pipeline job is active; idle behavior is unchanged
(never-expire until invalidated).

Uses the same isolation contract as the other app-importing tests:
dotenv stubbed, DB_PATH pointed at tmp, real env scrubbed — importing
app at collection time with the developer's .env loaded leaked the
real Immich key/URL into every test session.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Import a fresh app module isolated from the developer's env."""
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    for var in ("IMMICH_API_KEY", "IMMICH_BASE_URL", "IMMICH_PUBLIC_URL",
                "KNN_INVENTORY_DIR", "CORS_ORIGINS"):
        monkeypatch.delenv(var, raising=False)
    monkeypatch.chdir(tmp_path)
    import dotenv  # noqa: E402
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(db_path)
    storage._GLOBAL_NEAR_DUP_CACHE = {}
    import app  # noqa: E402
    return app


def _seed(app, key: str, age_seconds: float) -> None:
    app._results_cache.clear()
    app._results_cache_put(key, {"results": [1], "pagination": {"total": 1}})
    # Backdate the entry to simulate a payload cached long ago.
    value, _ts = app._results_cache[key]
    app._results_cache[key] = (value, time.monotonic() - age_seconds)


def test_stale_entry_served_when_idle(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "_get_active_job", lambda: None)
    _seed(app, "k", age_seconds=9999)
    assert app._results_cache_get("k") is not None


def test_stale_entry_missed_while_run_active(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "_get_active_job", lambda: {"job_id": 1})
    _seed(app, "k", age_seconds=9999)
    assert app._results_cache_get("k") is None
    # And the stale entry is dropped, so the next PUT isn't blocked by
    # a soon-to-be-evicted ghost.
    assert "k" not in app._results_cache


def test_fresh_entry_served_while_run_active(app_module, monkeypatch):
    app = app_module
    monkeypatch.setattr(app, "_get_active_job", lambda: {"job_id": 1})
    _seed(app, "k", age_seconds=0)
    assert app._results_cache_get("k") is not None
