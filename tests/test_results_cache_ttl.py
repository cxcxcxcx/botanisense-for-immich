"""Regression tests for the /api/results cache TTL while a run is
active.

The cache used to pin its first mid-run payload (usually the empty
pre-run grid) for the entire multi-hour pipeline run, because entries
only expired when _run_and_store invalidated them AFTER the run
finished. Now entries expire after _RESULTS_CACHE_RUNNING_TTL seconds
while a pipeline job is active; idle behavior is unchanged
(never-expire until invalidated).
"""

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import app


def _seed(key: str, age_seconds: float) -> None:
    app._results_cache.clear()
    app._results_cache_put(key, {"results": [1], "pagination": {"total": 1}})
    # Backdate the entry to simulate a payload cached long ago.
    value, _ts = app._results_cache[key]
    app._results_cache[key] = (value, time.monotonic() - age_seconds)


def test_stale_entry_served_when_idle(monkeypatch):
    monkeypatch.setattr(app, "_get_active_job", lambda: None)
    _seed("k", age_seconds=9999)
    assert app._results_cache_get("k") is not None


def test_stale_entry_missed_while_run_active(monkeypatch):
    monkeypatch.setattr(app, "_get_active_job", lambda: {"job_id": 1})
    _seed("k", age_seconds=9999)
    assert app._results_cache_get("k") is None
    # And the stale entry is dropped, so the next PUT isn't blocked by
    # a soon-to-be-evicted ghost.
    assert "k" not in app._results_cache


def test_fresh_entry_served_while_run_active(monkeypatch):
    monkeypatch.setattr(app, "_get_active_job", lambda: {"job_id": 1})
    _seed("k", age_seconds=0)
    assert app._results_cache_get("k") is not None
