"""Regression test: /api/reclassify must not deadlock on _results_lock.

`_invalidate_results_cache()` acquires the module-level `_results_lock`
itself. Three call sites used to invoke it *while already holding* that
lock. `threading.Lock` is not reentrant, so the second acquire blocked
forever — on the asyncio event-loop thread, which wedged the entire
server: every subsequent request, including /api/health, hung until
uvicorn was killed.

The user-visible symptom was "click Re-run KNN, nothing happens" — the
request never returned and the app stopped responding.

This test calls the invalidator the way those sites do (holding no
lock) and asserts it completes, then asserts the lock is free
afterwards. A re-introduced double-acquire hangs here instead of in
production.
"""

from __future__ import annotations

import sys
import threading
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def _run_with_timeout(fn, seconds: float = 5.0) -> bool:
    """Run `fn` on a daemon thread. Returns True if it finished."""
    done = threading.Event()

    def target():
        fn()
        done.set()

    t = threading.Thread(target=target, daemon=True)
    t.start()
    return done.wait(seconds)


@pytest.fixture
def app_mod(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "state.db"))
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import app
    return app


def test_invalidate_results_cache_terminates(app_mod):
    app_mod._results_cache["all:sort=rank"] = {"x": 1}
    finished = _run_with_timeout(lambda: app_mod._invalidate_results_cache("all:"))
    assert finished, "_invalidate_results_cache deadlocked"
    assert "all:sort=rank" not in app_mod._results_cache


def test_results_lock_is_free_after_invalidation(app_mod):
    """If a caller leaks the lock, the next request blocks. Pin that
    the lock is released."""
    app_mod._invalidate_results_cache("all:")
    acquired = app_mod._results_lock.acquire(timeout=2.0)
    assert acquired, "_results_lock still held after invalidation"
    app_mod._results_lock.release()


def test_double_acquire_would_deadlock(app_mod):
    """Documents *why* the call sites must not hold the lock: this is
    what they used to do, and it hangs. If this ever stops hanging
    (e.g. the lock became an RLock), the guard comments at the call
    sites can be relaxed."""
    def held_then_invalidate():
        with app_mod._results_lock:
            app_mod._invalidate_results_cache("all:")

    finished = _run_with_timeout(held_then_invalidate, seconds=1.5)
    assert not finished, (
        "_results_lock appears reentrant now — the no-lock-held "
        "requirement at the call sites may no longer apply"
    )
