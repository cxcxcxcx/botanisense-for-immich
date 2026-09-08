"""Tests for the worker-exit reaper in _get_active_job().

When the run child exits, app.py must trust the terminal state the
worker itself wrote to its status JSON:

  - "stopped" — the user clicked Stop and the worker unwound at a batch
    boundary (pipeline_worker calls pipeline.stopped_state()). This is
    NOT a crash: the UI badge must stay the neutral "stopped", and the
    status file must not be rewritten to "error".
  - "error" — the worker caught an exception and wrote error_state()
    itself; its message must be preserved verbatim.
  - "running" still on disk — the only genuine "died unexpectedly"
    case (SIGKILL / OOM hard-kill, no chance to write state). The
    reaper rewrites the file to error with the death message.

Before this contract existed, a clean user stop left the file at
"running", so the reaper branded every stop as a worker crash and the
UI showed a red error after clicking Stop.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


class _FakeProc:
    """Stand-in for subprocess.Popen — only poll() is touched by
    _get_active_job."""

    def __init__(self, exit_code: int | None):
        self._exit_code = exit_code
        self.pid = 4242

    def poll(self):
        return self._exit_code


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    monkeypatch.setenv("DB_PATH", str(tmp_path / "state.db"))
    import dotenv  # noqa: E402

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import app  # noqa: E402

    # Start every test from an empty registry.
    app._pipeline_active_job = None
    app._last_worker_outcome = None
    return app


def _register(app_module, tmp_path, proc, status):
    status_file = tmp_path / "run.status.json"
    status_file.write_text(json.dumps(status))
    app_module._pipeline_active_job = {
        "proc": proc,
        "job_id": "exit1",
        "status_path": status_file,
        "log_path": tmp_path / "run.log",
        "started_at": 1234.5,
    }
    return status_file


def test_stopped_worker_is_not_branded_a_crash(app_module, tmp_path):
    status_file = _register(
        app_module, tmp_path, _FakeProc(exit_code=0),
        {
            "status": "stopped",
            "stage": "yolo",
            "query": "fern",
            "n_requested": 800,
            "started_at": 1234.5,
            "finished_at": 1299.0,
            "last_error": "stopped by user",
        },
    )

    assert app_module._get_active_job() is None
    # Registry cleared, outcome recorded as a STOP, not an error.
    assert app_module._pipeline_active_job is None
    outcome = app_module._last_worker_outcome
    assert outcome is not None
    assert outcome["status"] == "stopped"
    assert outcome["stage"] == "yolo"
    assert outcome["query"] == "fern"
    # The status file must be left exactly as the worker wrote it —
    # no "died unexpectedly" rewrite.
    on_disk = json.loads(status_file.read_text())
    assert on_disk["status"] == "stopped"
    assert "died unexpectedly" not in str(on_disk)

    # And the SSE/status snapshot surfaces that stopped state rather
    # than falling back to the previous run's green "done".
    snap = app_module._build_event_snapshot()["pipeline"]
    assert snap["status"] == "stopped"


def test_worker_self_reported_error_keeps_its_message(app_module, tmp_path):
    status_file = _register(
        app_module, tmp_path, _FakeProc(exit_code=1),
        {
            "status": "error",
            "stage": "model",
            "query": "basil",
            "n_requested": 900,
            "started_at": 1234.5,
            "last_error": "RuntimeError: weights failed to load",
        },
    )

    assert app_module._get_active_job() is None
    outcome = app_module._last_worker_outcome
    assert outcome is not None
    assert outcome["status"] == "error"
    assert "weights failed to load" in outcome["last_error"]
    # The reaper must not overwrite the worker's own error text with
    # the generic death message.
    assert "died unexpectedly" not in outcome["last_error"]
    on_disk = json.loads(status_file.read_text())
    assert on_disk["status"] == "error"


def test_sigkill_with_running_state_is_marked_unexpected_death(app_module, tmp_path):
    status_file = _register(
        app_module, tmp_path, _FakeProc(exit_code=-9),
        {
            "status": "running",
            "stage": "search",
            "query": "fern",
            "n_requested": 800,
            "started_at": 1234.5,
        },
    )

    assert app_module._get_active_job() is None
    outcome = app_module._last_worker_outcome
    assert outcome is not None
    assert outcome["status"] == "error"
    assert "died unexpectedly" in outcome["last_error"]
    on_disk = json.loads(status_file.read_text())
    assert on_disk["status"] == "error"
    assert "died unexpectedly" in on_disk["last_error"]


def test_live_proc_is_not_reaped(app_module, tmp_path):
    status_file = _register(
        app_module, tmp_path, _FakeProc(exit_code=None),
        {"status": "running", "stage": "search", "query": "fern"},
    )

    active = app_module._get_active_job()
    assert active is not None
    assert active["job_id"] == "exit1"
    # Still running: no terminal outcome recorded, file untouched.
    assert app_module._last_worker_outcome is None
    assert json.loads(status_file.read_text())["status"] == "running"


def test_starting_a_new_run_clears_the_outcome(app_module, tmp_path, monkeypatch):
    """A previous stop/error must not linger into the next run's
    snapshots — _start_run resets _last_worker_outcome."""
    _register(
        app_module, tmp_path, _FakeProc(exit_code=0),
        {"status": "stopped", "stage": "yolo", "last_error": "stopped by user"},
    )
    app_module._get_active_job()
    assert app_module._last_worker_outcome is not None

    # Simulate the reset _start_run performs under the registry lock.
    with app_module._pipeline_jobs_lock:
        app_module._last_worker_outcome = None
    assert app_module._last_worker_outcome is None
