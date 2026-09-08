"""Tests for /api/events pipeline state during the run-child boot window.

POST /api/run returns as soon as the child is spawned, but the child
needs a few seconds (interpreter boot + torch import) before it writes
its first status file. In that window _build_event_snapshot used to
fall back to the parent's pipeline module state — which still holds the
PREVIOUS run's terminal snapshot, so the SSE told freshly-started
clients "done". The Run button re-enabled itself seconds after the
click (ParamsForm's awaiting latch had nothing real to hold on to) and
the status panel showed the old run's numbers as if they were the new
run's.

Contract pinned here:

1. Active child + no status file yet → an explicit "running"/"starting"
   snapshot carrying the child's job_id and spawn time.
2. Active child + status file → the file wins (with job_id stamped).
3. No active child → the last completed run passes through unchanged.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture()
def app_module(tmp_path, monkeypatch):
    """Import app.py in isolation (no repo-root .env / config leakage)."""
    monkeypatch.setenv("DB_PATH", str(tmp_path / "state.db"))
    import dotenv  # noqa: E402

    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import app  # noqa: E402
    return app


def _active(tmp_path, *, status_file: Path | None = None):
    # proc is never touched by _build_event_snapshot (liveness is
    # checked inside the real _get_active_job, which these tests
    # replace), so a bare dict is enough.
    return {
        "proc": None,
        "job_id": "bootstate1",
        "status_path": status_file or (tmp_path / "missing.status.json"),
        "log_path": tmp_path / "missing.log",
        "started_at": 1234.5,
    }


def test_boot_window_reports_starting_not_previous_run(app_module, tmp_path, monkeypatch):
    monkeypatch.setattr(app_module, "_get_active_job", lambda: _active(tmp_path))
    snap = app_module._build_event_snapshot()["pipeline"]
    assert snap["status"] == "running"
    assert snap["stage"] == "starting"
    assert snap["job_id"] == "bootstate1"
    assert snap["started_at"] == 1234.5
    assert snap["finished_at"] is None


def test_status_file_wins_once_written(app_module, tmp_path, monkeypatch):
    status_file = tmp_path / "written.status.json"
    status_file.write_text(json.dumps({
        "status": "running", "stage": "species",
        "chunk_idx": 2, "chunk_total": 9,
        "started_at": 1234.5, "finished_at": None,
    }))
    monkeypatch.setattr(
        app_module, "_get_active_job", lambda: _active(tmp_path, status_file=status_file)
    )
    snap = app_module._build_event_snapshot()["pipeline"]
    assert snap["status"] == "running"
    assert snap["stage"] == "species"
    assert snap["chunk_idx"] == 2
    assert snap["job_id"] == "bootstate1"


def test_unparsable_status_file_falls_back_to_starting(app_module, tmp_path, monkeypatch):
    status_file = tmp_path / "broken.status.json"
    status_file.write_text("{not json")
    monkeypatch.setattr(
        app_module, "_get_active_job", lambda: _active(tmp_path, status_file=status_file)
    )
    snap = app_module._build_event_snapshot()["pipeline"]
    assert snap["status"] == "running"
    assert snap["stage"] == "starting"


def test_no_active_job_keeps_completed_run(app_module, monkeypatch):
    completed = {"status": "done", "stage": "", "n_new": 7, "started_at": 1.0,
                 "finished_at": 2.0}
    monkeypatch.setattr(app_module, "_get_active_job", lambda: None)
    monkeypatch.setattr(app_module, "_last_completed_run", lambda: completed)
    snap = app_module._build_event_snapshot()["pipeline"]
    assert snap == completed
