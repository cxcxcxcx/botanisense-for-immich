"""Tests for web-UI connection setup and the config.toml removal.

app.py used to hard-exit at import when IMMICH_API_KEY / IMMICH_BASE_URL
were unset, so an unconfigured container crash-looped and the only way
to configure the app was editing files. Now the app boots unconfigured
and POST /api/settings/immich saves the connection into SQLite
(album_meta key "immich_connection"); accessors resolve env > DB.

These tests pin the user-visible contracts:

1. An unconfigured boot serves /api/health (immich.ok false) and the
   Immich-touching endpoints answer 409 with an actionable message.
2. Saving from the web UI persists the pair, never echoes the key back,
   and takes effect immediately.
3. Environment variables still pin the connection (env > DB).
4. A legacy config.toml is migrated (base_url → DB, [knn] → env) rather
   than silently ignored.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

# URL that fails to connect instantly (port 9 = discard, refused on
# loopback) — the POST handler live-pings the URL it saved.
DEAD_URL = "http://127.0.0.1:9"


def _boot(tmp_path, monkeypatch, *, env=None, legacy_config=None):
    """Import a fresh app module configured ONLY by the given inputs.

    Deletes the IMMICH_* variables so a developer's exported env can't
    leak into the result, and chdirs into tmp_path so no repo-root
    config.toml is found. Returns (TestClient, app module, db path).
    """
    db_path = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(db_path))
    for var in ("IMMICH_API_KEY", "IMMICH_BASE_URL", "IMMICH_PUBLIC_URL",
                "KNN_INVENTORY_DIR", "CORS_ORIGINS"):
        monkeypatch.delenv(var, raising=False)
    for key, value in (env or {}).items():
        monkeypatch.setenv(key, value)
    monkeypatch.chdir(tmp_path)
    if legacy_config is not None:
        (tmp_path / "config.toml").write_text(legacy_config)
    # app.py calls load_dotenv() at import, which would re-import the
    # developer's repo-root .env (find_dotenv resolves from the module's
    # directory, not the CWD) and defeat the delenv calls above. Stub it.
    import dotenv  # noqa: E402
    monkeypatch.setattr(dotenv, "load_dotenv", lambda *a, **k: False)
    for mod in ("app", "storage", "clusters"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    storage.init_db(db_path)
    storage._GLOBAL_NEAR_DUP_CACHE = {}
    import app  # noqa: E402
    from fastapi.testclient import TestClient  # noqa: E402
    return TestClient(app.app), app, db_path


def test_unconfigured_boot_serves_health(tmp_path, monkeypatch):
    client, app, _ = _boot(tmp_path, monkeypatch)
    with client as c:
        r = c.get("/api/health")
    assert r.status_code == 200
    body = r.json()
    assert body["db"]["ok"] is True
    assert body["immich"]["ok"] is False
    assert body["immich"]["error"] == "not configured"
    assert body["api_key"]["error"] == "not configured"


def test_unconfigured_config_reports_none(tmp_path, monkeypatch):
    client, app, _ = _boot(tmp_path, monkeypatch)
    with client as c:
        body = c.get("/api/config").json()
    assert body["base_url"] == ""
    assert body["api_key_set"] is False
    assert body["source"] == "none"
    assert body["env_pinned"] is False


def test_immich_endpoints_409_when_unconfigured(tmp_path, monkeypatch):
    client, app, _ = _boot(tmp_path, monkeypatch)
    with client as c:
        run = c.get("/api/refresh")
        thumb = c.get("/thumb/some-asset")
    assert run.status_code == 409
    assert "not configured" in run.json()["detail"]
    assert thumb.status_code == 409


def test_save_connection_persists_and_never_echoes_key(tmp_path, monkeypatch):
    client, app, db_path = _boot(tmp_path, monkeypatch)
    with client as c:
        r = c.post(
            "/api/settings/immich",
            json={"base_url": DEAD_URL + "/", "api_key": "sekret-key"},
        )
    assert r.status_code == 200
    body = r.json()
    assert body["saved"] is True
    assert body["env_pinned"] is False
    # The live check ran and failed (dead URL) — the form shows that.
    assert body["health"]["immich"]["ok"] is False
    # The key must not appear anywhere in the response payload.
    assert "sekret-key" not in r.text

    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        saved = storage.get_connection_settings(conn)
    # Trailing slash stripped; key stored for server-side use.
    assert saved == {"base_url": DEAD_URL, "api_key": "sekret-key"}

    with client as c:
        cfg = c.get("/api/config").json()
    assert cfg["base_url"] == DEAD_URL
    assert cfg["api_key_set"] is True
    assert cfg["source"] == "db"


def test_blank_key_keeps_saved_key(tmp_path, monkeypatch):
    client, app, db_path = _boot(tmp_path, monkeypatch)
    with client as c:
        first = c.post(
            "/api/settings/immich",
            json={"base_url": DEAD_URL, "api_key": "original-key"},
        )
        assert first.status_code == 200
        # Same-URL edit: blank api_key means "keep what's saved".
        second = c.post(
            "/api/settings/immich",
            json={"base_url": DEAD_URL, "api_key": ""},
        )
    assert second.status_code == 200
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        saved = storage.get_connection_settings(conn)
    assert saved == {"base_url": DEAD_URL, "api_key": "original-key"}


def test_blank_key_with_new_url_is_422(tmp_path, monkeypatch):
    """Blank key + CHANGED url must be refused: pairing the saved key
    with a new host would send that key to whoever controls the URL."""
    client, app, db_path = _boot(tmp_path, monkeypatch)
    with client as c:
        first = c.post(
            "/api/settings/immich",
            json={"base_url": DEAD_URL, "api_key": "original-key"},
        )
        assert first.status_code == 200
        second = c.post(
            "/api/settings/immich",
            json={"base_url": "http://127.0.0.1:20", "api_key": ""},
        )
    assert second.status_code == 422
    assert "key" in second.json()["detail"].lower()
    # The saved connection is untouched.
    import storage  # noqa: E402
    with storage.connect(db_path) as conn:
        saved = storage.get_connection_settings(conn)
    assert saved == {"base_url": DEAD_URL, "api_key": "original-key"}


def test_no_key_anywhere_is_422(tmp_path, monkeypatch):
    client, app, _ = _boot(tmp_path, monkeypatch)
    with client as c:
        r = c.post("/api/settings/immich", json={"base_url": DEAD_URL})
    assert r.status_code == 422
    assert "API key" in r.json()["detail"]


@pytest.mark.parametrize("bad_url", [
    "ftp://immich:2283",
    "http://user:pass@immich:2283",
    "not a url",
    "",
])
def test_invalid_url_is_422(tmp_path, monkeypatch, bad_url):
    client, app, _ = _boot(tmp_path, monkeypatch)
    with client as c:
        r = c.post(
            "/api/settings/immich",
            json={"base_url": bad_url, "api_key": "k"},
        )
    assert r.status_code == 422


def test_env_pins_over_saved_db_row(tmp_path, monkeypatch):
    env_url = "http://env-immich:2283"
    client, app, db_path = _boot(
        tmp_path,
        monkeypatch,
        env={"IMMICH_BASE_URL": env_url, "IMMICH_API_KEY": "env-key"},
    )
    with client as c:
        r = c.post(
            "/api/settings/immich",
            json={"base_url": DEAD_URL, "api_key": "db-key"},
        )
        assert r.status_code == 200
        assert r.json()["env_pinned"] is True
        cfg = c.get("/api/config").json()
    # Env still wins; the DB row waits in the wings.
    assert cfg["base_url"] == env_url
    assert cfg["source"] == "env"
    assert cfg["env_pinned"] is True


def test_legacy_config_toml_is_migrated(tmp_path, monkeypatch):
    legacy = (
        "[immich]\n"
        f'base_url = "{DEAD_URL}"\n'
        "\n"
        "[knn]\n"
        'inventory_dir = "/tmp/legacy-knn"\n'
    )
    client, app, _ = _boot(tmp_path, monkeypatch, legacy_config=legacy)
    # [immich].base_url moved into the DB (source "db"), because the
    # file is no longer read on later boots.
    with client as c:
        cfg = c.get("/api/config").json()
    assert cfg["base_url"] == DEAD_URL
    assert cfg["source"] == "db"
    # [knn].inventory_dir injected into the environment for this process.
    import os  # noqa: E402
    assert os.environ["KNN_INVENTORY_DIR"] == "/tmp/legacy-knn"
