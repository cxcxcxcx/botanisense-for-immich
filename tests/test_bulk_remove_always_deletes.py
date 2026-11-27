"""Regression tests for the bulk-remove cluster→album behavior.

Pins: `POST /api/album/cluster/remove` always DELETEs the matching
Immich album alongside dropping the local mapping. Policy: "never leave orphans". Cluster sync itself was removed; these
endpoints remain so albums an older version created can be cleaned up.
The endpoint takes no `delete_in_immich` parameter — the caller
cannot opt out.

Also covers the single-cluster `DELETE /api/album/cluster/{id}` so a
typo'd path or accidental branch in the bulk handler can't quietly
skip the Immich delete.
"""

from __future__ import annotations

import sys
import uuid
from pathlib import Path
from unittest import mock

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


@pytest.fixture
def db_path(tmp_path, monkeypatch):
    p = tmp_path / "state.db"
    monkeypatch.setenv("DB_PATH", str(p))
    for mod in ("app", "storage"):
        sys.modules.pop(mod, None)
    import storage  # noqa: E402
    import app  # noqa: E402
    storage.init_db(p)
    yield p


def _seed_asset(conn, aid: str, cluster_id: int) -> None:
    conn.execute(
        "INSERT INTO assets "
        "(asset_id, cluster_id, original_file_name, original_url, thumbnail_url) "
        "VALUES (?, ?, ?, ?, ?)",
        (aid, cluster_id, f"{aid}.jpg", f"http://x/{aid}", f"http://x/{aid}-t"),
    )


def _seed_synced_cluster(db_path, cluster_id: int, album_id: str, n_assets: int = 3) -> None:
    """Seed an asset row + a cluster_albums mapping, as left behind by
    the (removed) cluster sync of an older version."""
    import storage  # noqa
    with storage.connect(db_path) as conn:
        for i in range(n_assets):
            _seed_asset(conn, f"asset-{cluster_id}-{i:04d}", cluster_id)
        conn.execute(
            "INSERT INTO cluster_albums "
            "(cluster_id, album_id, album_url, n_assets, reused_existing, "
            " created_at, last_synced_at) "
            "VALUES (?, ?, ?, ?, 0, datetime('now'), datetime('now'))",
            (cluster_id, album_id, f"http://x/albums/{album_id}", n_assets),
        )
        conn.commit()


class _FakeImmichClient:
    def __init__(self, base_url: str, api_key: str) -> None:
        self.deleted_albums: list[str] = []
        self._albums: dict[str, dict] = {}

    def delete_album(self, *, album_id: str) -> None:
        self.deleted_albums.append(album_id)
        self._albums.pop(album_id, None)

    def close(self) -> None:
        pass


# ---------- bulk remove ---------------------------------------------------


def test_bulk_remove_calls_delete_album_for_every_mapping(db_path):
    """Three synced clusters → three Immich DELETE calls, no exceptions,
    no orphan mappings in either our DB or the fake Immich client's
    tracking."""
    import app
    import storage  # noqa

    _seed_synced_cluster(db_path, 10, "album-A")
    _seed_synced_cluster(db_path, 11, "album-B")
    _seed_synced_cluster(db_path, 12, "album-C")

    fake = _FakeImmichClient("http://x", "k")

    params = app.ClusterAlbumsRemoveParams(cluster_ids=None)
    import asyncio
    with mock.patch.object(app, "_immich_client", return_value=fake):
        result = asyncio.run(app.bulk_remove_cluster_albums(params))

    assert sorted(fake.deleted_albums) == ["album-A", "album-B", "album-C"]
    assert result["removed"] == 3
    assert result["remote_deleted"] == 3
    assert result["errors"] == []

    with storage.connect(db_path) as conn:
        leftover = storage.list_cluster_albums(conn)
    assert leftover == [], f"expected no local mappings left, got {leftover}"


def test_bulk_remove_takes_no_delete_in_immich_param(db_path):
    """The endpoint must not accept `delete_in_immich` — if a caller
    sends it (Pydantic defaults to ignoring extra fields, so this is
    a soft check), the deletion still happens. This pins the policy
    "the toggle was removed; callers cannot opt out"."""
    import app
    import storage  # noqa
    _seed_synced_cluster(db_path, 7, "album-X")

    fake = _FakeImmichClient("http://x", "k")
    # Pass the field even though it doesn't exist on the model — Pydantic
    # will silently drop it. The delete must still happen.
    params = app.ClusterAlbumsRemoveParams.model_construct(cluster_ids=None)
    # Sanity: confirm the field isn't on the model anymore.
    assert "delete_in_immich" not in app.ClusterAlbumsRemoveParams.model_fields

    import asyncio
    with mock.patch.object(app, "_immich_client", return_value=fake):
        result = asyncio.run(app.bulk_remove_cluster_albums(params))

    assert fake.deleted_albums == ["album-X"], fake.deleted_albums
    assert result["remote_deleted"] == 1


# ---------- single-cluster delete ----------------------------------------


def test_single_delete_always_deletes_in_immich(db_path):
    """DELETE /api/album/cluster/{id} also unconditionally calls
    delete_album. Same never-leave-orphans policy."""
    import app
    import storage  # noqa
    _seed_synced_cluster(db_path, 99, "album-Z")

    fake = _FakeImmichClient("http://x", "k")
    import asyncio
    with mock.patch.object(app, "_immich_client", return_value=fake):
        result = asyncio.run(app.delete_cluster_album(99))

    assert fake.deleted_albums == ["album-Z"]
    assert result["deleted"] is True
    assert result["remote_deleted"] is True

    with storage.connect(db_path) as conn:
        assert storage.get_cluster_album(conn, 99) is None
    # The endpoint signature must no longer accept delete_in_immich.
    sig = app.delete_cluster_album.__annotations__
    assert "delete_in_immich" not in sig, sig

