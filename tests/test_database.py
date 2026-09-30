"""Unit tests for the DuckDB persistence layer (engine/database.py)."""

from __future__ import annotations

from datetime import datetime, timezone

import pytest

from engine.database import Database
from engine.database import MediaItem, crop_to_pair, floats_to_sql, gps_to_pair


def _sample_item(item_id: str = "abc") -> MediaItem:
    return MediaItem(
        item_id=item_id,
        file_path="/tmp/photo.jpg",
        file_size=12345,
        file_format="JPEG",
        camera_make="Canon",
        camera_model="Canon EOS R5",
        camera_serial="SN12345",
        timestamp_utc=datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc),
        gps=(35.68, 139.69),
        p_hash=0x1234567890ABCDEF,
    )


def test_schema_created(db):
    # A freshly created DB exposes the media_items and scan_log tables.
    assert db.count() == 0
    tables = [r[0] for r in db._conn.execute(
        "SELECT table_name FROM information_schema.tables"
    ).fetchall()]
    assert "media_items" in tables
    assert "scan_log" in tables


def test_upsert_item_roundtrip(db):
    item = _sample_item()
    db.upsert_item(item)
    assert db.count() == 1
    fetched = db.item("abc")
    assert fetched is not None
    assert fetched.file_path == "/tmp/photo.jpg"
    assert fetched.file_size == 12345
    assert fetched.camera_make == "Canon"
    assert fetched.camera_serial == "SN12345"
    assert fetched.p_hash == 0x1234567890ABCDEF
    assert fetched.gps == pytest.approx((35.68, 139.69))
    assert fetched.timestamp_utc == datetime(2024, 6, 1, 12, 0, 0, tzinfo=timezone.utc)


def test_upsert_item_is_idempotent(db):
    """Upserting the same item_id twice must not create a duplicate row."""
    db.upsert_item(_sample_item())
    db.upsert_item(_sample_item())  # same id
    assert db.count() == 1


def test_upsert_item_updates_fields(db):
    item = _sample_item()
    db.upsert_item(item)
    # Re-upsert with updated metadata (e.g. scanner ran again).
    updated = _sample_item()
    updated.camera_model = "Canon EOS R5 II"
    updated.file_size = 99999
    db.upsert_item(updated)
    assert db.count() == 1
    fetched = db.item("abc")
    assert fetched.camera_model == "Canon EOS R5 II"
    assert fetched.file_size == 99999


def test_embeddings_upsert_and_fetch(db):
    db.upsert_item(_sample_item())
    vec = [0.1, 0.2, 0.3, 0.4]
    db.upsert_embeddings("abc", dinov2_vector=vec, aesthetic_score=0.85)
    fetched = db.item("abc")
    assert fetched.dinov2_vector == pytest.approx(vec)
    assert fetched.aesthetic_score == pytest.approx(0.85)
    assert "abc" not in db.pending_embeddings()  # now embedded, so NOT pending
    assert db.pending_embeddings() == []
    emb = db.all_embeddings()
    assert emb == [("abc", pytest.approx(vec))]


def test_pending_embeddings(db):
    db.upsert_item(_sample_item())
    assert db.pending_embeddings() == ["abc"]


def test_nearest_neighbors(db):
    db.upsert_item(_sample_item(item_id="a"))
    db.upsert_item(_sample_item(item_id="b"))
    db.upsert_embeddings("a", dinov2_vector=[1.0, 0.0, 0.0])
    db.upsert_embeddings("b", dinov2_vector=[0.0, 1.0, 0.0])
    results = db.nearest_neighbors([1.0, 0.0, 0.0], limit=2)
    assert results[0][0] == "a"
    assert results[0][1] == pytest.approx(1.0, abs=1e-6)
    assert results[1][0] == "b"
    assert results[1][1] == pytest.approx(0.0, abs=1e-6)


def test_nearest_neighbors_exclude(db):
    db.upsert_item(_sample_item(item_id="a"))
    db.upsert_embeddings("a", dinov2_vector=[1.0, 0.0])
    results = db.nearest_neighbors([1.0, 0.0], limit=2, exclude={"a"})
    assert results == []


def test_upsert_metadata_partial(db):
    db.upsert_item(_sample_item())
    db.upsert_metadata(
        "abc", camera_make="Nikon", gps=(48.85, 2.35), timestamp_utc=None
    )
    fetched = db.item("abc")
    assert fetched.camera_make == "Nikon"
    assert fetched.gps == pytest.approx((48.85, 2.35))
    # Untouched fields remain.
    assert fetched.file_size == 12345


def test_event_and_duplicate_assignment(db):
    db.upsert_item(_sample_item())
    db.set_event("abc", "event-1")
    assert db.item("abc").event_id == "event-1"
    db.set_duplicate_group(["abc"], "dup-1")
    assert db.item("abc").duplicate_group_id == "dup-1"


def test_batch_upsert(db):
    items = [_sample_item(str(i)) for i in range(10)]
    for it in items:
        it.file_path = f"/tmp/{it.item_id}.jpg"
    n = db.batch_upsert_items(items)
    assert n == 10
    assert db.count() == 10


def test_log_scan(db):
    db.log_scan("/tmp/a.jpg", "ok")
    db.log_scan("/tmp/b.jpg", "error", message="bad header")
    rows = db._conn.execute(
        "SELECT path, status, message FROM scan_log ORDER BY id"
    ).fetchall()
    assert rows == [
        ("/tmp/a.jpg", "ok", None),
        ("/tmp/b.jpg", "error", "bad header"),
    ]


# -- pure helper tests ------------------------------------------------------
def test_gps_pair_normalises_lonlat():
    # Some callers pass (lon, lat); we always store (lat, lon).
    assert gps_to_pair((139.69, 35.68)) == (139.69, 35.68)


def test_crop_to_pair_pads_and_clips():
    assert crop_to_pair((1, 2, 3)) == [1.0, 2.0, 3.0, 0.0]
    assert crop_to_pair((1, 2, 3, 4, 5)) == [1.0, 2.0, 3.0, 4.0]
    assert crop_to_pair(None) is None


def test_floats_to_sql_roundtrips():
    lit = floats_to_sql([1.0, 2.5, 3.0])
    assert lit.startswith("[") and lit.endswith("]")
    # NULL stays NULL (not a bare string).
    assert floats_to_sql(None) is None
