"""Unit tests for the Phase 2 multi-signal clustering engine.

These are fast, offline, and dependency-free (no hdbscan/sklearn). They build
in-memory :class:`MediaItem` rows directly and assert the clustering semantics
called for in the blueprint:

* photos taken on the same day merge into one event;
* a time gap larger than ``max_span_hours`` splits a run;
* photos taken by different camera bodies on the same day do NOT merge
  (camera-body anchoring);
* a co-located same-clock-window on different bodies DOES merge when GPS is
  within ``gps_co_locate_meters``;
* photos with no time and no GPS are anchored into events by visual embedding
  proximity;
* singleton clusters below ``min_count`` get no event_id;
* oversized events are split by earliest timestamp.
"""

from __future__ import annotations

from datetime import datetime, timezone

import numpy as np

from engine.clustering import (
    GPS_CO_LOCATE_METERS,
    body_signature,
    haversine_meters,
    run_clustering,
)
from engine.database import Database, MediaItem


# ---------------------------------------------------------------------------
# Builders
# ---------------------------------------------------------------------------
def _item(item_id: str, *, at: datetime | None = None, gps=None, make="Canon", model="R5", serial="SN1", fmt="JPEG"):
    return MediaItem(
        item_id=item_id,
        file_path=f"/tmp/{item_id}{fmt.lower()}",
        file_format=fmt,
        camera_make=make,
        camera_model=model,
        camera_serial=serial,
        timestamp_utc=at,
        gps=gps,
    )


def _day(n: int, hour: int = 10, minute: int = 0, *, tz=timezone.utc):
    return datetime(2024, 6, n, hour, minute, tzinfo=tz)


def _ids(results) -> set[str]:
    out: set[str] = set()
    for r in results:
        out.update(r.item_ids)
    return out


def _find(results, item_id: str):
    for r in results:
        if item_id in r.item_ids:
            return r
    return None


def _upsert_many(db, items):
    for it in items:
        db.upsert_item(it)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_haversine_roughly_correct():
    # Tokyo -> slightly further. ~1 km apart should be ~1 km.
    d = haversine_meters(35.68, 139.69, 35.69, 139.69)
    assert 800 < d < 1200


def test_body_signature_prefers_serial():
    a = _item("a", make="Canon", model="R5", serial="SN1")
    b = _item("b", make="Canon", model="R5", serial="SN2")
    assert body_signature(a) != body_signature(b)
    c = _item("c", make="Canon", model="R5", serial="SN1")
    assert body_signature(a) == body_signature(c)


def test_body_signature_falls_back_to_format():
    x = _item("x", make=None, model=None, serial=None, fmt="NEF")
    y = _item("y", make=None, model=None, serial=None, fmt="NEF")
    assert body_signature(x) == body_signature(y)
    z = _item("z", make=None, model=None, serial=None, fmt="ARW")
    assert body_signature(x) != body_signature(z)


# ---------------------------------------------------------------------------
# core clustering scenarios
# ---------------------------------------------------------------------------
def test_same_day_same_body_merges():
    db = Database(":memory:")
    items = [_item(f"a{i}", at=_day(1, hour=10 + i)) for i in range(4)]
    _upsert_many(db, items)
    results = run_clustering(db)
    assert len(results) == 1
    assert _ids(results) == {"a0", "a1", "a2", "a3"}


def test_time_gap_splits_run():
    db = Database(":memory:")
    # 09:00, then a 30h gap to the next day at 01:00.
    items = [
        _item("a", at=_day(1, hour=9)),
        _item("b", at=_day(1, hour=10)),
        _item("c", at=_day(2, hour=7)),  # >24h after b's 10:00 -> gap
    ]
    _upsert_many(db, items)
    results = run_clustering(db)
    # a and b are within a day; c is a separate event.
    assert _find(results, "a") is not None and _find(results, "b") is not None
    r_a = _find(results, "a")
    r_c = _find(results, "c")
    assert r_a is not r_c
    assert "c" not in r_a.item_ids


def test_same_day_different_bodies_do_not_merge():
    db = Database(":memory:")
    a = _item("a", at=_day(1, hour=10), serial="SN1")
    b = _item("b", at=_day(1, hour=11), serial="SN2")
    _upsert_many(db, [a, b])
    results = run_clustering(db)
    # Same day, different bodies -> two singletons, both below min_count(2).
    for it in ("a", "b"):
        assert _find(results, it) is None
    # Nothing assigned an event.
    for r in results:
        assert r.size < 2


def test_different_bodies_merge_when_gps_colocated():
    db = Database(":memory:")
    # Same clock window (~5 min apart), different bodies, ~10 m apart GPS.
    a = _item("a", at=_day(1, hour=10, minute=0), gps=(35.68, 139.69), serial="SN1")
    b = _item("b", at=_day(1, hour=10, minute=5), gps=(35.68, 139.6901), serial="SN2")
    _upsert_many(db, [a, b])
    results = run_clustering(db)
    assert len(results) == 1
    assert _ids(results) == {"a", "b"}


def test_same_body_same_time_merge_gps_irrelevant():
    db = Database(":memory:")
    # Same body, same time, wildly different GPS -> still merges (body matches).
    a = _item("a", at=_day(1, hour=10), gps=(35.68, 139.69), serial="SN1")
    b = _item("b", at=_day(1, hour=10), gps=(48.85, 2.35), serial="SN1")
    _upsert_many(db, [a, b])
    results = run_clustering(db)
    assert len(results) == 1


def test_exifless_photos_anchored_by_visual():
    db = Database(":memory:")
    # No time, no GPS -> must be clustered purely by DINOv2 proximity.
    a = _item("a", at=None, gps=None)
    b = _item("b", at=None, gps=None)
    c = _item("c", at=None, gps=None)
    _upsert_many(db, [a, b, c])
    # Two near-identical vectors (a,b) and one far away (c).
    vecs = {
        "a": [1.0, 0.0, 0.0],
        "b": [1.0, 1e-4, 0.0],  # cosine dist ~5e-9 <= epsilon
        "c": [0.0, 1.0, 0.0],  # orthogonal -> distance 1.0 > epsilon
    }
    for iid, vec in vecs.items():
        db.upsert_embeddings(iid, dinov2_vector=vec)
    results = run_clustering(db)
    # One event holding a+b, anchored by visual; c is a singleton below min_count.
    visual_events = [r for r in results if r.anchored_by_visual]
    assert len(visual_events) == 1
    vev = visual_events[0]
    assert set(vev.item_ids) == {"a", "b"}


def test_visual_no_components_below_min_count():
    db = Database(":memory:")
    a = _item("a", at=None, gps=None)
    _upsert_many(db, [a])
    db.upsert_embeddings("a", dinov2_vector=[1.0, 0.0])
    results = run_clustering(db)
    # A single EXIF-less photo can't form a cluster -> no event_id.
    assert _find(results, "a") is None


def test_oversized_event_is_split():
    db = Database(":memory:")
    # 10 items same day same body -> exceeds MAX_MEMBERS? No, 10 < 10_000.
    # Force a smaller guard via the parameter to exercise the split path.
    items = [_item(f"o{i}", at=_day(1, hour=10 + i)) for i in range(5)]
    _upsert_many(db, items)
    results = run_clustering(db, max_members=2)
    total = sum(r.size for r in results)
    assert total == 5
    for r in results:
        assert r.size <= 2


def test_all_no_embeddings_no_visual_events():
    db = Database(":memory:")
    items = [
        _item("a", at=_day(1, hour=10)),
        _item("b", at=_day(1, hour=11)),
    ]
    _upsert_many(db, items)
    results = run_clustering(db)
    assert len(results) == 1
    assert _ids(results) == {"a", "b"}


def test_set_event_written_to_db():
    db = Database(":memory:")
    items = [_item(f"d{i}", at=_day(1, hour=10 + i)) for i in range(3)]
    _upsert_many(db, items)
    run_clustering(db)
    for it in items:
        assert db.item(it.item_id).event_id is not None


def test_no_items_empty_results():
    db = Database(":memory:")
    assert run_clustering(db) == []


def test_min_count_excludes_singletons():
    db = Database(":memory:")
    # Two on day 1 (same body -> merge), one lone on day 2 (no group).
    items = [
        _item("a", at=_day(1, hour=10)),
        _item("b", at=_day(1, hour=11)),
        _item("c", at=_day(3, hour=10)),
    ]
    _upsert_many(db, items)
    results = run_clustering(db)
    assert len(results) == 1
    assert _ids(results) == {"a", "b"}
    assert _find(results, "c") is None
