"""Unit tests for the Phase 2 two-tier deduplication engine.

These are fast, offline, and dependency-free. They build in-memory
:class:`MediaItem` rows with synthetic pHashs and DINOv2 vectors directly, then
assert the blueprint's two-tier semantics:

* same pHash AND high cosine similarity  -> confirmed duplicate;
* similar pHash but low cosine similarity -> rejected (fast tier false positive);
* identical images collapse into one group with a chosen Master;
* a RAW beats a JPG Master in the same group (quality scoring);
* transitive merges (A~B, B~C) form a single group;
* singletons and below-threshold pairs form no group;
* ``run_dedup`` persists ``duplicate_group_id`` and ``is_primary_master``.
"""

from __future__ import annotations

from engine.database import Database, MediaItem
from engine.dedup import (
    EMBEDDING_THRESHOLD,
    PHASH_THRESHOLD,
    cosine_similarity,
    find_duplicate_groups,
    is_raw,
    phash_hamming,
    quality_score,
    run_dedup,
)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------
def _item(item_id: str, *, file_format="JPEG", p_hash=None, vector=None, make="Canon", model="R5", serial="SN1"):
    return MediaItem(
        item_id=item_id,
        file_path=f"/tmp/{item_id}.{file_format.lower()}",
        file_format=file_format,
        file_size=1000 if file_format != "JPEG" else 500,
        camera_make=make,
        camera_model=model,
        camera_serial=serial,
        p_hash=p_hash,
        dinov2_vector=vector,
    )


def _identical_vec(dim: int = 32) -> list[float]:
    v = [0.0] * dim
    v[0] = 1.0
    return v


def _close_vec(dim: int = 32, jitter: float = 1e-4) -> list[float]:
    v = _identical_vec(dim)
    v[1] = jitter
    return v


def _far_vec(dim: int = 32) -> list[float]:
    v = [0.0] * dim
    v[1] = 1.0
    return v


def _upsert_many(db, items):
    for it in items:
        db.upsert_item(it)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def test_phash_hamming_exact_and_missing():
    assert phash_hamming(0b1010, 0b1010) == 0
    assert phash_hamming(0b1111, 0b0000) == 4
    assert phash_hamming(None, 5) == 65
    assert phash_hamming(5, None) == 65


def test_cosine_identical_and_orthogonal():
    a = _identical_vec()
    assert cosine_similarity(a, a) == 1.0
    assert cosine_similarity(a, _far_vec()) == 0.0
    assert cosine_similarity(None, a) is None
    # A zero vector is tolerated, not a crash.
    assert cosine_similarity([0.0] * len(a), a) == 0.0


def test_is_raw_formats():
    assert is_raw("CR2")
    assert is_raw(".cr3")
    assert is_raw("NEF")
    assert is_raw("DNG")
    assert not is_raw("JPEG")
    assert not is_raw(None)


def test_quality_score_raw_beats_jpg():
    raw = _item("raw", file_format="CR2", p_hash=1, vector=_identical_vec())
    jpg = _item("jpg", file_format="JPEG", p_hash=1, vector=_identical_vec())
    assert quality_score(raw) > quality_score(jpg)


# ---------------------------------------------------------------------------
# two-tier grouping
# ---------------------------------------------------------------------------
def test_identical_images_form_one_group():
    db = Database(":memory:")
    items = [
        _item("a", p_hash=0xABCD, vector=_identical_vec()),
        _item("b", p_hash=0xABCD, vector=_identical_vec()),
        _item("c", p_hash=0xABCD, vector=_identical_vec()),
    ]
    _upsert_many(db, items)
    groups, stats = find_duplicate_groups(db)
    assert len(groups) == 1
    assert set(groups[0].item_ids) == {"a", "b", "c"}
    assert stats.groups == 1
    assert stats.derivatives == 2


def test_same_phash_but_distinct_embeddings_rejected():
    """The fast tier is a superset; the slow tier must filter false positives.

    All three share a pHash (fast tier would flag them) but their embeddings are
    orthogonal, so the two-tier filter must *not* confirm them as duplicates.
    """
    db = Database(":memory:")
    items = [
        _item("a", p_hash=0xABCD, vector=_identical_vec()),
        _item("b", p_hash=0xABCD, vector=_far_vec()),
        _item("c", p_hash=0xABCD, vector=_identical_vec(dim=32)),
    ]
    # Make a and c share a near-identical embedding; b is orthogonal.
    items[2].dinov2_vector = _identical_vec()
    items[1].dinov2_vector = _far_vec()
    _upsert_many(db, items)
    groups, stats = find_duplicate_groups(db)
    # b is a false positive on pHash; only {a, c} confirm.
    assert len(groups) == 1
    assert set(groups[0].item_ids) == {"a", "c"}


def test_transitive_merge_single_group():
    """A~B and B~C (each confirmed) => one group of three."""
    db = Database(":memory:")
    a = _item("a", p_hash=0x1000, vector=_identical_vec())
    b = _item("b", p_hash=0x1100, vector=_close_vec())   # close to a on both tiers
    c = _item("c", p_hash=0x1100, vector=_close_vec())   # close to b on both tiers
    _upsert_many(db, [a, b, c])
    groups, _ = find_duplicate_groups(db)
    assert len(groups) == 1
    assert set(groups[0].item_ids) == {"a", "b", "c"}


def test_two_separate_groups():
    db = Database(":memory:")
    items = [
        _item("a", p_hash=0x11111111, vector=_identical_vec()),
        _item("b", p_hash=0x11111111, vector=_identical_vec()),
        _item("c", p_hash=0x22222222, vector=_identical_vec(dim=64)),
        _item("d", p_hash=0x22222222, vector=_identical_vec(dim=64)),
    ]
    _upsert_many(db, items)
    groups, _ = find_duplicate_groups(db)
    assert len(groups) == 2
    group_ids = {frozenset(g.item_ids) for g in groups}
    assert group_ids == {frozenset({"a", "b"}), frozenset({"c", "d"})}


def test_phash_missing_never_grouped():
    db = Database(":memory:")
    items = [
        _item("a", p_hash=None, vector=_identical_vec()),
        _item("b", p_hash=None, vector=_identical_vec()),
    ]
    _upsert_many(db, items)
    groups, _ = find_duplicate_groups(db)
    # Missing pHash means the fast tier never fires, so no group forms even
    # though the embeddings are identical.
    assert groups == []


def test_singleton_below_min_group():
    db = Database(":memory:")
    _upsert_many(db, [_item("solo", p_hash=0x99, vector=_identical_vec())])
    groups, _ = find_duplicate_groups(db)
    assert groups == []


def test_no_items():
    db = Database(":memory:")
    groups, stats = find_duplicate_groups(db)
    assert groups == []
    assert stats.scanned == 0


# ---------------------------------------------------------------------------
# master selection
# ---------------------------------------------------------------------------
def test_master_prefers_raw_over_jpg():
    db = Database(":memory:")
    jpg = _item("jpg", file_format="JPEG", p_hash=0xBEEF, vector=_identical_vec())
    raw = _item("raw", file_format="CR2", p_hash=0xBEEF, vector=_identical_vec())
    _upsert_many(db, [jpg, raw])
    groups, _ = find_duplicate_groups(db)
    assert len(groups) == 1
    assert groups[0].master_item_id == "raw"
    assert groups[0].derivative_item_ids == ["jpg"]


def test_master_deterministic():
    """Same inputs in a different insertion order pick the same Master."""
    db1 = Database(":memory:")
    db2 = Database(":memory:")
    db1_batch = [
        _item("a", file_format="CR2", p_hash=0x1, vector=_identical_vec()),
        _item("b", file_format="JPEG", p_hash=0x1, vector=_identical_vec()),
    ]
    db2_batch = list(reversed(db1_batch))
    _upsert_many(db1, db1_batch)
    _upsert_many(db2, db2_batch)
    g1, _ = find_duplicate_groups(db1)
    g2, _ = find_duplicate_groups(db2)
    assert g1[0].master_item_id == g2[0].master_item_id
    # RAW wins regardless of insertion order.
    assert g1[0].master_item_id == "a"


# ---------------------------------------------------------------------------
# persistence
# ---------------------------------------------------------------------------
def test_run_dedup_writes_to_db():
    db = Database(":memory:")
    jpg = _item("jpg", file_format="JPEG", p_hash=0xC0DE, vector=_identical_vec())
    raw = _item("raw", file_format="CR2", p_hash=0xC0DE, vector=_identical_vec())
    _upsert_many(db, [jpg, raw])
    groups, stats = run_dedup(db)
    assert len(groups) == 1
    gid = groups[0].group_id

    jpg_row = db.item("jpg")
    raw_row = db.item("raw")
    assert jpg_row.duplicate_group_id == gid
    assert raw_row.duplicate_group_id == gid
    # Master (RAW) flagged primary; JPG marked derivative.
    assert raw_row.is_primary_master is True
    assert jpg_row.is_primary_master is False
    assert stats.masters == 1
    assert stats.derivatives == 1


def test_phash_threshold_boundary():
    """Exactly PHASH_THRESHOLD bits apart still counts; one more does not.

    ``(1 << N) - 1`` is the integer with the low ``N`` bits set, so its Hamming
    distance from 0 is exactly ``N``.
    """
    db = Database(":memory:")
    a = _item("a", p_hash=0, vector=_identical_vec())
    b = _item("b", p_hash=(1 << PHASH_THRESHOLD) - 1, vector=_identical_vec())  # exactly at threshold
    _upsert_many(db, [a, b])
    groups, _ = find_duplicate_groups(db)
    assert len(groups) == 1

    # One bit beyond the threshold -> not confirmed.
    db2 = Database(":memory:")
    c = _item("c", p_hash=0, vector=_identical_vec())
    d = _item("d", p_hash=(1 << (PHASH_THRESHOLD + 1)) - 1, vector=_identical_vec())
    _upsert_many(db2, [c, d])
    groups2, _ = find_duplicate_groups(db2)
    assert groups2 == []


def test_embedding_threshold_boundary():
    """Two embeddings just above and just below the cosine threshold."""
    import math

    def vec_at_cosine(target: float, dim: int = 32) -> list[float]:
        # Unit vector whose overlap with [1,0,...] is exactly ``target``.
        return [target, math.sqrt(1.0 - target * target)] + [0.0] * (dim - 2)

    db = Database(":memory:")
    a = _item("a", p_hash=0x5, vector=_identical_vec())
    b = _item("b", p_hash=0x5, vector=vec_at_cosine(EMBEDDING_THRESHOLD + 0.005))
    _upsert_many(db, [a, b])
    above, _ = find_duplicate_groups(db)
    assert len(above) == 1  # 0.975 >= 0.97 -> confirmed

    db2 = Database(":memory:")
    c = _item("c", p_hash=0x5, vector=_identical_vec())
    d = _item("d", p_hash=0x5, vector=vec_at_cosine(EMBEDDING_THRESHOLD - 0.005))
    _upsert_many(db2, [c, d])
    below, _ = find_duplicate_groups(db2)
    assert below == []  # 0.965 < 0.97 -> rejected


def test_pairs_checked_counters():
    db = Database(":memory:")
    a = _item("a", p_hash=0x0000FFFF, vector=_identical_vec())
    b = _item("b", p_hash=0x0000FFFF, vector=_identical_vec())
    c = _item("c", p_hash=0xFFFF0000, vector=_identical_vec())  # pHash far from a/b
    _upsert_many(db, [a, b, c])
    _, stats = find_duplicate_groups(db)
    # Only {a, b} pass the fast tier; c's pHash is too far.
    assert stats.pairs_checked == 1
    assert stats.pairs_confirmed == 1
