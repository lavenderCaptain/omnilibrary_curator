"""Unit tests for the Phase 2 embeddings worker.

These are fast and offline: the DINOv2 model is exercised against a single
32x32 synthetic JPEG, and the worker's contract (embedding dim, per-file
result shape, non-destructive RAW preview) is asserted. The expensive real-RAW
embeddings are covered by ``test_embed_real_raw_sample`` which is skipped
unless genuine samples exist.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

import engine.embeddings as em
from engine.database import Database
from engine.scanner import scan_directory


def _make_jpeg(size=(64, 64), color=(140, 90, 200)) -> Image.Image:
    return Image.fromarray(np.full((size[1], size[0], 3), color, np.uint8), "RGB")


# -- constants -------------------------------------------------------------
def test_embedding_dim_matches_dinov2_vits14():
    assert em.EMBEDDING_DIM == 384


# -- worker lifecycle ------------------------------------------------------
def test_worker_reports_missing_file_as_error(tmp_path):
    worker = em.EmbeddingWorker(batch_size=4)
    missing = tmp_path / "does_not_exist.jpg"
    results = list(worker.process([missing]))
    # A file that can't be decoded surfaces as a result with an error,
    # not an exception — so one bad file can't abort a long pass.
    assert len(results) == 1
    assert results[0].error is not None


def test_worker_loads_on_real_image(sample_jpeg_bytes, tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(sample_jpeg_bytes)
    worker = em.EmbeddingWorker(batch_size=2)
    results = list(worker.process([p]))
    assert len(results) == 1
    r = results[0]
    assert r.error is None
    assert r.dinov2_vector is not None
    assert len(r.dinov2_vector) == em.EMBEDDING_DIM


def test_embedding_batch_is_normalized(sample_jpeg_bytes, tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(sample_jpeg_bytes)
    worker = em.EmbeddingWorker(batch_size=2)
    img = Image.open(p).convert("RGB")
    emb = worker.embed_batch([img])[0]
    assert len(emb) == em.EMBEDDING_DIM
    import math

    assert abs(math.sqrt(sum(v * v for v in emb)) - 1.0) < 1e-4


def test_phash_matches_imagehash(sample_jpeg_bytes, tmp_path):
    import imagehash

    p = tmp_path / "photo.jpg"
    p.write_bytes(sample_jpeg_bytes)
    worker = em.EmbeddingWorker()
    img = Image.open(p).convert("RGB")
    h = worker.phash(img)
    # Compare against the canonical hex form of imagehash's output. This is the
    # stable contract regardless of whether the installed build defines __int__.
    canonical = int(str(imagehash.phash(img)), 16)
    assert h == canonical
    # A 64-bit hash never exceeds the unsigned 64-bit range.
    assert 0 <= h < 2**64


def test_phash_none_on_bad_input(tmp_path):
    worker = em.EmbeddingWorker()
    # PIL raises on garbage; our phash must return None, not raise.
    assert worker.phash(None) is None


def test_aesthetic_score_in_range(tmp_path):
    worker = em.EmbeddingWorker()
    img = _make_jpeg()
    score = worker.aesthetic_score(img)
    assert 0.0 <= score <= 10.0


def test_stats_counter(sample_jpeg_bytes, tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(sample_jpeg_bytes)
    worker = em.EmbeddingWorker()
    results = list(worker.process([p]))
    stats = worker.collect_stats(results)
    assert stats.processed == 1
    assert stats.ok == 1
    assert stats.failed == 0


# -- end-to-end with the Database layer ------------------------------------
def test_run_embeddings_writes_to_db(sample_jpeg_bytes, tmp_path):
    p = tmp_path / "photo.jpg"
    p.write_bytes(sample_jpeg_bytes)
    db = Database(":memory:")
    # Upsert a row first so run_embeddings has an item_id to write against.
    from engine.database import MediaItem

    item = MediaItem(file_path=str(p))
    db.upsert_item(item)

    worker = em.EmbeddingWorker(batch_size=1)
    results, stats = em.run_embeddings(
        db, worker, [p], item_ids=[item.item_id]
    )
    assert stats.ok == 1
    written = db.item(item.item_id)
    assert written.dinov2_vector is not None
    assert len(written.dinov2_vector) == em.EMBEDDING_DIM
    assert written.p_hash is not None
    assert written.aesthetic_score is not None


def test_run_embeddings_skips_errors(tmp_path):
    db = Database(":memory:")
    from engine.database import MediaItem

    missing = tmp_path / "nope.jpg"
    item = MediaItem(file_path=str(missing))
    db.upsert_item(item)

    worker = em.EmbeddingWorker()
    results, stats = em.run_embeddings(
        db, worker, [missing], item_ids=[item.item_id]
    )
    assert stats.failed == 1
    assert db.item(item.item_id).dinov2_vector is None


# -- RAW preview path (uses the embedded-JPEG extractor) -------------------
def test_embed_raw_uses_embedded_preview(sample_dng_bytes, tmp_path):
    p = tmp_path / "shot.dng"
    p.write_bytes(sample_dng_bytes)
    worker = em.EmbeddingWorker()
    results = list(worker.process([p]))
    r = results[0]
    assert r.error is None
    assert r.source_image is not None
    assert r.dinov2_vector is not None
    # The source RAW is left intact (fake header bytes preserved).
    assert p.read_bytes().startswith(b"RAWFILE")


# -- real samples (skipped without genuine photos) -------------------------
def test_embed_real_raw_sample(any_sample_raw, tmp_path):
    """A genuine camera RAW yields a 384-d embedding via its embedded preview."""
    worker = em.EmbeddingWorker()
    results = list(worker.process([any_sample_raw]))
    assert len(results) == 1
    assert results[0].error is None
    assert results[0].dinov2_vector is not None
    assert len(results[0].dinov2_vector) == em.EMBEDDING_DIM
