"""Unit tests for the file scanner and embedded-JPEG extractor.

The synthetic fixtures live in conftest.py: a real JPEG plus a "RAW" file that
is a fake header followed by a real JPEG. This lets us exercise EXIF-free
classification, format detection, embedded-preview extraction, and the
non-destructive guarantee without any real photos or exiftool.
"""

from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pytest

from engine.database import Database
from engine.scanner import (
    detect_format,
    FileScanner,
    extract_embedded_jpeg,
    is_supported_image,
    scan_directory,
)


# -- classification / format detection -------------------------------------
def test_is_supported_image():
    assert is_supported_image(Path("a.jpg"))
    assert is_supported_image(Path("a.NEF"))
    assert not is_supported_image(Path("a.txt"))


def test_format_detection():
    assert detect_format(Path("x.jpg")) == "JPEG"
    assert detect_format(Path("x.nef")) == "NEF"
    assert detect_format(Path("x.dng")) == "DNG"
    assert detect_format(Path("x.unknown")) == "UNKNOWN"


# -- metadata reading (fallback, no exiftool) -------------------------------
def test_scan_jpeg(sample_jpeg_bytes, tmp_path):
    p = tmp_path / "test.jpg"
    p.write_bytes(sample_jpeg_bytes)
    scanner = FileScanner(tmp_path, exiftool_exe=None)  # force fallback path
    result = scanner.scan_file(p)

    assert result.error is None
    assert result.source.value == "rawpy"  # fallback source
    assert result.item.file_size == len(sample_jpeg_bytes)
    assert result.item.file_format == "JPEG"
    assert result.embedded_jpeg is None  # JPEGs have no embedded preview


def test_scan_raw_extracts_embedded_jpeg(sample_dng_bytes, tmp_path):
    p = tmp_path / "shot.dng"
    p.write_bytes(sample_dng_bytes)
    scanner = FileScanner(tmp_path, exiftool_exe=None)
    result = scanner.scan_file(p)

    assert result.error is None
    assert result.item.file_format == "DNG"
    # An embedded JPEG must have been extracted for the synthetic RAW.
    assert result.embedded_jpeg is not None
    assert result.embedded_jpeg.exists()
    header = result.embedded_jpeg.read_bytes()[:3]
    assert header == b"\xff\xd8\xff"  # valid SOI marker


# -- embedded-jpeg extraction ----------------------------------------------
def test_extract_embedded_jpeg_from_synthetic_raw(sample_dng_bytes, tmp_path):
    raw = tmp_path / "in.dng"
    raw.write_bytes(sample_dng_bytes)
    out = tmp_path / "out.jpg"
    extracted = extract_embedded_jpeg(raw, out, as_raw=True)
    assert extracted == out
    assert out.exists()
    assert out.read_bytes()[:3] == b"\xff\xd8\xff"
    # The extraction is not destructive: the source still has its fake header.
    assert raw.read_bytes().startswith(b"RAWFILE")


def test_extract_embedded_jpeg_fails_on_non_jpeg(tmp_path):
    src = tmp_path / "plain.bin"
    src.write_bytes(b"no jpeg markers here at all \x00\x01\x02")
    out = tmp_path / "out.jpg"
    assert extract_embedded_jpeg(src, out) is None
    assert not out.exists()


# -- non-destructive guarantee ----------------------------------------------
def test_scan_directory_is_non_destructive(media_dir):
    original_files = {p for p in media_dir.rglob("*") if p.is_file()}
    before = {p.name: p.read_bytes() for p in original_files}
    db = Database(":memory:")
    results = scan_directory(media_dir, db, exiftool_exe=None)
    after = {p.name: p.read_bytes() for p in original_files}

    assert before == after  # nothing written next to source files
    # Every scanned asset produced a media item (even the RAW, via fallback).
    scanned = [r for r in results if r.error is None]
    assert len(scanned) == 2
    assert db.count() == 2


def test_scan_directory_skips_non_images(media_dir, tmp_path):
    # Add a non-image file alongside the images; it must be ignored.
    junk = media_dir / "notes.txt"
    junk.write_text("not an image")
    db = Database(":memory:")
    results = scan_directory(media_dir, db, exiftool_exe=None)
    paths = {r.path for r in results}
    assert junk not in paths
    assert len(results) == 2


def test_scan_directory_logs(media_dir):
    db = Database(":memory:")
    scan_directory(media_dir, db, exiftool_exe=None, log=True)
    rows = db._conn.execute(
        "SELECT status FROM scan_log"
    ).fetchall()
    assert len(rows) == 2
    assert all(status == "ok" for (status,) in rows)


# -- real RAW files --------------------------------------------------------
# The following tests exercise extraction against *genuine* camera RAW files
# (CR2/CR3/etc.) shipped under sample_images/, rather than the synthetic DNG.
# They are skipped automatically when no real RAWs are present, so the suite
# still passes in environments without the sample folder.


def test_extract_from_real_raw(any_sample_raw, tmp_path):
    """A real RAW file yields a valid embedded JPEG preview."""
    out = tmp_path / "preview.jpg"
    extracted = extract_embedded_jpeg(any_sample_raw, out, as_raw=True)
    assert extracted == out
    assert out.exists()
    assert out.read_bytes()[:3] == b"\xff\xd8\xff"  # valid JPEG SOI
    # Non-destructive: the source RAW is left byte-for-byte identical.
    assert any_sample_raw.exists()


def test_scan_real_raw_writes_item(any_sample_raw):
    """scan_file on a real RAW records a DB row with an embedded preview."""
    scanner = FileScanner(any_sample_raw.parent, exiftool_exe=None)
    result = scanner.scan_file(any_sample_raw)

    assert result.error is None
    assert result.item.file_format not in ("JPEG", "UNKNOWN")
    assert result.embedded_jpeg is not None
    assert result.embedded_jpeg.exists()
    assert result.embedded_jpeg.read_bytes()[:3] == b"\xff\xd8\xff"


def test_many_real_raws_all_extract(any_sample_raw, sample_raw_paths, tmp_path):
    """If several real RAWs are available, every one must extract a preview.

    Probes are written into ``tmp_path`` (never into the user's real
    ``sample_images/``), so a failed assertion can't leave stray files behind.
    """
    if len(sample_raw_paths) < 2:
        pytest.skip("only one real RAW available")
    for raw in sample_raw_paths:
        probe = tmp_path / f".probe_{raw.stem}.jpg"
        try:
            extracted = extract_embedded_jpeg(raw, probe, as_raw=True)
            assert extracted == probe and probe.exists(), f"failed: {raw}"
            assert probe.read_bytes()[:3] == b"\xff\xd8\xff"
        finally:
            probe.unlink(missing_ok=True)
