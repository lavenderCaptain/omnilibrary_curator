"""Shared pytest fixtures for the Omnilibrary Phase 1 test-suite.

The fixtures generate *synthetic* assets on disk so the suite runs fully
offline with no real photos and no exiftool:

* ``.jpg``  -- a tiny real JPEG produced by Pillow.
* ``.dng``  -- a synthetic RAW whose bytes are ``<fake header> + <real jpeg>``.
             This exercises the embedded-JPEG marker-fallback path used when
             libraw (rawpy) cannot decode an exotic RAW.
"""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image
from pathlib import Path

from engine.database import Database


def _make_jpeg_bytes(color=(200, 120, 80), size=(32, 32)) -> bytes:
    arr = np.full((size[1], size[0], 3), color, dtype=np.uint8)
    img = Image.fromarray(arr, "RGB")
    import io

    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=90)
    return buf.getvalue()


@pytest.fixture
def sample_jpeg_bytes() -> bytes:
    """A minimal but valid standalone JPEG (SOI/EOI present)."""
    return _make_jpeg_bytes()


@pytest.fixture
def sample_dng_bytes(sample_jpeg_bytes: bytes) -> bytes:
    """A synthetic "RAW" that is really a fake header followed by a real JPEG."""
    header = b"RAWFILE\x00\x01\x02\x03-synthetic-dng-for-tests-"
    # Pad to a fixed width to mimic how RAW files carry a preview embedded.
    header = header.ljust(64, b"\x00")
    return header + sample_jpeg_bytes


@pytest.fixture
def media_dir(tmp_path):
    """A directory tree with one JPEG and one synthetic RAW file."""
    (tmp_path / "sub").mkdir()
    jpeg = tmp_path / "photo.jpg"
    dng = tmp_path / "sub" / "shot.dng"
    jpeg.write_bytes(_make_jpeg_bytes())
    dng.write_bytes(
        (b"RAWFILE\x00-synthetic-raw-with-embedded-jpeg-"
         + b"\x00" * 48 + _make_jpeg_bytes())
    )
    return tmp_path


# ---------------------------------------------------------------------------
# Real-sample fixtures
#
# ``sample_images/`` is a symlink to a folder of genuine photos (CR2/CR3 RAW
# and JPEG). The RAWs let us prove that embedded-JPEG extraction works against
# *real* camera files, not just the synthetic ``.dng`` fixture above. The
# fixtures are optional: if the symlink / volume is not present the suite still
# passes, and any test that needs a RAW is skipped.
# ---------------------------------------------------------------------------
import os as _os

# Real-photo folder shipped alongside the project (symlink to an external disk).
SAMPLE_IMAGES_DIR = _os.path.join(_os.path.dirname(__file__), "..", "sample_images")


def _iter_samples(root, exts):
    """Yield every file under ``root`` whose extension is in ``exts`` (lowercase)."""
    found = []
    for dirpath, _dirs, files in _os.walk(root):
        for name in files:
            if name.lower().endswith(exts):
                found.append(Path(_os.path.join(dirpath, name)))
    return sorted(found)


def _sample_raws():
    if not _os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    return _iter_samples(SAMPLE_IMAGES_DIR, (".cr2", ".cr3", ".dng", ".arw", ".nef", ".orf"))


def _sample_jpgs():
    if not _os.path.isdir(SAMPLE_IMAGES_DIR):
        return []
    return _iter_samples(SAMPLE_IMAGES_DIR, (".jpg", ".jpeg"))


@pytest.fixture
def sample_raw_paths():
    """Absolute paths to the real RAW files under ``sample_images/`` (may be empty)."""
    return _sample_raws()


@pytest.fixture
def sample_jpg_paths():
    """Absolute paths to the real JPEG files under ``sample_images/`` (may be empty)."""
    return _sample_jpgs()


@pytest.fixture
def any_sample_raw(sample_raw_paths):
    """One real RAW file, or ``pytest.skip`` if none are available."""
    if not sample_raw_paths:
        pytest.skip("no real RAW files available (sample_images/ missing)")
    return sample_raw_paths[0]


@pytest.fixture
def sample_images_dir():
    return SAMPLE_IMAGES_DIR


@pytest.fixture
def db() -> Database:
    """An in-memory database (fresh per test)."""
    return Database(":memory:")
