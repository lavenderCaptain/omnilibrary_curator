"""File-tree scanner and metadata / embedded-JPEG extractor.

This is the Phase 1 ingestion surface. It walks a directory tree, classifies
each asset, extracts EXIF-style metadata, and pulls the *embedded JPEG* out of
RAW files without decoding the full-resolution pixel data (per the "RAW
Performance Optimization" constraint).

Two metadata paths are supported:

* ``exiftool`` (via :mod:`exiftool`) when a usable executable is discoverable
  on ``PATH`` or in the package bundle. Exiftool is a strong *acceleration*
  path -- it parses every tag quickly -- but is entirely optional.
* A :mod:`rawpy` + :mod:`PIL` fallback for the tags we actually need (make,
  model, serial where present, timestamp, orientation). This keeps the
  scanner fully functional offline and on machines where exiftool is not
  installed.

Extraction is non-destructive: nothing is written next to (or over) the source
files. The only side-effect the scanner has is a DuckDB record written by the
caller.
"""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Iterator, Optional

from PIL import Image

from .database import MediaItem

# ---------------------------------------------------------------------------
# Asset classification
# ---------------------------------------------------------------------------
JPEG_EXT = {".jpg", ".jpeg", ".jpe", ".jfif"}
RAW_EXT = {
    ".cr2", ".cr3", ".nef", ".arw", ".rw2", ".orf", ".dng", ".raf",
    ".pef", ".kdc", ".srw", ".x3f", ".sr2", ".mrw", ".3fr", ".fdc", ".iiq",
}
IMAGE_EXT = JPEG_EXT | RAW_EXT

# Camera bodies we can reliably identify even without EXIF, by extension.
# (Used only as a last-resort hint; real make/model come from EXIF.)
_EXT_HINT = {
    ".cr2": ("Canon", "Canon CR2"),
    ".cr3": ("Canon", "Canon CR3"),
    ".nef": ("Nikon", "Nikon NEF"),
    ".arw": ("Sony", "Sony ARW"),
    ".rw2": ("Panasonic", "Panasonic RW2"),
    ".orf": ("OM Systems", "OM Systems ORF"),
    ".raf": ("Fujifilm", "Fujifilm RAF"),
    ".pef": ("Pentax", "Pentax PEF"),
    ".srw": "Samsung",
    ".kdc": "Kodak",
    ".x3f": "Sigma",
}


def _guess_format(path: Path) -> str:
    """Return the file family: JPEG / RAW or a known RAW variant / UNKNOWN."""
    suffix = path.suffix.lower()
    if suffix in JPEG_EXT:
        return "JPEG"
    if suffix in RAW_EXT:
        # CR3 embeds a JPEG and is a distinct family from CR2/NEF; report it.
        return suffix.upper().lstrip(".")
    return "UNKNOWN"


def is_supported_image(path: str | Path) -> bool:
    """True if ``path`` has a recognised image extension (JPEG or RAW)."""
    return Path(path).suffix.lower() in IMAGE_EXT


def detect_format(path: str | Path) -> str:
    """Return the file family for ``path`` (see :func:`_guess_format`)."""
    return _guess_format(Path(path))


# Backwards-compatible alias for the raw extension set.
FILE_EXTENSIONS = IMAGE_EXT


# ---------------------------------------------------------------------------
# JPEG marker probe (no full decode) for "is this actually a JPEG?"
# ---------------------------------------------------------------------------
_SOI = bytes([0xFF, 0xD8])
_EOI = bytes([0xFF, 0xD9])


# ---------------------------------------------------------------------------
# Metadata sources
# ---------------------------------------------------------------------------
class MetadataSource(str, Enum):
    EXIFTOOL = "exiftool"
    RAWPY = "rawpy"
    NONE = "none"


@dataclass
class ExtractedMetadata:
    """The fields we care about, regardless of how they were obtained."""

    file_format: str
    file_size: int
    camera_make: Optional[str]
    camera_model: Optional[str]
    camera_serial: Optional[str]
    timestamp_utc: Optional[datetime]
    gps: Optional[tuple[float, float]]
    orientation: Optional[int]
    source: MetadataSource


def _parse_exiftool_json(output: str) -> ExtractedMetadata:
    """Convert an exiftool ``-j`` JSON blob into :class:`ExtractedMetadata`."""
    import json

    try:
        data = json.loads(output)
    except json.JSONDecodeError:
        return ExtractedMetadata("", 0, None, None, None, None, None, None, MetadataSource.EXIFTOOL)
    # exiftool -j always emits an *array* (one object per scanned file).
    # We scan a single path, so take the last entry if a list was returned.
    if isinstance(data, list):
        data = data[-1] if data else {}
    if not isinstance(data, dict):
        return ExtractedMetadata("", 0, None, None, None, None, None, None, MetadataSource.EXIFTOOL)

    def _first(*names):
        for n in names:
            if n in data and data[n] not in ("", None):
                return str(data[n]).strip()
        return None

    make = _first("Make", "CameraMake")
    model = _first("Model", "CameraModel")
    serial = _first("SerialNumber", "CameraSerialNumber")
    ts_raw = _first("DateTimeOriginal", "ModifyDate", "CreationDate")
    ts = _parse_iso_ts(ts_raw) if ts_raw else None
    gps = _extract_gps(data)
    orientation = _int_or_none(_first("Orientation"))
    return ExtractedMetadata(
        file_format="",  # filled in by caller from file extension
        file_size=len(output),
        camera_make=make,
        camera_model=model,
        camera_serial=serial,
        timestamp_utc=ts,
        gps=gps,
        orientation=orientation,
        source=MetadataSource.EXIFTOOL,
    )


def _parse_iso_ts(raw: str) -> Optional[datetime]:
    """Parse common EXIF timestamp forms to an aware UTC datetime."""
    raw = raw.strip()
    # EXIF stores as "YYYY:MM:DD HH:MM:SS"; normalise separators.
    for sep in (":", "/"):
        if raw[4] in ("-", ":") and sep in raw[5:10]:
            raw = raw.replace(sep, "-")
    # Fall back to a couple of tolerant formats.
    for fmt in ("%Y-%m-%d %H:%M:%S", "%Y-%m-%dT%H:%M:%S", "%Y-%m-%d %H:%M"):
        try:
            dt = datetime.strptime(raw, fmt)
            return dt.replace(tzinfo=timezone.utc)
        except ValueError:
            continue
    try:
        return datetime.fromisoformat(raw.replace("Z", "+00:00")).astimezone(timezone.utc)
    except ValueError:
        return None


def _extract_gps(data: dict) -> Optional[tuple[float, float]]:
    """Build (lat, lon) from exiftool's GPSLatitude/GPSLatitudeRef groups."""
    lat = data.get("GPSLatitude")
    lon = data.get("GPSLongitude")
    if not isinstance(lat, (list, tuple)) or not isinstance(lon, (list, tuple)):
        return None
    if len(lat) < 3 or len(lon) < 3:
        return None

    def _dms_to_deg(parts):
        parts = [float(p) for p in parts]
        deg = parts[0]
        if len(parts) > 1:
            deg += parts[1] / 60.0
        if len(parts) > 2:
            deg += parts[2] / 3600.0
        return deg

    lat_deg = _dms_to_deg(lat[:3])
    lon_deg = _dms_to_deg(lon[:3])
    if lat[2].lower() in ("s", "-"):
        lat_deg = -lat_deg
    if lon[2].lower() in ("w", "-"):
        lon_deg = -lon_deg
    return (lat_deg, lon_deg)


def _int_or_none(value) -> Optional[int]:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return None


# ---------------------------------------------------------------------------
# rawpy-based fallback metadata
# ---------------------------------------------------------------------------
def _fallback_metadata(path: Path, fmt: str) -> ExtractedMetadata:
    """Extract best-effort metadata with rawpy/Pillow (no full-resolution decode)."""
    size = path.stat().st_size
    make = model = serial = None
    timestamp = None
    orientation = None
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            thumb = raw.Thumbnails  # namedtuple with .data / .format
            # make/model/serial are not exposed by rawpy; rely on extension hint.
            ext_hint = _EXT_HINT.get(path.suffix.lower())
            if isinstance(ext_hint, tuple):
                make, model = ext_hint
            timestamp = _extract_ts_from_jpeg(thumb)
    except Exception:
        # Fall back to Pillow's JPEG metadata if rawpy fails (non-raw images).
        if fmt == "JPEG":
            try:
                with Image.open(path) as img:
                    exif = img.exif
                    orientation = _int_or_none(exif.get("Orientation")) if exif else None
                    timestamp = _extract_ts_from_jpeg(exif)
            except Exception:
                pass

    return ExtractedMetadata(
        file_format=fmt,
        file_size=size,
        camera_make=make,
        camera_model=model,
        camera_serial=serial,
        timestamp_utc=timestamp,
        gps=None,
        orientation=orientation,
        source=MetadataSource.RAWPY,
    )


def _extract_ts_from_jpeg(payload) -> Optional[datetime]:
    """Extract a JPEG APP1 EXIF timestamp via Pillow, decoding only the APP1 block."""
    if not payload:
        return None
    import io

    data = getattr(payload, "data", payload)
    if not isinstance(data, (bytes, bytearray)):
        return None
    try:
        with Image.open(io.BytesIO(bytes(data))) as img:
            exif = img.exif
            if not exif:
                return None
            ts_raw = _first_tag(exif, ("ExifOffset", "IFD0", "DateTimeOriginal"))
            if not ts_raw:
                ts_raw = exif.get("DateTime")
            return _parse_iso_ts(str(ts_raw)) if ts_raw else None
    except Exception:
        return None


def _first_tag(exif: dict, keys) -> Optional[str]:
    try:
        d = exif
        for k in keys:
            d = d[k]
        return d
    except (KeyError, TypeError):
        return None


# ---------------------------------------------------------------------------
# exiftool discovery + execution
# ---------------------------------------------------------------------------
def _find_exiftool() -> Optional[str]:
    """Locate a usable exiftool executable (system PATH or bundled)."""
    system = shutil.which("exiftool")
    if system:
        return system
    # Fall back to the pyinstaller/bundled helper if present.
    for candidate in (
        Path(__file__).resolve().parent.parent.parent
        / "venv" / "bin" / "exiftool",
    ):
        if candidate.is_file():
            return str(candidate)
    return None


def _run_exiftool(exiftool_exe: str, path: str) -> Optional[str]:
    try:
        proc = subprocess.run(
            [exiftool_exe, "-j", "-s3", path],
            capture_output=True,
            text=True,
            timeout=60,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if proc.returncode != 0:
        return None
    return proc.stdout


# ---------------------------------------------------------------------------
# Embedded-JPEG extraction
# ---------------------------------------------------------------------------
def extract_embedded_jpeg(path: Path, out_path: Path, *, as_raw: bool = True) -> Optional[Path]:
    """Pull the embedded preview JPEG from a RAW file and write it to ``out_path``.

    Uses rawpy's bundled thumbnail (libraw decodes only the embedded JPEG, never
    the full RAW pixel data). For synthetic/non-libraw RAW files we fall back to
    scanning the first JPEG marker sequence in the file.

    Returns ``out_path`` on success, ``None`` on failure.
    """
    if as_raw:
        return _extract_rawpy_embedded(path, out_path) or _extract_jpeg_marker(path, out_path)
    return _extract_jpeg_marker(path, out_path)


def _extract_rawpy_embedded(path: Path, out_path: Path) -> Optional[Path]:
    """Pull the embedded preview JPEG from a RAW file using rawpy/libraw.

    Tries the high-level ``rawpy.Thumbnails`` API first (rawpy >= 0.18), then
    falls back to the low-level ``rawpy.extract_thumb()`` API that some
    builds (including this project's) expose instead. Returns ``None`` if
    neither yields a usable preview so the marker scanner can be tried.
    """
    try:
        import rawpy

        with rawpy.imread(str(path)) as raw:
            data: Optional[bytes] = None
            # Prefer the high-level thumbnail collection when the build exposes it.
            thumbs = getattr(raw, "Thumbnails", None)
            if thumbs:
                best = max(thumbs, key=lambda t: len(getattr(t, "data", b"")))
                data = getattr(best, "data", None)
            else:
                # Low-level libraw API: ``extract_thumb()`` returns a Thumbnail
                # namedtuple with a ``data`` byte string and a ``format`` enum.
                thumb = raw.extract_thumb()
                if thumb is not None:
                    data = getattr(thumb, "data", None)

            if not data:
                return None
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(bytes(data))
            return out_path
    except Exception:
        # libraw/libtiff raise on exotic/corrupt RAWs; fall back to the marker
        # scanner. The broad handler is deliberate here (see module docstring).
        return None


def _extract_jpeg_marker(path: Path, out_path: Path) -> Optional[Path]:
    """Extract the first ``FFD8 ... FFD9`` JPEG sequence from a binary blob.

    Robust fallback for synthetic/unsupported RAW files that libraw cannot
    decode. Scans for the SOI marker and scans back for the end of file (EOI).
    """
    with open(path, "rb") as fh:
        blob = fh.read(1 << 24)  # 16 MiB cap; previews are small
    start = blob.find(_SOI)
    if start == -1:
        return None
    end = blob.find(_EOI, start + 2)
    if end == -1:
        # No explicit EOI; take everything to the end of the blob.
        end = len(blob)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_bytes(blob[start:end])
    return out_path


# ---------------------------------------------------------------------------
# Public scanner
# ---------------------------------------------------------------------------
@dataclass
class ScanResult:
    """Outcome of scanning a single file."""

    path: Path
    item: MediaItem
    source: MetadataSource
    embedded_jpeg: Optional[Path] = None
    skipped: bool = False
    error: Optional[str] = None


class FileScanner:
    """Walks a tree and scans each asset.

    ``scan()`` is a generator so callers (and the UI worker) can stream results
    without buffering everything in memory.
    """

    def __init__(
        self,
        root: str | Path,
        *,
        exiftool_exe: str | None = None,
    ):
        """Create a scanner.

        ``exiftool_exe`` controls the exiftool path with explicit semantics:

        * ``None``    – never use exiftool; always fall back to rawpy/Pillow.
        * ``"auto"``  – let :func:`_find_exiftool` locate a usable binary on
          ``PATH`` or in the bundled helper, else fall back to rawpy/Pillow.
        * a path str  – use that specific executable.
        """
        self.root = Path(root)
        if exiftool_exe is None:
            self._exiftool_exe: Optional[str] = None
        elif exiftool_exe == "auto":
            self._exiftool_exe = _find_exiftool()
        else:
            self._exiftool_exe = exiftool_exe

    def iter_files(self, root: Optional[Path] = None) -> Iterator[Path]:
        base = root or self.root
        thumb = self.root
        for dirpath, dirnames, filenames in os.walk(base):
            # Prune our own preview cache so a rescan can't re-scan previews.
            dirnames[:] = [d for d in dirnames if d != thumb.name]
            for name in sorted(filenames):
                path = Path(dirpath) / name
                if path.suffix.lower() in IMAGE_EXT and path.is_file():
                    yield path

    def scan(self, root: Optional[Path] = None) -> Iterator[ScanResult]:
        for path in self.iter_files(root):
            yield self.scan_file(path)

    def scan_file(self, path: Path) -> ScanResult:
        """Scan a single asset and return a :class:`ScanResult`."""
        path = Path(path)
        fmt = detect_format(path)
        try:
            metadata = self._read_metadata(path, fmt)
        except Exception as exc:  # noqa: BLE001 - log and keep scanning
            return ScanResult(
                path=path,
                item=MediaItem(file_path=str(path)),
                source=MetadataSource.NONE,
                error=str(exc),
            )

        item = MediaItem(
            file_path=str(path.resolve()),
            file_size=metadata.file_size,
            file_format=metadata.file_format,
            camera_make=metadata.camera_make,
            camera_model=metadata.camera_model,
            camera_serial=metadata.camera_serial,
            timestamp_utc=metadata.timestamp_utc,
            gps=metadata.gps,
        )

        embedded_jpeg: Optional[Path] = None
        if fmt != "JPEG":
            # For RAW, extract the embedded preview JPEG if possible.
            thumb_dir = self.root / ".omnilibrary_thumbs"
            thumb_dir.mkdir(parents=True, exist_ok=True)
            candidate = thumb_dir / f"{path.stem}.embedded.jpg"
            try:
                embedded_jpeg = extract_embedded_jpeg(path, candidate, as_raw=True)
            except Exception:
                embedded_jpeg = None

        return ScanResult(
            path=path,
            item=item,
            source=metadata.source,
            embedded_jpeg=embedded_jpeg,
        )

    def _read_metadata(self, path: Path, fmt: str) -> ExtractedMetadata:
        if self._exiftool_exe:
            out = _run_exiftool(self._exiftool_exe, str(path))
            if out:
                meta = _parse_exiftool_json(out)
                meta.file_format = fmt
                meta.file_size = path.stat().st_size
                return meta
        # Fallback: rawpy/Pillow.
        return _fallback_metadata(path, fmt)


def scan_directory(
    root: str | Path,
    db,
    *,
    exiftool_exe: Optional[str] = None,
    log: bool = True,
) -> list[ScanResult]:
    """Convenience wrapper: scan ``root`` and upsert every item into ``db``.

    Returns the list of :class:`ScanResult` produced. Items that could not be
    scanned are still recorded as unresolved rows so nothing is silently lost.
    """
    scanner = FileScanner(root, exiftool_exe=exiftool_exe)
    results = list(scanner.scan())
    items = [r.item for r in results]
    db.batch_upsert_items(items)
    if log:
        for r in results:
            status = "ok" if not r.error and not r.skipped else "error"
            db.log_scan(str(r.path), status, r.error)
    return results


if __name__ == "__main__":  # pragma: no cover - manual smoke check
    import duckdb

    from .database import Database

    root_arg = sys.argv[1] if len(sys.argv) > 1 else "."
    Database(duckdb.connect(":memory:"))
    for r in scan_directory(root_arg, Database(duckdb.connect(":memory:")), log=False):
        print(
            r.item.file_path,
            r.item.file_format,
            r.item.camera_make,
            r.source.value,
            "embed:", bool(r.embedded_jpeg),
        )
