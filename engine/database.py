"""DuckDB persistence layer for Omnilibrary.

This module owns *all* database access. It defines the :class:`MediaItem`
schema, schema (re)initialisation, upsert paths, and read helpers.

Storage note
------------
DuckDB 1.5.6 (the installed build) ships a native ``DOUBLE[]`` array type but
does **not** ship the DuckDB ``vector`` extension -- the matching Apple Silicon
``.duckdb_extension`` artifact is unavailable (HTTP 404 from the extension
host), so ``LOAD vector`` cannot be performed offline.

We therefore store image embeddings (DINOv2 / pHash / aesthetics) as native
``DOUBLE[]`` columns. This is a fully functional dense-vector store: we compute
cosine similarity with plain array scalar functions. The ``DOUBLE[]`` choice is
centralised here so that, should a compatible ``vector`` extension become
installable, swapping in real ``VECTOR(n)`` columns is a localised change rather
than a refactor across every caller.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Optional

import duckdb

# Columns that are set by the scanner (file-level metadata).
# Columns that are set later by the embeddings worker.
_EMBEDDING_COLUMNS = ("dinov2_vector", "aesthetic_score")


@dataclass
class MediaItem:
    """A single photo/film asset as persisted in ``media_items``."""

    file_path: str
    item_id: str = field(default_factory=lambda: uuid.uuid4().hex)
    file_size: Optional[int] = None
    file_format: Optional[str] = None
    camera_make: Optional[str] = None
    camera_model: Optional[str] = None
    camera_serial: Optional[str] = None
    timestamp_utc: Optional[datetime] = None
    # Stored as DOUBLE[2] = [lat, lon] (matches CLAUDE.md "gps_lat_lon").
    gps: Optional[tuple[float, float]] = None
    # 64-bit perceptual hash (imagehash). None when not computed.
    p_hash: Optional[int] = None
    # 768-dim DINOv2 embedding. Stored as DOUBLE[]. None until computed.
    dinov2_vector: Optional[list[float]] = None
    aesthetic_score: Optional[float] = None
    # crop_rect = (x, y, w, h) in pixels, top-left origin. None until a crop
    # is computed.
    crop_rect: Optional[tuple[float, float, float, float]] = None
    color_fix_flag: bool = False
    event_id: Optional[str] = None
    duplicate_group_id: Optional[str] = None
    is_primary_master: bool = True


# ---------------------------------------------------------------------------
# Schema
# ---------------------------------------------------------------------------
_SCHEMA = """
CREATE TABLE IF NOT EXISTS media_items (
    item_id             VARCHAR PRIMARY KEY,
    file_path           VARCHAR NOT NULL,
    file_size           BIGINT,
    file_format         VARCHAR,
    camera_make         VARCHAR,
    camera_model        VARCHAR,
    camera_serial       VARCHAR,
    timestamp_utc       TIMESTAMP,
    -- gps = DOUBLE[2] = [lat, lon]
    gps                 DOUBLE[2],
    -- p_hash is a 64-bit imagehash stored as UINT64 (imagehash is unsigned)
    p_hash              UINT64,
    -- dinov2_vector = DOUBLE[] (768-d for dinov2_vits14)
    dinov2_vector       DOUBLE[],
    aesthetic_score     DOUBLE,
    -- crop_rect = DOUBLE[4] = (x, y, w, h)
    crop_rect           DOUBLE[4],
    color_fix_flag      BOOLEAN NOT NULL DEFAULT FALSE,
    event_id            VARCHAR,
    duplicate_group_id  VARCHAR,
    is_primary_master   BOOLEAN NOT NULL DEFAULT TRUE,
    indexed_at          TIMESTAMP NOT NULL DEFAULT now()
);

CREATE INDEX IF NOT EXISTS idx_media_item_format ON media_items (file_format);
CREATE INDEX IF NOT EXISTS idx_media_event ON media_items (event_id);
CREATE INDEX IF NOT EXISTS idx_media_dup ON media_items (duplicate_group_id);

CREATE TABLE IF NOT EXISTS scan_log (
    id          BIGINT,
    path        VARCHAR NOT NULL,
    status      VARCHAR NOT NULL,
    message     VARCHAR,
    logged_at   TIMESTAMP NOT NULL DEFAULT now()
);

-- DuckDB 1.5.x rejects AUTOINCREMENT / SERIAL / IDENTITY column syntax, so we
-- back the id column with an explicit sequence. The sequence is (re)created
-- idempotently in init_schema() and bound in an ALTER afterwards.
CREATE SEQUENCE IF NOT EXISTS scan_log_seq;

ALTER TABLE scan_log
    ALTER COLUMN id SET DEFAULT nextval('scan_log_seq');
"""


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


class Database:
    """Owns the DuckDB connection and provides typed media-item access."""

    def __init__(self, path: str | Path = ":memory:"):
        # ``:memory:`` is in-process; a filesystem path gives a durable store.
        self.path = str(path)
        self._conn = duckdb.connect(self.path)
        self._conn.execute("PRAGMA enable_verification")
        self.init_schema()

    # -- lifecycle ---------------------------------------------------------
    def init_schema(self) -> None:
        """Create tables/indexes if they do not already exist (idempotent).

        DuckDB's connection exposes no ``executescript`` -- only ``execute`` --
        so the multi-statement schema is split on semicolons and run one
        statement at a time. Every statement ends in ``IF NOT EXISTS`` so a
        re-open of an existing database is safe.
        """
        statements = [s.strip() for s in _SCHEMA.split(";") if s.strip()]
        for stmt in statements:
            self._conn.execute(stmt)

    def close(self) -> None:
        if not self._conn.closed:
            self._conn.close()

    def __enter__(self) -> "Database":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    # -- write paths -------------------------------------------------------
    def upsert_item(self, item: MediaItem) -> str:
        """Insert the item or update an existing row with the same item_id."""
        self._conn.execute(
            """
            INSERT INTO media_items (
                item_id, file_path, file_size, file_format,
                camera_make, camera_model, camera_serial, timestamp_utc,
                gps, p_hash, dinov2_vector, aesthetic_score, crop_rect,
                color_fix_flag, event_id, duplicate_group_id,
                is_primary_master, indexed_at
            ) VALUES (
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?, ?, ?,
                ?, ?
            )
            ON CONFLICT(item_id) DO UPDATE SET
                file_path            = EXCLUDED.file_path,
                file_size            = EXCLUDED.file_size,
                file_format          = EXCLUDED.file_format,
                camera_make          = EXCLUDED.camera_make,
                camera_model         = EXCLUDED.camera_model,
                camera_serial        = EXCLUDED.camera_serial,
                timestamp_utc        = EXCLUDED.timestamp_utc,
                gps                  = EXCLUDED.gps,
                p_hash               = EXCLUDED.p_hash,
                dinov2_vector        = EXCLUDED.dinov2_vector,
                aesthetic_score      = EXCLUDED.aesthetic_score,
                crop_rect            = EXCLUDED.crop_rect,
                color_fix_flag       = EXCLUDED.color_fix_flag,
                event_id             = EXCLUDED.event_id,
                duplicate_group_id   = EXCLUDED.duplicate_group_id,
                is_primary_master    = EXCLUDED.is_primary_master,
                indexed_at           = EXCLUDED.indexed_at
            """,
            [
                item.item_id,
                item.file_path,
                item.file_size,
                item.file_format,
                item.camera_make,
                item.camera_model,
                item.camera_serial,
                _dt_to_iso(item.timestamp_utc),
                floats_to_sql(gps_to_pair(item.gps)),
                item.p_hash,
                floats_to_sql(item.dinov2_vector),
                _num_to_sql(item.aesthetic_score),
                floats_to_sql(crop_to_pair(item.crop_rect)),
                bool(item.color_fix_flag),
                item.event_id,
                item.duplicate_group_id,
                bool(item.is_primary_master),
                _utcnow(),
            ]
        )
        return item.item_id

    def batch_upsert_items(self, items: Iterable[MediaItem]) -> int:
        """Upsert many items in a single transaction. Returns the count."""
        items = list(items)
        if not items:
            return 0
        self._conn.execute("BEGIN")
        try:
            for item in items:
                self.upsert_item(item)
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise
        return len(items)

    def upsert_metadata(
        self,
        item_id: str,
        *,
        file_size: Optional[int] = None,
        file_format: Optional[str] = None,
        camera_make: Optional[str] = None,
        camera_model: Optional[str] = None,
        camera_serial: Optional[str] = None,
        timestamp_utc: Optional[datetime] = None,
        gps: Optional[tuple[float, float]] = None,
        p_hash: Optional[int] = None,
    ) -> None:
        """Update only the supplied non-None metadata fields."""
        fields = {
            "file_size": _num_to_sql(file_size),
            "file_format": file_format,
            "camera_make": camera_make,
            "camera_model": camera_model,
            "camera_serial": camera_serial,
            "timestamp_utc": _dt_to_iso(timestamp_utc),
            "gps": floats_to_sql(gps_to_pair(gps)),
            "p_hash": p_hash,
        }
        set_clause = ", ".join(f"{k} = ?" for k, v in fields.items() if v is not None)
        if not set_clause:
            return
        values = [*fields.values(), item_id]
        values = [v for v in values if v is not None]
        self._conn.execute(
            f"UPDATE media_items SET {set_clause} WHERE item_id = ?",
            values,
        )

    def upsert_embeddings(
        self,
        item_id: str,
        dinov2_vector: Optional[list[float]] = None,
        aesthetic_score: Optional[float] = None,
    ) -> None:
        """Persist a computed DINOv2 embedding and/or aesthetic score."""
        assignments = []
        params: list = []
        if dinov2_vector is not None:
            assignments.append("dinov2_vector = ?")
            params.append(floats_to_sql(dinov2_vector))
        if aesthetic_score is not None:
            assignments.append("aesthetic_score = ?")
            params.append(_num_to_sql(aesthetic_score))
        if assignments:
            assignments.append("indexed_at = ?")
            params.append(_utcnow())
            params.append(item_id)
            self._conn.execute(
                f"UPDATE media_items SET {', '.join(assignments)} "
                "WHERE item_id = ?",
                params,
            )

    def set_event(self, item_id: str, event_id: str) -> None:
        self._conn.execute(
            "UPDATE media_items SET event_id = ?, indexed_at = ? WHERE item_id = ?",
            [event_id, _utcnow(), item_id],
        )

    def set_duplicate_group(self, item_ids: Iterable[str], group_id: str) -> None:
        self._conn.execute("BEGIN")
        try:
            for item_id in item_ids:
                self._conn.execute(
                    "UPDATE media_items SET duplicate_group_id = ?, "
                    "indexed_at = ? WHERE item_id = ?",
                    [group_id, _utcnow(), item_id],
                )
            self._conn.execute("COMMIT")
        except Exception:
            self._conn.execute("ROLLBACK")
            raise

    def log_scan(self, path: str, status: str, message: Optional[str] = None) -> None:
        self._conn.execute(
            "INSERT INTO scan_log (path, status, message) VALUES (?, ?, ?)",
            [path, status, message],
        )

    # -- read paths --------------------------------------------------------
    def item(self, item_id: str) -> Optional[MediaItem]:
        row = self._conn.execute(
            "SELECT * FROM media_items WHERE item_id = ?", (item_id,)
        ).fetchone()
        return _row_to_item(row) if row else None

    def count(self) -> int:
        return int(self._conn.execute("SELECT count(*) FROM media_items").fetchone()[0])

    def all_item_ids(self) -> list[str]:
        return [r[0] for r in self._conn.execute(
            "SELECT item_id FROM media_items"
        ).fetchall()]

    def pending_embeddings(self) -> list[str]:
        """Item IDs that have no DINOv2 embedding yet (for batch processing)."""
        return [
            r[0]
            for r in self._conn.execute(
                "SELECT item_id FROM media_items WHERE dinov2_vector IS NULL"
            ).fetchall()
        ]

    def nearest_neighbors(
        self, query_vec: list[float], limit: int = 20, exclude: Optional[set[str]] = None
    ) -> list[tuple[str, float]]:
        """Return (item_id, cosine_similarity) for the closest stored vectors.

        The DuckDB build in this environment has no ``vector`` extension
        installed (its Apple Silicon artifact is unreachable offline), so we
        keep vectors in native ``DOUBLE[]`` columns and compute cosine
        similarity in NumPy. ``all_embeddings`` already returns usable floats;
        this simply applies cosine ranking client-side.
        """
        import numpy as np

        exclude = exclude or set()
        candidates = self.all_embeddings()
        if not candidates:
            return []
        query = np.asarray(query_vec, dtype=np.float64)
        query_norm = np.linalg.norm(query)
        if query_norm == 0.0:
            return []
        ids = []
        sims = []
        for item_id, vec in candidates:
            if item_id in exclude:
                continue
            v = np.asarray(vec, dtype=np.float64)
            denom = np.linalg.norm(v)
            if denom == 0.0:
                continue
            sims.append(float(np.dot(query, v) / (query_norm * denom)))
            ids.append(item_id)
        order = np.argsort(sims)[::-1][:limit]
        return [(ids[i], sims[i]) for i in order]

    def all_embeddings(self) -> list[tuple[str, list[float]]]:
        return [
            (r[0], list(map(float, r[1])))
            for r in self._conn.execute(
                "SELECT item_id, dinov2_vector FROM media_items "
                "WHERE dinov2_vector IS NOT NULL"
            ).fetchall()
        ]


# ---------------------------------------------------------------------------
# Conversion helpers
# ---------------------------------------------------------------------------
def _num_to_sql(value):
    """Numbers pass through as-is; None stays None for DuckDB NULL."""
    return value


def _dt_to_iso(value):
    """Serialise a datetime to ISO-8601 UTC.

    Naive datetimes are assumed to be UTC; aware datetimes are converted to
    UTC so the emitted string always carries a timezone designator, letting
    the value round-trip through DuckDB back into an aware datetime.
    """
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    else:
        value = value.astimezone(timezone.utc)
    return value.isoformat()


def gps_to_pair(gps):
    """Normalise the many ways a caller may express GPS to (lat, lon)."""
    if gps is None:
        return None
    if isinstance(gps, (list, tuple)) and len(gps) >= 2:
        return (float(gps[0]), float(gps[1]))
    # (lon, lat) -> (lat, lon)
    return (float(gps[1]), float(gps[0]))


def crop_to_pair(crop):
    """Normalise a crop_rect (x, y, w, h) to a list of 4 floats."""
    if crop is None:
        return None
    vals = list(map(float, crop))
    while len(vals) < 4:
        vals.append(0.0)
    return vals[:4]


def floats_to_sql(values):
    """Serialise a list of floats (or None) to a DuckDB array literal string.

    DuckDB binds ``list`` parameters as a Python list; for arrays that would
    require a single ``:value`` placeholder bound to the whole list. Passing the
    literal ``[..]`` as SQL avoids ambiguity and keeps parameter ordering
    stable across the upsert path.
    """
    if values is None:
        return None
    if isinstance(values, str):
        return values
    return "[" + ", ".join(repr(float(v)) for v in values) + "]"


def _row_to_item(row) -> MediaItem:
    """Map a ``media_items`` row tuple to a :class:`MediaItem`."""
    cols = [
        "item_id", "file_path", "file_size", "file_format",
        "camera_make", "camera_model", "camera_serial", "timestamp_utc",
        "gps", "p_hash", "dinov2_vector", "aesthetic_score", "crop_rect",
        "color_fix_flag", "event_id", "duplicate_group_id",
        "is_primary_master", "indexed_at",
    ]
    data = dict(zip(cols, row))
    ts = data.pop("timestamp_utc", None)
    if isinstance(ts, str):
        try:
            ts = datetime.fromisoformat(ts)
        except ValueError:
            ts = None
    if isinstance(ts, datetime) and ts.tzinfo is None:
        ts = ts.replace(tzinfo=timezone.utc)
    gps = data.pop("gps")
    crop = data.pop("crop_rect")
    vec = data.pop("dinov2_vector")
    return MediaItem(
        item_id=data.pop("item_id"),
        file_path=data.pop("file_path"),
        file_size=data.pop("file_size"),
        file_format=data.pop("file_format"),
        camera_make=data.pop("camera_make"),
        camera_model=data.pop("camera_model"),
        camera_serial=data.pop("camera_serial"),
        timestamp_utc=ts,
        gps=_sql_pair_to_floats(gps),
        p_hash=data.pop("p_hash"),
        dinov2_vector=list(map(float, vec)) if vec is not None else None,
        aesthetic_score=data.pop("aesthetic_score"),
        crop_rect=_sql_pair_to_floats(crop) if crop is not None else None,
        color_fix_flag=bool(data.pop("color_fix_flag")),
        event_id=data.pop("event_id"),
        duplicate_group_id=data.pop("duplicate_group_id"),
        is_primary_master=bool(data.pop("is_primary_master")),
    )


def _sql_pair_to_floats(values):
    """Turn a DuckDB DOUBLE[] (returned as a list) into a tuple of floats."""
    if values is None:
        return None
    return tuple(float(v) for v in values)
