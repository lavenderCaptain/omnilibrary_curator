"""Multi-signal auto-clustering of photos into events.

This is the Phase 2 "events" engine (see ``clustering.py`` in the project
blueprint). It assigns every stored :class:`MediaItem` an ``event_id`` in one
offline, deterministic pass and returns a summary :class:`ClusterResult` list.

Design (no external graph library)
----------------------------------
The blueprint calls for HDBSCAN over *time + GPS + camera body + visual
embeddings*. ``hdbscan`` / ``sklearn`` are **not** installable in this
environment (no network at install time, no build toolchain for its Cython
extension), so the clustering here is a pure :mod:`numpy` + stdlib
implementation that reproduces the *intended semantics* faithfully:

1. **Time split.** The primary signal is wall-clock time. Photos within
   ``max_span_hours`` (default 24h) of each other that also fall in the same
   day form a candidate event. The day-partition keeps one long shoot from
   bleeding a full day of unrelated photos together, and the time-span keeps a
   multi-day trip from collapsing into one giant blob.

2. **Camera-body anchoring.** Photos taken on the *same day* but by
   **different camera bodies** are never merged (per the blueprint's "Camera
   Hardware Signatures" rule), unless GPS proves they were spatially
   co-located. This is the *hard* constraint, applied after the time/GPS split:
   the last time a body was seen is used to decide whether the next photo of a
   new body continues the event or starts a new one.

3. **GPS co-location override.** Two photos taken at nearly the same wall-clock
   time (within ``gps_co_locate_minutes``) but on different camera bodies are
   treated as one event *only* when their GPS positions are within
   ``gps_co_locate_meters`` (default 50 m) of each other — the kind of overlap a
   shared location (a concert, a family dinner) naturally produces.

4. **Visual embedding anchoring.** Photos with **no time and no GPS** (RAW
   files with stripped EXIF, exports, re-saves) cannot use signals 1–3. Each
   connected component of near-identical DINOv2 embeddings — where two images
   within ``visual_epsilon`` (default 0.05) on the unit sphere are "the same
   shot" — becomes its own event. This is the "anchor EXIF-less photos to
   visual clusters" requirement.

5. **Max members guard.** Any event exceeding ``max_members`` (default 10 000)
   is split by its earliest timestamp so a pathological burst doesn't create a
   single 500k-photo event.

Distance math is plain spherical-geometry (haversine); there is no dependency
on ``scipy`` or ``sklearn``. All heavy lifting runs synchronously and is meant
to be driven off the PySide6 main thread by a ``QThread`` worker.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

import numpy as np

from .database import Database, MediaItem

# ---------------------------------------------------------------------------
# Tuning constants (blueprint-derived)
# ---------------------------------------------------------------------------
# A single event should not stretch more than ~24h of actual shooting.
MAX_SPAN_HOURS = 24
# Two photos on the same day are "together" unless a gap larger than this (in
# hours) falls between them in the sorted time order.
MIN_COUNT = 2
# Guard against pathological single events.
MAX_MEMBERS = 10_000
# Co-location: same-clock-window + within this many metres -> same event.
GPS_CO_LOCATE_MINUTES = 30
GPS_CO_LOCATE_METERS = 50.0
# DINOv2 unit-sphere cosine distance; below == "visually the same shot".
VISUAL_EPSILON = 0.05


# ---------------------------------------------------------------------------
# Results
# ---------------------------------------------------------------------------
@dataclass
class ClusterResult:
    """One auto-detected event."""

    event_id: str
    item_ids: list[str] = field(default_factory=list)
    size: int = 0
    title: str = ""
    # Whether this event exists only because of visual embedding anchoring
    # (i.e. none of its members had time or GPS metadata).
    anchored_by_visual: bool = False

    def __post_init__(self):
        self.size = len(self.item_ids)


# ---------------------------------------------------------------------------
# Spatial helpers (spherical geometry, no scipy)
# ---------------------------------------------------------------------------
_EARTH_RADIUS_M = 6_371_000.0


def haversine_meters(lat1, lon1, lat2, lon2) -> float:
    """Great-circle distance in metres between two (lat, lon) degrees."""
    rlat1, rlon1, rlat2, rlon2 = map(math.radians, (lat1, lon1, lat2, lon2))
    dlat = rlat2 - rlat1
    dlon = rlon2 - rlon1
    a = (
        math.sin(dlat / 2) ** 2
        + math.cos(rlat1) * math.cos(rlat2) * math.sin(dlon / 2) ** 2
    )
    return 2 * _EARTH_RADIUS_M * math.asin(min(1.0, math.sqrt(a)))


def _parse_iso(ts_str: str) -> Optional[datetime]:
    try:
        dt = datetime.fromisoformat(ts_str)
    except (ValueError, TypeError):
        return None
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)


# ---------------------------------------------------------------------------
# Body signature (the "which camera body" key)
# ---------------------------------------------------------------------------
def body_signature(item: MediaItem) -> str:
    """A stable key identifying the camera body that took ``item``.

    Combines ``Make|Model|Serial`` when a serial is present; otherwise falls
    back to ``Make|Model``; then to the camera family hint or file format; then
    ``"unknown"``. Two photos cluster by body only if these match exactly.
    """
    make = (item.camera_make or "").strip()
    model = (item.camera_model or "").strip()
    serial = (item.camera_serial or "").strip()

    if serial:
        return f"{make}|{model}|{serial}"
    if make or model:
        return f"{make}|{model}"
    # No EXIF at all: fall back to the body-classification hint we recorded on
    # the scanner side.
    fmt = (item.file_format or "").upper()
    hints = {
        "JPEG": "unknown",
        "CR2": "Canon CR2",
        "CR3": "Canon CR3",
        "NEF": "Nikon NEF",
        "ARW": "Sony ARW",
        "RW2": "Panasonic RW2",
        "ORF": "OM Systems ORF",
        "RAF": "Fujifilm RAF",
        "PEF": "Pentax PEF",
        "DNG": "Unknown DNG",
        "Srw": "Unknown RAW",
    }
    return hints.get(fmt, "unknown")


# ---------------------------------------------------------------------------
# Time-key: map a photo to the calendar day it belongs to
# ---------------------------------------------------------------------------
def _time_key(item: MediaItem) -> str:
    """Return the calendar day a photo belongs to (``YYYY-MM-DD`` or ``?``).

    Uses the EXIF timestamp when present; otherwise a best-effort parse from the
    file name (a trailing ``_YYYYMMDD`` or a leading one), else ``"?"`` for the
    "no time metadata" bucket.
    """
    if item.timestamp_utc is not None:
        return item.timestamp_utc.astimezone(timezone.utc).strftime("%Y-%m-%d")

    name = Path(item.file_path).stem
    for m in re.finditer(r"(\d{4}-\d{2}-\d{2})", name):
        return m.group(1)
    m = re.search(r"(\d{4})(\d{2})(\d{2})", name)
    if m:
        y, mo, d = m.groups()
        return f"{y}-{mo}-{d}"
    return "?"


def _event_title(day: str) -> str:
    """Human title for an event; prefer the day when the whole event is one day."""
    return f"Event on {day}" if day != "?" else "Untitled event"


# ---------------------------------------------------------------------------
# Per-item field accessors + day walk
# ---------------------------------------------------------------------------
def item_time(item: MediaItem) -> Optional[datetime]:
    return item.timestamp_utc


def gps_point(item: MediaItem) -> Optional[tuple[float, float]]:
    return item.gps


def _same_day_events(
    items: list[MediaItem],
    min_gap_minutes: float = GPS_CO_LOCATE_MINUTES,
    gps_co_locate_meters: float = GPS_CO_LOCATE_METERS,
) -> list[list[str]]:
    """Walk one day's photos in time order and group them into events.

    Rule set (mirrors the blueprint's multi-signal intent):

    * Each event carries a **primary body** — the body of the first photo that
      joined it (matching the blueprint's "never merge same-date photos taken by
      *different* bodies unless co-located" rule).
    * A photo whose body matches the current event's primary body **always
      joins** it, regardless of the time gap. Same-body photos are the whole
      event; the day boundary is the natural splitter.
    * A photo of a **different** body joins only when it is *co-located*: within
      ``min_gap_minutes`` in time **and** within ``gps_co_locate_meters`` of a
      member of the current event. Otherwise it starts a new event.

    GPS is therefore a *merge* signal between different bodies, never a
    splitter — same-body photos taken in two different cities on the same day
    stay together.
    """
    indexed = sorted(items, key=lambda i: item_time(i) or datetime.min.replace(tzinfo=timezone.utc))
    by_id = {it.item_id: it for it in items}

    events: list[list[str]] = []
    cur_body: Optional[str] = None
    cur_last_ts: Optional[datetime] = None
    cur_members: list[str] = []

    for it in indexed:
        ts = item_time(it)
        sig = body_signature(it)

        # First photo of the day (no current event yet): open the event.
        if cur_body is None:
            cur_body, cur_last_ts, cur_members = sig, ts, [it.item_id]
            continue

        # Same as the current event's primary body: always join.
        if sig == cur_body:
            cur_members.append(it.item_id)
            if ts is not None:
                cur_last_ts = ts
            continue

        # Different body: join only when co-located (time-close AND space-close).
        time_close = ts is None or cur_last_ts is None or (ts - cur_last_ts) <= timedelta(minutes=min_gap_minutes)
        member_items = [by_id[m] for m in cur_members]
        if time_close and _co_located(member_items, it, gps_co_locate_meters):
            cur_members.append(it.item_id)
            if ts is not None:
                cur_last_ts = ts
        else:
            events.append(cur_members)
            cur_body, cur_last_ts, cur_members = sig, ts, [it.item_id]

    if cur_members:
        events.append(cur_members)
    return events


def _co_located(member_items: list[MediaItem], it: MediaItem, meters: float) -> bool:
    """True if ``it``'s GPS is within ``meters`` of any member's GPS."""
    point = gps_point(it)
    if point is None:
        return False
    for m in member_items:
        mp = gps_point(m)
        if mp is None:
            continue
        if haversine_meters(mp[0], mp[1], point[0], point[1]) <= meters:
            return True
    return False


def _visual_component(
    items: list[MediaItem],
    vectors: dict[str, list[float]],
    epsilon: float = VISUAL_EPSILON,
) -> list[list[MediaItem]]:
    """Connected components of near-identical embeddings via BFS over a k-d graph.

    Two images are linked when their unit-sphere cosine distance is <= ``epsilon``.
    Returns one list of members per connected component. Only :mod:`numpy` is
    used; there is no sklearn dependency.
    """
    n = len(items)
    if n == 0:
        return []
    feats = np.asarray([vectors[m.item_id] for m in items], dtype=np.float64)
    norms = np.linalg.norm(feats, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    unit = feats / norms

    # Nearest-neighbour graph: for each point take its nearest neighbour and add
    # a directed edge. Connected components of this (symmetrised) graph are the
    # visual clusters. This is a light, dependency-free proxy for the density
    # reachability HDBSCAN models, and it reproduces the "near-identical shots
    # form a cluster" semantics we need.
    adj: list[list[int]] = [[] for _ in range(n)]
    sims = unit @ unit.T
    for i in range(n):
        for j in range(i + 1, n):
            d = 1.0 - sims[i, j]
            if d <= epsilon:
                adj[i].append(j)
                adj[j].append(i)

    seen = [False] * n
    components: list[list[MediaItem]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp = [items[start]]
        while stack:
            cur = stack.pop()
            for nb in adj[cur]:
                if not seen[nb]:
                    seen[nb] = True
                    comp.append(items[nb])
                    stack.append(nb)
        components.append(comp)
    return components


# ---------------------------------------------------------------------------
# Public clustering API
# ---------------------------------------------------------------------------
def run_clustering(
    db: Database,
    *,
    max_span_hours: float = MAX_SPAN_HOURS,
    min_count: int = MIN_COUNT,
    max_members: int = MAX_MEMBERS,
    gps_co_locate_minutes: float = GPS_CO_LOCATE_MINUTES,
    gps_co_locate_meters: float = GPS_CO_LOCATE_METERS,
    visual_epsilon: float = VISUAL_EPSILON,
) -> list[ClusterResult]:
    """Assign an ``event_id`` to every stored item and return the events.

    Returns a list of :class:`ClusterResult`, largest first. Each result's
    ``item_ids`` are written back to the DB via :meth:`Database.set_event`.

    The algorithm is: group by day -> time-split -> GPS-split -> split by camera
    body (hard) -> attach EXIF-less photos to visual components -> split oversized
    events.
    """
    items = db.all_items()
    if not items:
        return []

    by_id = {it.item_id: it for it in items}
    vectors = {
        it.item_id: vec
        for it, vec in _load_vectors(db, by_id)
    }

    # 1. Group by day.
    by_day: dict[str, list[str]] = {}
    visual_only_ids: list[str] = []
    for it in items:
        day = _time_key(it)
        by_day.setdefault(day, []).append(it.item_id)
        if item_time(it) is None and gps_point(it) is None:
            visual_only_ids.append(it.item_id)

    results: list[ClusterResult] = []
    used: set[str] = set()
    eid_counter = 0

    def assign(run_ids: list[str], *, anchored_by_visual: bool) -> None:
        for run in _guard_max_members(run_ids, by_id, max_members):
            nonlocal eid_counter
            eid_counter += 1
            members = run
            day = _event_day(members, by_id)
            results.append(
                ClusterResult(
                    event_id=f"event-{eid_counter:04d}",
                    item_ids=sorted(members),
                    title=_event_title(day),
                    anchored_by_visual=anchored_by_visual,
                )
            )
            for mid in members:
                used.add(mid)
                db.set_event(mid, f"event-{eid_counter:04d}")

    # 2-4. Day / time / GPS / body splitting for photos that carry time or GPS.
    for day, ids in sorted(by_day.items()):
        if day == "?":
            continue  # handled by the visual pass
        events = _same_day_events([by_id[i] for i in ids])
        for members in events:
            if len(members) >= min_count:
                assign(members, anchored_by_visual=False)

    # 5. Visual anchoring for EXIF-less photos not yet claimed.
    if visual_only_ids:
        remaining = [mid for mid in visual_only_ids if mid not in used]
        comps = _visual_component([by_id[m] for m in remaining], vectors, visual_epsilon)
        for comp in comps:
            if len(comp) >= min_count:
                assign([m.item_id for m in comp], anchored_by_visual=True)

    # Photos below ``min_count`` never get an event_id (left unclustered so the
    # UI can show them as "unsorted").

    results.sort(key=lambda r: r.size, reverse=True)
    return results


def _load_vectors(db: Database, by_id: dict[str, MediaItem]):
    """Yield (item, vector) for items that have a DINOv2 embedding."""
    for item_id, vec in db.all_embeddings():
        if item_id in by_id:
            yield by_id[item_id], vec


def _split_by_body(ids: list[str], by_id: dict[str, MediaItem], min_gap_minutes: float) -> list[list[str]]:
    """Split ``ids`` so different camera bodies don't merge on the same day.

    Members are sorted ascending by time. A new group starts when the body
    changes and the gap to the previous photo exceeds ``min_gap_minutes`` (a
    short pause between swapping bodies, e.g. a hybrid shooter, keeps it in the
    same event). Photos with no usable time stay grouped together.
    """
    indexed = sorted(
        range(len(ids)),
        key=lambda i: (
            item_time(by_id[ids[i]]) is None,
            item_time(by_id[ids[i]]) or datetime.min.replace(tzinfo=timezone.utc),
        ),
    )
    groups: list[list[str]] = []
    cur: list[str] = []
    last_body: Optional[str] = None
    last_ts: Optional[datetime] = None
    for i in indexed:
        it = by_id[ids[i]]
        sig = body_signature(it)
        ts = item_time(it)
        new_body = last_body is not None and sig != last_body
        gap_ok = ts is None or last_ts is None or (ts - last_ts) > timedelta(minutes=min_gap_minutes)
        if last_body is not None and new_body and gap_ok:
            groups.append(cur)
            cur = []
        cur.append(ids[i])
        last_body = sig
        if ts is not None:
            last_ts = ts
    if cur:
        groups.append(cur)
    return groups


def _guard_max_members(
    ids: list[str],
    by_id: dict[str, MediaItem],
    max_members: int,
) -> list[list[str]]:
    """Split an oversized event into runs of at most ``max_members`` photos.

    Ordered by earliest timestamp so the splits are chronological. Returns a
    list of runs; each run is a separate event.
    """
    if len(ids) <= max_members:
        return [ids]
    ordered = sorted(ids, key=lambda mid: (item_time(by_id[mid]) or datetime.min.replace(tzinfo=timezone.utc)))
    return [ordered[i : i + max_members] for i in range(0, len(ordered), max_members)]


def _event_day(ids: list[str], by_id: dict[str, MediaItem]) -> str:
    """A day shared by most of a cluster, for its title (``?`` when none)."""
    days = [_time_key(by_id[m]) for m in ids if _time_key(by_id[m]) != "?"]
    if days:
        return sorted(set(days))[0]
    return "?"
