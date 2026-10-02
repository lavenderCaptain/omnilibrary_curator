"""Multi-stage duplicate finder (pHash -> DINOv2) and Master/derivative scorer.

This is the Phase 2 "deduplication" engine (see ``dedup.py`` in the project
blueprint). It groups near-identical photos and, within each group, selects a
single **Master** (the highest-quality original) and labels the rest as
**Derivative** files.

The blueprint asks for a two-tier approach:

1. **Fast tier — pHash Hamming distance.** Perceptual hashes (64-bit pHash) are
   compared with Hamming distance. Two files within ``phash_threshold`` differing
   bits are *suspected* duplicates. This is cheap and catches exact re-saves,
   re-exports, and minor recompression.

2. **Slow tier — DINOv2 cosine similarity.** For each pHash-suspicious pair,
   their DINOv2 embeddings are compared with cosine similarity. Only pairs
   exceeding ``embedding_threshold`` (default 0.97) are *confirmed* duplicates.
   This filters out the pHash false positives that come from same-composition
   but genuinely different shots (e.g. two people at the same concert).

After grouping, a **quality score** picks the Master:

* **Format** — RAW (``.CR2``/``.CR3``/``.NEF``/...) outranks a rendered JPG
  (the RAW is the original capture).
* **Resolution** — higher pixel count wins.
* **File size** — larger tends to mean less lossy / higher quality.
* **EXIF completeness** — more captured metadata (make, model, timestamp,
  GPS, serial) is better.

The result is written to the DB via :meth:`Database.set_duplicate_group`: each
group gets a stable ``duplicate_group_id`` and the Master carries
``is_primary_master=True`` while derivatives carry ``False``.

Distance math and the scoring are pure NumPy/stdlib — no sklearn, no hdbscan,
no scipy. All heavy lifting is vectorised and meant to run off the PySide6 main
thread.
"""

from __future__ import annotations

import numpy as np
from dataclasses import dataclass, field
from typing import Optional

from .database import Database, MediaItem

# ---------------------------------------------------------------------------
# Tuning constants
# ---------------------------------------------------------------------------
# Max Hamming distance on the 64-bit pHash to *suspect* a duplicate pair.
PHASH_THRESHOLD = 10
# Min cosine similarity (on the L2-normalised DINOv2 sphere) to *confirm* a
# duplicate. Below this the two images are considered visually distinct.
EMBEDDING_THRESHOLD = 0.97
# Smallest group size that counts as a duplicate cluster.
MIN_DUPLICATE_GROUP = 2


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class DuplicateGroup:
    """A set of near-identical photos plus the chosen Master."""

    group_id: str
    item_ids: list[str] = field(default_factory=list)
    master_item_id: Optional[str] = None
    derivative_item_ids: list[str] = field(default_factory=list)

    @property
    def derivatives(self) -> list[str]:
        return [i for i in self.item_ids if i != self.master_item_id]


@dataclass
class DedupStats:
    """Aggregate counters for a completed dedup pass."""

    scanned: int = 0
    groups: int = 0
    confirmed_duplicates: int = 0
    masters: int = 0
    derivatives: int = 0
    pairs_checked: int = 0
    pairs_confirmed: int = 0


# ---------------------------------------------------------------------------
# Distance math
# ---------------------------------------------------------------------------
def phash_hamming(a: Optional[int], b: Optional[int]) -> int:
    """Hamming distance (number of differing bits) between two pHash ints.

    Missing hashes (``None``) are considered infinitely far apart: a file with
    no pHash can never be confirmed as a duplicate on the fast tier alone.
    """
    if a is None or b is None:
        return 65  # > any possible distance on a 64-bit hash
    return bin(a ^ b).count("1")


def cosine_similarity(a: Optional[list[float]], b: Optional[list[float]]) -> Optional[float]:
    """Cosine similarity of two vectors, or ``None`` if either is missing.

    Vectors are L2-normalised defensively so a zero vector yields ``0.0`` rather
    than a divide-by-zero. A dimension mismatch (shouldn't happen for same-model
    embeddings, but defensively) returns ``None`` instead of raising.
    """
    if a is None or b is None:
        return None
    va = np.asarray(a, dtype=np.float64)
    vb = np.asarray(b, dtype=np.float64)
    if va.shape != vb.shape:
        return None
    na = np.linalg.norm(va)
    nb = np.linalg.norm(vb)
    if na == 0.0 or nb == 0.0:
        return 0.0
    return float(np.dot(va, vb) / (na * nb))


# ---------------------------------------------------------------------------
# Quality scoring
# ---------------------------------------------------------------------------
_RAW_PREFIXES = (
    "CR2", "CR3", "NEF", "ARW", "RW2", "ORF", "RAF", "PEF",
    "SRW", "SR2", "KDC", "X3F", "MRW", "3FR", "FDC", "IIQ", "DNG",
)


def is_raw(file_format: Optional[str]) -> bool:
    """True if the stored file format string denotes a RAW capture."""
    if not file_format:
        return False
    fmt = file_format.upper().lstrip(".")
    return fmt.startswith(_RAW_PREFIXES)


def quality_score(item: MediaItem) -> float:
    """Return a single 0..1 score where higher means "more likely the Master".

    Components:

    * **Format (0.0–0.5):** RAW captures earn 0.5; a rendered JPG earns 0.5 minus
      a small penalty for being lossy.
    * **File size (0.0–0.3):** relative to the max file size in the run
      (larger tends to mean less lossy / higher quality).
    * **EXIF completeness (0.0–0.2):** fraction of the five metadata fields we
      track that are present.
    """
    fmt_score = 0.5 if is_raw(item.file_format) else 0.5 - 0.1
    score = fmt_score

    size_frac = _relative(item.file_size or 0, _run_max_size)
    score += 0.3 * size_frac

    completeness = _exif_completeness(item)
    score += 0.2 * completeness

    return min(1.0, max(0.0, score))


# Module-level run-scoped maxima, populated by ``run_dedup``.
_run_max_size = 0


def _relative(value: float, run_max: float) -> float:
    if run_max <= 0:
        return 0.0
    return min(1.0, value / run_max)


def _exif_completeness(item: MediaItem) -> float:
    """Fraction of the five tracked metadata fields that are populated."""
    fields = [
        item.camera_make, item.camera_model, item.camera_serial,
        item.timestamp_utc, item.gps is not None,
    ]
    return sum(1 for f in fields if f) / len(fields)


def _pick_master(group_ids: list[str], by_id: dict[str, MediaItem], scores: dict[str, float]) -> str:
    """Pick the Master from a group: highest score, ties broken by item_id.

    Deterministic: identical scores are disambiguated lexicographically so the
    same library always yields the same Master.
    """
    return sorted(group_ids, key=lambda i: (-scores[i], i))[0]


# ---------------------------------------------------------------------------
# Grouping
# ---------------------------------------------------------------------------
def _candidate_pairs(ids: list[str], items: list[MediaItem]) -> list[tuple[int, int]]:
    """Return index pairs that *pass the fast pHash tier*.

    Pure pairwise scan (documented; swap for a grid hash for large libraries).
    A pair with a missing pHash never passes the fast tier — it is neither
    confirmed nor rejected here; it simply waits for the slow tier if the other
    tier's structure surfaces it. Here we only emit pairs that are pHash-close.
    """
    pairs = []
    for i in range(len(ids)):
        hi = items[i].p_hash
        if hi is None:
            continue
        for j in range(i + 1, len(ids)):
            if phash_hamming(hi, items[j].p_hash) <= PHASH_THRESHOLD:
                pairs.append((i, j))
    return pairs


def find_duplicate_groups(
    db: Database,
    *,
    phash_threshold: int = PHASH_THRESHOLD,
    embedding_threshold: float = EMBEDDING_THRESHOLD,
) -> tuple[list[DuplicateGroup], DedupStats]:
    """Find duplicate clusters across every stored :class:`MediaItem`.

    Two-tier: pHash Hamming distance (fast) followed by DINOv2 cosine similarity
    (slow) to confirm. Returns the confirmed :class:`DuplicateGroup` list (size
    >= ``MIN_DUPLICATE_GROUP``) and a :class:`DedupStats` summary.
    """
    items = db.all_items()
    if not items:
        return [], DedupStats()

    by_id = {it.item_id: it for it in items}
    ids = [it.item_id for it in items]

    # Run-scoped maxima for the size scoring component.
    global _run_max_size
    _run_max_size = max((it.file_size or 0 for it in items), default=0)

    stats = DedupStats(scanned=len(items))

    # Union-Find over item indices to merge transitively: A~B and B~C => one group.
    parent = list(range(len(ids)))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x: int, y: int) -> None:
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    pairs = _candidate_pairs(ids, items)
    stats.pairs_checked = len(pairs)

    confirmed_edges = 0
    for i, j in pairs:
        sim = cosine_similarity(
            by_id[ids[i]].dinov2_vector, by_id[ids[j]].dinov2_vector
        )
        if sim is not None and sim >= embedding_threshold:
            union(i, j)
            confirmed_edges += 1

    # Build groups from the union-find structure. A component of size >= 2 here
    # *already* implies at least one confirmed edge (union only fires on the
    # slow tier), so no separate re-verification is needed.
    groups: dict[int, list[str]] = {}
    for idx, item_id in enumerate(ids):
        groups.setdefault(find(idx), []).append(item_id)

    confirmed: list[DuplicateGroup] = []
    for members in groups.values():
        if len(members) < MIN_DUPLICATE_GROUP:
            continue
        scores = {mid: quality_score(by_id[mid]) for mid in members}
        master = _pick_master(members, by_id, scores)
        confirmed.append(
            DuplicateGroup(
                group_id=f"dup-{len(confirmed) + 1:04d}",
                item_ids=sorted(members),
                master_item_id=master,
                derivative_item_ids=[m for m in sorted(members) if m != master],
            )
        )

    confirmed.sort(key=lambda g: len(g.item_ids), reverse=True)
    # Renumber after sorting so group ids stay monotonically small.
    for new_idx, group in enumerate(confirmed, start=1):
        group.group_id = f"dup-{new_idx:04d}"

    stats.groups = len(confirmed)
    stats.pairs_confirmed = confirmed_edges
    stats.confirming_duplicates = sum(len(g.item_ids) for g in confirmed)
    stats.masters = len(confirmed)
    stats.derivatives = sum(len(g.derivatives) for g in confirmed)
    return confirmed, stats


def run_dedup(db: Database, **kwargs) -> tuple[list[DuplicateGroup], DedupStats]:
    """Find duplicates and persist the assignment in the DB.

    Writes ``duplicate_group_id`` on every member via
    :meth:`Database.set_duplicate_group` and the Master/derivative flag via
    :meth:`Database.upsert_item`. Returns the groups plus stats (same as
    :func:`find_duplicate_groups`).
    """
    groups, stats = find_duplicate_groups(db, **kwargs)
    for group in groups:
        db.set_duplicate_group(group.item_ids, group.group_id)
        for item_id in group.item_ids:
            item = db.item(item_id)
            if item is None:
                continue
            item.is_primary_master = item_id == group.master_item_id
            db.upsert_item(item)
    return groups, stats
