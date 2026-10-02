"""Non-destructive XMP sidecar writing for Lightroom ingestion.

This is the Phase 4 export bridge of the project blueprint. It renders a
standard Adobe XMP sidecar for a :class:`MediaItem` and writes it **next to**
the source asset. It never modifies the source image in any way (the
"Non-Destructive Operations" constraint) — the sidecar is the only artifact,
and it lives beside the file (``photo.CR2`` -> ``photo.CR2.xmp``).

The sidecar encodes the four things Phase 4 of the blueprint exports:

* **Event title** -> ``lr:Headline`` (an IPTC headline) and ``dc:title`` plus a
  ``lr:Hierarchy:Event`` block that Lightroom uses to build its *Events* system.
* **Hierarchical keywords** -> ``crs:HierarchicalSubject`` (with the
  ``crs:Level1/2/3`` children Lightroom expects) and ``dc:subject`` bag entries,
  e.g. ``Events|Japan 2024|Day 01``.
* **Star rating** -> ``xmp:Rating`` and ``lr:Star`` (0-5, 0 == no rating).
* **Duplicate flag** -> ``lr:Flag`` and ``crs:DuplicateOf`` so Lightroom can
  stack derivatives under their master and mark them for deletion.

Namespaces used (all part of the Adobe/XMP/Lightroom/Camera-Raw spec):

====== ==================================================================
prefix namespace
====== ==================================================================
xap    ``http://ns.adobe.com/xap/1.0/``
xmp    ``http://ns.adobe.com/xap/1.0/``
dc     ``http://purl.org/dc/elements/1.1/``
crs    ``http://ns.adobe.com/camera-raw-settings/1.0/``
lr     ``http://ns.adobe.com/lightroom/1.0/``
rdf    ``http://www.w3.org/1999/02/22-rdf-syntax-ns#``
====== ==================================================================

The writer is stdlib-only. :func:`render_xmp` produces a deterministic,
parseable XMP XML string assembled by hand (so the namespace prefixes come out
exactly as Lightroom expects); :func:`write_sidecar` is the thin
disk-persistence wrapper around it.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

from engine.database import MediaItem

# ---------------------------------------------------------------------------
# XMP namespaces
# ---------------------------------------------------------------------------
_XMP_NS = "http://ns.adobe.com/xap/1.0/"
_DC_NS = "http://purl.org/dc/elements/1.1/"
_CRS_NS = "http://ns.adobe.com/camera-raw-settings/1.0/"
_LR_NS = "http://ns.adobe.com/lightroom/1.0/"
_RDF_NS = "http://www.w3.org/1999/02/22-rdf-syntax-ns#"

# Default (no-stars) rating. Never exported.
_NO_STAR = 0

# The xmlns:rdf attribute appended to every <RDF>/<Description> wrapper.
_RDF_ATTR = f' xmlns:rdf="{_RDF_NS}"'


# ---------------------------------------------------------------------------
# The approved export edit for a single asset
# ---------------------------------------------------------------------------
# The ``MediaItem`` only carries the *structural* signals (duplicate group,
# primary-master flag, aesthetic score). The human-approved values below (a
# real event title, a chosen star rating) are supplied by the caller.
class XmpExport:
    """The human-approved edit to export for a single asset.

    Every field is optional. A ``None`` field means "don't write anything for
    it". :meth:`XmpExport.derive` fills in sensible defaults from the
    :class:`MediaItem` for anything left unspecified.
    """

    def __init__(
        self,
        event_title: Optional[str] = None,
        event_hierarchy: Optional[str] = None,
        star_rating: Optional[int] = None,
        flag: Optional[str] = None,
        keywords: Optional[Sequence[str]] = None,
    ):
        self.event_title = event_title
        self.event_hierarchy = event_hierarchy
        self.star_rating = star_rating
        self.flag = flag
        self.keywords = list(keywords) if keywords else []

    def derive(self, item: MediaItem) -> "XmpExport":
        """Return a copy of ``self`` with defaults filled in from ``item``.

        Only the fields left ``None`` by the caller are backfilled. The
        duplicate group / primary-master signals drive ``flag``; nothing else is
        auto-derived because the export is meant to be human-approved.
        """
        flag = self.flag
        if flag is None and getattr(item, "duplicate_group_id", None):
            # Inside a duplicate group: the master keeps its flag, derivatives
            # are marked for deletion in Lightroom.
            flag = "Yes" if getattr(item, "is_primary_master", True) else "No"
        return XmpExport(
            event_title=self.event_title,
            event_hierarchy=self.event_hierarchy,
            star_rating=self.star_rating,
            flag=flag,
            keywords=self.keywords,
        )


# ---------------------------------------------------------------------------
# XML helpers
# ---------------------------------------------------------------------------
def _escape(text: str) -> str:
    """Escape text for safe inclusion in an XMP text node."""
    return (
        text.replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
        .replace("'", "&apos;")
    )


def _hierarchy_levels(hierarchy: Optional[str]) -> list[str]:
    """Yield the individual levels of a hierarchical event path.

    ``"Events|Japan 2024|Day 01"`` -> ``["Events", "Japan 2024", "Day 01"]``.
    An empty/``None`` path yields nothing.
    """
    if not hierarchy:
        return []
    return [level.strip() for level in hierarchy.split("|") if level.strip()]


def _star_attr(value: Optional[int]) -> str:
    """Normalise an optional 0-5 star rating; reject out-of-range values."""
    if value is None:
        return "0"
    value = int(value)
    if value < 0 or value > 5:
        raise ValueError(f"star rating out of range (0-5): {value!r}")
    return str(value)


def _decl(prefixes: Sequence[str]) -> str:
    """Return the ``xmlns:...`` attribute string for the given prefixes.

    The ``xmp`` and ``xap`` prefixes resolve to the same namespace, so an
    ``xmp`` prefix is emitted as ``xmlns:xap``.
    """
    def ns(p: str) -> str:
        if p in ("xmp", "xap"):
            return _XMP_NS
        if p == "dc":
            return _DC_NS
        if p == "crs":
            return _CRS_NS
        if p == "lr":
            return _LR_NS
        if p == "rdf":
            return _RDF_NS
        raise KeyError(f"unknown XMP prefix: {p!r}")

    return "".join(f' xmlns:{p}="{ns(p)}"' for p in prefixes)


def _element(tag: str, text: Optional[str] = None) -> str:
    """Render a namespaced element ``<prefix:Local>text</prefix:Local>``."""
    prefix, _, local = tag.partition(":")
    body = _escape(text) if text is not None else ""
    if body:
        return f"<{tag}>{body}</{tag}>"
    return f"<{tag}/>"


# ---------------------------------------------------------------------------
# Rendering
# ---------------------------------------------------------------------------
def render_xmp(item: MediaItem, edit: XmpExport) -> str:
    """Render the XMP sidecar XML (as a UTF-8 string) for ``item`` + ``edit``.

    The returned string starts with the ``<?xml?>`` prologue and is a complete,
    parseable XMP document. The caller persists it via :func:`write_sidecar`.
    """
    export = edit.derive(item)
    lines = ['<?xml version="1.0" encoding="UTF-8"?>',
             '<xmp:xmpmeta' + _decl(("xmp",)) + '>']

    lines.append(f"  <rdf:RDF{_RDF_ATTR}>")

    # ---- LR description: rating, flag, headline, event hierarchy ---------
    lr_body = ""
    if export.flag:
        lr_body += _element("lr:Flag", export.flag)
    if export.event_title:
        lr_body += _element("lr:Headline", export.event_title)
    if export.event_hierarchy:
        event_title = _element("dc:title", export.event_title or export.event_hierarchy)
        lr_body += (
            "<lr:Hierarchy>"
            + f"<lr:Hierarchy:Event>{event_title}</lr:Hierarchy:Event>"
            + "</lr:Hierarchy>"
        )
    star_str = _star_attr(export.star_rating)
    rating_attr = f' xmp:Rating="{star_str}"' if export.star_rating and int(export.star_rating) > 0 else ""
    if export.star_rating and int(export.star_rating) > 0:
        lr_body += _element("lr:Star", star_str)
    lr_desc = _decl(("lr", "xap")) + f' xap:CreatorTool="Omnilibrary 1.0"' + rating_attr
    lines.append(f"  <rdf:Description{_RDF_ATTR}{lr_desc}>{lr_body}</rdf:Description>")

    # ---- CR description: hierarchical subject + duplicate marker ---------
    # ``crs:HierarchicalSubject`` is the Camera-Raw convention Lightroom reads
    # to populate its keyword system; it needs ``crs:Level1/2/3`` children.
    if export.event_hierarchy or export.keywords or export.flag == "No":
        crs_body = ""
        for idx, level in enumerate(_hierarchy_levels(export.event_hierarchy), start=1):
            crs_body += _element(f"crs:Level{idx}", level)
        for kw in export.keywords:
            crs_body += _element("crs:HierarchicalSubject", kw)
        if export.flag == "No":
            crs_body += _element("crs:DuplicateOf", "true")
        crs_desc = _decl(("crs",))
        lines.append(f"  <rdf:Description{_RDF_ATTR}{crs_desc}>{crs_body}</rdf:Description>")

    # ---- DC description: title + keyword bag -----------------------------
    bag_items = _hierarchy_levels(export.event_hierarchy) + list(export.keywords)
    dc_body = ""
    if export.event_title:
        dc_body += _element("dc:title", export.event_title)
    if bag_items:
        bag = "".join(_element("rdf:li", kw) for kw in bag_items)
        dc_body += f"<dc:subject><rdf:Bag>{bag}</rdf:Bag></dc:subject>"
    dc_desc = _decl(("dc",))
    lines.append(f"  <rdf:Description{_RDF_ATTR}{dc_desc}>{dc_body}</rdf:Description>")

    lines.append("  </rdf:RDF>")
    lines.append("</xmp:xmpmeta>")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def sidecar_path_for(item: MediaItem, *, out_dir: Optional[Path] = None) -> Path:
    """Return the sidecar path for ``item``.

    The sidecar sits next to the source by default (``photo.CR2`` ->
    ``photo.CR2.xmp``). ``out_dir`` lets a caller redirect sidecars to a
    dedicated cache instead of scattering them beside the source.
    """
    source = Path(item.file_path)
    base = source.with_name(source.name + ".xmp")
    return base if out_dir is None else Path(out_dir) / base.name


def write_sidecar(
    item: MediaItem,
    edit: XmpExport,
    *,
    out_dir: Optional[Path] = None,
    sidecar_name: Optional[str] = None,
) -> Path:
    """Render and persist the XMP sidecar for ``item``; return its path.

    The source image is never opened, read, or modified. The sidecar is written
    to ``sidecar_name`` (or :func:`sidecar_path_for` when omitted). When
    ``out_dir`` is given the sidecar lands there; otherwise beside the source.
    """
    target = Path(sidecar_name) if sidecar_name else Path(item.file_path).with_name(
        Path(item.file_path).name + ".xmp"
    )
    out_path = (Path(out_dir) / target.name) if out_dir else target

    out_path.parent.mkdir(parents=True, exist_ok=True)
    xml = render_xmp(item, edit)
    out_path.write_text(xml, encoding="UTF-8")
    return out_path


def write_sidecars(
    items: Sequence[MediaItem],
    edit: XmpExport,
    *,
    out_dir: Optional[Path] = None,
    sidecar_name: Optional[str] = None,
) -> list[Path]:
    """Write a sidecar for every item in ``items``; return the written paths.

    A single literal ``sidecar_name`` is reused for every item when
    ``out_dir`` is set (useful when each item is a distinct export). When
    ``out_dir`` is omitted, each sidecar sits beside its source, so names never
    collide.
    """
    written = []
    for item in items:
        written.append(write_sidecar(item, edit, out_dir=out_dir, sidecar_name=sidecar_name))
    return written


# ---------------------------------------------------------------------------
# Public names
# ---------------------------------------------------------------------------
__all__ = [
    "XmpExport",
    "render_xmp",
    "sidecar_path_for",
    "write_sidecar",
    "write_sidecars",
]
