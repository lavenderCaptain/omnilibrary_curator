"""Tests for the non-destructive XMP sidecar writer (Phase 4)."""

from __future__ import annotations

import xml.etree.ElementTree as ET

import pytest

from engine.database import MediaItem
from xmp.writer import XmpExport, render_xmp, sidecar_path_for, write_sidecar, write_sidecars


def _item(**overrides):
    data = dict(
        file_path="/tmp/photos/CIMG0010.CR2",
        file_format="CR2",
        event_id=None,
        duplicate_group_id=None,
        is_primary_master=True,
    )
    data.update(overrides)
    return MediaItem(**data)


# ---------------------------------------------------------------------------
# Rendering structure
# ---------------------------------------------------------------------------
def test_render_is_valid_xmp_with_root_element():
    xml = render_xmp(_item(), XmpExport())
    assert xml.startswith("<?xml")
    root = ET.fromstring(xml)
    assert root.tag == "{http://ns.adobe.com/xap/1.0/}xmpmeta"


def test_render_standalone_sidecar_parses():
    """A sidecar with no human edits is still a valid, empty document."""
    xml = render_xmp(_item(), XmpExport())
    ET.fromstring(xml)  # must not raise


def test_star_rating_written_when_present():
    xml = render_xmp(_item(), XmpExport(event_title="Japan 2024", star_rating=4))
    assert "lr:Headline" in xml and "Japan 2024" in xml
    assert "xmp:Rating=\"4\"" in xml
    assert ">4<" in xml  # lr:Star text node


def test_zero_rating_not_written():
    xml = render_xmp(_item(), XmpExport(star_rating=0))
    assert "xmp:Rating" not in xml
    assert ">0<" not in xml


def test_out_of_range_star_raises():
    with pytest.raises(ValueError):
        render_xmp(_item(), XmpExport(star_rating=6))


def test_event_hierarchy_builds_lhierarchy_and_kws():
    xml = render_xmp(
        _item(),
        XmpExport(event_hierarchy="Events|Japan 2024|Day 01", event_title="Japan 2024"),
    )
    assert "lr:Hierarchy:Event" in xml
    assert ">Day 01<" in xml  # crs:Level3
    assert ">Japan 2024<" in xml  # crs:Level2


def test_duplicate_flag_written_for_derivative():
    """A non-master inside a group is flagged No and marked DuplicateOf."""
    xml = render_xmp(
        _item(duplicate_group_id="g1", is_primary_master=False),
        XmpExport(),
    )
    assert "lr:Flag" in xml and "No" in xml
    assert "crs:DuplicateOf" in xml


def test_master_flag_left_when_explicit():
    xml = render_xmp(
        _item(duplicate_group_id="g1", is_primary_master=True),
        XmpExport(flag="Yes"),
    )
    assert "lr:Flag" in xml


def test_flag_default_none_when_no_group():
    """Without a duplicate group, no flag is auto-written."""
    xml = render_xmp(_item(), XmpExport())
    assert "lr:Flag" not in xml


def test_sidecar_does_not_reference_source_file_contents():
    """The source path is used only to derive the sidecar path, never read."""
    xml = render_xmp(_item(file_path="/tmp/photos/CIMG0010.CR2"), XmpExport())
    # A bare filename must not leak into the sidecar body.
    assert "CIMG0010.CR2" not in xml


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------
def test_write_sidecar_next_to_source(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    item = _item(file_path=str(tmp_path / "photos" / "CIMG0010.CR2"))
    out = write_sidecar(item, XmpExport(event_title="T"))
    assert out == tmp_path / "photos" / "CIMG0010.CR2.xmp"
    assert out.exists()
    assert "T" in out.read_text()


def test_sidecar_path_for_out_dir(tmp_path):
    item = _item(file_path="/tmp/photos/CIMG0010.CR2")
    out = sidecar_path_for(item, out_dir=tmp_path)
    assert out == tmp_path / "CIMG0010.CR2.xmp"


def test_write_sidecar_does_not_touch_source(tmp_path):
    src = tmp_path / "CIMG0010.CR2"
    src.write_bytes(b"\x00\x01\x02RAW")
    write_sidecar(_item(file_path=str(src)), XmpExport(event_title="T"))
    assert src.read_bytes() == b"\x00\x01\x02RAW"  # untouched


def test_write_sidecars(tmp_path):
    item_a = _item(file_path=str(tmp_path / "a.CR2"))
    item_b = _item(file_path=str(tmp_path / "b.CR2"))
    written = write_sidecars([item_a, item_b], XmpExport(event_title="T"))
    assert len(written) == 2
    assert all(p.suffix == ".xmp" for p in written)


# ---------------------------------------------------------------------------
# Defaults
# ---------------------------------------------------------------------------
def test_derive_fills_flag_from_group():
    export = XmpExport().derive(_item(duplicate_group_id="g1", is_primary_master=False))
    assert export.flag == "No"
    export = XmpExport().derive(_item(duplicate_group_id="g1", is_primary_master=True))
    assert export.flag == "Yes"


def test_derive_leaves_flag_none_without_group():
    assert XmpExport().derive(_item()).flag is None


def test_hierarchy_levels_parser():
    from xmp.writer import _hierarchy_levels

    assert _hierarchy_levels("Events|Japan 2024|Day 01") == [
        "Events",
        "Japan 2024",
        "Day 01",
    ]
    assert _hierarchy_levels(None) == []
    assert _hierarchy_levels("") == []
