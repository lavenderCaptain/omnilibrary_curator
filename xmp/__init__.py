"""Non-destructive XMP sidecar writing for Lightroom ingestion."""

from xmp.writer import (
    XmpExport,
    render_xmp,
    sidecar_path_for,
    write_sidecar,
    write_sidecars,
)

__all__ = [
    "XmpExport",
    "render_xmp",
    "sidecar_path_for",
    "write_sidecar",
    "write_sidecars",
]
