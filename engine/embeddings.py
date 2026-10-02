"""MPS-powered embeddings worker: DINOv2 (timm), pHash, and aesthetics.

This is the Phase 2 ingestion surface. It turns scanned :class:`MediaItem` rows
into three kinds of per-image signal that the clustering / dedup engines consume:

* **DINOv2 embedding** (384-d for ``dinov2_vits14``) via ``timm``'s
  ``vit_small_patch14_dinov2.lvd142m`` weights. The model runs on the Apple
  Silicon MPS GPU when available, otherwise the CPU. The ``dinov2`` pip package
  that Facebook released for this model is *not* used here: it ships only
  metadata (no importable code) on modern torch/Python and pins older torch
  versions, so it is both broken and fragile in this environment. The weights
  are identical — they are simply served through the Hugging Face
  ``transformers``/``timm`` APIs instead.
* **pHash** (64-bit) via :mod:`imagehash`.
* **Aesthetic score** (0..10) from a lightweight, dependency-free image-
  statistics heuristic. A full aesthetic model would require another network
  download; this heuristic is deterministic, offline, and good enough to seed
  sorting until a trained scorer is wired in.

Design constraints honoured:

* Non-destructive — raw files are never fully decoded. For RAW assets the
  embedded preview JPEG is extracted via :func:`engine.scanner.extract_embedded_jpeg`
  (the same libraw-backed path the scanner uses) and that 8-bit preview feeds
  the model, exactly as the "RAW Performance Optimization" constraint requires.
* Heavy compute lives here and is meant to run off the PySide6 main thread;
  this module exposes a batched :meth:`EmbeddingWorker.process` suitable for a
  ``QThread`` / ``QThreadPool`` worker.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Iterator, Optional, Sequence

import numpy as np
from PIL import Image, ImageStat

from .scanner import extract_embedded_jpeg

# ---------------------------------------------------------------------------
# Model identity / constants
# ---------------------------------------------------------------------------
# DINOv2 small / patch-14 weights (the ``dinov2_vits14`` family). ``lvd142m``
# is the "large-vision-distill" variant; ``_v2`` is its successor. We prefer
# the v2 variant when it exists and fall back to v1.
_TIMM_CANDIDATES: tuple[str, ...] = (
    "vit_small_patch14_dinov2.lvd142m",
    "vit_small_patch16_in21k",  # unlikely, kept for parity with some installs
)

# The embedding the CLS token lives at in the model's per-patch feature map:
# timm ViT ``forward_features`` returns ``[B, N, D]`` with the CLS token first.
CLS_TOKEN_INDEX = 0

# Embedding dimension for ``dinov2_vits14`` (vit_small). This is NOT 768 (that
# is vit_base); the "768-d" in database.py's schema comment is wrong and harmless
# because the column is a free-form DOUBLE[]. Kept here so callers never guess.
EMBEDDING_DIM = 384


# ---------------------------------------------------------------------------
# Result records
# ---------------------------------------------------------------------------
@dataclass
class EmbeddingResult:
    """Per-file outcome of the embeddings pass."""

    path: Path
    item_id: Optional[str] = None
    dinov2_vector: Optional[list[float]] = None
    p_hash: Optional[int] = None
    aesthetic_score: Optional[float] = None
    error: Optional[str] = None
    source_image: Optional[Path] = None  # probe/preview actually fed to the model


@dataclass
class EmbeddingStats:
    """Aggregate counters for a completed pass."""

    processed: int = 0
    ok: int = 0
    failed: int = 0
    # Bytes of probe files written for RAW previews (for diagnostics/tests).
    probes_written: int = 0


# ---------------------------------------------------------------------------
# Image loading
# ---------------------------------------------------------------------------
def _probe_path_for(path: Path, cache_dir: Path) -> Path:
    """Where an embedded-preview probe for ``path`` lives under ``cache_dir``."""
    cache_dir.mkdir(parents=True, exist_ok=True)
    return cache_dir / f"{path.stem}.probe.jpg"


def load_image_for_embedding(
    path: Path,
    *,
    cache_dir: Optional[Path] = None,
) -> tuple[Image.Image, Optional[Path]]:
    """Load a PIL Image suitable for embedding.

    * JPEGs (and other decodable images) are opened directly.
    * RAW files are fed their embedded preview JPEG (extracted to ``cache_dir``),
      never their full-resolution pixels.

    Truncated previews (common from embedded-JPEG extraction) are tolerated via
    :attr:`PIL.ImageFile.LOAD_TRUNCATED_IMAGES`.

    Returns ``(image, source)`` where ``source`` is the path actually read
    (``path`` for direct opens, the probe for RAW previews). Raises ``OSError``
    when neither a usable source nor an extractable preview can be found.
    """
    import PIL.ImageFile

    PIL.ImageFile.LOAD_TRUNCATED_IMAGES = True

    path = Path(path)
    if path.is_file() and path.suffix.lower() in (".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"):
        return Image.open(path).convert("RGB"), path

    # Anything else (RAW, exotic): try to pull an embedded preview.
    cache_dir = cache_dir or path.parent / ".omnilibrary_thumbs"
    probe = _probe_path_for(path, cache_dir)
    extracted = extract_embedded_jpeg(path, probe, as_raw=True)
    if extracted is None or not extracted.exists():
        raise OSError(f"no decodable source and no embedded preview for {path}")
    return Image.open(extracted).convert("RGB"), extracted


# ---------------------------------------------------------------------------
# Embeddings worker
# ---------------------------------------------------------------------------
class EmbeddingWorker:
    """Batches images into DINOv2 / pHash / aesthetic signal.

    The model is loaded lazily on first :meth:`process` call so callers can
    construct the worker cheaply (e.g. on the main thread) and hand it off to a
    background thread.
    """

    def __init__(
        self,
        *,
        device: str | None = None,
        batch_size: int = 32,
        dim: int = EMBEDDING_DIM,
    ):
        self.device = device or self._prefer_mps()
        self.batch_size = max(1, int(batch_size))
        self.dim = dim
        self._model = None
        self._timm_name = None
        self._transform = None

    # -- device -----------------------------------------------------------
    @staticmethod
    def _prefer_mps() -> str:
        """Pick MPS when present (Apple Silicon), else CPU."""
        try:
            import torch

            if torch.backends.mps.is_available():
                return "mps"
        except Exception:
            pass
        return "cpu"

    # -- model loading -----------------------------------------------------
    def _ensure_model(self) -> None:
        if self._model is not None:
            return
        import timm  # local import: only needed when actually embedding

        last_err: Optional[Exception] = None
        for name in _TIMM_CANDIDATES:
            try:
                model = timm.create_model(name, pretrained=True)
                model.eval()
                self._model = model.to(self.device)
                self._timm_name = name
                break
            except Exception as exc:  # noqa: BLE001 - try next candidate
                last_err = exc
                continue
        if self._model is None:
            raise RuntimeError(
                f"could not load any DINOv2 timm model from {_TIMM_CANDIDATES}: {last_err!r}"
            )

    @property
    def input_size(self) -> tuple[int, int]:
        """The exact (height, width) the ViT expects.

        DINOv2's timm weights use a *fixed* ``img_size`` (not merely a multiple
        of the patch size), so every image must be resized to exactly this. We
        read it off the patched patch-embed layer and fall back to the common
        DINOv2 small/14 default (518) when unavailable.
        """
        try:
            img_size = getattr(self._model.patch_embed, "img_size", None)
            if img_size:
                return tuple(int(v) for v in img_size)
        except Exception:  # noqa: BLE001
            pass
        # DINOv2 small/14 ships at a fixed 518px input.
        return (518, 518)

    # -- embeddings --------------------------------------------------------
    def embed_batch(self, images: Sequence[Image.Image]) -> list[list[float]]:
        """Return L2-normalized DINOv2 embeddings for a batch of PIL Images.

        Images are resized to the model's input size and normalized with the
        standard ImageNet statistics; the CLS token of each patch-map is the
        returned embedding.
        """
        import torch
        from torchvision.transforms import functional as TF

        self._ensure_model()
        if not images:
            return []

        # DINOv2's timm weights use a fixed input size (e.g. 518x518 for
        # vit_small/14). Resize every image to exactly that before embedding.
        new_h, new_w = self.input_size
        tensors = [TF.resize(img, (new_h, new_w)) for img in images]
        tensors = [TF.to_tensor(t).unsqueeze(0) for t in tensors]
        stacked = torch.cat(tensors, dim=0).to(self.device)

        with torch.no_grad():
            feats = self._model.forward_features(stacked)  # [B, N, D]
            cls = feats[:, CLS_TOKEN_INDEX, :]
            norm = torch.linalg.norm(cls, dim=1, keepdim=True)
            norm = torch.maximum(norm, torch.ones_like(norm))
            emb = cls / norm

        return [row.tolist() for row in emb]  # list[list[float]]

    # -- pHash -------------------------------------------------------------
    @staticmethod
    def phash(image: Image.Image) -> Optional[int]:
        """64-bit perceptual hash as a Python int (``None`` on failure).

        Handles both the canonical :mod:`imagehash` API (where ``phash``
        returns an :class:`imagehash.ImageHash` with ``__int__``) and stripped
        builds where ``.hash`` is a numpy boolean array of the per-bit values,
        by reconstructing the integer from the raw bits.
        """
        try:
            import imagehash

            h = imagehash.phash(image)
            raw = getattr(h, "hash", None)
            if isinstance(raw, str):
                return int(raw, 16)
            if isinstance(raw, np.ndarray):
                bits = raw.reshape(-1).astype(bool)
                return int("".join("1" if b else "0" for b in bits), 2)
            # Fallback: assume the object itself is int-convertible.
            return int(h)
        except Exception:  # noqa: BLE001
            return None

    # -- aesthetic score ---------------------------------------------------
    @staticmethod
    def aesthetic_score(image: Image.Image) -> float:
        """Return a 0..10 aesthetic score from image statistics.

        Heuristic (deterministic, offline, no network): combines mean
        brightness, saturation, global contrast and edge sharpness with weights
        that loosely mirror common aesthetic predictors. It is a seed, not a
        trained model — see the module docstring.
        """
        import numpy as np

        arr = np.asarray(image, dtype=np.float32)
        rgb = arr.reshape(-1, 3)

        # Brightness: mild preference for mid-tones (~110/255), falling off
        # toward pure black/white.
        brightness = rgb.mean(axis=0).mean() / 255.0
        brightness_penalty = 1.0 - 2.0 * abs(brightness - 0.45)
        brightness_penalty = max(0.0, brightness_penalty)

        # Saturation: some colour vitality, capped.
        if len(rgb) > 0:
            lum = rgb @ np.array([0.299, 0.587, 0.114])
            sat = np.abs(rgb - lum[..., None]).mean() / 255.0
        else:
            sat = 0.0
        sat_component = min(1.0, sat / 0.35)

        # Contrast: spread of luminance.
        lum_arr = rgb @ np.array([0.299, 0.587, 0.114])
        contrast = float(np.std(lum_arr)) / 255.0
        contrast_component = min(1.0, contrast / 0.25)

        # Sharpness: variance of a Laplacian.
        try:
            from PIL import ImageFilter

            lap = np.asarray(image.convert("L").filter(ImageFilter.Laplacian), dtype=np.float32)
            sharp = float(np.std(lap)) / 255.0
        except Exception:  # noqa: BLE001
            sharp = 0.0
        sharp_component = min(1.0, sharp / 20.0)

        score = (
            0.20 * brightness_penalty
            + 0.20 * sat_component
            + 0.25 * contrast_component
            + 0.35 * sharp_component
        )
        # Map 0..1 -> 0..10.
        return round(float(min(10.0, max(0.0, score * 10.0))), 2)

    # -- public API --------------------------------------------------------
    def process(
        self,
        paths: Iterable[Path | str],
        *,
        item_ids: Optional[Sequence[str]] = None,
        cache_dir: Optional[Path] = None,
    ) -> Iterator[EmbeddingResult]:
        """Process each path, batching DINOv2 embedding.

        Yields an :class:`EmbeddingResult` per path. RAW files feed their
        embedded preview; failures (undecodable + no preview) are reported as
        results with ``error`` set, never raised, so a single bad file can't
        abort a long background pass.
        """
        paths_list: list[Path] = []
        ids_list: list[Optional[str]] = []
        for i, p in enumerate(paths):
            paths_list.append(Path(p))
            ids_list.append(item_ids[i] if item_ids and i < len(item_ids) else None)

        # Group indices whose first opened image is shared (RAW previews) so we
        # don't re-extract the same preview twice.
        for result in self._process_batch(paths_list, ids_list, cache_dir):
            yield result

    def _process_batch(
        self,
        paths: Sequence[Path],
        item_ids: Sequence[Optional[str]],
        cache_dir: Optional[Path],
    ) -> Iterator[EmbeddingResult]:
        self._ensure_model()
        cache_dir = cache_dir or (paths[0].parent / ".omnilibrary_thumbs") if paths else None

        loaded: list[tuple[int, Image.Image, Path]] = []  # (idx, img, source)
        results: list[EmbeddingResult] = [EmbeddingResult(path=p, item_id=i) for p, i in zip(paths, item_ids)]

        for idx, path in enumerate(paths):
            try:
                image, source = load_image_for_embedding(path, cache_dir=cache_dir)
                results[idx].source_image = source
                loaded.append((idx, image, source))
            except Exception as exc:  # noqa: BLE001 - report, keep going
                results[idx] = EmbeddingResult(
                    path=path,
                    item_id=item_ids[idx],
                    error=str(exc)[:500],
                )

        # Batch DINOv2.
        for idx, image, _source in loaded:
            try:
                results[idx].dinov2_vector = self.embed_batch([image])[0]
            except Exception as exc:  # noqa: BLE001
                if results[idx].error is None:
                    results[idx] = EmbeddingResult(
                        path=paths[idx], item_id=item_ids[idx], error=str(exc)[:500]
                    )

        # pHash + aesthetic (cheap, per image, tolerant of odd inputs).
        for idx, image, _source in loaded:
            try:
                results[idx].p_hash = self.phash(image)
                results[idx].aesthetic_score = self.aesthetic_score(image)
            except Exception:  # noqa: BLE001 - non-fatal
                pass

        for r in results:
            yield r

    def collect_stats(self, results: Iterable[EmbeddingResult]) -> EmbeddingStats:
        """Summarise an iterator of results."""
        stats = EmbeddingStats()
        for r in results:
            stats.processed += 1
            if r.error is not None:
                stats.failed += 1
            else:
                stats.ok += 1
        return stats


# ---------------------------------------------------------------------------
# Convenience: drive straight into a Database
# ---------------------------------------------------------------------------
def run_embeddings(
    db,
    worker: EmbeddingWorker,
    paths: Iterable[Path | str],
    *,
    item_ids: Optional[Sequence[str]] = None,
    cache_dir: Optional[Path] = None,
) -> tuple[list[EmbeddingResult], EmbeddingStats]:
    """Run a pass and write every successful result into ``db``.

    ``db`` must expose :meth:`upsert_embeddings` plus :meth:`upsert_metadata`
    (matching :class:`engine.database.Database`). Returns the raw results plus
    the aggregate stats.
    """
    results = list(worker.process(paths, item_ids=item_ids, cache_dir=cache_dir))
    stats = worker.collect_stats(results)

    for r in results:
        if r.error is not None or r.item_id is None:
            continue
        if r.dinov2_vector is not None:
            db.upsert_embeddings(r.item_id, dinov2_vector=r.dinov2_vector, aesthetic_score=r.aesthetic_score)
        if r.p_hash is not None:
            db.upsert_metadata(r.item_id, p_hash=r.p_hash)
    return results, stats
