"""Image near-duplicate checks with bounded decode budgets.

Every image is checked against per-file byte, dimension, pixel and frame limits
before its pixels are decoded, and one audit shares a single pixel budget across
both split folders. A refused image is counted with a stable reason instead of
being silently treated as clean or as a generic unreadable file. Pillow's own
decompression-bomb protection stays in place; LeakLens adds a stricter preflight
and turns a bomb warning into a refusal rather than ignoring it.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .availability import EngineUnavailable, probe
from .evidence import Evidence
from .leakage import Finding

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
IMAGE_FORMATS = frozenset({"PNG", "JPEG", "BMP", "GIF", "WEBP", "TIFF"})

MAX_IMAGE_BYTES = 64 * 1024 * 1024  # an 8k photographic image stays well below
MAX_IMAGE_SIDE = 20_000  # beyond any consumer camera or dataset image
MAX_IMAGE_PIXELS = 64_000_000  # 64 MP; Pillow's own limit is left untouched
MAX_IMAGE_FRAMES = 64  # animation is counted, never fully decoded
MAX_AUDIT_PIXELS = 2_000_000_000  # ~10k images at 200k pixels each


class ImageUnavailable(EngineUnavailable):
    """Pillow is not installed; image checks cannot run."""


class ImageRefused(Exception):
    """An image exceeded a decode budget; ``reason`` is stable and reportable."""

    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


@dataclass
class ImageBudget:
    """Pixel budget shared by every image in one audit."""

    max_pixels: int = MAX_AUDIT_PIXELS
    pixels_used: int = 0
    refused: dict[str, int] = field(default_factory=dict)

    def check(self, pixels: int) -> None:
        if self.pixels_used + pixels > self.max_pixels:
            raise ImageRefused("budget_exhausted")

    def add(self, pixels: int) -> None:
        self.pixels_used += pixels

    def refuse(self, reason: str) -> None:
        self.refused[reason] = self.refused.get(reason, 0) + 1


def _pil():
    return probe(
        "PIL.Image",
        "images",
        purpose="image checks",
        label="Pillow",
        error=ImageUnavailable,
    )


def dhash(path, size=8, *, budget=None) -> int:
    """Difference hash: size*size bits, set where a pixel is brighter than its right neighbour.

    Raises :class:`ImageRefused` when the file exceeds a decode budget. Raw
    decoder exceptions may propagate: ``hash_dir`` is the fail-closed entry
    point for untrusted directories.
    """
    Image = _pil()
    p = Path(path)
    if p.stat().st_size > MAX_IMAGE_BYTES:
        raise ImageRefused("too_large")
    budget = budget if budget is not None else ImageBudget()
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("error", Image.DecompressionBombWarning)
            with Image.open(p) as im:
                if im.format not in IMAGE_FORMATS:
                    raise ImageRefused("unsupported_format")
                # n_frames can extend a GIF's logical screen, so read it before
                # the final size check.
                if getattr(im, "n_frames", 1) > MAX_IMAGE_FRAMES:
                    raise ImageRefused("too_many_frames")
                width, height = im.size
                if width > MAX_IMAGE_SIDE or height > MAX_IMAGE_SIDE:
                    raise ImageRefused("too_large_dimensions")
                pixels = width * height
                if pixels > MAX_IMAGE_PIXELS:
                    raise ImageRefused("too_many_pixels")
                budget.check(pixels)
                with (
                    im.convert("L") as gray,
                    gray.resize((size + 1, size), Image.Resampling.LANCZOS) as small,
                ):
                    g = np.asarray(small, dtype=np.int16)
                budget.add(pixels)
    except (Image.DecompressionBombWarning, Image.DecompressionBombError):
        raise ImageRefused("decompression_bomb") from None
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_dir(folder, *, budget=None):
    """Hash every image under folder.

    Returns ``({relative path: hash}, unreadable count)``; refusals are counted
    by reason on ``budget.refused`` so they never look like clean images or like
    ordinary unreadable files. Pass a budget to inspect refusals; the default
    budget is not returned.
    """
    _pil()
    root = Path(folder)
    budget = budget if budget is not None else ImageBudget()
    hashes, skipped = {}, 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            continue
        try:
            hashes[p.relative_to(root).as_posix()] = dhash(p, budget=budget)
        except ImageRefused as exc:
            budget.refuse(exc.reason)
        except Exception:  # noqa: BLE001 - a decoder must never crash the audit
            skipped += 1
    return hashes, skipped


def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + (
        (x >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    x = (x + (x >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (x * np.uint64(0x0101010101010101)) >> np.uint64(56)


def nearest(train: dict, test: dict, chunk=64):
    """For each test image, the closest train image and its Hamming distance."""
    if not train or not test:
        return []
    names = list(train)
    tr = np.array([train[n] for n in names], dtype=np.uint64)
    te_names = list(test)
    te = np.array([test[n] for n in te_names], dtype=np.uint64)
    out = []
    for i in range(0, len(te), chunk):
        d = _popcount(te[i : i + chunk, None] ^ tr[None, :])
        j = d.argmin(axis=1)
        for k, (jj, dist) in enumerate(zip(j, d[np.arange(len(j)), j])):
            out.append(
                {"test": te_names[i + k], "train": names[jj], "distance": int(dist)}
            )
    return out


def image_overlap(train_dir, test_dir, max_distance=6, *, budget=None):
    """Near-duplicate images across two split folders, by dHash Hamming distance.

    Returns (findings, pairs, stats); pairs are the matches sorted by distance.
    One pixel budget is shared by both folders unless a caller supplies one.
    """
    budget = budget if budget is not None else ImageBudget()
    tr, s1 = hash_dir(train_dir, budget=budget)
    te, s2 = hash_dir(test_dir, budget=budget)
    pairs = sorted(
        (m for m in nearest(tr, te) if m["distance"] <= max_distance),
        key=lambda m: (m["distance"], m["test"]),
    )
    stats = {
        "train_images": len(tr),
        "test_images": len(te),
        "unreadable": s1 + s2,
        "refused": dict(sorted(budget.refused.items())),
        "decoded_pixels": budget.pixels_used,
        "near_duplicates": len(pairs),
    }
    fs = []
    if pairs:
        n = len(pairs)
        fs.append(
            Finding(
                "high",
                "image_overlap",
                "*",
                f"{n} test images ({n / len(te):.1%}) near-duplicate a train image (dHash distance <= {max_distance})",
                evidence=(
                    Evidence(
                        source="image",
                        kind="duplicate",
                        description=(
                            "near-identical image pairs appear in both the train and "
                            "test splits"
                        ),
                        observed=(
                            f"{n} test images within dHash distance {max_distance}"
                        ),
                        expected="0 image pairs within the distance threshold",
                    ),
                ),
            )
        )
    return fs, pairs, stats
