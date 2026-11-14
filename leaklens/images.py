"""Image near-duplicate checks with bounded decode budgets.

Every image is checked against per-file byte, dimension, pixel and frame limits
before its pixels are decoded, and one audit shares a single pixel budget across
both split folders. A refused image is counted with a stable reason instead of
being silently treated as clean or as a generic unreadable file. Pillow's own
decompression-bomb protection stays in place; LeakLens adds a stricter preflight
and turns a bomb warning into a refusal rather than ignoring it.
"""

from __future__ import annotations

import hashlib
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .availability import EngineUnavailable, probe
from .evidence import Evidence
from .leakage import Finding
from .near_duplicate import clusters_from_pairs

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}
IMAGE_FORMATS = frozenset({"PNG", "JPEG", "BMP", "GIF", "WEBP", "TIFF"})

MAX_IMAGE_BYTES = 64 * 1024 * 1024  # an 8k photographic image stays well below
MAX_IMAGE_SIDE = 20_000  # beyond any consumer camera or dataset image
MAX_IMAGE_PIXELS = 64_000_000  # 64 MP; Pillow's own limit is left untouched
MAX_IMAGE_FRAMES = 64  # animation is counted, never fully decoded
MAX_AUDIT_PIXELS = 2_000_000_000  # ~10k images at 200k pixels each
DIGEST_CHUNK = 1024 * 1024
PAIR_CHUNK = 128  # rows per vectorised distance block; keeps peak memory bounded


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


def _scan(folder, budget, *, with_digest):
    """Hash every image under folder; optionally digest the raw bytes too."""
    _pil()
    root = Path(folder)
    budget = budget if budget is not None else ImageBudget()
    hashes, digests, skipped = {}, {}, 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            continue
        relative = p.relative_to(root).as_posix()
        try:
            value = dhash(p, budget=budget)
        except ImageRefused as exc:
            budget.refuse(exc.reason)
            continue
        except Exception:  # noqa: BLE001 - a decoder must never crash the audit
            skipped += 1
            continue
        hashes[relative] = value
        # Digest after decoding so a refused file is never read again.
        if with_digest:
            try:
                digests[relative] = file_digest(p)
            except OSError:  # a file can vanish or fail between decode and read
                del hashes[relative]
                skipped += 1
    return hashes, digests, skipped


def hash_dir(folder, *, budget=None):
    """Hash every image under folder.

    Returns ``({relative path: hash}, unreadable count)``; refusals are counted
    by reason on ``budget.refused`` so they never look like clean images or like
    ordinary unreadable files. Pass a budget to inspect refusals; the default
    budget is not returned.
    """
    hashes, _digests, skipped = _scan(folder, budget, with_digest=False)
    return hashes, skipped


def sign_dir(folder, *, budget=None):
    """Hash and SHA-256 every image under folder.

    Returns ``({relative path: hash}, {relative path: sha256}, unreadable)``.
    Only images that passed every decode budget are digested, so exact
    duplicates are never claimed for a refused file.
    """
    return _scan(folder, budget, with_digest=True)


def file_digest(path, chunk=DIGEST_CHUNK) -> str:
    """SHA-256 of the file bytes; identity is byte equality, not pixels."""
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(chunk), b""):
            digest.update(block)
    return digest.hexdigest()


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


def hamming(left: int, right: int) -> int:
    """Bit distance between two 64-bit difference hashes."""
    return (int(left) ^ int(right)).bit_count()


def _popcount_array(values):
    one = np.uint64(1)
    values = values - ((values >> one) & np.uint64(0x5555555555555555))
    values = (values & np.uint64(0x3333333333333333)) + (
        (values >> np.uint64(2)) & np.uint64(0x3333333333333333)
    )
    values = (values + (values >> np.uint64(4))) & np.uint64(0x0F0F0F0F0F0F0F0F)
    return (values * np.uint64(0x0101010101010101)) >> np.uint64(56)


def pairs_within(signatures: dict, max_distance: int, chunk=PAIR_CHUNK):
    """Yield sorted id pairs whose Hamming distance is at most ``max_distance``.

    Exhaustive and vectorised: a 64-bit hash has no sketch to band, and byte
    banding would still emit a quarter of all pairs as Python tuples. Blocks of
    rows keep peak memory bounded, and pairs are streamed so a repetitive
    collection never materialises its quadratic edge set.
    """
    if not 0 <= max_distance <= 64:
        raise ValueError("max_distance must be between 0 and 64")
    names = sorted(signatures)
    values = np.array([signatures[name] for name in names], dtype=np.uint64)
    for start in range(0, len(names), chunk):
        block = values[start : start + chunk]
        distances = _popcount_array(block[:, None] ^ values[None, :])
        rows, columns = np.nonzero(distances <= max_distance)
        for row, column in zip(rows.tolist(), columns.tolist()):
            left = start + row
            if column > left:  # each unordered pair exactly once
                yield names[left], names[column]


@dataclass(frozen=True, slots=True)
class ImageCluster:
    """One connected component of near-duplicate image relationships."""

    key: str
    members: tuple[str, ...]
    train: int
    test: int
    exact_pairs: int

    def to_dict(self) -> dict:
        return {
            "key": self.key,
            "members": list(self.members),
            "train": self.train,
            "test": self.test,
            "exact_pairs": self.exact_pairs,
        }


@dataclass(frozen=True, slots=True)
class ImageAudit:
    """Pair view (unchanged) plus the exact and clustered views (M2)."""

    findings: list
    pairs: list
    stats: dict
    clusters: tuple[ImageCluster, ...] = ()
    exact_groups: tuple[dict, ...] = ()
    uninformative: int = 0

    @property
    def exact_duplicates(self) -> int:
        """Test images whose bytes also appear in the train folder."""
        return sum(len(group["test"]) for group in self.exact_groups)


def _informative(value: int) -> bool:
    """A uniform image hashes to all zeros or all ones and cannot be clustered."""
    bits = hamming(value, 0)
    return 0 < bits < 64


def _exact_groups(train_digests: dict, test_digests: dict) -> tuple[dict, ...]:
    """Digests present in both splits; byte equality only."""
    by_digest: dict[str, dict[str, list[str]]] = {}
    for relative, digest in train_digests.items():
        by_digest.setdefault(digest, {"train": [], "test": []})["train"].append(
            relative
        )
    for relative, digest in test_digests.items():
        by_digest.setdefault(digest, {"train": [], "test": []})["test"].append(relative)
    return tuple(
        {
            "digest": digest,
            "train": sorted(parts["train"]),
            "test": sorted(parts["test"]),
        }
        for digest, parts in sorted(by_digest.items())
        if parts["train"] and parts["test"]
    )


def cluster_signatures(
    train: dict, test: dict, train_digests: dict, test_digests: dict, max_distance: int
) -> tuple[tuple[ImageCluster, ...], tuple[dict, ...], int]:
    """Exact groups plus connected-component clusters over already-hashed images."""
    combined = {f"train:{name}": value for name, value in train.items()}
    combined.update({f"test:{name}": value for name, value in test.items()})
    informative = {
        name: value for name, value in combined.items() if _informative(value)
    }
    digests = {f"train:{name}": digest for name, digest in train_digests.items()}
    digests.update({f"test:{name}": digest for name, digest in test_digests.items()})

    clusters = []
    for base in clusters_from_pairs(
        informative, pairs_within(informative, max_distance)
    ):
        counts: dict[str, list[int]] = {}
        for member in base.members:
            pair = counts.setdefault(digests[member], [0, 0])
            pair[0 if member.startswith("train:") else 1] += 1
        clusters.append(
            ImageCluster(
                key=base.key,
                members=base.members,
                train=sum(pair[0] for pair in counts.values()),
                test=sum(pair[1] for pair in counts.values()),
                exact_pairs=sum(pair[0] * pair[1] for pair in counts.values()),
            )
        )
    exact = _exact_groups(train_digests, test_digests)
    return tuple(clusters), exact, len(combined) - len(informative)


def _summary_findings(pairs: list, test_count: int, max_distance: int) -> list:
    if not pairs:
        return []
    n = len(pairs)
    return [
        Finding(
            "high",
            "image_overlap",
            "*",
            f"{n} test images ({n / test_count:.1%}) near-duplicate a train image (dHash distance <= {max_distance})",
            evidence=(
                Evidence(
                    source="image",
                    kind="duplicate",
                    description=(
                        "near-identical image pairs appear in both the train and "
                        "test splits"
                    ),
                    observed=(f"{n} test images within dHash distance {max_distance}"),
                    expected="0 image pairs within the distance threshold",
                ),
            ),
        )
    ]


def analyze_images(
    train_dir, test_dir, max_distance=6, *, budget=None, with_clusters=True
) -> ImageAudit:
    """Audit two split folders; ``with_clusters`` adds the M2 grouping views.

    ``image_overlap`` keeps the original cost by disabling the extra digest
    read and the pairwise clustering pass.
    """
    budget = budget if budget is not None else ImageBudget()
    if with_clusters:
        train, train_digests, unreadable_train = sign_dir(train_dir, budget=budget)
        test, test_digests, unreadable_test = sign_dir(test_dir, budget=budget)
    else:
        train, unreadable_train = hash_dir(train_dir, budget=budget)
        test, unreadable_test = hash_dir(test_dir, budget=budget)
        train_digests, test_digests = {}, {}
    pairs = sorted(
        (m for m in nearest(train, test) if m["distance"] <= max_distance),
        key=lambda m: (m["distance"], m["test"]),
    )
    stats = {
        "train_images": len(train),
        "test_images": len(test),
        "unreadable": unreadable_train + unreadable_test,
        "refused": dict(sorted(budget.refused.items())),
        "decoded_pixels": budget.pixels_used,
        "near_duplicates": len(pairs),
    }
    findings = _summary_findings(pairs, len(test), max_distance)
    if not with_clusters:
        return ImageAudit(findings=findings, pairs=pairs, stats=stats)
    clusters, exact, uninformative = cluster_signatures(
        train, test, train_digests, test_digests, max_distance
    )
    return ImageAudit(
        findings=findings,
        pairs=pairs,
        stats=stats,
        clusters=clusters,
        exact_groups=exact,
        uninformative=uninformative,
    )


def image_overlap(train_dir, test_dir, max_distance=6, *, budget=None):
    """Near-duplicate images across two split folders, by dHash Hamming distance.

    Returns (findings, pairs, stats); pairs are the matches sorted by distance.
    One pixel budget is shared by both folders unless a caller supplies one.
    """
    audit = analyze_images(
        train_dir, test_dir, max_distance, budget=budget, with_clusters=False
    )
    return audit.findings, audit.pairs, audit.stats
