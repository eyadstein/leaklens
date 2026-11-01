from __future__ import annotations

from pathlib import Path

import numpy as np

from .leakage import Finding

IMAGE_SUFFIXES = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".webp", ".tif", ".tiff"}


def _pil():
    try:
        from PIL import Image
    except ImportError:
        raise ImportError('image checks need Pillow: pip install "leaklens-ml[images]"') from None
    return Image


def dhash(path, size=8) -> int:
    """Difference hash: size*size bits, set where a pixel is brighter than its right neighbour."""
    Image = _pil()
    with Image.open(path) as im:
        g = np.asarray(im.convert("L").resize((size + 1, size), Image.Resampling.LANCZOS), dtype=np.int16)
    bits = (g[:, 1:] > g[:, :-1]).ravel()
    return int("".join("1" if b else "0" for b in bits), 2)


def hash_dir(folder):
    """Hash every image under folder. Returns ({relative path: hash}, unreadable file count)."""
    Image = _pil()
    root = Path(folder)
    hashes, skipped = {}, 0
    for p in sorted(root.rglob("*")):
        if p.suffix.lower() not in IMAGE_SUFFIXES or not p.is_file():
            continue
        try:
            hashes[p.relative_to(root).as_posix()] = dhash(p)
        except (OSError, Image.DecompressionBombError):
            skipped += 1
    return hashes, skipped


def _popcount(x):
    x = x - ((x >> np.uint64(1)) & np.uint64(0x5555555555555555))
    x = (x & np.uint64(0x3333333333333333)) + ((x >> np.uint64(2)) & np.uint64(0x3333333333333333))
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
            out.append({"test": te_names[i + k], "train": names[jj], "distance": int(dist)})
    return out


def image_overlap(train_dir, test_dir, max_distance=6):
    """Near-duplicate images across two split folders, by dHash Hamming distance.

    Returns (findings, pairs, stats); pairs are the matches sorted by distance.
    """
    tr, s1 = hash_dir(train_dir)
    te, s2 = hash_dir(test_dir)
    pairs = sorted((m for m in nearest(tr, te) if m["distance"] <= max_distance), key=lambda m: (m["distance"], m["test"]))
    stats = {"train_images": len(tr), "test_images": len(te), "unreadable": s1 + s2, "near_duplicates": len(pairs)}
    fs = []
    if pairs:
        n = len(pairs)
        fs.append(Finding("high", "image_overlap", "*", f"{n} test images ({n / len(te):.1%}) near-duplicate a train image (dHash distance <= {max_distance})"))
    return fs, pairs, stats
