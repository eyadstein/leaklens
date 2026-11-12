"""Serialized-artifact refusal: pickle and archive magic, never a load (M2)."""

from __future__ import annotations

import os

UNSAFE_EXTENSIONS = frozenset(
    {".pkl", ".pickle", ".joblib", ".pt", ".pth", ".bin", ".ckpt"}
)
ARCHIVE_EXTENSIONS = frozenset(
    {
        ".zip",
        ".tar",
        ".gz",
        ".tgz",
        ".bz2",
        ".xz",
        ".7z",
        ".rar",
        ".cab",
        ".iso",
        ".lz4",
        ".zst",
        ".zstd",
        ".jar",
        ".war",
        ".whl",
        ".apk",
        ".xlsx",
        ".xls",
        ".npz",
    }
)
ARCHIVE_MAGICS = (
    b"PK\x03\x04",
    b"PK\x05\x06",
    b"PK\x07\x08",
    b"\x1f\x8b",
    b"BZh",
    b"\xfd7zXZ\x00",
    b"7z\xbc\xaf\x27\x1c",
    b"Rar!\x1a\x07",
    b"MSCF",
    b"CD001",
    b"\x04\x22\x4d\x18",
    b"\x28\xb5\x2f\xfd",
    b"ustar",
)


def looks_like_pickle(head: bytes) -> bool:
    """True for pickle protocol 2-5 magic; protocols 0/1 are text-only."""
    return len(head) >= 2 and head[0] == 0x80 and 2 <= head[1] <= 5


def looks_like_archive(head: bytes) -> bool:
    """True for the archive/compression magics the intake refuses."""
    return any(head.startswith(magic) for magic in ARCHIVE_MAGICS)


def unsafe_reason(name: str, head: bytes) -> str | None:
    """``"pickle"``/``"archive"`` for a refused artifact, else ``None``.

    Magic bytes are checked before the extension, so a pickle renamed to
    ``.safetensors`` or ``.onnx`` is still refused.
    """
    if looks_like_pickle(head):
        return "pickle"
    suffix = os.path.splitext(name)[1].lower()
    if suffix in UNSAFE_EXTENSIONS:  # modern torch.save writes a zip under .pt
        return "pickle"
    if looks_like_archive(head) or suffix in ARCHIVE_EXTENSIONS:
        return "archive"
    return None
