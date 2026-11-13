"""safetensors header parsing: tensor index and metadata only (Phase 16 M2).

The format is ``u64 header length | JSON header | raw tensor bytes``. Only the
header is decoded; weight bytes are never mapped, copied or interpreted.
"""

from __future__ import annotations

import hashlib
import json
import struct
from dataclasses import dataclass

from . import ArtifactParseError

MAX_HEADER = 16 * 1024 * 1024
MAX_TENSORS = 100_000
MAX_RECORDED_TENSORS = 32
MAX_METADATA_KEYS = 32
MAX_NAME = 300


@dataclass(frozen=True, slots=True)
class SafetensorsMeta:
    tensor_count: int
    tensors: tuple[str, ...]
    dtypes: dict[str, int]
    metadata: dict[str, str]
    metadata_truncated: bool
    header_digest: str
    data_bytes: int


def _no_duplicates(pairs: list[tuple[str, object]]) -> dict[str, object]:
    out: dict[str, object] = {}
    for key, value in pairs:
        if key in out:
            raise ArtifactParseError(f"duplicate header key {key[:MAX_NAME]!r}")
        out[key] = value
    return out


def _offsets(entry: object) -> tuple[int, int]:
    if not isinstance(entry, dict):
        raise ArtifactParseError("a tensor entry must be an object")
    dtype = entry.get("dtype")
    shape = entry.get("shape")
    offsets = entry.get("data_offsets")
    if not isinstance(dtype, str) or not isinstance(shape, list):
        raise ArtifactParseError("a tensor entry needs a dtype and a shape list")
    if (
        not isinstance(offsets, list)
        or len(offsets) != 2
        or not all(
            isinstance(item, int) and not isinstance(item, bool) for item in offsets
        )
    ):
        raise ArtifactParseError("a tensor entry needs two integer data_offsets")
    start, end = offsets
    if start < 0 or end < start:
        raise ArtifactParseError("a tensor offset range is invalid")
    return start, end


def parse_safetensors(data: bytes) -> SafetensorsMeta:
    """Parse the header, validate offsets and summarize the tensor index."""
    if len(data) < 8:
        raise ArtifactParseError("a safetensors file needs an 8-byte header length")
    (length,) = struct.unpack("<Q", data[:8])
    if length == 0 or length > MAX_HEADER:
        raise ArtifactParseError("the safetensors header length is outside the cap")
    if 8 + length > len(data):
        raise ArtifactParseError("the safetensors header is longer than the file")
    raw = data[8 : 8 + length]
    try:
        header = json.loads(raw.decode("utf-8"), object_pairs_hook=_no_duplicates)
    except (UnicodeDecodeError, json.JSONDecodeError, RecursionError) as error:
        raise ArtifactParseError("the safetensors header is not valid JSON") from error
    if not isinstance(header, dict):
        raise ArtifactParseError("the safetensors header must be a JSON object")
    region = len(data) - 8 - length

    raw_metadata = header.pop("__metadata__", {})
    if not isinstance(raw_metadata, dict):
        raise ArtifactParseError("__metadata__ must be an object")
    metadata: dict[str, str] = {}
    truncated = False
    for key in sorted(raw_metadata)[:MAX_METADATA_KEYS]:
        value = raw_metadata[key]
        if not isinstance(value, str):
            raise ArtifactParseError("safetensors metadata values must be strings")
        text = value[:MAX_NAME]
        truncated = truncated or len(text) != len(value)
        metadata[key[:MAX_NAME]] = text
    truncated = truncated or len(raw_metadata) > MAX_METADATA_KEYS

    if len(header) > MAX_TENSORS:
        raise ArtifactParseError("the tensor index exceeds the tensor cap")
    intervals: list[tuple[int, int]] = []
    dtypes: dict[str, int] = {}
    for name in sorted(header):
        entry = header[name]
        start, end = _offsets(entry)
        if end > region:
            raise ArtifactParseError("a tensor offset range is past the data region")
        intervals.append((start, end))
        dtype = str(entry["dtype"])
        dtypes[dtype] = dtypes.get(dtype, 0) + 1
    previous_end = 0
    for start, end in sorted(intervals):
        if start < previous_end:
            raise ArtifactParseError("tensor offset ranges overlap")
        previous_end = max(previous_end, end)

    names = tuple(name[:MAX_NAME] for name in sorted(header)[:MAX_RECORDED_TENSORS])
    return SafetensorsMeta(
        tensor_count=len(header),
        tensors=names,
        dtypes=dict(sorted(dtypes.items())),
        metadata=metadata,
        metadata_truncated=truncated,
        header_digest=hashlib.sha256(raw).hexdigest(),
        data_bytes=region,
    )
