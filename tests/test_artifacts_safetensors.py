"""Phase 16 M2: safetensors header parsing without execution."""

from __future__ import annotations

import json
import struct

import pytest

from leaklens.analysis.artifacts import ArtifactParseError
from leaklens.analysis.artifacts.safetensors import parse_safetensors


def _header(payload: dict) -> bytes:
    raw = json.dumps(payload).encode("utf-8")
    return struct.pack("<Q", len(raw)) + raw


def _tensor(offsets=(0, 16), dtype="F32", shape=(2, 2)) -> dict:
    return {"dtype": dtype, "shape": list(shape), "data_offsets": list(offsets)}


def test_a_valid_header_yields_tensors_and_metadata():
    data = (
        _header({"w": _tensor(), "__metadata__": {"format": "pt", "model": "tiny"}})
        + b"\x00" * 16
    )
    meta = parse_safetensors(data)
    assert meta.tensor_count == 1
    assert meta.tensors == ("w",)
    assert meta.metadata["format"] == "pt"
    assert len(meta.header_digest) == 64


def test_an_empty_file_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        parse_safetensors(b"")


def test_a_truncated_header_is_a_parse_error():
    data = _header({"w": _tensor()}) + b"\x00" * 16
    with pytest.raises(ArtifactParseError):
        parse_safetensors(data[:10])


def test_a_header_longer_than_the_file_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        parse_safetensors(struct.pack("<Q", 999) + b"{}")


def test_a_header_over_the_cap_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        parse_safetensors(struct.pack("<Q", 64 * 1024 * 1024))


def test_invalid_json_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        parse_safetensors(struct.pack("<Q", 3) + b"{no" + b"\x00" * 4)


def test_non_utf8_header_is_a_parse_error():
    raw = b'{"w": "\xff"}'
    with pytest.raises(ArtifactParseError):
        parse_safetensors(struct.pack("<Q", len(raw)) + raw)


def test_offsets_beyond_the_data_region_are_a_parse_error():
    data = _header({"w": _tensor((0, 64))}) + b"\x00" * 16
    with pytest.raises(ArtifactParseError):
        parse_safetensors(data)


def test_negative_and_overlapping_offsets_are_a_parse_error():
    with pytest.raises(ArtifactParseError):
        parse_safetensors(_header({"w": _tensor((-1, 4))}) + b"\x00" * 16)
    with pytest.raises(ArtifactParseError):
        parse_safetensors(
            _header({"a": _tensor((0, 8)), "b": _tensor((4, 12))}) + b"\x00" * 16
        )


def test_duplicate_json_keys_are_a_parse_error():
    raw = b'{"w": {"dtype": "F32", "shape": [1], "data_offsets": [0, 4]}, "w": {}}'
    with pytest.raises(ArtifactParseError):
        parse_safetensors(struct.pack("<Q", len(raw)) + raw + b"\x00" * 4)


def test_metadata_values_are_bounded_and_recorded_as_truncated():
    data = (
        _header({"w": _tensor(), "__metadata__": {"config": "x" * 10_000}})
        + b"\x00" * 16
    )
    meta = parse_safetensors(data)
    assert len(meta.metadata["config"]) == 300
    assert meta.metadata_truncated is True


def test_the_recorded_tensor_list_is_bounded():
    tensors = {f"t{index}": _tensor((index, index + 1)) for index in range(40)}
    data = _header(tensors) + b"\x00" * 40
    meta = parse_safetensors(data)
    assert meta.tensor_count == 40
    assert len(meta.tensors) == 32
    assert meta.tensors[0] == "t0"
