"""Phase 16 M6: ONNX protobuf metadata without an ONNX runtime."""

from __future__ import annotations

import pytest

from leaklens.analysis.artifacts import ArtifactParseError, onnx


def _varint(value: int) -> bytes:
    out = bytearray()
    while True:
        byte = value & 0x7F
        value >>= 7
        if value:
            out.append(byte | 0x80)
        else:
            out.append(byte)
            return bytes(out)


def _field(number: int, wire: int, payload: bytes | int) -> bytes:
    key = _varint((number << 3) | wire)
    if wire == 2:
        return key + _varint(len(payload)) + payload
    if wire == 0:
        return key + _varint(int(payload))
    return key + payload


def _string(number: int, text: str) -> bytes:
    return _field(number, 2, text.encode("utf-8"))


def _tensor(name: str, dims: tuple[int, ...] = (2, 3)) -> bytes:
    packed = b"".join(_varint(dim) for dim in dims)
    return _string(8, name) + _field(1, 2, packed) + _field(2, 0, 1)


def _value_info(name: str) -> bytes:
    return _string(1, name)


def _graph() -> bytes:
    return (
        _string(2, "main_graph")
        + _field(5, 2, _tensor("weight"))
        + _field(11, 2, _value_info("input_ids"))
        + _field(12, 2, _value_info("logits"))
    )


def _model() -> bytes:
    return (
        _string(2, "leaklens-test")
        + _field(7, 2, _graph())
        + _field(8, 2, _field(2, 0, 17))
    )


def test_graph_inputs_outputs_opset_and_initializers_are_parsed():
    meta = onnx.parse_onnx(_model())
    assert meta.producer == "leaklens-test"
    assert meta.opset == (17,)
    assert meta.inputs == ("input_ids",)
    assert meta.outputs == ("logits",)
    assert meta.initializer_count == 1
    assert meta.initializers == ("weight",)
    assert len(meta.digest) == 64


def test_an_empty_or_garbage_model_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(b"")
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(b"not a protobuf model")


def test_a_truncated_model_is_a_parse_error():
    data = _model()
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(data[: len(data) // 2])


def test_a_declared_length_past_the_buffer_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(_field(7, 2, b"\x0a\x64x"))


def test_a_varint_longer_than_ten_bytes_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(b"\x80" * 11)


def test_an_unknown_wire_type_is_a_parse_error():
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(_varint((1 << 3) | 7))


def test_the_size_cap_is_enforced_before_parsing():
    with pytest.raises(ArtifactParseError):
        onnx.parse_onnx(b"\x00" * 9, max_bytes=8)


def test_the_depth_cap_rejects_an_over_deep_message():
    with pytest.raises(ArtifactParseError):
        onnx._read_fields(b"", 0, 0, depth=onnx.MAX_PROTO_DEPTH + 1)


# ---------------------------------------------------------------------------
# External tensor data (#630)
# ---------------------------------------------------------------------------


def _external_tensor(name: str, location: str, *, data_location: int = 1) -> bytes:
    entry = _string(1, "location") + _string(2, location)
    return (
        _string(8, name)
        + _field(2, 0, 1)
        + _field(13, 2, entry)
        + _field(14, 0, data_location)
    )


def _model_with(initializer: bytes) -> bytes:
    graph = (
        _string(2, "main_graph")
        + _field(5, 2, initializer)
        + _field(11, 2, _value_info("input_ids"))
    )
    return (
        _string(2, "leaklens-test")
        + _field(7, 2, graph)
        + _field(8, 2, _field(2, 0, 17))
    )


def test_external_tensor_data_is_reported_without_opening_it():
    meta = onnx.parse_onnx(_model_with(_external_tensor("weight", "weights.bin")))
    assert meta.initializer_count == 1
    assert meta.external_data_count == 1
    assert meta.external_data == ("weights.bin",)


def test_a_model_without_external_data_is_unchanged():
    meta = onnx.parse_onnx(_model())
    assert meta.external_data_count == 0
    assert meta.external_data == ()
    from leaklens.analysis.artifacts import formats

    summary = formats._summary(meta)
    assert summary == {
        "producer": "leaklens-test",
        "graph": "main_graph",
        "opset": [17],
        "inputs": ["input_ids"],
        "outputs": ["logits"],
        "initializer_count": 1,
    }


def test_the_summary_names_the_external_locations():
    from leaklens.analysis.artifacts import formats

    summary = formats._summary(
        onnx.parse_onnx(_model_with(_external_tensor("weight", "weights.bin")))
    )
    assert summary["external_data_count"] == 1
    assert summary["external_data"] == ["weights.bin"]


def test_recorded_external_locations_are_capped():
    tensors = b"".join(
        _field(5, 2, _external_tensor(f"w{index}", f"shard-{index}.bin"))
        for index in range(onnx.MAX_RECORDED_NAMES + 3)
    )
    graph = _string(2, "main_graph") + tensors
    meta = onnx.parse_onnx(_string(2, "p") + _field(7, 2, graph))
    assert meta.external_data_count == onnx.MAX_RECORDED_NAMES + 3
    assert len(meta.external_data) == onnx.MAX_RECORDED_NAMES


def test_a_malformed_external_entry_does_not_crash_the_parser():
    # Field 13 with a body that is not a valid StringStringEntryProto: the tensor
    # is still reported as external, but no location is invented.
    broken = _string(8, "weight") + _field(13, 2, b"\x0a\x7f") + _field(14, 0, 1)
    meta = onnx.parse_onnx(_model_with(broken))
    assert meta.initializer_count == 1
    assert meta.external_data_count == 1
    assert meta.external_data == ()
