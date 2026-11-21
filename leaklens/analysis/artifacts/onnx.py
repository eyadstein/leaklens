"""ONNX metadata via a bounded protobuf reader; no runtime, no protobuf dep (M6).

Only the fields LeakLens reports are decoded: producer, graph name, opset
imports, initializer names and graph inputs/outputs. Every read is length- and
depth-checked against the buffer, so a malformed model is a parse error.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from . import ArtifactParseError

MAX_ONNX_BYTES = 64 * 1024 * 1024
MAX_PROTO_FIELDS = 200_000
MAX_PROTO_DEPTH = 16
MAX_RECORDED_NAMES = 32
MAX_NAME = 300

_WIRE_VARINT = 0
_WIRE_64BIT = 1
_WIRE_LENGTH = 2
_WIRE_32BIT = 5


@dataclass(frozen=True, slots=True)
class OnnxMeta:
    producer: str
    graph_name: str
    opset: tuple[int, ...]
    inputs: tuple[str, ...]
    outputs: tuple[str, ...]
    initializer_count: int
    initializers: tuple[str, ...]
    digest: str
    external_data_count: int = 0
    external_data: tuple[str, ...] = ()


def _varint(data: bytes, pos: int, end: int) -> tuple[int, int]:
    value = 0
    for shift in range(0, 70, 7):
        if pos >= end:
            raise ArtifactParseError("a protobuf varint runs past the buffer")
        byte = data[pos]
        pos += 1
        value |= (byte & 0x7F) << shift
        if not byte & 0x80:
            return value, pos
    raise ArtifactParseError("a protobuf varint is longer than ten bytes")


def _read_fields(
    data: bytes, start: int, end: int, *, depth: int
) -> list[tuple[int, int, bytes | int]]:
    """Decode one message body into ``(field number, wire type, payload)``."""
    if depth > MAX_PROTO_DEPTH:
        raise ArtifactParseError("the ONNX message nesting exceeds the depth cap")
    fields: list[tuple[int, int, bytes | int]] = []
    pos = start
    while pos < end:
        key, pos = _varint(data, pos, end)
        number, wire = key >> 3, key & 0x07
        if number == 0:
            raise ArtifactParseError("a protobuf field number must be positive")
        if wire == _WIRE_VARINT:
            payload, pos = _varint(data, pos, end)
        elif wire == _WIRE_LENGTH:
            size, pos = _varint(data, pos, end)
            if pos + size > end:
                raise ArtifactParseError("a protobuf length runs past the buffer")
            payload = data[pos : pos + size]
            pos += size
        elif wire == _WIRE_64BIT:
            if pos + 8 > end:
                raise ArtifactParseError("a protobuf fixed64 runs past the buffer")
            payload = data[pos : pos + 8]
            pos += 8
        elif wire == _WIRE_32BIT:
            if pos + 4 > end:
                raise ArtifactParseError("a protobuf fixed32 runs past the buffer")
            payload = data[pos : pos + 4]
            pos += 4
        else:
            raise ArtifactParseError("a protobuf wire type is not supported")
        fields.append((number, wire, payload))
        if len(fields) > MAX_PROTO_FIELDS:
            raise ArtifactParseError("the protobuf message exceeds the field cap")
    if pos != end:
        raise ArtifactParseError("the protobuf message did not end on a field boundary")
    return fields


def _body(payload: bytes | int) -> bytes:
    if not isinstance(payload, bytes):
        raise ArtifactParseError("an expected protobuf message was a varint")
    return payload


def _string(payload: bytes | int) -> str:
    try:
        return _body(payload).decode("utf-8")[:MAX_NAME]
    except UnicodeDecodeError as error:
        raise ArtifactParseError("a protobuf string is not UTF-8") from error


def _first_string(body: bytes, field: int, depth: int) -> str:
    """The first length-delimited string at ``field`` inside ``body``."""
    for number, wire, value in _read_fields(body, 0, len(body), depth=depth):
        if number == field and wire == _WIRE_LENGTH:
            return _string(value)
    return ""


def _external_data(tensor: bytes) -> tuple[bool, str]:
    """``(weights live outside the file, first declared location)``.

    Only the declaration is read: the referenced shard is never opened,
    resolved or hashed. A TensorProto that cannot be decoded is not treated as
    external, so a malformed model stays a parse error at the caller.
    """
    external = False
    location = ""
    for number, wire, payload in _read_fields(tensor, 0, len(tensor), depth=3):
        if number == 14 and wire == _WIRE_VARINT:  # TensorProto.data_location
            external = external or int(payload) == 1  # 1 == EXTERNAL
        elif number == 13 and wire == _WIRE_LENGTH:  # TensorProto.external_data
            external = True
            try:
                entry = _body(payload)
                key = _first_string(entry, 1, 4)
                value = _first_string(entry, 2, 4)
            except ArtifactParseError:
                # A broken entry still proves the tensor is external; only the
                # location string is lost.
                continue
            if key == "location" and not location:
                location = value
    return external, location


def parse_onnx(data: bytes, *, max_bytes: int = MAX_ONNX_BYTES) -> OnnxMeta:
    """Decode the reported ModelProto fields from ``data`` within the caps."""
    if len(data) > max_bytes:
        raise ArtifactParseError("the ONNX model exceeds the byte cap")
    if not data:
        raise ArtifactParseError("an ONNX model is empty")
    producer = ""
    graph_name = ""
    opset: list[int] = []
    inputs: list[str] = []
    outputs: list[str] = []
    initializers: list[str] = []
    initializer_count = 0
    external_data_count = 0
    external_data: list[str] = []
    for number, wire, payload in _read_fields(data, 0, len(data), depth=0):
        if wire != _WIRE_LENGTH:
            continue
        body = _body(payload)
        if number == 2:  # ModelProto.producer_name
            producer = _string(payload)
        elif number == 7:  # ModelProto.graph
            for inner, inner_wire, value in _read_fields(body, 0, len(body), depth=1):
                if inner_wire != _WIRE_LENGTH:
                    continue
                if inner == 2:  # GraphProto.name
                    graph_name = _string(value)
                elif inner == 5:  # GraphProto.initializer
                    initializer_count += 1
                    if len(initializers) < MAX_RECORDED_NAMES:
                        initializers.append(_first_string(_body(value), 8, 2))
                    try:
                        external, location = _external_data(_body(value))
                    except ArtifactParseError:
                        # A malformed TensorProto stays in the initializer count
                        # without claiming anything about its storage.
                        external, location = False, ""
                    if external:
                        external_data_count += 1
                        if location and len(external_data) < MAX_RECORDED_NAMES:
                            external_data.append(location)
                elif inner in (11, 12):  # GraphProto.input / output
                    name = _first_string(_body(value), 1, 2)
                    if name:
                        (inputs if inner == 11 else outputs).append(name)
        elif number == 8:  # ModelProto.opset_import
            for inner, inner_wire, value in _read_fields(body, 0, len(body), depth=1):
                if inner == 2 and inner_wire == _WIRE_VARINT:
                    opset.append(int(value))
    return OnnxMeta(
        producer=producer,
        graph_name=graph_name,
        opset=tuple(sorted(opset)),
        inputs=tuple(inputs[:MAX_RECORDED_NAMES]),
        outputs=tuple(outputs[:MAX_RECORDED_NAMES]),
        initializer_count=initializer_count,
        initializers=tuple(initializers),
        digest=hashlib.sha256(data).hexdigest(),
        external_data_count=external_data_count,
        external_data=tuple(external_data),
    )
