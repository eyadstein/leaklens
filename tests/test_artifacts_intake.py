"""Phase 16 M-S1/M2: one typed artifact intake and a hostile corpus."""

from __future__ import annotations

import json
import os
import struct
from pathlib import Path

import pytest

from leaklens.analysis.artifacts import formats

HOSTILE = [
    ("zip-traversal.zip", b"PK\x03\x04../../etc/passwd"),
    ("zip-bomb.zip", b"PK\x03\x04" + b"\x00" * 64),
    ("zip-encrypted.zip", b"PK\x03\x04\x01\x00"),
    ("xlsx-workbook.xlsx", b"PK\x03\x04"),
    ("jar.jar", b"PK\x03\x04"),
    ("tar-traversal.tar", b"ustar\x0000../../etc/passwd"),
    ("gzip-bomb.gz", b"\x1f\x8b\x08\x00" + b"\x00" * 32),
    ("bzip2.bz2", b"BZh9"),
    ("xz.xz", b"\xfd7zXZ\x00"),
    ("seven.7z", b"7z\xbc\xaf\x27\x1c"),
    ("rar.rar", b"Rar!\x1a\x07\x00"),
    ("cab.cab", b"MSCF"),
    ("iso.iso", b"CD001"),
    ("lz4.lz4", b"\x04\x22\x4d\x18"),
    ("zstd.zst", b"\x28\xb5\x2f\xfd"),
    ("pickle-protocol2.pkl", b"\x80\x02payload"),
    ("pickle-protocol4.pickle", b"\x80\x04payload"),
    ("pickle-protocol5.joblib", b"\x80\x05payload"),
    ("torch-zip.pt", b"PK\x03\x04\x14\x00\x00\x00"),
    ("torch-legacy.pth", b"\x80\x02payload"),
    ("weights.bin", b"\x00\x01\x02"),
    ("checkpoint.ckpt", b"\x80\x04payload"),
    ("renamed.safetensors", b"\x80\x04payload"),
    ("renamed.onnx", b"\x80\x05payload"),
    ("no-extension", b"\x80\x04payload"),
    ("truncated.safetensors", b"\xff\xff\xff\xff"),
    ("huge-header.safetensors", struct.pack("<Q", 10**9)),
    ("bad.onnx", b"\xff\xff\xff\xff"),
    ("empty.safetensors", b""),
    ("wrong-magic.json", b"\x00\x01\x02"),
]


def _write(root: Path, name: str, data: bytes) -> Path:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


@pytest.mark.parametrize(("name", "data"), HOSTILE)
def test_a_hostile_artifact_is_refused_with_a_typed_reason_and_no_write(
    tmp_path, name, data
):
    path = _write(tmp_path, name, data)
    before = {item for item in tmp_path.rglob("*")}
    meta, finding, skip = formats.inspect_artifact(tmp_path, path)
    after = {item for item in tmp_path.rglob("*")}
    assert before == after
    assert (finding is not None) or (skip is not None and meta is None)
    assert meta is None
    if skip is not None:
        assert isinstance(skip, str) and skip


def test_a_valid_safetensors_file_yields_metadata_and_no_finding(tmp_path):
    header = json.dumps(
        {"w": {"dtype": "F32", "shape": [2], "data_offsets": [0, 8]}}
    ).encode()
    path = _write(
        tmp_path,
        "model.safetensors",
        struct.pack("<Q", len(header)) + header + b"\x00" * 8,
    )
    meta, finding, skip = formats.inspect_artifact(tmp_path, path)
    assert finding is None
    assert skip is None
    assert meta is not None
    assert meta.format == "safetensors"
    assert meta.summary["tensor_count"] == 1


def test_artifact_json_is_parsed_with_a_digest(tmp_path):
    path = _write(
        tmp_path,
        "model.metadata.json",
        b'{"framework": "torch", "params": 42}',
    )
    meta, finding, skip = formats.inspect_artifact(tmp_path, path)
    assert finding is None
    assert skip is None
    assert meta is not None
    assert meta.format == "json"
    assert len(meta.digest) == 64


def test_duplicate_json_keys_are_refused(tmp_path):
    path = _write(tmp_path, "model.metadata.json", b'{"a": 1, "a": 2}')
    meta, finding, skip = formats.inspect_artifact(tmp_path, path)
    assert meta is None and finding is None
    assert skip == "artifact_unreadable"


def test_a_deeply_nested_artifact_json_is_a_typed_skip(tmp_path):
    path = _write(tmp_path, "model.metadata.json", b'{"a":' * 1500 + b"1" + b"}" * 1500)
    meta, finding, skip = formats.inspect_artifact(tmp_path, path)
    assert meta is None and finding is None
    assert skip == "artifact_unreadable"


def test_an_oversized_artifact_is_skipped(tmp_path):
    path = _write(tmp_path, "model.safetensors", b"\x00" * 32)
    meta, finding, skip = formats.inspect_artifact(tmp_path, path, max_bytes=8)
    assert meta is None and finding is None
    assert skip == "artifact_too_large"


def test_a_symlink_outside_the_root_is_skipped(tmp_path):
    root = tmp_path / "repo"
    root.mkdir()
    outside = tmp_path / "outside.bin"
    outside.write_bytes(b"\x00\x01")
    link = root / "link.bin"
    try:
        os.symlink(outside, link)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available on this platform")
    meta, finding, skip = formats.inspect_artifact(root, link)
    assert meta is None and finding is None
    assert skip == "artifact_outside_root"


def test_a_zip_saved_torch_checkpoint_is_a_finding_not_a_skip(tmp_path):
    path = _write(tmp_path, "model.pt", b"PK\x03\x04\x14\x00\x00\x00")
    _, finding, skip = formats.inspect_artifact(tmp_path, path)
    assert skip is None
    assert finding is not None and finding.kind == "artifact_unsafe_format"


def test_pickle_and_archive_refusals_are_distinct(tmp_path):
    pickle_path = _write(tmp_path, "model.pkl", b"\x80\x04payload")
    _, finding, skip = formats.inspect_artifact(tmp_path, pickle_path)
    assert skip is None
    assert finding is not None
    assert finding.kind == "artifact_unsafe_format"
    assert finding.severity == "high"
    assert finding.diagnostic is True

    archive_path = _write(tmp_path, "model.zip", b"PK\x03\x04")
    _, finding, skip = formats.inspect_artifact(tmp_path, archive_path)
    assert finding is None
    assert skip == "archive_refused"
