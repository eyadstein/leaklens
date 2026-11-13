"""Phase 16 M2: serialized-artifact refusal without unpickling."""

from __future__ import annotations

import pytest

from leaklens.analysis.artifacts import pickle_guard


@pytest.mark.parametrize(
    "name",
    [
        "model.pkl",
        "model.pickle",
        "model.joblib",
        "model.pt",
        "model.pth",
        "model.bin",
        "model.ckpt",
    ],
)
def test_every_unsafe_extension_is_refused(name):
    assert pickle_guard.unsafe_reason(name, b"") is not None


@pytest.mark.parametrize(
    "protocol", [b"\x80\x02", b"\x80\x03", b"\x80\x04", b"\x80\x05"]
)
def test_pickle_protocol_magic_is_detected(protocol):
    assert pickle_guard.looks_like_pickle(protocol + b"payload")


def test_pickle_protocol_zero_and_one_are_refused_by_extension():
    for name in ("model.pkl", "model.pickle"):
        assert pickle_guard.unsafe_reason(name, b"(dp0\n") == "pickle"


def test_a_pickle_renamed_to_a_safe_extension_is_still_refused():
    assert (
        pickle_guard.unsafe_reason("model.safetensors", b"\x80\x04payload") == "pickle"
    )
    assert pickle_guard.unsafe_reason("model.onnx", b"\x80\x05payload") == "pickle"


def test_archive_magic_is_detected():
    assert pickle_guard.looks_like_archive(b"PK\x03\x04")
    assert pickle_guard.looks_like_archive(b"\x1f\x8b\x08")
    assert pickle_guard.unsafe_reason("model.zip", b"PK\x03\x04") == "archive"


def test_a_valid_safetensors_file_is_not_refused():
    assert pickle_guard.unsafe_reason("model.safetensors", b"\x10\x00\x00\x00") is None
