"""Containment regression tests for the one shared path helper (M-S0)."""

from __future__ import annotations

import os

import pytest

from leaklens.paths import PathOutsideRoot, contained, resolve_within


def test_root_and_nested_paths_are_inside(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    assert (
        resolve_within(tmp_path, "nested/train.csv") == (nested / "train.csv").resolve()
    )
    assert resolve_within(tmp_path, ".") == tmp_path.resolve()
    assert contained(tmp_path.resolve(), tmp_path.resolve())


def test_empty_path_is_refused(tmp_path):
    with pytest.raises(PathOutsideRoot, match="required"):
        resolve_within(tmp_path, "   ")


def test_traversal_is_refused(tmp_path):
    for value in ("../outside.csv", "..\\outside.csv", "../../outside.csv"):
        with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
            resolve_within(tmp_path, value)


def test_windows_separators_resolve_inside_on_every_platform(tmp_path):
    nested = tmp_path / "nested"
    nested.mkdir()
    assert (
        resolve_within(tmp_path, "nested\\train.csv")
        == (nested / "train.csv").resolve()
    )


def test_absolute_path_outside_is_refused(tmp_path):
    outside = (tmp_path.parent / "outside.csv").resolve()
    with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
        resolve_within(tmp_path, outside.as_posix())


def test_unc_and_device_paths_are_refused_before_resolution(tmp_path):
    # These must not touch the network; the anchor check runs before resolve().
    for value in ("//server/share/x.csv", "//?/C:/x.csv", r"\\server\share\x.csv"):
        with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
            resolve_within(tmp_path, value)


@pytest.mark.skipif(os.name != "nt", reason="Windows drive semantics")
def test_other_drives_and_drive_relative_paths_are_refused(tmp_path):
    assert tmp_path.drive
    other = "Z:/elsewhere/x.csv" if tmp_path.drive.upper() != "Z:" else "Y:/x.csv"
    for value in (other, "Z:x.csv" if tmp_path.drive.upper() != "Z:" else "Y:x.csv"):
        with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
            resolve_within(tmp_path, value)


def test_symlink_escape_is_refused(tmp_path):
    outside = tmp_path.parent / "outside-target.txt"
    outside.write_text("x", encoding="utf-8")
    link = tmp_path / "link.txt"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
        resolve_within(tmp_path, "link.txt")
