"""Containment regression tests for the one shared path helper (M-S0)."""

from __future__ import annotations

import os
from pathlib import Path

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


def test_a_symlinked_directory_escape_is_refused(tmp_path):
    outside = tmp_path.parent / "outside-dir"
    outside.mkdir(exist_ok=True)
    (outside / "leak.py").write_text("x = 1\n", encoding="utf-8")
    try:
        (tmp_path / "code").symlink_to(outside, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    for value in ("code", "code/leak.py"):
        with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
            resolve_within(tmp_path, value)


def test_a_symlink_loop_never_resolves_outside_the_root(tmp_path):
    try:
        (tmp_path / "a").symlink_to(tmp_path / "b")
        (tmp_path / "b").symlink_to(tmp_path / "a")
    except (OSError, NotImplementedError):
        pytest.skip("symlinks are not available")
    try:
        target = resolve_within(tmp_path, "a")
    except PathOutsideRoot:
        return  # refusing a loop is also safe
    assert contained(tmp_path.resolve(), target)


def test_embedded_nul_is_refused(tmp_path):
    for value in ("in\u0000put.csv", (tmp_path / "in\u0000put.csv").as_posix()):
        with pytest.raises(PathOutsideRoot, match="NUL"):
            resolve_within(tmp_path, value)


def test_a_sibling_sharing_the_root_prefix_is_refused(tmp_path):
    root = tmp_path / "root"
    root.mkdir()
    sibling = tmp_path / "root-other"
    sibling.mkdir()
    (sibling / "x.py").write_text("x = 1\n", encoding="utf-8")
    with pytest.raises(PathOutsideRoot, match="outside the workspace root"):
        resolve_within(root, "../root-other/x.py")


def test_a_nonexistent_target_inside_the_root_is_allowed(tmp_path):
    assert (
        resolve_within(tmp_path, "new/nested.csv")
        == (tmp_path / "new" / "nested.csv").resolve()
    )


def test_an_existing_file_keeps_its_canonical_path(tmp_path):
    target = tmp_path / "data.csv"
    target.write_text("x\n1\n", encoding="utf-8")
    assert resolve_within(tmp_path, "data.csv") == target.resolve()
    assert resolve_within(tmp_path, ".") == tmp_path.resolve()


def test_a_fifo_inside_the_root_is_contained_but_stays_a_fifo(tmp_path):
    if os.name == "nt":
        pytest.skip("POSIX named pipes")
    os.mkfifo(tmp_path / "pipe")
    assert resolve_within(tmp_path, "pipe") == (tmp_path / "pipe").resolve()


@pytest.mark.skipif(os.name == "nt", reason="POSIX root is /")
def test_the_filesystem_root_contains_its_children():
    assert resolve_within(Path("/"), "etc/hostname") == Path("/etc/hostname").resolve()
    assert resolve_within(Path("/"), ".") == Path("/")
