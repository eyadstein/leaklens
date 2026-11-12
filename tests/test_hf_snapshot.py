"""Local HuggingFace snapshot reading without importing ``datasets``."""

from __future__ import annotations

import sys

import pytest

pa = pytest.importorskip("pyarrow")
pq = pytest.importorskip("pyarrow.parquet")
ipc = pytest.importorskip("pyarrow.ipc")

from leaklens.integrations.hf import (  # noqa: E402
    SnapshotUnavailable,
    read_snapshot,
)

TRAIN = ("a" * 40, "b" * 40, "c" * 40)


def _snapshot(tmp_path):
    root = tmp_path / "snapshot"
    (root / "train").mkdir(parents=True)
    (root / "validation").mkdir()
    table = pa.table({"text": list(TRAIN), "label": [0, 1, 0]})
    pq.write_table(table, root / "train" / "data.parquet")
    pq.write_table(pa.table({"text": ["v" * 40]}), root / "validation" / "data.parquet")
    return root


def test_reads_parquet_without_importing_datasets(tmp_path):
    root = _snapshot(tmp_path)
    assert read_snapshot(root, field="text", split="train") == list(TRAIN)
    assert "datasets" not in sys.modules


def test_reads_arrow_stream_files(tmp_path):
    root = tmp_path / "snapshot"
    root.mkdir()
    table = pa.table({"text": list(TRAIN)})
    with (
        pa.OSFile(str(root / "train.arrow"), "wb") as sink,
        ipc.new_stream(sink, table.schema) as writer,
    ):
        writer.write_table(table)
    assert read_snapshot(root, field="text") == list(TRAIN)


def test_split_filter_and_row_cap(tmp_path):
    root = _snapshot(tmp_path)
    assert read_snapshot(root, field="text", split="validation") == ["v" * 40]
    assert read_snapshot(root, field="text", split="train", max_rows=2) == list(
        TRAIN[:2]
    )


def test_field_defaults_to_the_text_column(tmp_path):
    root = _snapshot(tmp_path)
    assert read_snapshot(root, split="train") == list(TRAIN)


def test_missing_field_is_an_error(tmp_path):
    root = _snapshot(tmp_path)
    with pytest.raises(ValueError, match="column"):
        read_snapshot(root, field="missing")


def test_missing_pyarrow_is_a_recorded_unavailable_extra(tmp_path, monkeypatch):
    root = _snapshot(tmp_path)
    monkeypatch.setitem(sys.modules, "pyarrow", None)
    monkeypatch.setitem(sys.modules, "pyarrow.parquet", None)
    with pytest.raises(SnapshotUnavailable, match="parquet"):
        read_snapshot(root, field="text")


def test_an_empty_snapshot_is_an_error(tmp_path):
    root = tmp_path / "empty"
    root.mkdir()
    with pytest.raises(ValueError, match="parquet"):
        read_snapshot(root, field="text")


def test_files_outside_the_snapshot_root_are_skipped(tmp_path):
    root = _snapshot(tmp_path)
    outside = tmp_path / "outside.parquet"
    pq.write_table(pa.table({"text": ["z" * 40]}), outside)
    link = root / "link.parquet"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    assert read_snapshot(root, field="text") == list(TRAIN)
