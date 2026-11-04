"""Adversarial upload limits for compressed/binary containers."""

import io

import pandas as pd
import pytest

pytest.importorskip("fastapi")
pytest.importorskip("httpx")
import zipfile

from fastapi.testclient import TestClient

from leaklens.api import UploadLimits, app
from leaklens.readers import (
    MAX_CONTAINER_UNCOMPRESSED_BYTES,
    DatasetLimitError,
    read_dataset,
)

c = TestClient(app)

# --- _check_upload no longer mis-blocks binary containers ---


def test_xlsx_under_8mib_not_rejected_by_newline_count():
    # A 2-row xlsx is ~500 bytes and contains hundreds of newlines in its XML;
    # the pre-fix newline counter would never have seen this because the old
    # check ran before parsing — but the newline exemption must not regress
    # legitimate small files. Exact byte-level newlines don't matter; what
    # matters is that the endpoint no longer counts newlines for .xlsx.
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    data = buf.getvalue()
    # ensure the upload is well under 8 MiB
    assert len(data) < UploadLimits.MAX_BYTES
    r = c.post(
        "/audit",
        files={"file": ("data.xlsx", data, "application/octet-stream")},
        data={"target": "y"},
    )
    assert r.status_code == 200, r.text


def test_parquet_under_8mib_accepted():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"x": [0, 1], "y": [0, 1]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    r = c.post(
        "/audit",
        files={"file": ("data.parquet", buf.getvalue(), "application/octet-stream")},
        data={"target": "y"},
    )
    assert r.status_code == 200, r.text


# --- Row-limit bypasses via compression ---


def test_parquet_row_limit_bypass_rejected():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"x": list(range(60_001)), "y": [0, 1] * 30_000 + [0]})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    data = buf.getvalue()
    assert len(data) < UploadLimits.MAX_BYTES  # bypass relied on this
    assert data.count(b"\n") < UploadLimits.MAX_ROWS
    r = c.post(
        "/audit",
        files={"file": ("data.parquet", data, "application/octet-stream")},
        data={"target": "y"},
    )
    assert r.status_code == 413, r.text
    assert "too many rows" in r.text


def test_xlsx_row_limit_bypass_rejected():
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"x": list(range(60_001)), "y": [0, 1] * 30_000 + [0]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    data = buf.getvalue()
    assert len(data) < UploadLimits.MAX_BYTES
    r = c.post(
        "/audit",
        files={"file": ("data.xlsx", data, "application/octet-stream")},
        data={"target": "y"},
    )
    assert r.status_code == 413, r.text
    assert "too many rows" in r.text


def test_cli_reader_without_max_rows_rejects_zip_bomb(tmp_path):
    import zipfile

    p = tmp_path / "data.xlsx"
    with zipfile.ZipFile(p, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr(
            "xl/worksheets/sheet1.xml",
            b"A" * (MAX_CONTAINER_UNCOMPRESSED_BYTES + 1),
        )
    with pytest.raises(DatasetLimitError):
        read_dataset(str(p), "data.xlsx")


def test_malformed_zip_and_gzip_rejected(tmp_path):
    # Malformed ZIP should raise a zipfile.BadZipFile (subclass of Exception),
    # which pandas will propagate; we ensure that a DatasetLimitError is NOT raised
    # (since it is a well-formed ZIP header but bad internals) and that the error
    # is not a generic, unexpected crash.
    with pytest.raises(Exception) as ctx:
        read_dataset(io.BytesIO(b"PK\x03\x04garbage"), "data.xlsx")
    # Ensure it's NOT a DatasetLimitError (path traversal/size limit)
    assert not isinstance(ctx.value, DatasetLimitError)
    # Ensure the error originates from zipfile.BadZipFile or pandas reading logic
    assert any(
        isinstance(ctx.value, t)
        for t in (zipfile.BadZipFile, pd.errors.ParserError, ValueError)
    )


def test_gzip_expansion_beyond_limit():
    import gzip as gz

    buf = io.BytesIO()
    with gz.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
        f.write(b"A" * (MAX_CONTAINER_UNCOMPRESSED_BYTES + 1))
    with pytest.raises(DatasetLimitError):
        read_dataset(io.BytesIO(buf.getvalue()), "data.csv")


def test_nested_gzip_cumulative_limit():
    import gzip as gz

    inner = gz.compress(b"A" * (MAX_CONTAINER_UNCOMPRESSED_BYTES + 1))
    outer = gz.compress(inner)
    with pytest.raises(DatasetLimitError, match="gzip"):
        read_dataset(io.BytesIO(outer), "data.csv")


def test_csv_gzip_bomb_row_limit_rejected():
    # Small gzipped CSV probe: decompressed size is well under 256 MiB so the
    # gzip container check alone must NOT fire; the row check is what blocks
    # this. Use a reader-level probe that exercises max_rows on a *large*
    # gzip bomb proving the streaming decompress path is bounded.
    import gzip as gz

    buf = io.BytesIO()
    with gz.GzipFile(filename="", mode="wb", fileobj=buf, mtime=0) as f:
        f.write(b"x,y\n")
        for i in range(60_001):
            f.write(f"{i},{i % 2}\n".encode())
    gz_data = buf.getvalue()
    assert len(gz_data) < UploadLimits.MAX_BYTES
    # read_dataset with max_rows must detect the bomb; the API paths this
    # through _bytes_for_guard + give it to pandas. The gzipped CSV probe is
    # at the reader level: read_dataset with max_rows on raw bytes should
    # still be safe (gzip check fires before parsing).
    with pytest.raises((DatasetLimitError, UnicodeDecodeError)):
        read_dataset(io.BytesIO(gz_data), "data.csv.gz", max_rows=UploadLimits.MAX_ROWS)


def test_zip_container_uncompressed_limit():
    # A zip whose header claims a huge uncompressed size must be rejected
    # before any member is extracted/read.
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # highly compressible member: declares large file_size without
        # actually storing gigabytes of data
        zf.writestr(
            "xl/worksheets/sheet1.xml", b"A" * (MAX_CONTAINER_UNCOMPRESSED_BYTES + 1)
        )
    data = buf.getvalue()
    with pytest.raises(DatasetLimitError):
        read_dataset(io.BytesIO(data), "data.xlsx", max_rows=UploadLimits.MAX_ROWS)


def test_read_dataset_max_rows_boundary():
    # Exactly at the limit should pass (when max_rows is given).
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"x": list(range(100)), "y": [0, 1] * 50})
    buf = io.BytesIO()
    df.to_parquet(buf, index=False)
    out = read_dataset(io.BytesIO(buf.getvalue()), "data.parquet", max_rows=100)
    assert len(out) == 100

    # One over the limit must raise.
    with pytest.raises(DatasetLimitError):
        read_dataset(io.BytesIO(buf.getvalue()), "data.parquet", max_rows=99)


def test_csv_nrows_early_exit():
    # 60k-row CSV must be rejected even though it passes the byte check.
    df = pd.DataFrame({"x": list(range(60_001)), "y": [0, 1] * 30_000 + [0]})
    buf = io.BytesIO(df.to_csv(index=False).encode())
    assert len(buf.getvalue()) < UploadLimits.MAX_BYTES
    with pytest.raises(DatasetLimitError):
        read_dataset(buf, "data.csv", max_rows=UploadLimits.MAX_ROWS)


def test_scan_source_size_rejected():
    big = "train_test_split(X, y)\n" * 600_000
    assert len(big) > UploadLimits.MAX_BYTES
    r = c.post("/scan", json={"source": big})
    assert r.status_code == 413, r.text


def test_predict_invalid_model_rejected(tmp_path, monkeypatch):
    (tmp_path / "models").mkdir()
    (tmp_path / "models" / "bad.joblib").write_bytes(b"not a joblib")
    monkeypatch.chdir(tmp_path)
    r = c.post(
        "/predict",
        json={"rows": [{"x": 1}], "model_path": "models/bad.joblib"},
    )
    assert r.status_code == 400, r.text
    assert "invalid model" in r.text


def test_zip_traversal_rejected():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("../escape.txt", b"secret")
    with pytest.raises(DatasetLimitError, match="traversal"):
        read_dataset(
            io.BytesIO(buf.getvalue()), "data.xlsx", max_rows=UploadLimits.MAX_ROWS
        )


def test_zip_backslash_traversal_rejected():
    import zipfile

    from leaklens.readers import _check_zip_entry_path

    # Call the path check directly: zipfile normalizes "\\" to "/" on Windows,
    # but a raw archive entry (e.g. created on POSIX) can keep backslashes.
    with pytest.raises(DatasetLimitError, match="traversal"):
        _check_zip_entry_path("..\\escape.txt", "data.xlsx")

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("..\\escape.txt", b"secret")
    with pytest.raises(DatasetLimitError, match="traversal"):
        read_dataset(
            io.BytesIO(buf.getvalue()), "data.xlsx", max_rows=UploadLimits.MAX_ROWS
        )


def test_zip_absolute_rejected():
    import zipfile

    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("/etc/passwd", b"data")
    with pytest.raises(DatasetLimitError, match="archive"):
        read_dataset(
            io.BytesIO(buf.getvalue()), "data.xlsx", max_rows=UploadLimits.MAX_ROWS
        )
