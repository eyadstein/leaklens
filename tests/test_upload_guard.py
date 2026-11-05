"""Upload validation tests for the public live-scan API."""

from __future__ import annotations

import io
import zipfile

import pandas as pd
import pytest

from leaklens.upload_guard import (
    GuardLimits,
    UploadRejected,
    available_formats,
    describe_dataset,
    guard_upload,
    read_guarded_dataset,
    sanitize_filename,
    validate_dataframe,
)


def limits(**overrides) -> GuardLimits:
    base = {
        "max_bytes": 1024 * 1024,
        "max_rows": 100,
        "max_columns": 20,
        "max_cells": 2000,
        "formats": ("csv", "json", "xlsx", "parquet", "feather"),
    }
    base.update(overrides)
    return GuardLimits(**base)


def csv_bytes(text: str = "a,b\n1,2\n3,4\n") -> bytes:
    return text.encode()


def rejected(
    callable_, status: int | None = None, match: str | None = None
) -> UploadRejected:
    with pytest.raises(UploadRejected) as excinfo:
        callable_()
    if status is not None:
        assert excinfo.value.status_code == status
    if match:
        assert match.lower() in excinfo.value.message.lower()
    return excinfo.value


# --- filenames ------------------------------------------------------------


def test_sanitize_filename_keeps_a_normal_name():
    assert sanitize_filename("my data.csv") == "my data.csv"
    assert sanitize_filename("résumé.csv") == "résumé.csv"


def test_sanitize_filename_rejects_paths_and_nul():
    rejected(lambda: sanitize_filename("../../etc/passwd"), 400)
    rejected(lambda: sanitize_filename("C:\\temp\\data.csv"), 400)
    rejected(lambda: sanitize_filename("data\x00.csv"), 400)


def test_sanitize_filename_removes_control_and_unsafe_characters():
    assert sanitize_filename("a<b>c.csv") == "a_b_c.csv"
    rejected(lambda: sanitize_filename("data\n.csv"), 400)


def test_sanitize_filename_bounds_length_and_keeps_extension():
    name = sanitize_filename("a" * 300 + ".csv", max_chars=40)
    assert len(name) <= 40
    assert name.endswith(".csv")


# --- basic format and content sniffing -----------------------------------


def test_guard_accepts_a_real_csv_and_ignores_mime():
    upload = guard_upload(
        "data.csv", csv_bytes(), limits(), content_type="application/x-evil"
    )
    assert upload.filename == "data.csv"
    assert upload.suffix == ".csv"
    assert upload.content_type == "application/x-evil"


def test_guard_rejects_unsupported_extension():
    rejected(
        lambda: guard_upload("data.txt", b"a,b\n1,2\n", limits()), 415, "unsupported"
    )


def test_guard_rejects_a_disabled_format():
    rejected(
        lambda: guard_upload("data.json", b'[{"a": 1}]', limits(formats=("csv",))),
        415,
        "not enabled",
    )


def test_guard_rejects_empty_and_oversized_payloads():
    rejected(lambda: guard_upload("data.csv", b"", limits()), 422, "empty")
    rejected(
        lambda: guard_upload(
            "data.csv", csv_bytes("a,b\n" + "1,2\n" * 20), limits(max_bytes=10)
        ),
        413,
        "larger",
    )


def test_guard_rejects_binary_content_in_a_csv():
    rejected(
        lambda: guard_upload("data.csv", b"\x00\x01\x02PK", limits()), 415, "binary"
    )
    rejected(
        lambda: guard_upload("data.csv", b"PK\x03\x04" + b"\x00" * 20, limits()), 415
    )


def test_guard_rejects_csv_without_a_header_and_row():
    rejected(lambda: guard_upload("data.csv", b"a,b\n", limits()), 415, "header")
    rejected(
        lambda: guard_upload("data.csv", b"just one column\n1\n2\n", limits()), 415
    )


def test_guard_rejects_content_that_does_not_match_the_extension():
    rejected(lambda: guard_upload("data.json", csv_bytes(), limits()), 415, "json")
    rejected(lambda: guard_upload("data.csv", b'[{"a": 1}]', limits()), 415, "csv")


def test_guard_accepts_record_and_column_json():
    assert guard_upload("a.json", b'[{"a": 1}, {"a": 2}]', limits()).suffix == ".json"
    assert (
        guard_upload("b.json", b'{"a": [1, 2], "b": [3, 4]}', limits()).suffix
        == ".json"
    )


@pytest.mark.parametrize(
    "payload",
    [
        b"5",
        b'"text"',
        b"[]",
        b"[1, 2]",
        b'[{"a": 1}, 2]',
        b'{"a": [1, 2], "b": [3]}',
        b'{"a": 1, "b": 2}',
        b"{not json}",
    ],
)
def test_guard_rejects_invalid_json(payload):
    rejected(lambda: guard_upload("data.json", payload, limits()), 415, "json")


# --- binary containers ----------------------------------------------------


def test_guard_accepts_a_real_xlsx_and_rejects_a_fake_zip():
    pytest.importorskip("openpyxl")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    buf = io.BytesIO()
    df.to_excel(buf, index=False)
    upload = guard_upload("data.xlsx", buf.getvalue(), limits())
    assert upload.suffix == ".xlsx"
    out = read_guarded_dataset(upload, limits(), "b", require_target=True)
    assert len(out) == 2

    fake = io.BytesIO()
    with zipfile.ZipFile(fake, "w") as zf:
        zf.writestr("[Content_Types].xml", b"")
    rejected(lambda: guard_upload("data.xlsx", fake.getvalue(), limits()), 415, "excel")


def test_guard_rejects_xlsx_archive_traversal_and_expansion():
    def archive(entries: dict[str, bytes]) -> bytes:
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
            for name, data in entries.items():
                zf.writestr(name, data)
        return buf.getvalue()

    traversal = archive(
        {
            "[Content_Types].xml": b"",
            "xl/workbook.xml": b"",
            "../escape.xml": b"secret",
        }
    )
    rejected(lambda: guard_upload("data.xlsx", traversal, limits()), 415, "unsafe")

    expanded = archive(
        {
            "[Content_Types].xml": b"",
            "xl/workbook.xml": b"x" * 200,
        }
    )
    rejected(
        lambda: guard_upload("data.xlsx", expanded, limits(max_uncompressed_bytes=100)),
        415,
        "decompression",
    )


def test_guard_accepts_parquet_and_feather_magic():
    pytest.importorskip("pyarrow")
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    for suffix in (".parquet", ".feather"):
        buf = io.BytesIO()
        if suffix == ".feather":
            # pandas forwards extra keywords to pyarrow, which has no index arg.
            df.to_feather(buf)
        else:
            df.to_parquet(buf, index=False)
        upload = guard_upload("data" + suffix, buf.getvalue(), limits())
        out = read_guarded_dataset(upload, limits(), "b", require_target=True)
        assert len(out) == 2


def test_guard_rejects_parquet_and_feather_with_wrong_magic():
    rejected(lambda: guard_upload("data.parquet", b"not parquet at all", limits()), 415)
    rejected(lambda: guard_upload("data.feather", b"not feather at all", limits()), 415)
    rejected(
        lambda: guard_upload("data.csv", b"PAR1" + b"\x00" * 8 + b"PAR1", limits()), 415
    )


# --- parsed frame limits --------------------------------------------------


def test_validate_dataframe_enforces_rows_columns_and_cells():
    rejected(
        lambda: validate_dataframe(pd.DataFrame({"a": range(5)}), limits(max_rows=4)),
        413,
        "rows",
    )
    rejected(
        lambda: validate_dataframe(
            pd.DataFrame({f"c{i}": [1] for i in range(5)}), limits(max_columns=4)
        ),
        413,
        "columns",
    )
    rejected(
        lambda: validate_dataframe(
            pd.DataFrame({f"c{i}": [1] * 4 for i in range(5)}), limits(max_cells=10)
        ),
        413,
        "cell",
    )


def test_validate_dataframe_requires_a_real_target():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    rejected(lambda: validate_dataframe(df, limits(), None, require_target=True), 422)
    rejected(
        lambda: validate_dataframe(df, limits(), "missing", require_target=True),
        422,
        "not found",
    )
    constant = pd.DataFrame({"a": [1, 2, 3], "b": [1, 1, 1]})
    rejected(
        lambda: validate_dataframe(constant, limits(), "b", require_target=True),
        422,
        "two distinct",
    )


def test_validate_dataframe_rejects_bad_column_names_and_duplicates():
    duplicate = pd.DataFrame([[1, 2]], columns=["a", "a"])
    rejected(lambda: validate_dataframe(duplicate, limits()), 422, "duplicate")
    control = pd.DataFrame({("bad\nname"): [1]})
    rejected(lambda: validate_dataframe(control, limits()), 422, "control")
    long_name = pd.DataFrame({("x" * 200): [1]})
    rejected(lambda: validate_dataframe(long_name, limits()), 422, "120")


def test_validate_dataframe_accepts_a_clean_frame():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [0, 1, 0]})
    validate_dataframe(df, limits(), "b", require_target=True)


def test_read_guarded_dataset_reports_limit_and_parse_failures(monkeypatch):
    upload = guard_upload(
        "data.csv", csv_bytes("a,b\n" + "1,2\n" * 20), limits(max_rows=5)
    )
    rejected(lambda: read_guarded_dataset(upload, limits(max_rows=5)), 413, "rows")

    def boom(*_args, **_kwargs):
        raise ValueError("internal parser detail")

    monkeypatch.setattr("leaklens.upload_guard.read_dataset", boom)
    upload = guard_upload("data.csv", csv_bytes(), limits())
    err = rejected(lambda: read_guarded_dataset(upload, limits()), 422, "parsed")
    assert "internal parser detail" not in err.message


def test_describe_dataset_bounds_preview_and_marks_candidates():
    df = pd.DataFrame(
        {
            "id": range(20),
            "target": [0, 1] * 10,
            "text": ["x" * 1000] * 20,
        }
    )
    described = describe_dataset(df, limits())
    assert described["rows"] == 20
    assert len(described["preview"]) == 5
    assert len(described["preview"][0]["text"]) <= 512
    assert "target" in described["target_candidates"]
    assert "id" not in described["target_candidates"]


def test_available_formats_reports_configuration_and_installation():
    formats = available_formats(limits(formats=("csv",)))
    by_extension = {item["extension"]: item for item in formats}
    assert by_extension[".csv"]["enabled"] is True
    assert by_extension[".json"]["configured"] is False
    assert by_extension[".xlsx"]["configured"] is False
