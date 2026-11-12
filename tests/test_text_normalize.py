"""Text normalisation and intake: Unicode, limits, malformed rows, boilerplate."""

from __future__ import annotations

import json

import pytest

from leaklens.analysis.text.normalize import (
    DEFAULT_MIN_CHARS,
    documents_from_texts,
    load_documents,
    normalize_text,
    text_digest,
)


def test_normalisation_folds_unicode_case_and_whitespace():
    assert normalize_text("  Hello\u00a0  WORLD\u3000!  ") == "hello world !"
    assert normalize_text("Stra\u00dfe") == "strasse"
    assert normalize_text("\uff21\uff22\uff23") == "abc"
    assert normalize_text("\n\t mixed \r\n lines \n") == "mixed lines"


def test_digest_is_normalisation_insensitive_and_stable():
    assert text_digest(" Hello  World ") == text_digest("hello world")
    assert text_digest("hello world") == text_digest("hello world")
    assert text_digest("hello world") != text_digest("hello worlds")
    assert len(text_digest("x")) == 16


def test_documents_keep_original_index_and_count_skips():
    docs, skipped = documents_from_texts(
        ["a" * 40, "short", "", None, "a" * 40], "train", min_chars=32
    )
    assert [doc.id for doc in docs] == ["train:0", "train:4"]
    assert [doc.split for doc in docs] == ["train", "train"]
    assert docs[0].digest == docs[1].digest
    assert skipped == 3


def test_min_chars_is_documented_and_configurable():
    assert DEFAULT_MIN_CHARS == 32
    docs, skipped = documents_from_texts(["abcd"], "test", min_chars=4)
    assert len(docs) == 1 and skipped == 0


def test_max_docs_and_max_chars_are_enforced():
    docs, skipped = documents_from_texts(
        ["a" * 40, "b" * 40, "c" * 40], "train", min_chars=10, max_docs=2
    )
    assert len(docs) == 2 and skipped == 1
    docs, skipped = documents_from_texts(
        ["x" * 50], "train", min_chars=10, max_chars=40
    )
    assert docs == [] and skipped == 1


def test_boilerplate_lines_are_removed_before_hashing():
    text = "Copyright 2026 Example Corp\n" + "real content " * 10
    docs, _ = documents_from_texts(
        [text],
        "train",
        min_chars=10,
        boilerplate=frozenset({"copyright 2026 example corp"}),
    )
    assert docs[0].text == ("real content " * 10).strip()
    plain, _ = documents_from_texts([text], "train", min_chars=10)
    assert docs[0].digest != plain[0].digest


def test_jsonl_field_extraction_and_malformed_lines(tmp_path):
    path = tmp_path / "train.jsonl"
    long = "a" * 40
    path.write_text(
        "\n".join(
            [
                json.dumps({"text": long}),
                "{not json",
                json.dumps({"other": long}),
                json.dumps({"text": "short"}),
                json.dumps({"text": "b" * 40}),
            ]
        ),
        encoding="utf-8",
    )
    docs, skipped = load_documents(path, "train", field="text", min_chars=20)
    assert [doc.id for doc in docs] == ["train:0", "train:4"]
    assert skipped == 3


def test_json_list_and_plain_text_files(tmp_path):
    json_path = tmp_path / "test.json"
    json_path.write_text(json.dumps(["a" * 40, {"text": "b" * 40}]), encoding="utf-8")
    docs, _ = load_documents(json_path, "test", field="text", min_chars=10)
    assert [doc.id for doc in docs] == ["test:0", "test:1"]

    text_path = tmp_path / "corpus.txt"
    text_path.write_text("a" * 40, encoding="utf-8")
    docs, _ = load_documents(text_path, "train", min_chars=10)
    assert [doc.id for doc in docs] == ["train:0"]


def test_encoding_errors_are_replaced_not_raised(tmp_path):
    path = tmp_path / "train.txt"
    path.write_bytes(b"\xff\xfe" + b"a" * 40)
    docs, _ = load_documents(path, "train", min_chars=10)
    assert len(docs) == 1


def test_a_file_with_no_readable_rows_is_an_error(tmp_path):
    path = tmp_path / "train.jsonl"
    path.write_text("{not json\n", encoding="utf-8")
    with pytest.raises(ValueError, match="readable"):
        load_documents(path, "train", field="text", min_chars=1)


def test_unsupported_format_is_an_error(tmp_path):
    path = tmp_path / "train.bin"
    path.write_bytes(b"abc")
    with pytest.raises(ValueError, match="unsupported"):
        load_documents(path, "train", min_chars=1)
