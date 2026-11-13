"""CLI surface parity for the text subcommand: JSON, exit codes, flags."""

from __future__ import annotations

import json

import numpy as np
import pytest

from leaklens.cli import main

BASE = (
    "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu xi "
    "omicron pi rho sigma tau upsilon phi chi psi omega"
)
OTHER = "one two three four five six seven eight nine ten eleven twelve thirteen"


def _corpus(tmp_path, train_rows, test_rows, name="data"):
    root = tmp_path / name
    root.mkdir(exist_ok=True)
    train = root / "train.jsonl"
    test = root / "test.jsonl"
    train.write_text(
        "\n".join(json.dumps({"text": row}) for row in train_rows), encoding="utf-8"
    )
    test.write_text(
        "\n".join(json.dumps({"text": row}) for row in test_rows), encoding="utf-8"
    )
    return train, test


def test_text_subcommand_reports_overlap_and_keeps_exit_zero(tmp_path, capsys):
    train, test = _corpus(tmp_path, [BASE, OTHER], [BASE, OTHER + " extra"])
    code = main(["text", str(train), str(test), "--json", "--min-chars", "10"])
    payload = json.loads(capsys.readouterr().out)

    assert code == 0
    assert payload["trust_score"] == 100
    kinds = {finding["kind"] for finding in payload["findings"]}
    assert "text_exact_duplicate" in kinds
    assert payload["documents"] == {"train": 2, "test": 2}
    assert BASE not in json.dumps(payload)


def test_fail_on_findings_is_opt_in(tmp_path, capsys):
    train, test = _corpus(tmp_path, [BASE], [BASE])
    args = ["text", str(train), str(test), "--json", "--min-chars", "10"]
    assert main(args) == 0
    assert main([*args, "--fail-on-findings"]) == 1


def test_clean_corpus_exits_zero_without_findings(tmp_path, capsys):
    train, test = _corpus(tmp_path, [BASE], [OTHER])
    code = main(["text", str(train), str(test), "--json", "--min-chars", "10"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert payload["findings"] == []


def test_write_then_use_eval_fingerprints(tmp_path, capsys):
    train, test = _corpus(tmp_path, [BASE, OTHER], [BASE])
    fingerprints = tmp_path / "eval.fp"
    main(
        [
            "text",
            str(train),
            str(test),
            "--json",
            "--min-chars",
            "10",
            "--write-fingerprints",
            str(fingerprints),
        ]
    )
    capsys.readouterr()
    assert fingerprints.read_text(encoding="utf-8").strip()

    code = main(
        [
            "text",
            str(train),
            str(test),
            "--json",
            "--min-chars",
            "10",
            "--eval-fingerprints",
            str(fingerprints),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(
        finding["kind"] == "text_eval_contamination" for finding in payload["findings"]
    )


def test_rag_flags_need_each_other(tmp_path):
    train, test = _corpus(tmp_path, [BASE], [OTHER])
    with pytest.raises(SystemExit):
        main(["text", str(train), str(test), "--rag", str(tmp_path)])
    with pytest.raises(SystemExit):
        main(["text", str(train), str(test), "--eval-path", "data/eval"])


def test_rag_scan_reports_a_shared_index(tmp_path, capsys):
    train, test = _corpus(tmp_path, [BASE], [OTHER])
    source = tmp_path / "rag" / "index.py"
    source.parent.mkdir()
    source.write_text(
        "from langchain_community.vectorstores import FAISS\n"
        "from langchain_community.document_loaders import DirectoryLoader\n"
        'def build(e):\n    return FAISS.from_documents(DirectoryLoader("data").load(), e)\n',
        encoding="utf-8",
    )
    (tmp_path / "data" / "eval").mkdir()
    code = main(
        [
            "text",
            str(train),
            str(test),
            "--json",
            "--min-chars",
            "10",
            "--rag",
            str(tmp_path),
            "--eval-path",
            "data/eval",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(
        finding["kind"] == "rag_index_leakage" for finding in payload["findings"]
    )


def test_embeddings_are_opt_in_and_need_a_model(tmp_path, capsys):
    train, test = _corpus(tmp_path, [BASE], [BASE])
    train_vectors = tmp_path / "train.npy"
    test_vectors = tmp_path / "test.npy"
    np.save(train_vectors, np.array([[1.0, 0.0]], dtype=np.float32))
    np.save(test_vectors, np.array([[1.0, 0.0]], dtype=np.float32))

    with pytest.raises(SystemExit):
        main(
            [
                "text",
                str(train),
                str(test),
                "--embeddings",
                str(train_vectors),
                str(test_vectors),
            ]
        )

    code = main(
        [
            "text",
            str(train),
            str(test),
            "--json",
            "--min-chars",
            "10",
            "--embeddings",
            str(train_vectors),
            str(test_vectors),
            "--embedding-model",
            "test-model",
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(
        finding["kind"] == "text_embedding_similarity"
        for finding in payload["findings"]
    )
    assert "test-model" in json.dumps(payload)


def test_boilerplate_lines_are_ignored(tmp_path, capsys):
    header = "Copyright 2026 Example Corp all rights reserved"
    train, test = _corpus(tmp_path, [header + "\n" + BASE], [BASE])
    boilerplate = tmp_path / "boilerplate.txt"
    boilerplate.write_text(header, encoding="utf-8")
    code = main(
        [
            "text",
            str(train),
            str(test),
            "--json",
            "--min-chars",
            "10",
            "--boilerplate",
            str(boilerplate),
        ]
    )
    payload = json.loads(capsys.readouterr().out)
    assert code == 0
    assert any(
        finding["kind"] == "text_exact_duplicate" for finding in payload["findings"]
    )


@pytest.mark.parametrize(
    "flags",
    [["--threshold", "1.5"], ["--shingle-size", "0"], ["--min-chars", "-1"]],
)
def test_invalid_numeric_flags_exit_two(tmp_path, flags):
    train, test = _corpus(tmp_path, [BASE], [BASE])
    with pytest.raises(SystemExit):
        main(["text", str(train), str(test), *flags])
