"""Every documented JSON surface validates against the packaged report-1 schema.

``tests/test_report_schema.py`` checks hand-written documents and
``tests/test_public_api.py`` checks the two HTTP endpoints. These tests run the
real CLI surfaces and validate the payload they print, so a producer that drifts
from the schema fails here instead of in a consumer.
"""

from __future__ import annotations

import json

import pandas as pd
import pytest
from jsonschema import Draft202012Validator

from leaklens.cli import main
from leaklens.report_schema import report_schema


def validate(payload: dict) -> None:
    schema = report_schema()
    assert payload["$schema"] == schema["$id"]
    Draft202012Validator(schema).validate(payload)


def printed(capsys) -> dict:
    return json.loads(capsys.readouterr().out)


def test_the_data_audit_json_validates(tmp_path, capsys):
    frame = tmp_path / "data.csv"
    pd.DataFrame({"y": [0, 1] * 10, "leak": [0, 1] * 10}).to_csv(frame, index=False)

    assert main([str(frame), "--target", "y", "--json"]) in (0, 1)
    payload = printed(capsys)
    validate(payload)
    assert payload["findings"]


def test_the_text_json_validates(tmp_path, capsys):
    train = tmp_path / "train.jsonl"
    test = tmp_path / "test.jsonl"
    body = "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu"
    train.write_text(json.dumps({"text": body}) + "\n", encoding="utf-8")
    test.write_text(json.dumps({"text": body}) + "\n", encoding="utf-8")

    assert main(["text", str(train), str(test), "--json", "--min-chars", "10"]) == 0
    payload = printed(capsys)
    validate(payload)
    assert payload["skipped"] == {"train": 0, "test": 0}


def test_the_dl_json_validates(tmp_path, monkeypatch, capsys):
    project = tmp_path / "project"
    project.mkdir()
    (project / "train.py").write_text("model.fit(X_train)\n", encoding="utf-8")
    monkeypatch.chdir(tmp_path)

    assert main(["dl", "project", "--json"]) == 0
    validate(printed(capsys))


def test_the_image_json_validates(tmp_path, monkeypatch, capsys):
    pytest.importorskip("PIL", reason="the images surface needs the images extra")
    from PIL import Image

    for split, name in (("train", "a.png"), ("test", "b.png")):
        (tmp_path / split).mkdir()
        Image.new("RGB", (16, 16), (10, 20, 30)).save(tmp_path / split / name)
    monkeypatch.chdir(tmp_path)

    assert main(["images", "train", "test", "--json"]) in (0, 1)
    payload = printed(capsys)
    validate(payload)
    assert payload["pairs"]


def test_the_splits_json_validates(tmp_path, capsys):
    dataset = tmp_path / "coco"
    for split, name in (("train", "a.jpg"), ("val", "b.jpg")):
        (dataset / "images" / split).mkdir(parents=True)
        (dataset / "images" / split / name).write_bytes(b"same bytes in both splits")
        declaration = {
            "images": [{"id": 1, "file_name": f"{split}/{name}"}],
            "annotations": [],
            "categories": [],
        }
        annotations = dataset / "annotations"
        annotations.mkdir(exist_ok=True)
        (annotations / f"instances_{split}.json").write_text(
            json.dumps(declaration), encoding="utf-8"
        )

    assert main(["splits", str(dataset), "--json"]) == 1
    payload = printed(capsys)
    validate(payload)
    assert payload["findings"][0]["kind"] == "image_split_overlap"
