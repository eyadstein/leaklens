"""Phase 17 M3: declared-split disjointness for COCO, YOLO and VOC layouts."""

from __future__ import annotations

import json
import os

import pytest

from leaklens import cli, image_splits


def _write(path, data=b"image-bytes"):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path


def _coco(root, split, names):
    payload = {
        "images": [{"id": index, "file_name": name} for index, name in enumerate(names)]
    }
    path = root / "annotations" / f"instances_{split}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")
    return path


def _voc(root, lists, images):
    for identifier, data in images.items():
        _write(root / "JPEGImages" / identifier, data)
    main = root / "ImageSets" / "Main"
    main.mkdir(parents=True, exist_ok=True)
    for split, identifiers in lists.items():
        (main / f"{split}.txt").write_text(
            "\n".join(identifiers) + ("\n" if identifiers else ""), encoding="utf-8"
        )


def _tree(root):
    return sorted(path.relative_to(root).as_posix() for path in root.rglob("*"))


# ---------------------------------------------------------------------------
# COCO
# ---------------------------------------------------------------------------


def test_coco_clean_splits_report_counts_and_no_findings(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _write(tmp_path / "images" / "val" / "b.jpg", b"b")
    _write(tmp_path / "images" / "test" / "c.jpg", b"c")
    _coco(tmp_path, "train", ["train/a.jpg"])
    _coco(tmp_path, "val", ["val/b.jpg"])
    _coco(tmp_path, "test", ["test/c.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert audit.ok
    assert audit.stats["format"] == "coco"
    assert audit.findings == []
    assert audit.stats["partitions"]["train"] == {"declared": 1, "unique": 1}
    assert audit.stats["partitions"]["val"] == {"declared": 1, "unique": 1}
    assert audit.stats["verified_images"] == 3


def test_coco_identical_bytes_under_different_names_is_a_high_finding(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"same")
    _write(tmp_path / "images" / "val" / "copy.jpg", b"same")
    _coco(tmp_path, "train", ["train/a.jpg"])
    _coco(tmp_path, "val", ["val/copy.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert len(audit.findings) == 1
    finding = audit.findings[0]
    assert (finding.severity, finding.kind, finding.column) == (
        "high",
        "image_split_overlap",
        "*",
    )
    assert finding.diagnostic is False
    assert audit.stats["cross_split_groups"] == 1
    assert audit.cross_groups[0]["partitions"] == {
        "train": ["images/train/a.jpg"],
        "val": ["images/val/copy.jpg"],
    }


def test_coco_same_digest_in_all_three_partitions_is_one_group(tmp_path):
    for split in ("train", "val", "test"):
        _write(tmp_path / "images" / split / f"{split}.jpg", b"same")
        _coco(tmp_path, split, [f"{split}/{split}.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert len(audit.findings) == 1
    assert list(audit.cross_groups[0]["partitions"]) == ["train", "val", "test"]


def test_coco_within_split_duplicate_is_not_a_cross_split_finding(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"same")
    _write(tmp_path / "images" / "train" / "copy.jpg", b"same")
    _write(tmp_path / "images" / "val" / "b.jpg", b"other")
    _coco(tmp_path, "train", ["train/a.jpg", "train/copy.jpg"])
    _coco(tmp_path, "val", ["val/b.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert audit.findings == []
    assert audit.stats["within_split_groups"] == 1
    assert audit.cross_groups == ()


def test_coco_same_basename_with_different_bytes_is_not_a_finding(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"train-bytes")
    _write(tmp_path / "images" / "val" / "a.jpg", b"val-bytes")
    _coco(tmp_path, "train", ["train/a.jpg"])
    _coco(tmp_path, "val", ["val/a.jpg"])

    assert image_splits.audit_splits(tmp_path).findings == []


def test_coco_image_id_is_not_used_as_content_identity(tmp_path):
    # The same integer id with different file names is not a conflict.
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_train.json").write_text(
        json.dumps({"images": [{"id": 1, "file_name": "train/a.jpg"}]}),
        encoding="utf-8",
    )
    (tmp_path / "annotations" / "instances_val.json").write_text(
        json.dumps({"images": [{"id": 1, "file_name": "val/b.jpg"}]}), encoding="utf-8"
    )
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _write(tmp_path / "images" / "val" / "b.jpg", b"b")

    assert image_splits.audit_splits(tmp_path).findings == []


def test_coco_repeated_declaration_counts_once(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_train.json").write_text(
        json.dumps(
            {
                "images": [
                    {"id": 1, "file_name": "train/a.jpg"},
                    {"id": 2, "file_name": "train/a.jpg"},
                ]
            }
        ),
        encoding="utf-8",
    )

    audit = image_splits.audit_splits(tmp_path)

    assert audit.findings == []
    assert audit.stats["partitions"]["train"] == {"declared": 2, "unique": 1}


def test_coco_missing_and_traversing_paths_are_unresolved(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _coco(tmp_path, "train", ["train/a.jpg", "missing.jpg", "../../outside.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert audit.findings == []
    assert audit.stats["unresolved"] == {"missing_asset": 1, "outside_root": 1}
    assert audit.stats["verified_images"] == 1


def test_coco_invalid_json_invalid_encoding_and_no_split_token(tmp_path):
    annotations = tmp_path / "annotations"
    annotations.mkdir()
    (annotations / "broken_train.json").write_text("{not json", encoding="utf-8")
    (annotations / "val.json").write_bytes(b"\xff\xfe\x00")
    (annotations / "labels.json").write_text(
        json.dumps({"images": []}), encoding="utf-8"
    )
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _coco(tmp_path, "train", ["train/a.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert audit.findings == []
    assert audit.stats["unresolved"]["invalid_json"] == 1
    assert audit.stats["unresolved"]["invalid_encoding"] == 1
    assert any("no train/val/test token" in text for text in audit.stats["limitations"])


def test_coco_non_image_file_name_is_unresolved(tmp_path):
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "instances_train.json").write_text(
        json.dumps({"images": [{"id": 1, "file_name": "train/labels.txt"}]}),
        encoding="utf-8",
    )
    _write(tmp_path / "train" / "labels.txt", b"not an image")

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"unsupported_suffix": 1}
    assert audit.ok is False


def test_coco_oversized_manifest_is_a_limitation(tmp_path, monkeypatch):
    monkeypatch.setattr(image_splits, "MAX_MANIFEST_BYTES", 16)
    (tmp_path / "annotations").mkdir()
    payload = json.dumps({"images": [{"id": 1, "file_name": "a.jpg"}] * 10})
    (tmp_path / "annotations" / "instances_train.json").write_text(
        payload, encoding="utf-8"
    )

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"too_large": 1}
    assert any("too_large" in text for text in audit.stats["limitations"])


def test_declaration_cap_stops_reading_and_says_so(tmp_path, monkeypatch):
    monkeypatch.setattr(image_splits, "MAX_DECLARATIONS", 2)
    for index in range(5):
        _write(tmp_path / "images" / "train" / f"{index}.jpg", bytes([index]))
    _coco(tmp_path, "train", [f"train/{index}.jpg" for index in range(5)])

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["declarations"] == 2
    assert any("declaration cap" in text for text in audit.stats["limitations"])


# ---------------------------------------------------------------------------
# YOLO
# ---------------------------------------------------------------------------


def _yolo(tmp_path, config):
    (tmp_path / "data.yaml").write_text(json.dumps(config), encoding="utf-8")


def test_yolo_directory_declarations(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _write(tmp_path / "images" / "val" / "b.jpg", b"b")
    _yolo(tmp_path, {"train": "images/train", "val": "images/val", "names": {0: "cat"}})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.ok
    assert audit.findings == []
    assert audit.stats["format"] == "yolo"
    assert audit.stats["verified_images"] == 2


def test_yolo_text_list_declarations(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _write(tmp_path / "images" / "val" / "a-copy.jpg", b"a")
    (tmp_path / "lists").mkdir()
    (tmp_path / "lists" / "train.txt").write_text(
        "images/train/a.jpg\n", encoding="utf-8"
    )
    (tmp_path / "lists" / "val.txt").write_text(
        "images/val/a-copy.jpg\n", encoding="utf-8"
    )
    _yolo(tmp_path, {"train": "lists/train.txt", "val": "lists/val.txt"})

    audit = image_splits.audit_splits(tmp_path)

    assert len(audit.findings) == 1
    assert audit.cross_groups[0]["partitions"] == {
        "train": ["images/train/a.jpg"],
        "val": ["images/val/a-copy.jpg"],
    }


def test_yolo_absolute_path_inside_root_is_accepted(tmp_path):
    image = _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _yolo(tmp_path, {"train": str(image.resolve())})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["verified_images"] == 1


def test_yolo_absolute_path_outside_root_is_refused(tmp_path):
    outside = _write(tmp_path.parent / "outside" / "a.jpg", b"a")
    _yolo(tmp_path, {"train": str(outside.resolve())})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"outside_root": 1}
    assert audit.ok is False


def test_yolo_remote_declaration_is_reported_not_fetched(tmp_path):
    _yolo(tmp_path, {"train": "https://example.invalid/train"})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"remote_path": 1}
    assert any("never fetched" in text for text in audit.stats["limitations"])


def test_yolo_unsafe_tag_is_a_typed_limitation_and_never_executed(tmp_path):
    marker = tmp_path / "executed.txt"
    (tmp_path / "data.yaml").write_text(
        f"train: !!python/object/apply:os.system ['echo x > {marker}']\n",
        encoding="utf-8",
    )

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"yaml_error": 1}
    assert not marker.exists()


def test_yolo_unsupported_declaration_forms_are_reported(tmp_path):
    _yolo(tmp_path, {"train": {"unexpected": "mapping"}, "val": [1, 2]})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"]["unsupported_declaration"] == 3
    assert audit.ok is False


def test_yolo_missing_pyyaml_is_an_install_hint(tmp_path, monkeypatch):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _yolo(tmp_path, {"train": "images/train"})
    monkeypatch.setattr(image_splits.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(ImportError, match="leaklens-ml\\[yaml\\]"):
        image_splits.audit_splits(tmp_path)


def test_yolo_no_config_is_unusable(tmp_path):
    audit = image_splits.audit_splits(tmp_path, "yolo")

    assert audit.ok is False
    assert "data.yaml" in audit.stats["reason"]


# ---------------------------------------------------------------------------
# VOC
# ---------------------------------------------------------------------------


def test_voc_clean_lists_are_disjoint(tmp_path):
    _voc(
        tmp_path,
        {"train": ["000001"], "val": ["000002"], "test": ["000003"]},
        {"000001.jpg": b"a", "000002.jpg": b"b", "000003.jpg": b"c"},
    )

    audit = image_splits.audit_splits(tmp_path)

    assert audit.ok
    assert audit.stats["format"] == "voc"
    assert audit.findings == []
    assert audit.stats["verified_images"] == 3


def test_voc_extensionless_identifier_resolves_through_jpegimages(tmp_path):
    _voc(tmp_path, {"train": ["000001"]}, {"000001.png": b"a"})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["verified_images"] == 1
    assert audit.stats["unresolved"] == {}


def test_voc_identical_bytes_across_lists_is_high(tmp_path):
    _voc(
        tmp_path,
        {"train": ["000001"], "val": ["000002"]},
        {"000001.jpg": b"same", "000002.jpg": b"same"},
    )

    audit = image_splits.audit_splits(tmp_path)

    assert [f.kind for f in audit.findings] == ["image_split_overlap"]


def test_voc_duplicate_identifier_is_counted_once(tmp_path):
    _voc(tmp_path, {"train": ["000001", "000001"]}, {"000001.jpg": b"a"})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.findings == []
    assert audit.stats["partitions"]["train"] == {"declared": 2, "unique": 1}


def test_voc_ambiguous_extension_and_missing_identifier(tmp_path):
    _voc(
        tmp_path,
        {"train": ["000001", "000002", "000003.jpg"]},
        {"000001.jpg": b"a", "000001.png": b"b"},
    )

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"ambiguous_asset": 1, "missing_asset": 2}


def test_voc_trainval_is_a_documented_union_not_a_partition(tmp_path):
    _voc(
        tmp_path,
        {"train": ["000001"], "val": ["000002"], "trainval": ["000001", "000002"]},
        {"000001.jpg": b"a", "000002.jpg": b"b"},
    )

    audit = image_splits.audit_splits(tmp_path)

    assert audit.findings == []
    assert any("trainval.txt" in text for text in audit.stats["limitations"])
    assert audit.stats["partitions"]["train"]["unique"] == 1


def test_voc_trainval_alone_is_unusable(tmp_path):
    _voc(tmp_path, {"trainval": ["000001"]}, {"000001.jpg": b"a"})

    audit = image_splits.audit_splits(tmp_path)

    assert audit.ok is False
    assert any("cannot be separated" in text for text in audit.stats["limitations"])


# ---------------------------------------------------------------------------
# Disjointness details, determinism and security
# ---------------------------------------------------------------------------


def test_identity_conflict_without_a_digest_is_a_high_finding(tmp_path):
    _write(tmp_path / "images" / "a.jpg", b"a")
    _coco(tmp_path, "train", ["images/a.jpg"])
    _coco(tmp_path, "val", ["images/a.jpg"])

    audit = image_splits.audit_splits(tmp_path, max_digest_bytes=0)

    assert audit.stats["identity_conflicts"] == 1
    assert audit.findings[0].severity == "high"
    assert "is declared in train + val" in audit.findings[0].detail
    assert audit.findings[0].stable_key.startswith("path:images/a.jpg")
    assert audit.stats["unresolved"]["digest_budget_exhausted"] == 1


def test_multiple_paths_with_one_digest_are_all_reported(tmp_path):
    _write(tmp_path / "train-a.jpg", b"same")
    _write(tmp_path / "train-b.jpg", b"same")
    _write(tmp_path / "val-c.jpg", b"same")
    _coco(tmp_path, "train", ["train-a.jpg", "train-b.jpg"])
    _coco(tmp_path, "val", ["val-c.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert len(audit.findings) == 1
    assert audit.cross_groups[0]["partitions"]["train"] == [
        "train-a.jpg",
        "train-b.jpg",
    ]


def test_digest_budget_reports_a_partial_audit(tmp_path):
    _write(tmp_path / "train-a.jpg", b"same")
    _write(tmp_path / "val-a.jpg", b"same")
    _coco(tmp_path, "train", ["train-a.jpg"])
    _coco(tmp_path, "val", ["val-a.jpg"])

    audit = image_splits.audit_splits(tmp_path, max_digest_bytes=1)

    assert audit.stats["digested_images"] == 0
    assert audit.stats["unresolved"]["digest_budget_exhausted"] == 1
    assert any("digest budget" in text for text in audit.stats["limitations"])


def test_control_characters_in_declarations_are_unresolved(tmp_path):
    _write(tmp_path / "images" / "a.jpg", b"a")
    _coco(tmp_path, "train", ["images/a.jpg", "images/\x01bad.jpg"])

    audit = image_splits.audit_splits(tmp_path)

    assert audit.stats["unresolved"] == {"control_character": 1}
    assert audit.stats["verified_images"] == 1


def test_symlink_escaping_the_root_is_refused(tmp_path):
    outside = _write(tmp_path.parent / "outside-secret.jpg", b"secret")
    root = tmp_path / "dataset"
    try:
        root.mkdir()
        os.symlink(outside, root / "escape.jpg")
    except (OSError, NotImplementedError):
        pytest.skip("this platform cannot create symlinks without extra privileges")
    _yolo(root, {"train": "escape.jpg"})

    audit = image_splits.audit_splits(root, "auto")

    assert audit.stats["unresolved"] == {"outside_root": 1}


def test_repeated_runs_are_byte_identical_and_do_not_write(tmp_path):
    _write(tmp_path / "images" / "train" / "a.jpg", b"same")
    _write(tmp_path / "images" / "val" / "b.jpg", b"same")
    _coco(tmp_path, "train", ["train/a.jpg"])
    _coco(tmp_path, "val", ["val/b.jpg"])
    before = _tree(tmp_path)

    first = json.dumps(image_splits.audit_splits(tmp_path).to_dict(), sort_keys=True)
    second = json.dumps(image_splits.audit_splits(tmp_path).to_dict(), sort_keys=True)

    assert first == second
    assert _tree(tmp_path) == before


def test_unknown_layout_and_missing_directory_are_unusable(tmp_path):
    assert image_splits.audit_splits(tmp_path).stats["status"] == "unusable"
    assert (
        image_splits.audit_splits(tmp_path / "nope").stats["reason"]
        == "the dataset path is not a directory"
    )


# ---------------------------------------------------------------------------
# CLI integration
# ---------------------------------------------------------------------------


def test_cli_splits_reports_a_finding_and_exits_one(tmp_path, capsys):
    _write(tmp_path / "images" / "train" / "a.jpg", b"same")
    _write(tmp_path / "images" / "val" / "b.jpg", b"same")
    _coco(tmp_path, "train", ["train/a.jpg"])
    _coco(tmp_path, "val", ["val/b.jpg"])

    assert cli.main(["splits", str(tmp_path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is True
    assert payload["format"] == "coco"
    assert payload["findings"][0]["kind"] == "image_split_overlap"
    assert payload["findings"][0]["severity"] == "high"
    assert payload["trust_score"] == 65


def test_cli_splits_clean_exits_zero_and_unknown_exits_two(tmp_path, capsys):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _coco(tmp_path, "train", ["train/a.jpg"])

    assert cli.main(["splits", str(tmp_path)]) == 0
    assert "No cross-split image overlap found." in capsys.readouterr().out

    empty = tmp_path / "empty"
    empty.mkdir()
    assert cli.main(["splits", str(empty)]) == 2
    assert "No audit could run" in capsys.readouterr().out


def test_cli_splits_missing_pyyaml_is_a_usage_error(tmp_path, monkeypatch):
    _write(tmp_path / "images" / "train" / "a.jpg", b"a")
    _yolo(tmp_path, {"train": "images/train"})
    monkeypatch.setattr(image_splits.importlib.util, "find_spec", lambda name: None)

    with pytest.raises(SystemExit) as excinfo:
        cli.main(["splits", str(tmp_path)])

    assert excinfo.value.code == 2
