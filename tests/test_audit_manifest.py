import json
import os

import pytest

from leaklens import assemble, audit_manifest
from leaklens.cli import main
from leaklens.leakage import Finding


def _document(tmp_path, *, findings=None):
    source = tmp_path / "input.csv"
    source.write_text("uid\n1\n2\n", encoding="utf-8")
    sections = assemble.build(
        findings
        if findings is not None
        else [Finding("medium", "id_like", "uid", "98% unique values")]
    )
    return audit_manifest.build(
        sections,
        inputs=[source],
        options={"target": "y"},
        gate={"failed": False, "fail_under": None, "fail_on_new": False},
        root=tmp_path,
    )


def _write(tmp_path, document, name="audit.json"):
    path = tmp_path / name
    path.write_text(audit_manifest.dumps(document), encoding="utf-8", newline="\n")
    return path


@pytest.fixture(autouse=True)
def _run_in_tmp(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)


@pytest.fixture
def manifest(tmp_path):
    document = _document(tmp_path)
    return _write(tmp_path, document), document


def test_a_fresh_manifest_verifies(manifest):
    path, document = manifest
    assert audit_manifest.verify(path) == []
    assert document["schema"] == audit_manifest.SCHEMA
    assert document["schema_version"] == audit_manifest.SCHEMA_VERSION
    assert document["inputs"] == [
        {
            "path": "input.csv",
            "kind": "file",
            "digest": document["inputs"][0]["digest"],
        }
    ]
    assert document["inputs"][0]["digest"].startswith("sha256:")


def test_manifest_digest_is_stable_across_two_builds(tmp_path):
    first = _document(tmp_path)
    second = _document(tmp_path)
    assert first["manifest_digest"] == second["manifest_digest"]
    assert first["section_digests"] == second["section_digests"]
    changed = _document(tmp_path, findings=[Finding("high", "id_like", "uid", "x")])
    assert changed["manifest_digest"] != first["manifest_digest"]


def test_serialization_is_deterministic(tmp_path):
    first = _document(tmp_path)
    second = _document(tmp_path)
    second["generated_at"] = first["generated_at"]
    assert audit_manifest.dumps(first) == audit_manifest.dumps(second)
    assert audit_manifest.dumps(first).endswith("\n")


def test_key_reordering_and_whitespace_still_verify(tmp_path, manifest):
    path, document = manifest
    path.write_text(json.dumps(document, indent=4, sort_keys=False), encoding="utf-8")
    assert audit_manifest.verify(path) == []


def test_a_changed_score_is_reported(tmp_path, manifest):
    path, document = manifest
    document["result"]["trust_score"] = 0
    _write(tmp_path, document)
    issues = audit_manifest.verify(path)
    assert "trust_score changed since the manifest was written" in issues
    assert any(issue.startswith("manifest digest") for issue in issues)


def test_changed_evidence_is_reported(tmp_path, manifest):
    path, document = manifest
    document["result"]["findings"].append(
        Finding("high", "target_leakage", "y2", "copied").to_dict()
    )
    _write(tmp_path, document)
    assert "findings changed since the manifest was written" in (
        audit_manifest.verify(path)
    )


def test_a_changed_detector_version_is_reported(tmp_path, manifest):
    path, document = manifest
    document["detectors"][0]["introduced"] = "0.0.0"
    _write(tmp_path, document)
    assert "detectors changed since the manifest was written" in (
        audit_manifest.verify(path)
    )


def test_a_changed_gate_is_reported(tmp_path, manifest):
    path, document = manifest
    document["result"]["gate"]["failed"] = True
    _write(tmp_path, document)
    assert "gate changed since the manifest was written" in (
        audit_manifest.verify(path)
    )


def test_a_modified_input_is_reported(tmp_path, manifest):
    path, _ = manifest
    (tmp_path / "input.csv").write_text("uid\n3\n", encoding="utf-8")
    assert audit_manifest.verify(path) == ["modified: input.csv"]


def test_a_missing_input_is_reported(tmp_path, manifest):
    path, _ = manifest
    (tmp_path / "input.csv").unlink()
    assert audit_manifest.verify(path) == ["missing: input.csv"]


def test_an_unsafe_input_path_is_reported(tmp_path, manifest):
    path, document = manifest
    document["inputs"][0]["path"] = "../outside.csv"
    _write(tmp_path, document)
    issues = audit_manifest.verify(path)
    assert "unsafe path: ../outside.csv" in issues
    assert issues.count("unsafe path: ../outside.csv") == 1


def test_unknown_keys_fail_verification(tmp_path, manifest):
    path, document = manifest
    document["surprise"] = 1
    _write(tmp_path, document)
    assert "unknown key: surprise" in audit_manifest.verify(path)


def test_a_missing_manifest_digest_is_reported(tmp_path, manifest):
    path, document = manifest
    del document["manifest_digest"]
    _write(tmp_path, document)
    assert "manifest_digest is missing" in audit_manifest.verify(path)


def test_malformed_input_entries_are_reported(tmp_path, manifest):
    path, document = manifest
    document["inputs"] = ["not an object"]
    _write(tmp_path, document)
    assert "malformed input entry #0" in audit_manifest.verify(path)


def test_directory_inputs_are_digested(tmp_path):
    folder = tmp_path / "code"
    folder.mkdir()
    (folder / "a.py").write_text("x = 1\n", encoding="utf-8")
    (folder / "__pycache__").mkdir()
    (folder / "__pycache__" / "a.pyc").write_bytes(b"ignored")
    document = audit_manifest.build(assemble.build([]), inputs=[folder], root=tmp_path)
    path = _write(tmp_path, document)
    assert document["inputs"][0]["kind"] == "directory"
    assert audit_manifest.verify(path) == []
    (folder / "a.py").write_text("x = 2\n", encoding="utf-8")
    assert audit_manifest.verify(path) == ["modified: code"]


def test_directory_inputs_ignore_symlinks(tmp_path):
    folder = tmp_path / "tree"
    folder.mkdir()
    outside = tmp_path / "outside.txt"
    outside.write_text("original\n", encoding="utf-8")
    try:
        (folder / "link.txt").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available for this user")
    document = audit_manifest.build(assemble.build([]), inputs=[folder], root=tmp_path)
    path = _write(tmp_path, document)
    outside.write_text("changed\n", encoding="utf-8")
    assert audit_manifest.verify(path) == []


def test_verification_root_can_be_given_explicitly(tmp_path, manifest):
    path, _ = manifest
    elsewhere = tmp_path / "elsewhere"
    elsewhere.mkdir()
    issues = audit_manifest.verify(path, root=elsewhere)
    assert issues == ["missing: input.csv"]


@pytest.mark.parametrize("version", [0, 2, "1", None, True])
def test_unsupported_schema_versions_are_refused(tmp_path, manifest, version):
    path, document = manifest
    document["schema_version"] = version
    _write(tmp_path, document)
    with pytest.raises(audit_manifest.ManifestError, match="schema_version"):
        audit_manifest.verify(path)


def test_an_unknown_schema_is_refused(tmp_path, manifest):
    path, document = manifest
    document["schema"] = "something-else"
    _write(tmp_path, document)
    with pytest.raises(audit_manifest.ManifestError, match="schema"):
        audit_manifest.verify(path)


def test_malformed_json_is_refused(tmp_path):
    path = tmp_path / "broken.json"
    path.write_text("{", encoding="utf-8")
    with pytest.raises(audit_manifest.ManifestError, match="not valid JSON"):
        audit_manifest.verify(path)


def test_a_non_object_manifest_is_refused(tmp_path):
    path = tmp_path / "list.json"
    path.write_text("[]", encoding="utf-8")
    with pytest.raises(audit_manifest.ManifestError, match="not a JSON object"):
        audit_manifest.verify(path)


def test_a_missing_manifest_is_refused(tmp_path):
    with pytest.raises(audit_manifest.ManifestError, match="cannot read"):
        audit_manifest.verify(tmp_path / "absent.json")


def test_non_finite_numbers_are_refused(tmp_path):
    document = _document(tmp_path)
    document["options"]["threshold"] = float("nan")
    with pytest.raises(ValueError):
        audit_manifest.dumps(document)


def test_a_manifest_with_a_non_finite_number_is_refused(tmp_path, manifest):
    path, document = manifest
    document["options"]["threshold"] = float("nan")
    path.write_text(json.dumps(document), encoding="utf-8")
    with pytest.raises(audit_manifest.ManifestError, match="cannot be parsed"):
        audit_manifest.verify(path)


def test_cli_writes_a_manifest_that_verifies(tmp_path):
    source = tmp_path / "clean.py"
    source.write_text("x = 1\n", encoding="utf-8")
    path = tmp_path / "audit.json"
    assert main(["--scan", str(source), "--manifest", str(path)]) == 0
    assert path.exists()
    assert audit_manifest.verify(path) == []
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["inputs"][0]["path"] == "clean.py"
    assert document["options"]["threshold"] == 0.95
    assert document["result"]["gate"]["failed"] is False


def test_cli_verify_exit_codes(tmp_path, manifest):
    path, document = manifest
    assert main(["verify", str(path)]) == 0
    document["result"]["trust_score"] = 100
    _write(tmp_path, document)
    assert main(["verify", str(path)]) == 1
    (tmp_path / "audit.json").write_text("{", encoding="utf-8")
    assert main(["verify", str(path)]) == 2


def test_cli_verify_json_reports_issues(tmp_path, manifest, capsys):
    path, document = manifest
    document["result"]["trust_score"] = 100
    _write(tmp_path, document)
    capsys.readouterr()
    assert main(["verify", str(path), "--json"]) == 1
    payload = json.loads(capsys.readouterr().out)
    assert payload["ok"] is False
    assert "trust_score changed since the manifest was written" in payload["issues"]


def test_a_failed_audit_still_writes_a_manifest(tmp_path):
    source = tmp_path / "bad.py"
    source.write_text(
        "from sklearn.preprocessing import StandardScaler\n"
        "scaler = StandardScaler()\nscaler.fit(X_test)\n",
        encoding="utf-8",
    )
    path = tmp_path / "audit.json"
    assert main(["--scan", str(source), "--manifest", str(path)]) == 1
    document = json.loads(path.read_text(encoding="utf-8"))
    assert document["result"]["gate"]["failed"] is True
    assert audit_manifest.verify(path) == []


def test_a_duplicate_input_entry_cannot_hide_a_change(tmp_path, manifest):
    path, document = manifest
    document["inputs"].append({**document["inputs"][0], "digest": "sha256:" + "0" * 64})
    _write(tmp_path, document)
    assert "modified: input.csv" in audit_manifest.verify(path)


@pytest.mark.parametrize("declared", ["../outside.csv", "..\\outside.csv"])
def test_traversal_paths_are_reported(tmp_path, manifest, declared):
    path, document = manifest
    document["inputs"][0]["path"] = declared
    _write(tmp_path, document)
    assert f"unsafe path: {declared}" in audit_manifest.verify(path)


def test_an_absolute_outside_path_is_reported(tmp_path, manifest):
    path, document = manifest
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("uid\n1\n", encoding="utf-8")
    document["inputs"][0]["path"] = str(outside)
    _write(tmp_path, document)
    assert f"unsafe path: {outside}" in audit_manifest.verify(path)


def test_a_symlink_that_escapes_the_root_is_reported(tmp_path, manifest):
    path, document = manifest
    outside = tmp_path.parent / "outside.csv"
    outside.write_text("uid\n1\n", encoding="utf-8")
    try:
        (tmp_path / "link.csv").symlink_to(outside)
    except OSError:
        pytest.skip("symlinks are not available for this user")
    document["inputs"][0]["path"] = "link.csv"
    _write(tmp_path, document)
    assert "unsafe path: link.csv" in audit_manifest.verify(path)


def test_a_declared_file_that_became_a_directory_is_reported(tmp_path, manifest):
    path, _ = manifest
    (tmp_path / "input.csv").unlink()
    (tmp_path / "input.csv").mkdir()
    assert audit_manifest.verify(path) == ["not a regular file: input.csv"]


def test_a_declared_directory_that_became_a_file_is_reported(tmp_path):
    folder = tmp_path / "code"
    folder.mkdir()
    document = audit_manifest.build(assemble.build([]), inputs=[folder], root=tmp_path)
    path = _write(tmp_path, document)
    folder.rmdir()
    folder.write_text("x = 1\n", encoding="utf-8")
    assert audit_manifest.verify(path) == ["not a directory: code"]


@pytest.mark.skipif(os.name == "nt", reason="POSIX named pipes")
def test_a_non_regular_input_is_reported(tmp_path, manifest):
    path, document = manifest
    os.mkfifo(tmp_path / "pipe")
    document["inputs"][0]["path"] = "pipe"
    _write(tmp_path, document)
    assert "not a regular file: pipe" in audit_manifest.verify(path)


def test_a_tool_version_change_is_reported(tmp_path, manifest):
    path, document = manifest
    document["tool"]["version"] = "0.0.0"
    _write(tmp_path, document)
    assert "tool changed since the manifest was written" in audit_manifest.verify(path)


def test_a_duplicate_json_key_is_refused(tmp_path, manifest):
    path, document = manifest
    text = audit_manifest.dumps(document).replace(
        '"schema_version": 1,', '"schema_version": 1,\n  "schema_version": 1,', 1
    )
    path.write_text(text, encoding="utf-8")
    with pytest.raises(audit_manifest.ManifestError, match="duplicate JSON key"):
        audit_manifest.verify(path)


def test_a_path_with_a_nul_byte_is_reported(tmp_path, manifest):
    path, document = manifest
    document["inputs"][0]["path"] = "in\u0000put.csv"
    _write(tmp_path, document)
    assert any(
        issue.startswith(("cannot resolve", "cannot read"))
        for issue in audit_manifest.verify(path)
    )


def test_an_unlistable_directory_input_is_reported(tmp_path, manifest, monkeypatch):
    path, document = manifest
    (tmp_path / "code").mkdir()
    document["inputs"][0].update({"path": "code", "kind": "directory"})
    _write(tmp_path, document)

    def walk(root, onerror=None, followlinks=False):
        if onerror is not None:
            onerror(PermissionError("denied"))
        return iter(())

    monkeypatch.setattr(audit_manifest.os, "walk", walk)
    assert any(
        issue.startswith("cannot digest code") for issue in audit_manifest.verify(path)
    )


def test_an_oversized_manifest_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(audit_manifest, "MAX_MANIFEST_BYTES", 16)
    path = tmp_path / "big.json"
    path.write_text("x" * 17, encoding="utf-8")
    with pytest.raises(audit_manifest.ManifestError, match="larger than"):
        audit_manifest.verify(path)
