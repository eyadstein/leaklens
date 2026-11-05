"""Real-world benchmark regression tests (kept separate from synthetic bench)."""

import pathlib

import pytest

pytestmark = pytest.mark.skipif(
    not pathlib.Path("realworld/manifest.jsonl").exists(),
    reason="real-world corpus not fetched (run scripts/realworld.py)",
)


def test_realworld_eval_runs_and_reports():
    from leaklens.realworld_eval import evaluate

    m = evaluate()
    assert m["n"] == 60
    assert m["tp"] + m["fp"] + m["fn"] + m["tn"] >= 0
    assert 0.0 <= m["precision"] <= 1.0
    assert 0.0 <= m["recall"] <= 1.0
    # hand-reviewed confirmed leaks must all be flagged after fixes
    assert m["recall"] >= 0.8


def test_realworld_labels_cover_manifest():
    import json

    labels = json.loads(
        pathlib.Path("realworld/labels.json").read_text(encoding="utf-8")
    )
    manifest = pathlib.Path("realworld/manifest.jsonl").read_text(encoding="utf-8")
    ids = {json.loads(line)["id"] for line in manifest.splitlines() if line.strip()}
    assert set(labels) == ids


def test_realworld_notebooks_exist():
    nb_dir = pathlib.Path("realworld/notebooks")
    assert len(list(nb_dir.glob("nb_*.ipynb"))) == 60
