"""Phase 14 pipeline/MLOps engine: graph, lineage, checks and gate safety."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leaklens import cli, manifest, pipeline
from leaklens.fixes import FIXES
from leaklens.leakage import score
from leaklens.lineage import LineageEdge, sort_edges

CLEAN_DVC = """stages:
  split:
    cmd: python split.py
    deps: [data/raw.csv]
    outs: [data/train.csv, data/eval.csv]
  train:
    cmd: python train.py
    deps: [data/train.csv]
    params:
    - params.yaml:
      - seed
    outs: [model.pkl]
  evaluate:
    cmd: python evaluate.py
    deps: [data/eval.csv, model.pkl]
    outs: [metrics.json]
"""

SHARED_INPUT_DVC = """stages:
  train:
    cmd: python train.py
    deps: [data/all.csv]
    params:
    - params.yaml:
      - seed
    outs: [model.pkl]
  evaluate:
    cmd: python evaluate.py
    deps: [data/all.csv, model.pkl]
    outs: [metrics.json]
"""


def _write(root: Path, files: dict[str, str]) -> Path:
    for name, text in files.items():
        path = root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text, encoding="utf-8")
    return root


def _data(*names: str) -> dict[str, str]:
    return {name: "a,b\n1,2\n" for name in names}


def _clean(tmp_path: Path) -> Path:
    return _write(
        tmp_path,
        {
            "dvc.yaml": CLEAN_DVC,
            "params.yaml": "seed: 42\n",
            **_data("data/raw.csv", "data/train.csv", "data/eval.csv"),
        },
    )


def test_three_stage_graph_yields_the_expected_edges(tmp_path):
    report = pipeline.audit_pipeline(_clean(tmp_path))
    assert report.status == "ok"
    assert {stage.id for stage in report.stages} == {
        "dvc.yaml:split",
        "dvc.yaml:train",
        "dvc.yaml:evaluate",
    }
    assert report.unresolved == ()
    by_column = {edge.column: edge for edge in report.edges}
    assert by_column["data/raw.csv"].transform == "root input"
    assert by_column["data/train.csv"].transform == "dvc.yaml:split"
    assert by_column["data/eval.csv"].transform == "dvc.yaml:split"
    assert by_column["model.pkl"].transform == "dvc.yaml:train"
    assert {edge.confidence for edge in report.edges} == {"confirmed"}
    # M1: a dependency with no producer and no file is labelled unresolved.
    unresolved = pipeline.audit_pipeline(
        _write(
            tmp_path / "u",
            {
                "dvc.yaml": CLEAN_DVC.replace(
                    "deps: [data/train.csv]", "deps: [data/absent.csv]"
                ),
                "params.yaml": "seed: 42\n",
                **_data("data/raw.csv", "data/eval.csv"),
            },
        )
    )
    assert unresolved.unresolved == ("data/absent.csv",)
    missing = next(e for e in unresolved.edges if e.column == "data/absent.csv")
    assert missing.confidence == "unresolved"
    assert missing.transform == "unresolved"


def test_the_same_metadata_always_produces_the_same_graph(tmp_path):
    root = _clean(tmp_path)
    first = pipeline.audit_pipeline(root)
    second = pipeline.audit_pipeline(root)
    assert first.to_dict() == second.to_dict()
    assert list(first.edges) == list(sort_edges(first.edges))
    assert first.to_dict()["edges"] == sorted(
        first.to_dict()["edges"], key=lambda edge: tuple(edge.values())
    )


def test_shared_train_eval_input_is_high_with_evidence(tmp_path):
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "dvc.yaml": SHARED_INPUT_DVC,
                "params.yaml": "seed: 42\n",
                **_data("data/all.csv"),
            },
        )
    )
    overlap = [f for f in report.findings if f.kind == "pipeline_split_overlap"]
    assert len(overlap) == 1
    finding = overlap[0]
    assert finding.severity == "high"
    assert finding.column == "data/all.csv"
    assert "dvc.yaml:train" in finding.detail and "dvc.yaml:evaluate" in finding.detail
    assert finding.diagnostic is True
    evidence = finding.evidence[0]
    assert evidence.source == "code"
    assert evidence.kind == "overlap"
    assert evidence.location and evidence.observed and evidence.expected
    # The confirmed-lineage finding never changes the trust score.
    assert score([finding]) == 100
    assert score(report.findings) == 100


def test_an_eval_stage_may_consume_the_trained_model(tmp_path):
    """Consuming the model is normal; only data-derived inputs are suspicious."""
    report = pipeline.audit_pipeline(_clean(tmp_path))
    assert not [f for f in report.findings if f.kind == "pipeline_split_overlap"]


def test_unresolved_dependency_is_low_and_diagnostic(tmp_path):
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "dvc.yaml": CLEAN_DVC.replace(
                    "deps: [data/train.csv]", "deps: [data/absent.csv]"
                ),
                "params.yaml": "seed: 42\n",
                **_data("data/raw.csv", "data/eval.csv"),
            },
        )
    )
    unresolved = [f for f in report.findings if f.kind == "pipeline_unresolved_lineage"]
    assert [f.column for f in unresolved] == ["data/absent.csv"]
    assert unresolved[0].severity == "low"
    assert unresolved[0].diagnostic is True


def test_parameter_drift_compares_the_lock_with_the_current_file(tmp_path):
    root = _write(
        tmp_path,
        {
            "dvc.lock": (
                "stages:\n"
                "  train:\n"
                "    cmd: python train.py\n"
                "    deps: []\n"
                "    params:\n"
                "      params.yaml:\n"
                "        seed: 42\n"
                "        max_depth: 5\n"
                "    outs:\n"
                "    - path: model.pkl\n"
                "      md5: '11111111111111111111111111111111'\n"
            ),
            "params.yaml": "seed: 7\n",
        },
    )
    report = pipeline.audit_pipeline(root)
    drift = {f.column: f for f in report.findings if f.kind == "pipeline_param_drift"}
    assert set(drift) == {"params.yaml:seed", "params.yaml:max_depth"}
    assert all(f.severity == "low" and f.diagnostic for f in drift.values())
    assert "42" in drift["params.yaml:seed"].detail
    # The same stage also has no seed in its locked values, but it has one in
    # neither file's keys: no_random_state is not duplicated here.
    assert not [f for f in report.findings if f.kind == "no_random_state"]


def test_a_lock_cannot_make_the_audit_read_outside_the_root(tmp_path):
    """A DVC lock is untrusted input: a "../" param path is skipped, never read."""
    root = _write(
        tmp_path / "repo",
        {
            "dvc.lock": (
                "stages:\n"
                "  train:\n"
                "    cmd: python train.py\n"
                "    params:\n"
                "      ../outside.yaml:\n"
                "        secret: hunter2\n"
                "    outs:\n"
                "    - path: model.pkl\n"
                "      md5: '11111111111111111111111111111111'\n"
            ),
            **_data("data/train.csv"),
        },
    )
    _write(tmp_path, {"outside.yaml": "secret: hunter2\n"})
    report = pipeline.audit_pipeline(root)
    assert report.skipped["params_outside_root"] == 1
    assert "hunter2" not in json.dumps(report.to_dict())
    assert not [f for f in report.findings if f.kind == "pipeline_param_drift"]


def test_locked_output_without_a_digest_is_reported(tmp_path):
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "dvc.lock": (
                    "stages:\n"
                    "  train:\n"
                    "    cmd: python train.py\n"
                    "    outs:\n"
                    "    - path: model.pkl\n"
                )
            },
        )
    )
    digest = [f for f in report.findings if f.kind == "pipeline_missing_digest"]
    assert [f.column for f in digest] == ["model.pkl"]
    assert (
        digest[0].evidence[0].expected == "every locked output carries a content digest"
    )


def test_unpinned_requirements_are_low(tmp_path):
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "requirements.txt": "pandas>=2\nscikit-learn==1.5.0\n",
                "requirements/dev.txt": "black\n",
            },
        )
    )
    unpinned = [f for f in report.findings if f.kind == "pipeline_unpinned_dependency"]
    assert {f.column for f in unpinned} == {
        "requirements.txt",
        "requirements/dev.txt",
    }
    assert all(f.severity == "low" for f in unpinned)


def test_training_stage_without_a_seed_reuses_no_random_state(tmp_path):
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "dvc.yaml": (
                    "stages:\n"
                    "  train:\n"
                    "    cmd: python train.py\n"
                    "    deps: [data/train.csv]\n"
                    "    params:\n"
                    "    - params.yaml:\n"
                    "      - n_estimators\n"
                    "    outs: [model.pkl]\n"
                ),
                "params.yaml": "n_estimators: 100\n",
                **_data("data/train.csv"),
            },
        )
    )
    seeds = [f for f in report.findings if f.kind == "no_random_state"]
    assert [f.column for f in seeds] == ["dvc.yaml:train"]
    assert seeds[0].diagnostic is True
    # M4: the shared kind carries framework-neutral advice.
    advice = manifest.BY_KIND["no_random_state"]
    text = FIXES[advice.advice]
    assert "seed" in text
    assert "xgboost" in text.lower() and "torch" in text.lower()


def test_notebook_stages_follow_execution_order(tmp_path):
    cells = [
        {
            "cell_type": "code",
            "execution_count": 2,
            "source": 'df.to_csv("data/f.csv")',
        },
        {
            "cell_type": "code",
            "execution_count": 1,
            "source": 'pd.read_csv("data/raw.csv")',
        },
    ]
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "pipeline.ipynb": json.dumps({"cells": cells}),
                **_data("data/raw.csv"),
            },
        )
    )
    assert [stage.id for stage in report.stages] == [
        "pipeline.ipynb#exec-1",
        "pipeline.ipynb#exec-2",
    ]
    assert report.stages[1].source == "pipeline.ipynb[cell 0]"
    assert report.stages[1].outs == ("data/f.csv",)


def test_malformed_metadata_degrades_to_a_skip(tmp_path):
    report = pipeline.audit_pipeline(
        _write(
            tmp_path,
            {
                "dvc.yaml": "stages: [unclosed\n  train: {cmd: [",
                "pipeline.dag.json": '{"stages": [',
                "pipeline.ipynb": '{"cells": [}',
                "requirements.txt": "pandas\n",
            },
        )
    )
    assert report.skipped["malformed_yaml"] == 1
    assert report.skipped["malformed_json"] == 1
    assert report.skipped["malformed_notebook"] == 1
    assert [f.kind for f in report.findings] == ["pipeline_unpinned_dependency"]


def test_missing_yaml_degrades_for_a_directory_and_raises_for_a_named_file(
    tmp_path, monkeypatch
):
    message = 'pipeline metadata parsing need PyYAML: pip install "leaklens-ml[yaml]"'

    def unavailable():
        raise pipeline.PipelineUnavailable(message)

    monkeypatch.setattr(pipeline, "available", lambda: False)
    monkeypatch.setattr(pipeline, "require_yaml", unavailable)
    root = _write(tmp_path, {"dvc.yaml": CLEAN_DVC})

    report = pipeline.audit_pipeline(root)
    assert report.skipped["yaml_not_installed"] == 1
    assert report.stages == ()

    with pytest.raises(pipeline.PipelineUnavailable) as caught:
        pipeline.audit_pipeline(root / "dvc.yaml")
    assert 'pip install "leaklens-ml[yaml]"' in str(caught.value)
    assert issubclass(pipeline.PipelineUnavailable, ImportError)


def test_every_pipeline_kind_is_registered_unscored_and_diagnostic():
    entries = {
        kind: manifest.BY_KIND[kind]
        for kind in (
            "pipeline_split_overlap",
            "pipeline_unresolved_lineage",
            "pipeline_param_drift",
            "pipeline_missing_digest",
            "pipeline_unpinned_dependency",
        )
    }
    assert all(entry.scored is False for entry in entries.values())
    assert manifest.ENGINES["leaklens.pipeline"].surface == "code"
    assert set(entries) <= manifest.ENGINES["leaklens.pipeline"].kinds


def test_cli_pipeline_is_opt_in_and_never_fails_the_gate(tmp_path, monkeypatch, capsys):
    root = _write(
        tmp_path,
        {
            "dvc.yaml": SHARED_INPUT_DVC,
            "params.yaml": "seed: 42\n",
            **_data("data/all.csv"),
        },
    )
    monkeypatch.chdir(tmp_path)

    code = cli.main(["--scan", str(root), "--json"])
    plain = json.loads(capsys.readouterr().out)
    assert "pipeline" not in plain

    code = cli.main(["--scan", str(root), "--pipeline", "--json"])
    payload = json.loads(capsys.readouterr().out)
    assert code == 0  # a high pipeline diagnostic never fails the gate
    assert payload["pipeline"]["status"] == "ok"
    assert payload["trust_score"] == 100
    assert any(f["kind"] == "pipeline_split_overlap" for f in payload["findings"])
    assert all(
        f.get("diagnostic") is True
        for f in payload["findings"]
        if f["kind"].startswith("pipeline_")
    )


def test_cli_pipeline_requires_a_scan_target(tmp_path):
    """A dataset-only run must not silently ignore --pipeline."""
    with pytest.raises(SystemExit) as caught:
        cli.main(
            [
                "--train",
                str(_write(tmp_path, _data("data.csv")) / "data.csv"),
                "--pipeline",
            ]
        )
    assert caught.value.code == 2


def test_lineage_edge_is_validated_and_sorted():
    edge = LineageEdge("train", "data.csv", "split", "dvc.yaml:split")
    assert edge.to_dict() == {
        "stage": "train",
        "column": "data.csv",
        "transform": "split",
        "locator": "dvc.yaml:split",
        "confidence": "unresolved",
    }
    with pytest.raises(ValueError):
        LineageEdge("train", "data.csv", "split", "loc", confidence="probably")
    with pytest.raises(ValueError):
        LineageEdge("", "data.csv", "split", "loc")
    assert sort_edges([edge, LineageEdge("a", "b", "c", "d")])[0].stage == "a"
