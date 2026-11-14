"""Phase 16 M4/M5: framework config adapters and reproducibility checks."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from leaklens.analysis.dl import config, training

LIGHTNING_CLEAN = """
seed_everything: 42
trainer:
  max_epochs: 10
  precision: 16-mixed
  strategy: ddp
  deterministic: true
"""

LIGHTNING_PRECISION = """
seed_everything: 42
trainer:
  max_epochs: 10
  precision: 16-mixed
"""

LIGHTNING_EARLY_STOP = """
trainer:
  max_epochs: 5
  callbacks:
    - class_path: lightning.pytorch.callbacks.EarlyStopping
      init_args:
        monitor: val_loss
        patience: 5
"""

KERAS = """
epochs: 5
compile:
  optimizer: adam
  loss: sparse_categorical_crossentropy
  metrics: [accuracy]
"""

PYTORCH = """
num_epochs: 3
optimizer: adamw
lr: 0.0003
scheduler: cosine
"""

UNKNOWN = """
foo: bar
baz: 1
"""


def _write(tmp_path: Path, name: str, text: str) -> Path:
    path = tmp_path / name
    path.write_text(text, encoding="utf-8")
    return path


def test_a_clean_lightning_config_emits_no_findings(tmp_path):
    path = _write(tmp_path, "config.yaml", LIGHTNING_CLEAN)
    loaded = config.load_config(path, source="config.yaml")
    assert loaded is not None
    assert loaded.framework == "lightning"
    signals = config.config_signals(loaded)
    assert training.reproducibility_findings(signals) == []
    assert training.training_findings(signals) == []


def test_lightning_precision_without_determinism_is_a_reproducibility_finding(tmp_path):
    path = _write(tmp_path, "config.yaml", LIGHTNING_PRECISION)
    signals = config.config_signals(config.load_config(path, source="config.yaml"))
    findings = training.reproducibility_findings(signals)
    assert [finding.kind for finding in findings] == ["no_random_state"]
    assert "precision" in findings[0].evidence[0].observed
    assert findings[0].evidence[0].location == "config.yaml:trainer.precision"


def test_an_early_stopping_patience_longer_than_the_epochs_is_a_mismatch(tmp_path):
    path = _write(tmp_path, "config.yaml", LIGHTNING_EARLY_STOP)
    loaded = config.load_config(path, source="config.yaml")
    signals = config.config_signals(loaded)
    findings = training.training_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_training_protocol_mismatch"]
    assert "patience" in findings[0].evidence[0].location


def test_keras_and_pytorch_shapes_are_recognised(tmp_path):
    keras = config.load_config(_write(tmp_path, "keras.yaml", KERAS))
    pytorch = config.load_config(_write(tmp_path, "torch.yaml", PYTORCH))
    assert keras is not None and keras.framework == "keras"
    assert pytorch is not None and pytorch.framework == "pytorch"
    assert training.reproducibility_findings(config.config_signals(keras))
    assert config.config_signals(pytorch)


def test_an_unknown_framework_shape_is_skipped(tmp_path):
    loaded = config.load_config(_write(tmp_path, "unknown.yaml", UNKNOWN))
    assert loaded is not None
    assert loaded.framework == "unknown"
    assert config.config_signals(loaded) == []


def test_json_configs_need_no_yaml_module(tmp_path):
    path = _write(tmp_path, "config.json", json.dumps({"trainer": {"max_epochs": 3}}))
    loaded = config.load_config(path, yaml_mod=None, source="config.json")
    assert loaded is not None and loaded.framework == "lightning"
    signals = config.config_signals(loaded)
    assert [finding.kind for finding in training.reproducibility_findings(signals)] == [
        "no_random_state"
    ]


def test_malformed_yaml_is_a_typed_skip_not_a_crash(tmp_path):
    pytest.importorskip("yaml")
    path = _write(tmp_path, "config.yaml", "trainer: [unclosed\n")
    assert config.load_config(path) is None


def test_a_deeply_nested_config_is_a_typed_skip(tmp_path):
    path = _write(tmp_path, "config.json", '{"a":' * 1500 + "1" + "}" * 1500)
    assert config.load_config(path) is None


def test_a_recursive_yaml_anchor_is_a_typed_skip(tmp_path):
    pytest.importorskip("yaml")
    path = _write(tmp_path, "config.yaml", "a: &a {b: [*a]}\n")
    assert config.load_config(path) is None


def test_a_yaml_file_without_pyyaml_raises_the_typed_unavailable(tmp_path):
    path = _write(tmp_path, "config.yaml", LIGHTNING_CLEAN)
    with pytest.raises(config.DlUnavailable):
        config.load_config(path, yaml_mod=None)


def test_requirement_pins_are_reused_from_the_pipeline_family():
    text = "torch==2.1.0\nnumpy>=1.24\ntransformers\n-r extra.txt\n"
    findings = config.requirement_pin_findings(text, "requirements.txt")
    assert {finding.kind for finding in findings} == {"pipeline_unpinned_dependency"}
    assert {finding.column for finding in findings} == {"requirements.txt"}
    assert {finding.severity for finding in findings} == {"low"}
    assert all(finding.diagnostic for finding in findings)
    assert {finding.stable_key for finding in findings} == {
        "pin:requirements.txt:numpy",
        "pin:requirements.txt:transformers",
    }


def test_fully_pinned_requirements_are_clean():
    assert (
        config.requirement_pin_findings("torch==2.1.0\nnumpy==1.26.4\n", "r.txt") == []
    )


def test_config_findings_are_stable_across_loads(tmp_path):
    path = _write(tmp_path, "config.yaml", LIGHTNING_EARLY_STOP)
    first = training.training_findings(
        config.config_signals(config.load_config(path, source="config.yaml"))
    )
    second = training.training_findings(
        config.config_signals(config.load_config(path, source="config.yaml"))
    )
    assert [finding.fingerprint for finding in first] == [
        finding.fingerprint for finding in second
    ]
