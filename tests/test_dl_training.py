"""Phase 16 M1/M4: training-code signals and reproducibility findings."""

from __future__ import annotations

from pathlib import Path

from leaklens.analysis.dl import training

CLEAN = """
import torch
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

torch.manual_seed(42)
torch.use_deterministic_algorithms(True)

trainer = Trainer(max_epochs=10, precision="16-mixed", strategy="ddp")
stop = EarlyStopping(monitor="val_loss", patience=2)
"""

MISSING_SEED = """
from lightning.pytorch import Trainer

trainer = Trainer(max_epochs=3)
"""

PRECISION_NO_DETERMINISM = """
import torch
from lightning.pytorch import Trainer

torch.manual_seed(0)
trainer = Trainer(max_epochs=3, precision=16)
"""

DISTRIBUTED_NO_DETERMINISM = """
import torch
from lightning.pytorch import Trainer

torch.manual_seed(0)
trainer = Trainer(max_epochs=3, strategy="ddp")
"""

EARLY_STOP_MISMATCH = """
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping

trainer = Trainer(max_epochs=5)
stop = EarlyStopping(monitor="val_loss", patience=5)
"""

LR_ZERO = """
import torch

opt = torch.optim.Adam(model.parameters(), lr=0.0)
"""

SCHEDULER_ZERO = """
import torch

sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=0.1, total_steps=0)
"""


def _repo(tmp_path: Path, source: str, name: str = "train.py") -> Path:
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(source, encoding="utf-8")
    return tmp_path


def _signals(tmp_path: Path, source: str) -> list[training.TrainingSignal]:
    skipped: dict[str, int] = {}
    files = [tmp_path / "train.py"]
    return training.scan_python(tmp_path, files, skipped=skipped)


def test_scan_extracts_seed_determinism_precision_distributed_and_early_stop(tmp_path):
    _repo(tmp_path, CLEAN)
    signals = _signals(tmp_path, CLEAN)
    names = {signal.name for signal in signals}
    assert {
        "seed",
        "determinism",
        "precision",
        "distributed",
        "early_stopping",
    } <= names
    assert {signal.name for signal in signals if signal.line > 0} == names
    entry = next(signal for signal in signals if signal.name == "training_entry")
    assert entry.path == "train.py"
    assert entry.line == 9
    assert entry.location == "train.py:9"


def test_clean_training_config_emits_no_findings(tmp_path):
    _repo(tmp_path, CLEAN)
    signals = _signals(tmp_path, CLEAN)
    assert training.training_findings(signals) == []
    assert training.reproducibility_findings(signals) == []


def test_missing_seed_is_a_reproducibility_finding_at_the_entry_line(tmp_path):
    _repo(tmp_path, MISSING_SEED)
    signals = _signals(tmp_path, MISSING_SEED)
    findings = training.reproducibility_findings(signals)
    assert [finding.kind for finding in findings] == ["no_random_state"]
    finding = findings[0]
    assert finding.diagnostic is True
    assert finding.column == "train.py"
    assert finding.evidence[0].location == "train.py:4"
    assert "seed" in finding.evidence[0].description


def test_precision_without_a_determinism_flag_is_a_reproducibility_finding(tmp_path):
    _repo(tmp_path, PRECISION_NO_DETERMINISM)
    signals = _signals(tmp_path, PRECISION_NO_DETERMINISM)
    findings = training.reproducibility_findings(signals)
    assert [finding.kind for finding in findings] == ["no_random_state"]
    assert "precision" in findings[0].evidence[0].observed


def test_distributed_without_a_determinism_flag_is_a_reproducibility_finding(tmp_path):
    _repo(tmp_path, DISTRIBUTED_NO_DETERMINISM)
    signals = _signals(tmp_path, DISTRIBUTED_NO_DETERMINISM)
    findings = training.reproducibility_findings(signals)
    assert [finding.kind for finding in findings] == ["no_random_state"]
    assert "strategy=ddp" in findings[0].evidence[0].observed


def test_early_stopping_patience_that_can_never_fire_is_a_protocol_mismatch(tmp_path):
    _repo(tmp_path, EARLY_STOP_MISMATCH)
    signals = _signals(tmp_path, EARLY_STOP_MISMATCH)
    findings = training.training_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_training_protocol_mismatch"]
    assert findings[0].severity == "medium"
    assert findings[0].diagnostic is True
    assert "patience" in findings[0].detail


def test_zero_learning_rate_and_zero_scheduler_horizon_are_protocol_mismatches(
    tmp_path,
):
    _repo(tmp_path, LR_ZERO + SCHEDULER_ZERO)
    signals = _signals(tmp_path, LR_ZERO + SCHEDULER_ZERO)
    findings = training.training_findings(signals)
    keys = {finding.stable_key for finding in findings}
    assert keys == {
        "protocol:train.py:optimizer_lr",
        "protocol:train.py:scheduler_horizon",
    }


def test_malformed_python_degrades_to_a_skip(tmp_path):
    _repo(tmp_path, "def broken(:\n")
    skipped: dict[str, int] = {}
    signals = training.scan_python(tmp_path, [tmp_path / "train.py"], skipped=skipped)
    assert signals == []
    assert skipped == {"python_syntax_error": 1}


def test_two_scans_of_the_same_tree_produce_the_same_fingerprints(tmp_path):
    _repo(tmp_path, MISSING_SEED + EARLY_STOP_MISMATCH)
    first = training.scan_python(tmp_path, [tmp_path / "train.py"], skipped={})
    second = training.scan_python(tmp_path, [tmp_path / "train.py"], skipped={})
    assert [finding.fingerprint for finding in training.training_findings(first)] == [
        finding.fingerprint for finding in training.training_findings(second)
    ]


def test_full_precision_is_not_reported_as_mixed_precision(tmp_path):
    signals = _signals(
        tmp_path,
        "import torch\nfrom lightning.pytorch import Trainer\n"
        "torch.manual_seed(0)\ntrainer = Trainer(max_epochs=3, precision=32)\n",
    )
    assert training.reproducibility_findings(signals) == []
