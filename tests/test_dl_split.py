"""Phase 16 M3: train/eval split provenance for deep-learning code."""

from __future__ import annotations

from pathlib import Path

from leaklens.analysis.dl import training
from leaklens.leakage import score

SAME_DATASET = """
from torch.utils.data import DataLoader

dataset = MyDataset("data.pt")
train_loader = DataLoader(dataset, batch_size=32, shuffle=True)
val_loader = DataLoader(dataset, batch_size=32)
"""

OVERLAPPING_SUBSET = """
from torch.utils.data import Subset

full = MyDataset("full")
train_ds = Subset(full, [0, 1, 2, 3])
val_ds = Subset(full, [3, 4, 5])
"""

DISJOINT_SUBSET = """
from torch.utils.data import Subset

full = MyDataset("full")
train_ds = Subset(full, [0, 1, 2])
val_ds = Subset(full, [3, 4, 5])
"""

UNKNOWN_SUBSET = """
from torch.utils.data import Subset
import numpy as np

full = MyDataset("full")
order = np.arange(len(full))
train_ds = Subset(full, order[:80])
val_ds = Subset(full, order[80:])
"""

TRAIN_TRANSFORM_ON_VALIDATION = """
import torchvision.transforms as transforms

train_tf = transforms.Compose(
    [transforms.RandomResizedCrop(224), transforms.ToTensor()]
)
val_tf = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)
train_ds = ImageFolder("train", transform=train_tf)
val_ds = ImageFolder("val", transform=train_tf)
"""

INLINE_AUGMENTATION_ON_VALIDATION = """
import torchvision.transforms as transforms

val_ds = ImageFolder(
    "val",
    transform=transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.ToTensor()]
    ),
)
"""

DETERMINISTIC_VALIDATION = """
import torchvision.transforms as transforms

train_tf = transforms.Compose(
    [transforms.RandomResizedCrop(224), transforms.ToTensor()]
)
val_tf = transforms.Compose(
    [transforms.Resize(256), transforms.CenterCrop(224), transforms.ToTensor()]
)
train_ds = ImageFolder("train", transform=train_tf)
val_ds = ImageFolder("val", transform=val_tf)
"""

DISABLED_AUGMENTATION = """
import torchvision.transforms as transforms

val_tf = transforms.Compose(
    [transforms.RandomHorizontalFlip(p=0), transforms.ToTensor()]
)
val_ds = ImageFolder("val", transform=val_tf)
"""


def _repo(tmp_path: Path, source: str) -> tuple[Path, list[training.TrainingSignal]]:
    path = tmp_path / "train.py"
    path.write_text(source, encoding="utf-8")
    return tmp_path, training.scan_python(tmp_path, [path], skipped={})


def test_the_same_dataset_on_both_sides_is_a_high_split_finding(tmp_path):
    _, signals = _repo(tmp_path, SAME_DATASET)
    findings = training.split_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_split_overlap"]
    finding = findings[0]
    assert finding.severity == "high"
    assert finding.diagnostic is True
    assert finding.column == "train.py"
    assert "same dataset" in finding.detail
    locations = {evidence.location for evidence in finding.evidence}
    assert locations == {"train.py:5", "train.py:6"}
    assert score(findings) == 100


def test_overlapping_literal_subset_indices_are_reported(tmp_path):
    _, signals = _repo(tmp_path, OVERLAPPING_SUBSET)
    findings = training.split_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_split_overlap"]
    assert "1 shared" in findings[0].evidence[0].observed
    assert findings[0].stable_key == "split:train.py:index_overlap"


def test_disjoint_subset_indices_are_clean(tmp_path):
    _, signals = _repo(tmp_path, DISJOINT_SUBSET)
    assert training.split_findings(signals) == []


def test_unresolved_index_variables_are_not_guessed(tmp_path):
    _, signals = _repo(tmp_path, UNKNOWN_SUBSET)
    assert training.split_findings(signals) == []


def test_a_training_transform_on_validation_is_a_high_finding(tmp_path):
    _, signals = _repo(tmp_path, TRAIN_TRANSFORM_ON_VALIDATION)
    findings = training.split_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_split_overlap"]
    assert "train_tf" in findings[0].evidence[0].observed
    assert findings[0].stable_key == "split:train.py:eval_augmentation"
    assert findings[0].evidence[0].location == "train.py:11"


def test_an_inline_random_augmentation_on_validation_is_reported(tmp_path):
    _, signals = _repo(tmp_path, INLINE_AUGMENTATION_ON_VALIDATION)
    findings = training.split_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_split_overlap"]


def test_a_transform_assigned_through_an_attribute_is_reported(tmp_path):
    _, signals = _repo(
        tmp_path,
        "import torchvision.transforms as transforms\n\n"
        'val_ds = ImageFolder("val")\n'
        "val_ds.transform = transforms.RandomHorizontalFlip()\n",
    )
    findings = training.split_findings(signals)
    assert [finding.kind for finding in findings] == ["dl_split_overlap"]
    assert findings[0].stable_key == "split:train.py:eval_augmentation"
    assert findings[0].evidence[0].location == "train.py:4"


def test_a_deterministic_validation_transform_is_clean(tmp_path):
    _, signals = _repo(tmp_path, DETERMINISTIC_VALIDATION)
    assert training.split_findings(signals) == []


def test_a_disabled_random_augmentation_is_clean(tmp_path):
    _, signals = _repo(tmp_path, DISABLED_AUGMENTATION)
    assert training.split_findings(signals) == []


COUNTERFACTUAL_FACTORY = """
train_data = make_dataset()
test_data = make_dataset()
"""


def test_independent_dataset_factories_are_not_the_same_dataset(tmp_path):
    _, signals = _repo(tmp_path, COUNTERFACTUAL_FACTORY)
    assert training.split_findings(signals) == []


def test_split_findings_are_stable_across_scans(tmp_path):
    root, first = _repo(tmp_path, OVERLAPPING_SUBSET)
    second = training.scan_python(root, [root / "train.py"], skipped={})
    assert [f.fingerprint for f in training.split_findings(first)] == [
        f.fingerprint for f in training.split_findings(second)
    ]
