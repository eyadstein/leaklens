"""Static training-code signals and deterministic DL checks (Phase 16 M1/M3/M4).

The module parses Python with :mod:`ast`; it never imports PyTorch, Lightning,
TensorFlow or Keras and never executes scanned code. A signal is a static claim
with a repo-relative file and line. Findings derived from a signal are
``diagnostic=True``, so they never change ``leakage.score()`` or an exit gate.
"""

from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from ...evidence import Evidence
from ...leakage import Finding

MAX_PY_FILE_BYTES = 2 * 1024 * 1024

SEED_CALLS = frozenset(
    {
        "manual_seed",
        "manual_seed_all",
        "seed_everything",
        "set_seed",
        "set_random_seed",
        "seed",
    }
)
DETERMINISM_CALLS = frozenset({"use_deterministic_algorithms", "enable_op_determinism"})
TRAINER_CALLS = frozenset({"Trainer", "fit", "trainer.fit"})
EARLY_STOP_CALLS = frozenset({"EarlyStopping"})
OPTIMIZER_CALLS = frozenset(
    {
        "Adam",
        "AdamW",
        "SGD",
        "RMSprop",
        "Adagrad",
        "Adamax",
        "NAdam",
        "RAdam",
        "LBFGS",
        "Adafactor",
        "FusedAdam",
    }
)
SCHEDULER_CALLS = frozenset(
    {
        "StepLR",
        "MultiStepLR",
        "ExponentialLR",
        "CosineAnnealingLR",
        "CosineAnnealingWarmRestarts",
        "ReduceLROnPlateau",
        "OneCycleLR",
        "LambdaLR",
        "CyclicLR",
    }
)
DISTRIBUTED_CALLS = frozenset(
    {
        "DDP",
        "DistributedDataParallel",
        "FSDP",
        "FullyShardedDataParallel",
        "MirroredStrategy",
        "init_process_group",
    }
)
PRECISION_CALLS = frozenset({"autocast", "GradScaler"})
NON_DISTRIBUTED_VALUES = frozenset(
    {"", "auto", "cpu", "gpu", "tpu", "mps", "single", "none"}
)
RANDOM_AUGMENTATIONS = frozenset(
    {
        "RandomHorizontalFlip",
        "RandomVerticalFlip",
        "RandomCrop",
        "RandomResizedCrop",
        "RandomRotation",
        "RandomAffine",
        "RandomPerspective",
        "RandomErasing",
        "ColorJitter",
        "AutoAugment",
        "RandAugment",
        "TrivialAugmentWide",
    }
)


DATASET_WRAPPERS = frozenset(
    {
        "DataLoader",
        "DataLoader2",
        "Subset",
        "Dataset",
        "IterableDataset",
        "TensorDataset",
        "ConcatDataset",
        "ImageFolder",
        "random_split",
    }
)
DATASET_TOKENS = frozenset({"ds", "dataset", "datasets", "loader", "loaders", "data"})
EVAL_TOKENS = frozenset({"val", "valid", "validation", "eval", "test"})


@dataclass(frozen=True, slots=True)
class TrainingSignal:
    """One static training signal with a file/line or config-field locator."""

    name: str
    value: str
    path: str
    line: int = 0
    field: str = ""
    role: str = ""
    indices: tuple[int, ...] | None = None

    @property
    def location(self) -> str:
        if self.line > 0:
            return f"{self.path}:{self.line}"
        if self.field:
            return f"{self.path}:{self.field}"
        return self.path

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "name": self.name,
            "value": self.value[:200],
            "path": self.path,
        }
        if self.line:
            payload["line"] = self.line
        if self.field:
            payload["field"] = self.field
        if self.role:
            payload["role"] = self.role
        return payload


def _skip(skipped: dict[str, int], reason: str) -> None:
    skipped[reason] = skipped.get(reason, 0) + 1


def _name(node: ast.AST) -> str:
    parts: list[str] = []
    while isinstance(node, ast.Attribute):
        parts.append(node.attr)
        node = node.value
    if isinstance(node, ast.Name):
        parts.append(node.id)
    return ".".join(reversed(parts))


def _last(name: str) -> str:
    return name.rsplit(".", 1)[-1]


def _literal(node: ast.AST | None) -> Any:
    if isinstance(node, ast.Constant):
        return node.value
    if (
        isinstance(node, ast.UnaryOp)
        and isinstance(node.op, ast.USub)
        and isinstance(node.operand, ast.Constant)
        and isinstance(node.operand.value, (int, float))
    ):
        return -node.operand.value
    if isinstance(node, (ast.List, ast.Tuple)):
        values = [_literal(item) for item in node.elts]
        if all(value is not None for value in values):
            return tuple(values)
    return None


def _kwarg(call: ast.Call, name: str) -> ast.AST | None:
    for keyword in call.keywords:
        if keyword.arg == name:
            return keyword.value
    return None


def _text(value: Any) -> str:
    if isinstance(value, (str, int, float, bool)):
        return str(value)
    return ""


def mixed_precision(value: Any) -> bool:
    """True when a precision setting means fp16/bf16/mixed precision."""
    text = str(value).lower()
    return any(marker in text for marker in ("16", "bf16", "mixed"))


def tokens(name: str) -> set[str]:
    return {token for token in name.lower().replace("-", "_").split("_") if token}


def _role(name: str) -> str:
    parts = tokens(name)
    if parts & {"train", "training"}:
        return "train"
    if parts & EVAL_TOKENS:
        return "eval"
    return ""


def _random_transform(node: ast.AST) -> bool:
    for child in ast.walk(node):
        if not isinstance(child, ast.Call):
            continue
        if _last(_name(child.func)) not in RANDOM_AUGMENTATIONS:
            continue
        if _literal(_kwarg(child, "p")) == 0:  # a disabled augmentation
            continue
        return True
    return False


def _is_dataset_value(name: str, value: ast.AST) -> bool:
    if tokens(name) & DATASET_TOKENS:
        return True
    if isinstance(value, ast.Call):
        head = _last(_name(value.func))
        return head in DATASET_WRAPPERS or head.endswith("Dataset")
    return False


def _dataset_shape(value: ast.AST) -> tuple[str, str, tuple[int, ...] | None]:
    """``(base expression, shape, literal indices)`` for one dataset binding."""
    node = value
    while (
        isinstance(node, ast.Call)
        and _last(_name(node.func)) in {"DataLoader", "DataLoader2"}
        and node.args
    ):
        node = node.args[0]
    if isinstance(node, ast.Call) and _last(_name(node.func)) == "Subset" and node.args:
        indices = _literal(node.args[1]) if len(node.args) > 1 else None
        if not isinstance(indices, tuple):
            indices = None
        source = node.args[0]
        shape = "subset_name" if isinstance(source, ast.Name) else "subset_call"
        return ast.unparse(source)[:200], shape, indices
    shape = "plain_name" if isinstance(node, ast.Name) else "plain_call"
    return ast.unparse(node)[:200], shape, None


def _transform_arg(value: ast.AST) -> ast.AST | None:
    if not isinstance(value, ast.Call):
        return None
    for key in ("transform", "train_transform"):
        node = _kwarg(value, key)
        if node is not None:
            return node
    return None


def _describe(value: ast.AST) -> str:
    return value.id if isinstance(value, ast.Name) else ast.unparse(value)[:120]


def _split_signals(tree: ast.AST, path: str) -> list[TrainingSignal]:
    """Dataset bindings and evaluation augmentations, with their line numbers."""
    assignments = [
        node
        for node in ast.walk(tree)
        if isinstance(node, (ast.Assign, ast.AnnAssign)) and node.value is not None
    ]
    random_names: dict[str, bool] = {}
    for node in assignments:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                random_names[target.id] = _random_transform(node.value)
    out: list[TrainingSignal] = []
    for node in assignments:
        targets = node.targets if isinstance(node, ast.Assign) else [node.target]
        for target in targets:
            if isinstance(target, ast.Name):
                role = _role(target.id)
                if role and _is_dataset_value(target.id, node.value):
                    base, shape, indices = _dataset_shape(node.value)
                    out.append(
                        TrainingSignal(
                            "split_binding",
                            base,
                            path,
                            line=node.lineno,
                            field=shape,
                            role=role,
                            indices=indices,
                        )
                    )
                if role == "eval":
                    transform = _transform_arg(node.value)
                    if transform is not None and _is_random(transform, random_names):
                        out.append(
                            TrainingSignal(
                                "eval_augmentation",
                                _describe(transform),
                                path,
                                line=node.lineno,
                            )
                        )
            elif (
                isinstance(target, ast.Attribute)
                and target.attr == "transform"
                and _role(_name(target.value)) == "eval"
                and _is_random(node.value, random_names)
            ):
                out.append(
                    TrainingSignal(
                        "eval_augmentation",
                        _describe(node.value),
                        path,
                        line=node.lineno,
                    )
                )
    return out


def _is_random(node: ast.AST, random_names: dict[str, bool]) -> bool:
    if isinstance(node, ast.Name):
        return random_names.get(node.id, False)
    return _random_transform(node)


def _file_signals(tree: ast.AST, path: str) -> list[TrainingSignal]:
    found: list[TrainingSignal] = []

    def add(name: str, value: Any, line: int) -> None:
        text = _text(value)
        if text:
            found.append(TrainingSignal(name, text[:120], path, line=line))

    for node in ast.walk(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                target_name = _name(target)
                if (
                    target_name.endswith(".deterministic")
                    and _literal(node.value) is True
                ):
                    add("determinism", "cudnn.deterministic=True", node.lineno)
            continue
        if not isinstance(node, ast.Call):
            continue
        name = _name(node.func)
        last = _last(name)
        line = getattr(node, "lineno", 0)
        if last in SEED_CALLS:
            add("seed", name, line)
        if last in DETERMINISM_CALLS:
            add("determinism", name, line)
        if last in DISTRIBUTED_CALLS:
            add("distributed", last, line)
        if last in PRECISION_CALLS:
            value = _literal(_kwarg(node, "dtype"))
            add("precision", value if value is not None else last, line)
        if last in TRAINER_CALLS and (
            last == "Trainer"
            or _kwarg(node, "epochs") is not None
            or _kwarg(node, "max_epochs") is not None
        ):
            add("training_entry", last, line)
            max_epochs = _literal(_kwarg(node, "max_epochs"))
            if max_epochs is None:
                max_epochs = _literal(_kwarg(node, "epochs"))
            if isinstance(max_epochs, int):
                add("max_epochs", max_epochs, line)
            precision = _literal(_kwarg(node, "precision"))
            if precision is not None and mixed_precision(precision):
                add("precision", precision, line)
            strategy = _literal(_kwarg(node, "strategy"))
            if strategy is not None and str(strategy).lower() not in (
                NON_DISTRIBUTED_VALUES
            ):
                add("distributed", strategy, line)
            devices = _literal(_kwarg(node, "devices"))
            if isinstance(devices, int) and devices > 1:
                add("distributed", f"devices={devices}", line)
            elif isinstance(devices, (list, tuple)) and len(devices) > 1:
                add("distributed", f"devices={len(devices)}", line)
            if _literal(_kwarg(node, "deterministic")) is True:
                add("determinism", "deterministic=True", line)
        if last in EARLY_STOP_CALLS:
            patience = _literal(_kwarg(node, "patience"))
            if isinstance(patience, int):
                add("early_stopping", patience, line)
        if last in OPTIMIZER_CALLS:
            lr = _literal(_kwarg(node, "lr"))
            if isinstance(lr, (int, float)):
                add("optimizer_lr", lr, line)
        if last in SCHEDULER_CALLS:
            for key in ("T_max", "total_steps", "step_size"):
                horizon = _literal(_kwarg(node, key))
                if isinstance(horizon, int):
                    add("scheduler_horizon", horizon, line)
                    break
        if last == "set_global_policy":
            policy = _literal(node.args[0]) if node.args else None
            if isinstance(policy, str) and ("float" in policy or "mixed" in policy):
                add("precision", policy, line)
    found.extend(_split_signals(tree, path))
    return found


def scan_python(
    root: str | Path,
    files: list[Path],
    *,
    skipped: dict[str, int],
    max_bytes: int = MAX_PY_FILE_BYTES,
) -> list[TrainingSignal]:
    """Parse each Python file once and return deterministic signals."""
    base = Path(root)
    out: list[TrainingSignal] = []
    for path in sorted(files):
        try:
            if path.stat().st_size > max_bytes:
                _skip(skipped, "python_file_too_large")
                continue
            data = path.read_bytes()
        except OSError:
            _skip(skipped, "python_unreadable")
            continue
        try:
            text = data.decode("utf-8")
        except UnicodeDecodeError:
            _skip(skipped, "python_not_utf8")
            continue
        try:
            tree = ast.parse(text)
        except (SyntaxError, ValueError, RecursionError):
            _skip(skipped, "python_syntax_error")
            continue
        try:
            rel = path.relative_to(base).as_posix()
        except ValueError:
            rel = path.name
        out.extend(_file_signals(tree, rel))
    return out


def split_findings(signals: list[TrainingSignal]) -> list[Finding]:
    """M3: shared or augmented train/eval data is a high split finding."""
    by_path: dict[str, list[TrainingSignal]] = defaultdict(list)
    for signal in signals:
        if signal.name in {"split_binding", "eval_augmentation"}:
            by_path[signal.path].append(signal)
    out: list[Finding] = []
    for path, items in sorted(by_path.items()):
        train = [item for item in items if item.role == "train"]
        evaluation = [item for item in items if item.role == "eval"]
        for tr in train:
            for ev in evaluation:
                if tr.value != ev.value:
                    continue
                if tr.field.endswith("_call") or ev.field.endswith("_call"):
                    continue  # two independent constructor calls are not one dataset
                if tr.field == "plain_name" and ev.field == "plain_name":
                    mode = "same_dataset"
                    observed = (
                        "the same dataset object is used for training and evaluation"
                    )
                    detail = (
                        "the same dataset is used for training and evaluation; "
                        "every evaluated sample was trained on"
                    )
                else:
                    if tr.indices is None or ev.indices is None:
                        continue  # unresolved index variables are never guessed
                    if tr.field == "plain_name":
                        shared = list(ev.indices)
                    elif ev.field == "plain_name":
                        shared = list(tr.indices)
                    else:
                        shared = sorted(set(tr.indices) & set(ev.indices))
                    if not shared:
                        continue
                    mode = "index_overlap"
                    observed = f"{len(shared)} shared sample indices, e.g. {shared[:5]}"
                    detail = (
                        f"training and evaluation draw from the same dataset with "
                        f"{len(shared)} shared sample indices"
                    )
                out.append(
                    Finding(
                        "high",
                        "dl_split_overlap",
                        path,
                        detail,
                        evidence=(
                            Evidence(
                                source="code",
                                kind="overlap",
                                description=detail,
                                location=ev.location,
                                observed=observed,
                                expected="disjoint training and evaluation samples",
                            ),
                            Evidence(
                                source="code",
                                kind="overlap",
                                description="the training side of the same pair",
                                location=tr.location,
                                observed=observed,
                                expected="disjoint training and evaluation samples",
                            ),
                        ),
                        stable_key=f"split:{path}:{mode}",
                        diagnostic=True,
                    )
                )
        for signal in items:
            if signal.name != "eval_augmentation":
                continue
            detail = (
                f"validation data is built with a random augmentation ({signal.value}); "
                f"the reported score is measured on randomized inputs"
            )
            out.append(
                Finding(
                    "high",
                    "dl_split_overlap",
                    path,
                    detail,
                    evidence=(
                        Evidence(
                            source="code",
                            kind="pattern",
                            description=(
                                "an evaluation dataset is constructed with a random "
                                "augmentation transform"
                            ),
                            location=signal.location,
                            observed=f"transform={signal.value}",
                            expected="a deterministic evaluation transform",
                        ),
                    ),
                    stable_key=f"split:{path}:eval_augmentation",
                    diagnostic=True,
                )
            )
    return _unique(out)


def _finding(
    kind: str,
    severity: str,
    column: str,
    detail: str,
    stable_key: str,
    evidence: Evidence,
) -> Finding:
    return Finding(
        severity,
        kind,
        column,
        detail,
        evidence=(evidence,),
        stable_key=stable_key[:180],
        diagnostic=True,
    )


def _unique(findings: list[Finding]) -> list[Finding]:
    unique: dict[str, Finding] = {}
    for finding in findings:
        unique.setdefault(finding.fingerprint, finding)
    return sorted(unique.values(), key=lambda f: (f.kind, f.column, f.detail))


def training_findings(signals: list[TrainingSignal]) -> list[Finding]:
    """Optimizer/scheduler/early-stopping consistency with the claimed protocol."""
    by_path: dict[str, list[TrainingSignal]] = defaultdict(list)
    for signal in signals:
        by_path[signal.path].append(signal)
    out: list[Finding] = []
    for path, items in sorted(by_path.items()):
        epochs = [
            int(signal.value)
            for signal in items
            if signal.name == "max_epochs" and signal.value.isdigit()
        ]
        for signal in items:
            if signal.name == "early_stopping" and epochs:
                patience = int(signal.value)
                limit = max(epochs)
                if patience >= limit:
                    out.append(
                        _finding(
                            "dl_training_protocol_mismatch",
                            "medium",
                            path,
                            f"early stopping patience={patience} is not smaller than "
                            f"max_epochs={limit}; the callback cannot fire before the "
                            f"run ends, so the declared early-stopping protocol is not "
                            f"the one the run follows",
                            f"protocol:{path}:early_stop",
                            Evidence(
                                source="code",
                                kind="pattern",
                                description=(
                                    "an early-stopping patience is greater than or equal "
                                    "to the configured epoch budget"
                                ),
                                location=signal.location,
                                observed=f"patience={patience}, max_epochs={limit}",
                                expected="patience < max_epochs",
                            ),
                        )
                    )
            if signal.name == "optimizer_lr" and float(signal.value) <= 0:
                out.append(
                    _finding(
                        "dl_training_protocol_mismatch",
                        "medium",
                        path,
                        f"optimizer learning rate is {signal.value}; a run with a "
                        f"non-positive learning rate cannot update the model",
                        f"protocol:{path}:optimizer_lr",
                        Evidence(
                            source="code",
                            kind="pattern",
                            description="an optimizer is configured with lr <= 0",
                            location=signal.location,
                            observed=f"lr={signal.value}",
                            expected="lr > 0",
                        ),
                    )
                )
            if signal.name == "scheduler_horizon" and float(signal.value) <= 0:
                out.append(
                    _finding(
                        "dl_training_protocol_mismatch",
                        "medium",
                        path,
                        f"scheduler horizon is {signal.value}; a non-positive "
                        f"T_max/total_steps/step_size cannot schedule the claimed "
                        f"training run",
                        f"protocol:{path}:scheduler_horizon",
                        Evidence(
                            source="code",
                            kind="pattern",
                            description="a scheduler is configured with a non-positive horizon",
                            location=signal.location,
                            observed=f"horizon={signal.value}",
                            expected="a positive T_max, total_steps or step_size",
                        ),
                    )
                )
    return _unique(out)


def _reproducibility_finding(
    kind: str,
    path: str,
    detail: str,
    stable_key: str,
    signal: TrainingSignal,
    observed: str,
    expected: str,
) -> Finding:
    return _finding(
        kind,
        "low",
        path,
        detail,
        stable_key,
        Evidence(
            source="code",
            kind="pattern",
            description=detail,
            location=signal.location,
            observed=observed,
            expected=expected,
        ),
    )


def reproducibility_findings(signals: list[TrainingSignal]) -> list[Finding]:
    """M4 checks: seeds, precision and distributed topology (existing kind)."""
    entries = [signal for signal in signals if signal.name == "training_entry"]
    if not entries:
        return []
    any_seed = any(signal.name == "seed" for signal in signals)
    any_determinism = any(signal.name == "determinism" for signal in signals)
    out: list[Finding] = []
    if not any_seed:
        for entry in entries:
            out.append(
                _reproducibility_finding(
                    "no_random_state",
                    entry.path,
                    "training is configured but no seed call was found in the "
                    "scanned files; the run cannot be reproduced from its own "
                    "configuration",
                    f"seed:{entry.path}",
                    entry,
                    "no torch.manual_seed / seed_everything / set_random_seed call",
                    "a recorded seed in the training entry point",
                )
            )
    if not any_determinism:
        for signal in signals:
            if signal.name == "precision":
                out.append(
                    _reproducibility_finding(
                        "no_random_state",
                        signal.path,
                        f"mixed precision ({signal.value}) is configured without a "
                        f"deterministic-algorithm flag; repeated runs may differ",
                        f"precision:{signal.path}",
                        signal,
                        f"precision={signal.value}, no determinism flag",
                        "torch.use_deterministic_algorithms / enable_op_determinism "
                        "and a recorded seed",
                    )
                )
            if signal.name == "distributed":
                out.append(
                    _reproducibility_finding(
                        "no_random_state",
                        signal.path,
                        f"distributed training ({signal.value}) is configured without a "
                        f"deterministic-algorithm flag; repeated runs may differ",
                        f"distributed:{signal.path}",
                        signal,
                        f"strategy={signal.value}, no determinism flag",
                        "a per-rank seed and a determinism flag recorded in the "
                        "training entry point",
                    )
                )
    return _unique(out)
