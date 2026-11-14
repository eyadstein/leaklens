"""Framework config adapters and reproducibility checks (Phase 16 M4/M5).

Three documented shapes are recognised: Lightning (``trainer``/``max_epochs``),
Keras (``compile``/``loss``/``metrics``) and raw PyTorch (``optimizer``/``lr``/
``num_epochs``). An unknown shape is skipped, never guessed. Configuration is
untrusted input: a malformed or oversized file is a typed skip.

No reproducibility family is added here: findings reuse ``no_random_state`` and
``pipeline_unpinned_dependency`` from Phase 14, with framework-neutral advice.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from importlib import util as importlib_util
from pathlib import Path
from typing import Any

from ...availability import EngineUnavailable
from ...evidence import Evidence
from ...leakage import Finding
from .training import TrainingSignal, mixed_precision

MAX_CONFIG_BYTES = 2 * 1024 * 1024

LIGHTNING_KEYS = frozenset(
    {"trainer", "seed_everything", "max_epochs", "accelerator", "strategy", "precision"}
)
KERAS_KEYS = frozenset(
    {"compile", "loss", "metrics", "validation_split", "keras", "fit"}
)
PYTORCH_KEYS = frozenset(
    {"lr", "learning_rate", "optimizer", "scheduler", "num_epochs", "batch_size"}
)
SEED_KEYS = frozenset(
    {"seed_everything", "seed", "random_seed", "manual_seed", "random_state"}
)
DETERMINISM_KEYS = frozenset(
    {"deterministic", "use_deterministic_algorithms", "cudnn_deterministic"}
)
MAX_EPOCH_KEYS = frozenset({"max_epochs", "epochs", "num_epochs"})
LR_KEYS = frozenset({"lr", "learning_rate"})
HORIZON_KEYS = frozenset({"t_max", "total_steps", "step_size"})
DEVICE_KEYS = frozenset({"devices", "num_processes", "world_size", "gpus"})
DISTRIBUTED_VALUES = frozenset({"ddp", "ddp2", "ddp_spawn", "fsdp", "deepspeed"})


class DlUnavailable(EngineUnavailable):
    """The `yaml` extra is required to read a YAML training config."""


@dataclass(frozen=True, slots=True)
class DlConfig:
    framework: str
    values: dict[str, Any]
    source: str


def _yaml_module() -> Any:
    if importlib_util.find_spec("yaml") is None:
        return None
    import yaml

    return yaml


def _keys(value: Any) -> set[str]:
    found: set[str] = set()
    if isinstance(value, dict):
        for key, child in value.items():
            found.add(str(key).lower())
            found |= _keys(child)
    elif isinstance(value, list):
        for child in value:
            found |= _keys(child)
    return found


def framework_of(values: dict[str, Any]) -> str:
    """The documented shape that best matches ``values``, or ``"unknown"``."""
    keys = _keys(values)
    scores = {
        "lightning": len(keys & LIGHTNING_KEYS),
        "keras": len(keys & KERAS_KEYS),
        "pytorch": len(keys & PYTORCH_KEYS),
    }
    best = max(scores.values())
    winners = [name for name, score in scores.items() if score == best and score > 0]
    return winners[0] if len(winners) == 1 else "unknown"


def load_config(
    path: str | Path,
    yaml_mod: Any = ...,
    *,
    source: str | None = None,
    max_bytes: int = MAX_CONFIG_BYTES,
) -> DlConfig | None:
    """Read one JSON/YAML config; ``None`` when malformed or unreadable."""
    file = Path(path)
    suffix = file.suffix.lower()
    if suffix in {".yaml", ".yml"} and yaml_mod is ...:
        yaml_mod = _yaml_module()
    if suffix in {".yaml", ".yml"} and yaml_mod is None:
        raise DlUnavailable(
            'training configs need PyYAML: pip install "leaklens-ml[yaml]"'
        )
    try:
        if file.stat().st_size > max_bytes:
            return None
        text = file.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError):
        return None
    try:
        values = (
            yaml_mod.safe_load(text)
            if suffix in {".yaml", ".yml"}
            else json.loads(text)
        )
        if not isinstance(values, dict):
            return None
        framework = framework_of(values)
    except Exception:  # noqa: BLE001 - any parser/recursion error is a typed skip
        return None
    return DlConfig(framework, values, source or file.name)


def requirement_pin_findings(text: str, source: str) -> list[Finding]:
    """Reuse the Phase 14 pin rule for a requirements file."""
    out: list[Finding] = []
    seen: set[str] = set()
    for raw in text.splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line or line.startswith("-"):
            continue
        package = line.split(";", 1)[0].strip()
        if "==" in package or "://" in package:
            continue
        name = package.split("[", 1)[0]
        for marker in ("===", ">=", "<=", "~=", "!=", ">", "<", "="):
            name = name.split(marker, 1)[0]
        name = name.strip()
        if not name or name in seen:
            continue
        seen.add(name)
        out.append(
            Finding(
                "low",
                "pipeline_unpinned_dependency",
                source,
                f"dependency {name!r} in {source!r} has no exact version pin; "
                f"re-running the pipeline can resolve a different version and "
                f"change the trained model",
                evidence=(
                    Evidence(
                        source="code",
                        kind="pattern",
                        description="a requirements entry has no exact version pin",
                        location=source,
                        observed=package[:200],
                        expected="package==version",
                    ),
                ),
                stable_key=f"pin:{source}:{name}"[:180],
                diagnostic=True,
            )
        )
    return out


def _distributed_value(value: Any) -> bool:
    text = str(value).lower()
    return any(marker in text for marker in DISTRIBUTED_VALUES)


def _signal(name: str, value: Any, path: str, field: str) -> TrainingSignal:
    return TrainingSignal(name, str(value)[:120], path, field=field)


def _walk(node: Any, dotted: str, path: str, out: list[TrainingSignal]) -> None:
    if isinstance(node, dict):
        for key in sorted(node, key=str):
            child = node[key]
            field = f"{dotted}.{key}" if dotted else str(key)
            lowered = str(key).lower()
            if isinstance(child, bool):
                if lowered in DETERMINISM_KEYS and child:
                    out.append(_signal("determinism", lowered, path, field))
            elif isinstance(child, (int, float)):
                if lowered in MAX_EPOCH_KEYS:
                    out.append(_signal("max_epochs", child, path, field))
                elif lowered in SEED_KEYS:
                    out.append(_signal("seed", child, path, field))
                elif lowered in LR_KEYS:
                    out.append(_signal("optimizer_lr", child, path, field))
                elif lowered in HORIZON_KEYS:
                    out.append(_signal("scheduler_horizon", child, path, field))
                elif lowered in DEVICE_KEYS and child > 1:
                    out.append(_signal("distributed", child, path, field))
                elif lowered == "precision" and mixed_precision(child):
                    out.append(_signal("precision", child, path, field))
            elif isinstance(child, str):
                if lowered in SEED_KEYS:
                    out.append(_signal("seed", child, path, field))
                elif lowered == "precision" and mixed_precision(child):
                    out.append(_signal("precision", child, path, field))
                elif lowered == "strategy" and _distributed_value(child):
                    out.append(_signal("distributed", child, path, field))
                elif lowered in LR_KEYS:
                    try:
                        out.append(_signal("optimizer_lr", float(child), path, field))
                    except ValueError:
                        pass
            _walk(child, field, path, out)
            if (
                str(key).lower() == "class_path"
                and isinstance(child, str)
                and "earlystopping" in child.lower()
            ):
                args = node.get("init_args")
                if isinstance(args, dict) and isinstance(args.get("patience"), int):
                    out.append(
                        _signal(
                            "early_stopping",
                            args["patience"],
                            path,
                            f"{field}.init_args.patience",
                        )
                    )
    elif isinstance(node, list):
        for index, child in enumerate(node):
            _walk(child, f"{dotted}[{index}]", path, out)


def config_signals(config: DlConfig) -> list[TrainingSignal]:
    """Extract the same signals the code scanner uses, with config field paths."""
    if config.framework == "unknown":
        return []
    entry = {"lightning": "trainer", "keras": "epochs", "pytorch": "num_epochs"}[
        config.framework
    ]
    out = [
        TrainingSignal("training_entry", config.framework, config.source, field=entry)
    ]
    _walk(config.values, "", config.source, out)
    return out
