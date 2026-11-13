# Phase 16 acceptance record (2026-11-13)

Scope is [AI_ML_EXPANSION_ROADMAP.md](AI_ML_EXPANSION_ROADMAP.md) §5, plus the
artifact half of the security milestones M-S1/M-S2 from
[ARTIFACT_SECURITY_STRATEGY.md](ARTIFACT_SECURITY_STRATEGY.md) §6. This record
states what shipped, what is measured and what remains limited. Phase 15 is
[PHASE15_ACCEPTANCE.md](PHASE15_ACCEPTANCE.md).

## Milestones

| Id | Milestone | Status | Evidence |
|---|---|---|---|
| M1 | Training-code signals (seeds, determinism, precision, distributed config, early stopping) | **Shipped** | `analysis/dl/training.py` (`scan_python`, `TrainingSignal` with repo-relative path plus line or config field); profile/precision/distributed/early-stop signals; `tests/test_dl_training.py` |
| M2 | Checkpoint metadata without execution; pickle refusal per M-S2 | **Shipped** | `analysis/artifacts/formats.py` (one typed intake), `pickle_guard.py` (magic before extension), `safetensors.py` (header/offsets only); `artifact_unsafe_format`; `tests/test_artifacts_safetensors.py`, `tests/test_artifacts_pickle_guard.py`; AST gate `tests/_meta/test_no_torch_load.py` |
| M3 | Split provenance: same file, overlapping indices, augmented evaluation | **Shipped** | `training.split_findings` (`same_dataset`, literal-`Subset` index intersection, random eval augmentation); a synthetic repo that validates on training augmentations is `high`; `tests/test_dl_split.py` |
| M4 | Reproducibility configuration only (precision, distributed topology, pins) | **Shipped** | Reuses Phase 14 kinds `no_random_state` and `pipeline_unpinned_dependency`; no new family, no second report format; `training.reproducibility_findings`, `config.requirement_pin_findings`; `tests/test_dl_config.py` |
| M5 | Framework adapters (PyTorch/Lightning/Keras config shapes) | **Shipped** | `analysis/dl/config.py` (`safe_load`/JSON, documented key shapes, `unknown` → no signals); malformed/oversized/non-object → typed skip; `tests/test_dl_config.py` |
| M6 | `onnx.py` graph metadata without a runtime | **Shipped** | `analysis/artifacts/onnx.py`: bounded protobuf reader for producer, graph name, opset, inputs/outputs and initializer names; no ONNX runtime or protobuf dependency; `tests/test_artifacts_onnx.py` |
| M7 | DL benchmark | **Shipped** | `leaklens/bench_dl.py` (32 labelled repositories, 6 families) → [DL_BENCHMARK.md](DL_BENCHMARK.md); `tests/test_bench_dl.py` |
| M-S1 | Decoder intake contract — **artifact half** | **Shipped (artifact half)** | One typed intake for safetensors/ONNX/JSON; every refusal is a typed skip with a reason; 30-entry hostile corpus (traversal, bomb, encrypted, wrong magic, pickle) produces no crash and no write (`tests/test_artifacts_intake.py`). The image-decoder half stays Phase 17. |
| M-S2 | Serialized-artifact policy enforced in code | **Shipped** | Pickle/`.pt`/`.pth`/`.bin`/`.ckpt`/`.joblib` produce `artifact_unsafe_format` without deserializing; safetensors/JSON/ONNX parse with caps and digests; the AST check resolves `import torch`, `import torch as T`, `from torch import load` and `from torch import load as L` and finds no load call in `leaklens/**/*.py`. |

## Acceptance criteria

| # | Criterion | Result | Evidence |
| ---: |---|---|---|
| 1 | LeakLens never imports torch/TensorFlow to audit a project | Pass | `tests/_meta/test_no_torch_load.py` parses every production module (function-local imports included) and asserts no forbidden top-level module; no framework appears in `pyproject.toml` dependencies. |
| 2 | No serialized artifact is unpickled; refusal is a first-class finding | Pass | Pickle magic and extensions are refused before any parser runs; `artifact_unsafe_format` is `high`, `diagnostic=True`; `tests/test_artifacts_pickle_guard.py`, `tests/test_artifacts_intake.py`. |
| 3 | Every claim cites file/line or checkpoint field | Pass | `Evidence.location` is `file:line` for code and `file:field` for config (`config_signals`); artifact findings cite the relative artifact path; no absolute root appears in `to_dict()` or a finding repr (`tests/test_dl_checkpoints.py`). |
| 4 | Config shapes degrade gracefully across framework versions | Pass | Unknown or ambiguous shapes produce zero signals; malformed YAML, oversized files and non-object documents are skips; missing PyYAML is a typed `DlUnavailable` (`tests/test_dl_config.py`). |

## Integration contract

- **Engine id**: `leaklens.dl` added to `manifest.ENGINES`; the frozen-id test was
  updated additively (no existing id changed).
- **Kinds**: three new kinds (`dl_split_overlap` high,
  `dl_training_protocol_mismatch` medium, `artifact_unsafe_format` high), all
  `requires=None`, `scored=False`, `diagnostic=True`; `artifact_unsafe_format`
  declares `modalities=("artifact",)`. M4 adds no kind.
- **Assembly**: the CLI builds one ordinary finding list and serializes it with
  `assemble.build`; no second scoring path and no new payload section outside the
  subcommand's `dl` report.
- **Exit behaviour**: `leaklens dl` exits `0` by default; `--fail-on-findings` is
  the only opt-in failure mode. Existing commands are unchanged.
- **New modules**: `leaklens/analysis/dl/` (`training.py`, `config.py`,
  `checkpoints.py`, `__init__.py`) and `leaklens/analysis/artifacts/`
  (`formats.py`, `safetensors.py`, `pickle_guard.py`, `onnx.py`), plus
  `leaklens/bench_dl.py`.
- **Dependencies**: none added. YAML configs reuse the existing `yaml` extra; ONNX
  is parsed from bytes without `onnx`/`onnxruntime`.
- **No version bump** and no release/tag in this phase.

## Measurement

`python -m leaklens.bench_dl` on the generating machine: 32 hand-written labelled
repositories, precision 1.00, recall 1.00, F1 1.00, 0 false positives, 0 false
negatives; 0.20 s end-to-end, 3.5 MiB peak traced memory. The pass criterion is
FP = 0 and recall ≥ 0.90 plus every declared skip observed. All numbers are
runner-dependent and are not a pass/fail gate. The set is curated, not a sample of
real projects: a regression guard, never an accuracy claim. Per-case rows,
families and the (empty) miss list are in [DL_BENCHMARK.md](DL_BENCHMARK.md).

## Limitations and deferred work

- **Static evidence, not proof of a run.** Findings describe what code and
  configuration claim. Seeds set in unscanned files, runtime-produced datasets and
  dynamically resolved transforms are invisible; the engine stays silent instead
  of guessing.
- **Split pairing is same-file.** A train/eval pair split across two files is not
  correlated in this phase.
- **Only documented config shapes are read.** New framework versions can fall to
  `unknown`, which is a skip rather than a false finding.
- **Archives are refused, never inspected.** LeakLens cannot say whether a member
  is safe for a downstream consumer; refusal is the product.
- **The image-decoder half of M-S1 is deferred to Phase 17**, together with
  `MAX_IMAGE_PIXELS` image-decode budgets (M-S5). This phase does not refactor
  `upload_guard`.
- **No API/workspace DL surface.** The roadmap does not require one; the CLI and
  the shared assembly path cover the milestone.
- **Scoring graduation is not claimed.** No labelled real-world deep-learning
  corpus was evaluated, so the new kinds stay `scored=False` (ADR-006).

Phase 16 does not require a deferred milestone from Phase 13–15. Phase 17
(computer vision) consumes the artifact intake for image decoders; Phase 18
(investigation/governance) is the next roadmap dependent.
