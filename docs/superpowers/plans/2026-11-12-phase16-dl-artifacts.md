# Phase 16 — Deep learning and artifact auditing

Source: `docs/AI_ML_EXPANSION_ROADMAP.md` §5 (lines 289–349). Dependencies: Phase 13;
Phase 14 for split provenance **and** reproducibility. Status at planning time:
`main` = `da86dc8fbd12e707f76c840df1a6dc689bc162d0` (= `origin/main`), clean except
the protected `.phase5-bench.json` and `graphify-out/`.

## Implementation boundary

### 1. Roadmap requirements (verbatim scope)

- Static analysis of PyTorch/Lightning/Keras **training code and configs**: seeds,
  determinism flags, precision, distributed config, early stopping,
  optimizer/scheduler/early-stopping consistency.
- Checkpoint metadata parsing **without execution**: `safetensors` header and JSON
  metadata; `pickle`/`.pt`/`.pth`/`.bin` refused with a finding.
- Train/eval split provenance for DL: same file, overlapping indices, augmented
  duplicates; validating on training augmentations is `high`.
- Reproducibility **configuration only** (precision mode, distributed topology,
  requirement pins), reusing Phase 14 reproducibility kinds; no new family and no
  second report format.
- Framework adapters (PyTorch/Lightning/Keras config shapes); unknown frameworks
  skip, never crash.
- `onnx.py` at M6: graph inputs/outputs and opset parsed from protobuf bytes with a
  size cap; no ONNX runtime installed or required.
- `bench_dl`: ≥20 labelled repositories, precision/recall published, misses listed.
- New: `analysis/dl/{config,checkpoints,training}.py`,
  `analysis/artifacts/{formats,safetensors,pickle_guard,onnx}.py`,
  `docs/DL_AUDIT.md`, `leaklens/bench_dl.py`, `docs/DL_BENCHMARK.md`.

Acceptance: (1) LeakLens never imports torch/TensorFlow to audit a project;
(2) no serialized artifact is unpickled and refusal is a first-class finding;
(3) every claim cites file/line or a checkpoint field; (4) config shapes degrade
gracefully across framework versions. Non-goals: no training, inference,
conversion, GPU diagnostics or reimplementation of framework behaviour.

### 2. Dependencies already satisfied

- `evidence.Evidence` (bounded fields, aggregate-only), `availability.EngineUnavailable`
  + `probe`, `paths.resolve_within`/`contained`, `leakage.Finding` + single
  `leakage.score()`, `assemble.build`, `manifest` catalogue + `ENGINES`,
  `tests/test_architecture.py`, `tests/_meta/test_detector_contract.py`,
  `tests/test_public_contract.py`.
- Phase 14 kinds and conventions: `no_random_state`,
  `pipeline_unpinned_dependency`, `pipeline_split_overlap`, diagnostics with
  `scored=False`, benchmark as regression guard, framework-neutral advice.
- Phase 15 conventions: opt-in CLI subcommand, `--fail-on-findings`, bounded
  evidence, deterministic fingerprints, published misses, no score/exit change.

### 3. Missing gates

No new dependency, no extra, no network, no GPU. PyYAML stays the existing `yaml`
extra and degrades to a skip; ONNX protobuf and safetensors are parsed from bytes
with the standard library. M5 is met by adapter functions over shapes LeakLens
documents, not by installing the frameworks.

The artifact engine also ships the **artifact half of M-S1** (which
`ARTIFACT_SECURITY_STRATEGY.md` §6 assigns to Phase 16): one typed intake contract
(magic bytes first, size/field budgets, a reason on every refusal), archives
refused without extraction, and a 30-case hostile artifact/archive corpus proving
no crash and no write. The image-decoder half of M-S1 stays with Phase 17, as the
same row says "then 17 (images)". M-S2 is met by the pickle refusal finding, the
safetensors/JSON capped parsers and the no-`torch.load` AST check.

### 4. Modules and public interfaces

| Module | Interface |
|---|---|
| `analysis/artifacts/__init__.py` | `ArtifactParseError`, `ArtifactMeta`, `SafetensorsMeta`, `OnnxMeta` re-exports; no parsing logic (avoids submodule cycles) |
| `analysis/artifacts/formats.py` | `format_of(name, head)`, `inspect_artifact(path, root, *, max_bytes)` -> `(ArtifactMeta | None, Finding | None, skip_reason | None)`, `inspect_json(data)`, `INTAKE_REFUSALS`; pickle/zip magic is checked **before** extension dispatch |
| `analysis/artifacts/safetensors.py` | `SafetensorsMeta(tensor_count, tensors, dtypes, metadata, metadata_truncated, header_digest)`, `parse_safetensors(data)` (header cap, offset validation, bounded recorded metadata) |
| `analysis/artifacts/pickle_guard.py` | `UNSAFE_EXTENSIONS`, `unsafe_reason(name, head) -> str | None`, `looks_like_pickle(head)`, `looks_like_archive(head)` |
| `analysis/artifacts/onnx.py` | `OnnxMeta(producer, opset, inputs, outputs, initializer_count, initializers)`, `parse_onnx(data)` (bounded protobuf reader: depth, field count, length-vs-remaining, varint width; no runtime) |
| `analysis/dl/config.py` | `DlConfig(framework, values, source)`, `load_config(path, yaml_mod)`, `framework_of(mapping)`, `config_signals(...)`, `requirement_pin_findings(...)`, `DlUnavailable(EngineUnavailable)` |
| `analysis/dl/training.py` | `TrainingSignal`, `scan_python(root, files)` (AST only, per-file byte cap), `split_findings(signals)`, `training_findings(signals)`, `reproducibility_findings(signals, *, source)` |
| `analysis/dl/checkpoints.py` | `CheckpointReport(artifacts, findings, skipped)`, `inspect_artifacts(root, files, *, max_bytes)` |
| `analysis/dl/__init__.py` | `DlReport(status, findings, signals, artifacts, skipped)` (`status` is `ok|no_artifacts`, the Phase 14 vocabulary; no absolute root is serialized), `audit_dl(root, *, strict_yaml=False, max_files, max_artifact_bytes)` |
| `leaklens/cli.py` | `dl(argv)` subcommand (`leaklens dl PATH [--json] [--fail-on-findings]`), dispatched before the main parser |
| `leaklens/bench_dl.py` | `_build_cases()`, `evaluate()`, `markdown()`, `main(argv)`; `_build_cases()` is pure and imports cleanly, cases materialise inside `evaluate()` |

New finding kinds (all `surface="code"`, `diagnostic=True`, `scored=False`, added
to `_DIAGNOSTIC_KINDS`):

| kind | severity | modality | milestone | benchmark |
|---|---|---|---|---|
| `dl_split_overlap` | high | code | M3 | `same dataset on both sides` |
| `dl_training_protocol_mismatch` | medium | code | M1 | `early stopping can never fire` |
| `artifact_unsafe_format` | high | artifact | M2 | `pickle checkpoint refused` |

M4 adds **no kind**: precision/distributed/seed checks emit the existing
`no_random_state`, and requirement pins emit the existing
`pipeline_unpinned_dependency`. One engine row `leaklens.dl` owns the three new
kinds plus the two reused kinds; advice keys are added to `FIXES`, and
`FIXES["no_random_state"]` is widened to seed + determinism + precision + topology
(framework-neutral, as Phase 14 M4 requires).

`dl_training_protocol_mismatch` covers the roadmap's optimizer/scheduler/early-stop
consistency: early-stopping patience that can never fire within `max_epochs`, an
optimizer learning rate `<= 0`, and a scheduler horizon (`T_max`/`total_steps`)
`<= 0`. Each is a literal-value static claim with file/line evidence.

### 5. Contracts

- **Evidence**: source `code`, kind `pattern`/`overlap`/`parse`; `location` is
  `relative/path.py:LINE` or `relative/config.yaml:FIELD`/`checkpoint.safetensors:__metadata__`;
  `description`/`observed`/`expected` stay within `MAX_TEXT`/`MAX_FIELD`; never raw
  file bytes, tensor data, weights, secrets or document text. Artifact metadata is
  summarized: tensor count, at most 32 recorded names (each capped), dtype counts,
  at most 32 metadata entries (values truncated to 300 chars), and a header/byte
  SHA-256 digest. No absolute machine path and no `root` field is serialized.
- **Identity**: `column` is the repo-relative POSIX file (or artifact) path;
  `stable_key` names the mode/symbol/field (never a line number or count), so
  fingerprints survive line moves and repeated counts.
- **Determinism**: sorted file walk, sorted signals/artifacts/findings, no
  timestamps or absolute machine paths in findings.
- **Scoring**: every new finding is `diagnostic=True`; `leakage.score()` is
  unchanged and remains the only score; the CLI exits 0 by default and 1 only with
  `--fail-on-findings`. Existing commands and gates are untouched.
- **Assembly**: `leaklens dl` builds one ordinary finding list and serializes with
  `assemble.build`; its JSON report adds a `dl` section (signals/artifacts/skips)
  beside the shared sections. No second report format.
- **Availability**: PyYAML is the only optional parser (`yaml` extra); without it a
  YAML file is skipped with `yaml_not_installed` and an actionable stderr line.
  No torch/TensorFlow/onnx import exists anywhere in the audit path.

### 6. Tests and adversarial cases

New: `tests/test_dl_training.py`, `tests/test_dl_split.py`, `tests/test_dl_config.py`,
`tests/test_artifacts_safetensors.py`, `tests/test_artifacts_pickle_guard.py`,
`tests/test_artifacts_onnx.py`, `tests/test_artifacts_intake.py` (30-case hostile
archive/artifact corpus), `tests/test_dl_checkpoints.py`, `tests/test_dl_cli.py`,
`tests/test_bench_dl.py`, `tests/_meta/test_no_torch_load.py`.
Updated: `tests/_meta/test_detector_contract.py` (benchmark cases + diagnostic set),
`tests/test_public_contract.py` (engine id + golden fingerprint per new kind).

Covered: determinism (same tree → same findings/fingerprints, byte-identical JSON,
no `%TEMP%` in output), bounded evidence, malformed/truncated artifacts, oversized
headers, pickle refusal matrix (protocols 0/1/4/5 and every unsafe extension,
including a pickle renamed `.safetensors`/`.onnx`/no-extension), safetensors
0-byte/magic-only/header-past-EOF/negative/overlapping offsets/duplicate keys/
non-UTF-8/boundary at `MAX_TENSORS`, ONNX garbage/truncation/depth/length-past-
remaining/varint overflow/size cap, malformed YAML/JSON, framework-version shapes,
unknown framework skip, symlink loop and symlink→outside-root and vanishing/
permission-denied files (typed skip, never a crash), input per-file byte cap,
prompt-injection-looking source text is inert (no instruction following), no
score/severity/exit change, no architecture cycle, no new cycle from
`analysis.artifacts` error location.

`tests/_meta/test_no_torch_load.py` statically resolves `import torch`,
`import torch as X`, `from torch import load` and `from torch import load as Y`
(including function-local imports) over `leaklens/**/*.py` and asserts no call; the
same resolver is proven on fixture sources inside `tests/`. It also asserts no
`import tensorflow`/`keras` and no ONNX runtime import in the audit path.

### 7. Benchmark and evaluation

`bench_dl` writes ≥20 labelled synthetic repositories to a temp dir and audits
them with `audit_dl`. `_build_cases()` is a pure case table (so the detector
contract test can import it); each case declares the exact `(kind, column)` truth
set and materialises its files only inside `evaluate()`. Metrics are TP/FP/FN,
precision, recall, F1, FP rate **overall and per finding kind**; misses are printed
and listed in `docs/DL_BENCHMARK.md`. Families: clean seeded Lightning, missing
seed, precision/distributed without determinism, early-stopping mismatch, optimizer/
scheduler mismatch, clean Keras, unknown framework, malformed YAML/JSON, same
train/eval dataset, overlapping `Subset` indices, train transform on validation,
clean disjoint subsets, pinned and unpinned requirements, clean safetensors, valid
JSON metadata, valid ONNX, pickle/`.pt`/`.bin` refusals, pickle renamed
`.safetensors`, truncated safetensors, oversized header, adversarial archive member.

Pass criterion: `fp == 0 and recall >= RECALL_FLOOR` on this curated set only, with
`RECALL_FLOOR` fixed before the first run. YAML cases are skipped with a recorded
reason when the `yaml` extra is absent; the published recall never silently counts
a skipped case as a hit. Artifact-parse timings and `tracemalloc` peak are recorded,
not gated (runner dependent); the declared byte caps are asserted on file size, not
on measured peak memory. No real-world accuracy claim.

### 8. Performance and resources

- `MAX_FILES = 5000`, `MAX_PY_FILES = 2000`, `MAX_PY_FILE_BYTES = 2 MiB`,
  `MAX_CONFIG_BYTES = 2 MiB`, `MAX_SAFETENSORS_HEADER = 16 MiB`,
  `MAX_ONNX_BYTES = 64 MiB`, `MAX_ARTIFACT_BYTES = 64 MiB`, `MAX_TENSORS = 100_000`,
  `MAX_RECORDED_TENSORS = 32`, `MAX_METADATA_KEYS = 32`, `MAX_NAME = 300`,
  `MAX_PROTO_FIELDS = 200_000`, `MAX_PROTO_DEPTH = 16`.
- Artifact reads are bounded: magic head, safetensors header only, ONNX up to the
  cap; larger files become a typed skip, never a full read. Every artifact counts
  against `MAX_FILES`; every `.py` file is size-checked before `ast.parse`.
- Archive formats are refused by magic without extraction; hostile archive bytes
  are inert.
- No parallelism, no cache, no network; one deterministic pass.

### 9. Explicitly excluded

- No torch/TensorFlow/onnx import or execution; no `torch.load`; no unpickling
  (refusal is the finding); no checkpoint conversion or weight mapping.
- No training, inference, GPU diagnostics, distributed launch or scheduler
  reimplementation.
- No new reproducibility family, no new report format, no second score.
- No workspace/API surface (not required by §5); CLI + shared assembly only.
- No new dependency or extra; PyYAML remains the existing `yaml` extra.
- No refactor of `images.image_overlap`; no Phase 14 deferred work.
- No scoring graduation; no version bump, tag or release.

### 10. Milestone sequence (TDD: RED → GREEN each)

1. **M1 training signals** — `test_dl_training.py`: seed/determinism/precision/
   distributed/early-stopping extraction, `dl_training_protocol_mismatch`, static
   file/line evidence; then `training.scan_python` + `training_findings`.
2. **M2 artifacts** — `test_artifacts_safetensors.py`, `pickle_guard`, `formats`,
   `test_artifacts_intake.py` (30-case hostile corpus), `_meta/test_no_torch_load.py`;
   then the safetensors/JSON parsers, the pickle-first refusal matrix, archive
   refusal and the magic/extensions guard.
3. **M3 split provenance** — `test_dl_split.py`: same dataset, overlapping indices,
   train transform on validation, clean negatives; then `split_findings` + manifest
   kind `dl_split_overlap`.
4. **M4 reproducibility config** — extend `test_dl_training.py`/`test_dl_config.py`:
   precision/distributed/seed checks emit `no_random_state`; requirement pins emit
   `pipeline_unpinned_dependency`; then `reproducibility_findings` (both code and
   config inputs).
5. **M5 adapters** — `test_dl_config.py`: Lightning/PyTorch/Keras shapes, version
   variants, unknown framework skip, malformed YAML/JSON skip, missing PyYAML;
   then `config.load_config`/`framework_of`.
6. **M6 ONNX** — `test_artifacts_onnx.py`: graph inputs/outputs/opset/initializers,
   truncation, garbage, size cap; then `onnx.parse_onnx` + `formats` dispatch.
7. **Orchestration + CLI** — `test_dl_checkpoints.py`, `test_dl_cli.py`; then
   `checkpoints.inspect_artifacts`, `audit_dl`, `cli.dl`, engine row, all manifest
   entries and advice keys.
8. **Benchmark** — `bench_dl.py`, `test_bench_dl.py`, generate
   `docs/DL_BENCHMARK.md`; verify ≥20 cases, metrics and misses.
9. **Docs/meta** — `DL_AUDIT.md`, `PHASE16_ACCEPTANCE.md`, roadmap status,
   `ARTIFACT_SECURITY_STRATEGY.md` §6 M-S1/M-S2 status, README/AGENTS/CI wiring,
   regenerate `DETECTORS.md` + site fixtures, golden fingerprints + engine id.
10. **Verification** — full suite after final review fixes, ruff, `git diff --check`,
    manifest check, self-scan, site check, every benchmark with digest comparison,
    Graphify after-snapshot, Ponytail full-diff review, focused commits, fast-forward
    push, CI/CodeQL/Pages read-back.

## Ponytail challenge (before implementation)

| # | Challenge | Disposition |
|---|---|---|
| 1 | A new `artifact` surface would be cleaner than using `code`. | Rejected: `SURFACES` is a frozen public contract; `code` surface + `artifact` modality already matches the SQL precedent (surface `code`, modality `sql`). No frozen id changes. |
| 2 | Three new kinds may be more than the roadmap needs. | Accepted as the minimum: one for the M3 high split finding, one for M1 protocol inconsistency, one for M2 refusal. M4 reuses existing kinds exactly as required. |
| 3 | A generic framework adapter registry is tempting. | Rejected: three small shape functions plus one skip are enough; no plugin ABI. |
| 4 | `parse_onnx` needs a protobuf dependency. | Rejected: a bounded varint/length-delimited reader over the fields we need is ~100 lines, stdlib only, and testable; no ONNX runtime, no `protobuf` dependency. |
| 5 | Detecting "augmented duplicates" in general is unsound. | Accepted: M3's acceptance narrows it to "validates on training augmentations"; we claim only the static patterns we can cite (same dataset path, overlapping literal `Subset` indices, train transform/augmentation passed to validation) and label them static evidence. |
| 6 | Requirement-pin logic would be duplicated from `pipeline.py`. | Accepted: ~15 lines in `config.py` because the architecture forbids `analysis` importing a root engine package; both emit the same existing kind. Documented as shared rules. |
| 7 | Scanning every `.bin` as a pickle may produce noise. | Accepted: the serialized-artifact policy says `.pt/.pth/.bin` are metadata-only; a refusal finding is the intended result, and valid safetensors/ONNX are dispatched by magic/extension first. |
| 8 | ONNX has no magic bytes, so extension-only dispatch can misclassify. | Accepted: `.onnx` is parsed and validated structurally; a parse failure becomes a skip, never a crash or a guess. |
| 9 | A YAML config needs PyYAML, an existing optional extra. | Accepted: degrade to a typed skip with an actionable install line, matching Phase 14; an explicitly named YAML file raises `DlUnavailable`, matching `PipelineUnavailable`. |
| 10 | The AST no-torch guard could be bypassed by `getattr(torch, "load")`. | Accepted and recorded: the required forms (`import torch`, `from torch import load`, aliases) are covered; dynamic attribute access is documented as residual risk, not silently claimed. |
| 11 | Evidence has no numeric count/ratio field. | Accepted: bounded counts/ratios stay inside `description`/`observed`/`expected`; no schema change. |
| 12 | Do we need `--fail-on-findings`? | Kept: every new kind is diagnostic, so the default exit must stay 0; the opt-in flag is the only failure mode, matching Phase 15. |
| 13 | Independent review: M-S1 is assigned to Phase 16 by the security strategy. | Accepted and added: one typed artifact-intake contract (magic first, budgets, a reason on every refusal), archives refused without extraction, plus a 30-case hostile artifact/archive corpus proving no crash and no write. The image-decoder half stays with Phase 17, exactly as the security row says "then 17 (images)". |
| 14 | Independent review: M-S2's JSON half and digest are missing. | Accepted: `inspect_json` parses artifact JSON with size/depth caps and records a SHA-256 digest; safetensors records a header digest. |
| 15 | Independent review: a pickle renamed `.safetensors`/`.onnx` must still be refused. | Accepted: pickle/archive magic is checked **before** extension dispatch; a parse failure after a clean magic check is a typed skip, not a silent success. |
| 16 | Independent review: optimizer/scheduler consistency is not covered. | Accepted: `dl_training_protocol_mismatch` adds literal `lr <= 0` and scheduler-horizon `<= 0` alongside the early-stopping case. |
| 17 | Independent review: `no_random_state` advice would be wrong for precision/topology. | Accepted: the shared advice text is widened (seed + deterministic algorithms + precision/topology, framework-neutral); generated docs are regenerated. |
| 18 | Independent review: no golden fingerprint for new kinds; `bench_dl` needs `_build_cases()`. | Accepted: a per-kind golden fingerprint test is added, and `bench_dl._build_cases()` is pure and registered in the detector contract. |
| 19 | Independent review: unbounded artifact metadata, ONNX recursion, machine paths, per-file caps. | Accepted: metadata is summarized/truncated, ONNX gets a depth cap and length-vs-remaining check, `root` is not serialized, and every `.py` file has a byte cap. |
| 20 | Independent review: M-S1's "generalise `upload_guard`" could mean refactoring it. | Rejected as a refactor: `upload_guard` is the dataset-upload boundary and changing it risks a shipped surface for no Phase 16 acceptance. The artifact intake follows its rules (magic bytes, budgets, typed reasons) without touching it; the 30-case corpus proves the refusal contract. |

## Ponytail full-diff review (after implementation)

Filled in after the implementation and full-suite verification; every finding and
its disposition is recorded in the Phase 16 report and `PHASE16_ACCEPTANCE.md`.
