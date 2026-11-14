# Deep-learning and artifact auditing

Phase 16 adds a static audit for deep-learning repositories that answers three
questions: **was the evaluation split disjoint, can the run be reproduced from its
own configuration, and what does the checkpoint actually prove?** It reads Python
training code with `ast`, documented framework configs, requirement declarations
and model metadata. It never imports torch/TensorFlow/Keras/ONNX, never unpickles a
checkpoint, never extracts an archive and never executes the scanned project.

The engine is `leaklens.dl`, driven by `leaklens dl DIR`. Every finding is
`diagnostic: true` and `scored: false`: it never changes `leakage.score()`, an
existing finding's severity or the exit code of an existing command. The
subcommand exits `0` by default and `1` only with `--fail-on-findings`.

## Quick start

```sh
leaklens dl .                       # audit a repository
leaklens dl . --json                 # machine-readable report
leaklens dl . --fail-on-findings     # opt-in: exit 1 when anything is reported
leaklens dl . --max-files 20000 --max-artifact-mb 128
```

| Kind | Severity | Signal |
|---|---|---|
| `dl_split_overlap` | high | train and eval bind the same dataset object, overlap literal `Subset` indices, or evaluation data is built with a random augmentation |
| `dl_training_protocol_mismatch` | medium | early-stopping patience ≥ `max_epochs`, optimizer `lr <= 0`, or scheduler horizon (`T_max`/`total_steps`/`step_size`) `<= 0` |
| `artifact_unsafe_format` | high | a pickle-based checkpoint or archive was refused before deserialization |
| `no_random_state` | low | training is configured without a recorded seed, or mixed precision / distributed training has no determinism flag |
| `pipeline_unpinned_dependency` | low | a `requirements*.txt` entry has no exact version pin |

The last two kinds are the Phase 14 reproducibility family, reused unchanged in
name and severity; Phase 16 adds no second family and no second report format.

## What is scanned

- `.py` files, parsed with `ast`; syntax errors and non-UTF-8 files are typed skips.
- `.yaml`/`.yml`/`.json` files whose name contains `config`, `hparams`,
  `hyperparams` or `params` (framework configs, M5).
- `requirements*.txt` (pins, M4).
- Model artifacts by extension: `.safetensors`, `.onnx`, `.pkl`, `.pickle`,
  `.joblib`, `.pt`, `.pth`, `.bin`, `.ckpt`, and the archive families `.zip`,
  `.tar`, `.gz`, `.tgz`, `.bz2`, `.xz`, `.7z`, `.rar`, `.npz`, `.xlsx`, `.jar`,
  `.whl`. A `.json` file whose stem contains `model`, `models`, `metadata` or
  `weights` is treated as artifact metadata when it is not also a config.

Directories named `.git`, `.hg`, `.svn`, `node_modules`, `__pycache__`, `.venv`,
`venv`, `.tox`, `.mypy_cache`, `.pytest_cache`, `.ruff_cache`, `.eggs`, `dist`,
`build` and `site-packages`, plus every dot-directory, are pruned without being
descended. A symlinked file that resolves outside the audit root is an
`outside_root` skip; symlinked directories are never followed.

## Split provenance (M3)

Two bindings are compared per file: a **train** name (`train`, `training`) and an
**eval** name (`val`, `valid`, `validation`, `eval`, `test`) bound to a dataset
value — a name containing `ds`, `dataset(s)`, `loader(s)` or `data`, or a call to
`DataLoader`, `Subset`, `Dataset`, `ImageFolder`, `random_split`, `TensorDataset`,
`ConcatDataset` or any `*Dataset` callable.

- **Same dataset object** (`DataLoader(dataset)` on both sides) is a high finding.
- **Literal `Subset` indices** are intersected. Disjoint literals are clean. If
  either index set is a variable (`order[:80]`), the pair is not guessed and no
  finding is produced.
- **Random evaluation augmentation**: an eval dataset constructed with a
  stochastic transform (`RandomCrop`, `RandomResizedCrop`, `RandomHorizontalFlip`,
  `ColorJitter`, `RandAugment`, …) is a high finding, whether the transform is
  inline or a named variable. `p=0` disables the augmentation and is clean;
  deterministic transforms (`Resize`, `CenterCrop`) are clean.

Only literal facts are reported. This is static evidence about what the code says,
not proof of which samples a run actually saw.

## Reproducibility (M4)

Reproducibility reuses `no_random_state`. A training entry (`Trainer`, `fit`, or a
call with `epochs`/`max_epochs`) in the scanned files requires a recorded seed
(`manual_seed`, `manual_seed_all`, `seed_everything`, `set_seed`,
`set_random_seed`) anywhere in the same audit. Mixed precision (`autocast`,
`GradScaler`, `Trainer(precision=...)`, `set_global_policy`, or a `precision`
config field) and distributed topology (`DDP`, `FSDP`, `init_process_group`,
`Trainer(strategy=..., devices=...)`, `MirroredStrategy`) additionally require a
determinism flag (`use_deterministic_algorithms`, `enable_op_determinism`,
`deterministic=True`, `cudnn.deterministic=True`) when one is not present.

Requirement pins reuse `pipeline_unpinned_dependency`: an entry without `==` (or a
URL) is reported once per package per file. The advice key is shared with the
pipeline family.

## Protocol consistency (M1)

Three contradictions between what the configuration claims and what the run can
do are reported as `dl_training_protocol_mismatch`:

- early-stopping `patience >= max_epochs` (the callback cannot fire before the run
  ends);
- optimizer `lr <= 0` (no update can happen);
- scheduler horizon `<= 0` (nothing can be scheduled).

## Framework adapters (M5)

`analysis/dl/config.py` recognises three documented shapes by recursive key
membership: Lightning (`trainer`, `seed_everything`, `max_epochs`, `strategy`,
`precision`), Keras (`compile`, `loss`, `metrics`, `fit`, `validation_split`) and
raw PyTorch (`optimizer`, `lr`, `scheduler`, `num_epochs`, `batch_size`). A file
that matches none of them, or matches several equally, is `unknown` and produces no
signals — never a crash and never a guess.

YAML parsing uses `safe_load` only and requires the existing `yaml` extra
(`pip install "leaklens-ml[yaml]"`). Without it, YAML configs are a
`yaml_not_installed` skip; JSON configs never need it. Malformed, oversized
(>2 MiB) and non-object documents are `config_unreadable` skips.

## Artifact intake (M2/M6, M-S2)

No artifact is deserialized. The intake (`analysis/artifacts/`) reads metadata only:

| Format | What is read | Caps |
|---|---|---|
| safetensors | 8-byte header length, JSON header, tensor names/dtypes/offsets, `__metadata__` | header ≤ 16 MiB, ≤ 100 000 tensors, ≤ 32 recorded names, offsets validated and non-overlapping |
| ONNX | protobuf fields only: producer, graph name, opset imports, graph inputs/outputs, initializer count/names | file ≤ 64 MiB, ≤ 200 000 fields, nesting ≤ 16, every length checked against the buffer |
| JSON | top-level type, key/item count, SHA-256 | ≤ 2 MiB, depth ≤ 64, duplicate keys rejected |

Pickle magic (`\x80\x02`–`\x80\x05`) is checked **before** the extension, so a
pickle renamed to `.safetensors` is still refused. Pickle extensions (`.pkl`,
`.pickle`, `.joblib`, `.pt`, `.pth`, `.bin`, `.ckpt`) without magic are refused by
extension; `.bin` weights cannot be proven safe and are refused too. A refusal is a
first-class high finding (`artifact_unsafe_format`) whose advice is to re-export
metadata as safetensors, ONNX or JSON.

Archives are refused without extraction (`archive_refused` skip), including
traversal, encrypted and bomb-shaped members. A truncated or malformed safe file
is an `artifact_unreadable` skip. Every refusal and skip is counted in the report's
`skipped` map, so nothing disappears into a log line.

The intake applies `paths.resolve_within`, so a symlink or path that resolves
outside the audit root is `artifact_outside_root`; foreign drives, UNC and device
anchors are rejected before `resolve()`.

## Evidence and privacy

`DlReport.to_dict()` carries `status` (`ok`/`no_artifacts`), bounded `signals`
(name, truncated value, repo-relative path, line or config field, role),
`artifacts` (relative path, format, digest and bounded metadata) and `skipped`.
Findings cite a repo-relative `file:line` or `file:field`; absolute roots never
appear, and checkpoint weights, row data and full source are never serialized.

Scanned content is inert data. A comment such as "ignore previous instructions"
is parsed as Python and changes nothing (covered by a benchmark case and a test).

## Scoring status

All five kinds are diagnostics. No labelled real-world deep-learning corpus was
evaluated, so none is scored and none graduates (ADR-006); `scored=False` is
asserted by the manifest contract tests. The curated benchmark in
[DL_BENCHMARK.md](DL_BENCHMARK.md) is a regression guard, not an accuracy claim.

## Limits

- **Static only.** Findings are claims about code and configuration. A seed set in
  an unscanned file, a dataset produced at runtime, or a transform resolved
  dynamically is invisible; the engine says nothing rather than guessing.
- **Same-file pairing.** Split provenance compares bindings inside one file. A
  train/eval pair defined in two files is not correlated in this phase.
- **Documented shapes only.** Framework configuration evolves; an unrecognised
  shape degrades to `unknown`, which is a skip, not a finding.
- **No model execution, conversion or inference**, no GPU diagnostics, no
  reimplementation of framework behaviour.
- **Archives are never opened.** LeakLens refuses them; it cannot say whether a
  member is safe for a downstream consumer.
- **Not a proof of training history.** A clean audit does not prove the reported
  score was honestly produced, only that the audited files contain no configured
  contradiction it can see.
