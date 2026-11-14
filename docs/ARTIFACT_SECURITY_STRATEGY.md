# Artifact security strategy

**Status:** design + verified assessment. Nothing in this document changes production behavior.
It records what was measured on 2026-11-10, what BLA can actually do, and which hardening
milestones are worth scheduling.

**Scope:** LeakLens's own release artifacts, the untrusted artifacts LeakLens ingests (datasets,
archives, model files, images, documents), and the supply chain around both.

**Non-goals:** obfuscating LeakLens source, claiming any artifact is impossible to reverse
engineer, DRM, licence enforcement, or hiding the algorithms. LeakLens is MIT-licensed: its
readability is a feature, not a defect.

---

## 1. What BLA is, verified

BLA (`bla-agents`) is installed as a development tool. Verified in this environment:

| Fact | Value | How verified |
|---|---|---|
| Version | 3.0.1 (Node v24.14.1, win32-x64) | `bla_doctor` |
| Engines ready | JS AST shield, Electron ASAR packer, WASM binary shield, Python `.pyc`/marshal shield, native ELF/Mach-O/PE shield, ZIP/APK archive shield, SQLite runtime, AST parser, asar packer | `bla_doctor` |
| Command surface | `harden`, `defend`, `inspect-release`, `audit`, `protect`, `verify`, `weave-plan`, `verify-weave`, `emit-headers`, `polymorph-report`, packaging lints (R1–R7, lanes, WASM, seal, panel-derive, AES-unwrap, hydrate-dump, XOR-mask), `lint-plaintext-secrets`, `benchmark`, `doctor` | `bla --help`, MCP tool list |
| Positioning | Counter-REA hardening: tarpits, semantic budget breakers, decoys | `bla --help`, `bla_doctor` limits list |

BLA is a **registrant-software and release-artifact hardening tool** aimed at JavaScript,
Electron, WASM, native binaries and Python bytecode. It is not an architectural design tool, not
a dependency auditor and not a data-security scanner. Its score of "REA exposure" measures how
cheaply an AI reverse-engineering agent could reconstruct logic; for an MIT Python library that
number being high is expected and acceptable.

### 1.1 Measured on LeakLens (read-only)

| Target | Command | Result | Interpretation |
|---|---|---|---|
| `leaklens/` package | `bla_audit` | `exposureScore` 98, `protectionStatus` UNPROTECTED, 1,756 functions, 1,780 obligation-ledger nodes, 0 resource limits triggered | Plain source distribution: readable by design. Not a defect for MIT. |
| `leaklens/ai/model.json` | `bla_audit` | unsupported format ("file not found or unsupported format") | BLA cannot audit a JSON model artifact. Its model coverage targets binaries/managed formats, not sklearn coefficient JSON. |
| `dist/leaklens_ml-0.13.0-py3-none-any.whl` | `bla_audit` | refused: "ZIP input cannot be rewritten safely: unsupported versions, timestamps, extra fields, comments, disks, or file attributes" | The audit path wants to rewrite the archive; standard `wheel`/`sdist` archives are not accepted. Audit the unpacked tree or the sdist instead. |
| `dist/` | `bla_inspect_release` | `filesInspected` 0, findings [], scope `local-map-and-js-ts-css-files-only` | Out of scope for a pure-Python wheel. Honest no-op, not a pass. |
| `leaklens/` | `bla_lint_plaintext_secrets` | 0 findings, `shouldRefuseBankrupt: false` | No cleartext secret heuristic hit in the shipped package. |
| repository | `bla_lint` | 1 refuse-level hit: `lane-independence` in `tests/site/demo-engine.test.mjs` | False positive in this context: the heuristic targets BLA-style unlock gates and flags a JS test double's per-lane comparison. Not a LeakLens security issue. No other gate fired. |

**Conclusion.** BLA's *hardening* features are not applicable and must not be applied: obfuscating
an MIT Python audit tool would damage debuggability, supply-chain trust, reproducibility and the
project's own auditability. BLA's *packaging and verification* features are applicable in a
narrow way: gate the release tree for accidental exposure (secrets, source maps, test corpora,
unshipped data), and verify release-artifact integrity. Those become milestones in §6.

---

## 2. Threat model

Assets: (a) the integrity of LeakLens's own releases; (b) the confidentiality of the user's data
and source; (c) the integrity of the audit result a user relies on; (d) the user's machine.

| # | Threat | Surface | Present control (verified) | Gap |
|---|---|---|---|---|
| T1 | Zip-slip / decompression bomb in an uploaded workbook | `upload_guard` | Rejects absolute and `..` entries, encrypted entries, >1,500 entries, >24 MiB uncompressed; 4 MiB upload cap | Only `.xlsx` containers are walked; future image/archive inputs need the same walk |
| T2 | Path traversal in upload names | `upload_guard.sanitize_filename` | Strips path components, NULs, caps length | — |
| T3 | Formula/CSV injection in exported reports | `report`, `sarif`, planned workspace CSV export | Values are HTML-escaped; no spreadsheet export today | The planned selection CSV export must neutralise cells starting with `= + - @` and quote per RFC 4180 before it ships; a test covers a malicious finding detail |
| T4 | Malicious serialized model (pickle/RCE) | future artifact engine; `POST /predict` | Scan path loads no pickle; `api.py` loads operator-placed joblib models from `./models/` only (see §2.2 G1) | Roadmap must forbid `pickle`, `torch.load` without `weights_only`, and request-time joblib loads; the prediction endpoint needs an explicit opt-in or a safe format; see M-S2 |
| T5 | Malicious image decoder payload | `images` extra (Pillow) | Pillow `>=9.1`; LeakLens preflight rejects oversized bytes, dimensions, pixels, frames and unsupported formats before decode; Pillow bomb warnings/errors are refusals | Video decoding and multi-frame content beyond the frame count are out of scope; limits are constants, not per-call options |
| T6 | Malicious `.pyc`/native payload in a scanned project | code scanner | Source is parsed, never imported or executed | Keep it that way: the artifact engine must parse only |
| T7 | Prompt injection via scanned source or dataset text | `--llm`, `--ai-llm` | Only sanitized finding metadata is sent; never raw rows or full source; schema-validated reply; deterministic fallback | Keep the ban on raw text; add an explicit injection test corpus |
| T8 | Workspace exposure beyond localhost | `workspace` | Binds `127.0.0.1`, documents that any reachable client can read the workspace root | No auth by design; keep `--host` an explicit, documented risk |
| T9 | Audit-result tampering / non-reproducibility | reports, manifests | Fingerprints, provenance records, artifact digests, deterministic scoring; opt-in unsigned audit manifest and stdlib-only offline `leaklens verify` (M-S3) | No signature or creator authentication; a deliberate rewrite can recompute the digest |
| T10 | Supply chain: dependency or action compromise | `pyproject.toml`, workflows | Dependabot, CodeQL, `scripts/generate_sbom.py`, `scripts/validate_release.py` | GitHub Actions are referenced by major tag (`actions/checkout@v7`), not by commit SHA |
| T11 | Accidental release of test corpora, benchmarks or fixtures | sdist/wheel | `bench_realworld` digest guard, `validate_release.py` | No explicit allowlist diff of wheel contents before publish (M-S4) |
| T12 | Bad dependency pulled in by an optional extra | extras | Extras are small and pinned by major (`sqlglot>=25`, `pillow>=9.1`, …) | New heavy extras (vision/DL/text) need a written dependency admission rule |

---

## 2.2 Open CodeQL alerts on `main` — verified triage

Independently retrieved on **2026-11-11** through the GitHub code-scanning API for
`eyadstein/leaklens` (`state=open`, all created 2026-11-09): **20 alerts**, matching the external
report. They are *reported alerts*, not confirmed exploitable vulnerabilities. Dispositions below
come from reading the source-to-sink path at each location, not from the alert text. Line numbers
are anchored to `717398c`, the HEAD this review was performed against.

| Rule | Severity | Count | Locations |
|---|---|---:|---|
| `py/unsafe-deserialization` | critical | 1 | `leaklens/api.py:325` |
| `py/path-injection` | high | 16 | `api.py:314,320`; `baseline.py:236`; `scanner.py:1019,1023`; `workspace.py:228,1107,1152,1154 (×2),1304,1306,1313,1928,1930,1932` |
| `py/stack-trace-exposure` | medium | 3 | `workspace.py:1680,1712,1716` |

### Disposition by group

**G1 — `joblib.load` on a request-derived path (`api.py:325`).** Source: `PredictReq.model_path`
→ `Path.cwd()/p` → `.resolve()` → containment check `models_root in resolved.parents` → existence
check → `joblib.load`. Sink-to-source reading: arbitrary path traversal is blocked by the
containment check, but the *deserialization* remains real — anyone who can place a file under
`./models/` (or win the resolve→load race) gets code execution in the API process. The code comment
already states the pickle risk. **Disposition: confirmed hazard class, not confirmed exploitable
in the documented deployment.** Fix belongs to the artifact policy (ADR-016): no
deserialize-at-request, a safe format allowlist (JSON/safetensors) or a model fixed at startup;
pickle is never accepted from a request. Until fixed, the honest description is "the operator must
trust `./models/`", which the docstring says.

**G2 — path injection through `WorkspaceManager.resolve` (`workspace.py:1107` and its 9 downstream
uses).** Source: HTTP query/body → `resolve()` → `Path(...).resolve()` → `is_relative_to(root)` →
raise. Every downstream sink (`is_file`, `stat`, `read_bytes`, `read_text`) receives the resolved,
contained path. No `..` escape exists, including the Windows separator case the comment documents.
CodeQL's `py/path-injection` does not model `resolve()` + `is_relative_to` as a sanitizer.
**Disposition: likely false positive for directory traversal; residual issues are real and
narrower:**
- a Windows UNC path (`\\host\share\...`) or drive-absolute path is resolved *before* rejection and
  may trigger network path resolution (SMB/NTLM exposure) on a LAN-exposed workspace;
- resolve→open is a TOCTOU window for an attacker who can already write inside the workspace root.

**G3 — no sink (`workspace.py:228`, `baseline.py:236`, `scanner.py:1019,1023`, `api.py:314,320`).**
`_relative()` only formats a display path; `Baseline.load()` and `scan_targets()` take operator
paths from the CLI/config; `api.py:314/320` are the containment gate itself. In this threat model
the caller is the principal who can already read the file. **Disposition: false positive
(informational).** The operator-supplied path is still validated for output-path containment where
config requires it.

**G4 — exception text to an HTTP client (`workspace.py:1678`, alert anchored at 1680).**
`str(ConfigError)` is returned in the `/api/config` payload, which can disclose a filesystem path
and parse detail to any client that can reach the port. Local-only default keeps impact low.
**Disposition: confirmed minor information disclosure.** Fix: generic message in the payload,
detail in the server log.

**G5 — `workspace.py:1712,1716`.** Both handlers return `manager.compare(...)`; the adjacent
`HTTPException(404, f"unknown {kind} session")` contains a fixed string, not exception text.
**Disposition: likely false positive** (imprecise mapping; confirm during remediation).

### Shared root causes

1. **One resolve+containment pattern not recognised as a sanitizer** produces 10 of 20 alerts
   (all in `WorkspaceManager.resolve`). The other six path alerts are operator-supplied paths or the
   containment gate itself and are informational. The correct response is a testable sanitizer plus
   regression tests, never suppression.
2. **Deserialize-at-request** in the prediction API (G1) — the only alert with a plausible RCE
   path in the documented threat model.
3. **Exception text echoed to clients** (G4) — a hygiene bug with a one-line fix.

### Rules for remediation

- No alert is dismissed without a written source-to-sink analysis and a test that proves the
  containment or the refusal.
- CodeQL stays as configured (no query exclusions, no path filters added to hide alerts).
- Remediation lands **before** new attack surface: agents, model loaders, integrations and extra
  file-processing engines are all new intake paths.
- Alerts are triaged in this document (or a successor) so the disposition survives the person.

### Regression coverage required

Containment tests for `..`, Windows separators, absolute drive paths, UNC paths and symlinks;
a refusal test for `joblib`/pickle artifacts; an API test asserting error bodies never contain
traceback text; and a scanner test that `scan_targets` cannot escape its root.

### Remediation status (Phase 13, 2026-11-11)

Behaviour changed on `main`; the next CodeQL run on the default branch was then read back through
the code-scanning API. Counts above are the pre-fix baseline.

- **G1 — fixed.** `/predict` no longer resolves or opens a request-supplied path. The model is fixed
  at startup (`LEAKLENS_MODEL_PATH`, default `models/best.joblib`), any other `model_path` gets 422
  before any filesystem work, and the load cache is keyed on the resolved operator path.
  `tests/test_api_predict.py` proves a request-selected path never reaches `joblib.load`.
- **G2 — hardened, not closed.** `leaklens/paths.py` owns `resolve_within`: a candidate whose
  drive/anchor differs from the root is refused *before* `resolve()`, so UNC/device paths cannot
  trigger network resolution just to be rejected; traversal, absolute-path and symlink escapes are
  still refused after resolution. `WorkspaceManager.resolve` and `WorkspaceManager._policy`
  delegate to it. `tests/test_paths.py` covers `..`, both separators, absolute paths, UNC/device
  paths, another drive and symlinks.
- **G4 — fixed.** `/api/config` returns the fixed string `configuration could not be loaded`; the
  `ConfigError` detail is written to the server log.
- **G3/G5 — unchanged by design** (no sink / fixed strings); they remain likely false positives.
  Containment and refusal are covered by tests rather than by an alert disposition alone.

### Verified alert outcome after the CodeQL run

Read back on 2026-11-11 at `8492a3c`. No alert was dismissed, suppressed or filtered:
five alerts are `fixed`, sixteen remain `open`, and one is new at the sanitizer helper itself.

Fixed automatically by the remediation:

| Alert | Rule | Location |
|---:|---|---|
| 1 | `py/path-injection` | `api.py:314` (old request path) |
| 2 | `py/path-injection` | `api.py:320` (old `models/` gate) |
| 3 | `py/unsafe-deserialization` | `api.py:325` (old `joblib.load`) |
| 7 | `py/path-injection` | `workspace.py:1107` (old `WorkspaceManager.resolve`) |
| 18 | `py/stack-trace-exposure` | `workspace.py:1680` (old `/api/config` body) |

Still open at `9b301bd` (14 `py/path-injection`, 2 `py/stack-trace-exposure`). Each is
dispositioned below from its source-to-sink path; none is dismissed or suppressed.

Re-read after Phase 14 at `fe75897`: **14 `py/path-injection` alerts are open, 0 alerts are
dismissed**, and both `py/stack-trace-exposure` alerts (19/20) are closed by the G5 fix. The
remaining sinks are `workspace.py:232,1149,1151,1301,1303,1310,1926,1928,1930`,
`scanner.py:882,886`, `baseline.py:236` and `paths.py:41`.

| Alert | Sink | Input class | Protection in the path | Disposition |
|---:|---|---|---|---|
| 4, 5 | `scanner.py:882` `is_file`, `:886` `rglob` | Operator (CLI path) or a workspace target already resolved by `resolve_within` | `scan_path` is a public API whose contract is "scan what the operator names"; the workspace never passes a raw request path | Documented risk: operator-controlled by design, no request-only path reaches it |
| 6 | `workspace.py:232` `resolve()` in `_relative` | Engine-produced result path | Values come from scan/audit results produced under `manager.root`; the `ValueError` fallback returns `as_posix()` | Documented: not request-controlled; unmodeled sanitizer |
| 8, 9, 10 | `workspace.py:1148` `exists`, `:1150` `is_file`/`is_dir` | Request path | `manager.resolve` → `paths.resolve_within` (rejects traversal, absolute escapes, foreign drive/anchor, symlink escape) | Unmodeled sanitizer; `tests/test_paths.py`, `tests/test_workspace.py` |
| 11, 12, 13 | `workspace.py:1925` `is_file`, `:1927` `stat`, `:1929` `read_text` (`/api/source`) | Request path | `manager.resolve` before every sink; preview capped by `MAX_SOURCE_BYTES` | Unmodeled sanitizer; containment tests |
| 14, 15, 16 | `workspace.py:1300` `is_file`, `:1302` `stat`, `:1309` `read_bytes` | Request path | `open_dataset` → `manager.resolve`, then the upload guard and size limit | Unmodeled sanitizer; containment + upload-limit tests |
| 17 | `baseline.py:236` `read_text` | Operator (`--baseline`, `[baseline].path`) | Operator chooses the file; reading it is the feature | Documented risk: not request-controlled |
| 19, 20 | `workspace.py:1709`, `:1713` compare responses | Request-triggered, exception provenance | The `except` branch used to echo `{exc}`; it now returns a fixed reason and logs the detail | **Remediated in the Phase 14 audit** (see below) |
| 21 | `paths.py:41` `resolve()` inside the sanitizer | Request path | Drive/anchor compared *before* `resolve()`; result containment-checked after | Unmodeled sanitizer: this is the check itself; `tests/test_paths.py` |

Alert 7's downstream readers remain because CodeQL does not model `resolve()` + containment as a
sanitizer; `paths.py:41` is the same unmodeled pattern, now with one home and regression tests
instead of ten scattered call sites. `py/path-injection` may also be reported for
`baseline.py`/`scanner.py` because CodeQL follows operator-supplied CLI/config paths into file reads.
Those are not request-controlled in this threat model, and no suppression or query filter was added.

### Phase 14 audit additions (2026-11-11)

- **G5 (alerts 19/20) — fixed.** `_compare_sessions` no longer interpolates exception text into the
  response; it returns `the saved findings cannot be compared` and logs the exception server-side,
  matching the G4 rule. `tests/test_response_parity.py` and `tests/test_workspace.py` cover the
  compare responses.
- **One containment helper.** `config._output_path` used a second `relative_to` containment check;
  it now calls `paths.contained`, so `paths.py` is the only containment implementation.
- **Complete-response parity.** `public_api.run_audit_job` did not run the missingness audit while
  the CLI and `/audit` did, so the same dataset produced different findings in the workspace.
  It now runs `audit_missingness` with the same inputs and returns the same section.
  `tests/test_response_parity.py` proves CLI, `/audit`, public API and workspace agree on finding
  identity, trust, counts and policy effects.


---

## 3. Trust boundaries in the future architecture

Five boundaries, each with an owner and a rule.

1. **Untrusted bytes → parsed structure.** One intake module owns every decoder. Rules: extension
   *and* magic-byte validation, size/entry/pixel/depth budgets, format allowlist, no archive member
   written to disk, no macro evaluation, no formula evaluation. Parsers are pure functions with
   typed errors. Every limit is recorded in the result so a report can state what was refused.
   The generalisation happens when the first new decoder lands (Phase 16/17), reusing
   `upload_guard`'s logic rather than creating a parallel one.
2. **Parsed structure → engine.** Engines never open user files directly; they receive validated
   references. This is what makes a vision engine safe to add.
3. **Engine → score.** Findings only. No engine may write a score, a fingerprint or a baseline
   entry; `leakage.score()` remains the single scoring path.
4. **Result → AI.** Evidence ids and aggregate metadata only, never raw rows, never full source.
   The AI layer may not create findings or evidence, and a malformed reply is discarded.
5. **Result → user.** Every output states the detector set, versions, what was skipped and why.
   A result without that context is not shippable.

---

## 4. Serialized-artifact policy (the rule that prevents an RCE class)

The artifact engine (Phase 16) will inspect model files. Formal policy:

| Format | Allowed | Rule |
|---|---|---|
| JSON (sklearn coefficients, configs) | Yes | Parse with `json`; enforce depth/size caps; digest |
| `safetensors` | Yes | Parse the header and tensor index only; never map weights for execution |
| `GGUF`, ONNX (protobuf), `numpy .npy/.npz` | Yes, if implemented | Metadata-first; `allow_pickle=False` for numpy |
| Pickle (`pickle`, `joblib`, `torch.save` legacy) | **Metadata only, never executed** | If an unfickler is ever written it runs in a separate process with no network, no filesystem writes and a hard timeout; default is to refuse with a finding |
| `.pt`/`.pth`/`.bin` | Metadata only | Refuse `torch.load`; the audit runs without PyTorch installed |
| Encrypted or password-protected archives | Refuse | Recorded as a skip, never retried silently |

Rationale: an auditor that executes a user's serialized model becomes the attack. LeakLens must be
safe to point at a hostile artifact, and reporting "this file is a pickle and I will not load it"
is a legitimate audit result.

---

## 5. Reproducible releases

Existing: version in two places (kept in sync), `scripts/generate_sbom.py`,
`scripts/validate_release.py`, release workflow publishing to PyPI, digest guards on benchmark
corpora, deterministic detectors.

Planned, in order:

1. **Wheel content allowlist.** `validate_release.py` compares the wheel's file list against an
   explicit allowlist derived from `leaklens/**` and package data, failing on anything else
   (fixtures, corpora, caches, `graphify-out/`, `.phase5-bench.json`).
2. **Action pinning.** Pin GitHub Actions to commit SHAs with Dependabot maintaining them. Small,
   mechanical, real risk reduction.
3. **Audit manifest.** A JSON document produced with every audit (optional flag) containing input
   digests, tool version, detector ids/versions, options, policy/baseline digests, evidence
   digests, score and gate decision. Verifiable offline with a documented digest recipe. It is
   evidence, not a certificate — it proves what LeakLens saw and concluded, not that the model is
   good.
4. **Optional BLA packaging gate.** A non-blocking, developer-run job (`bla lint` with the known
   false positive allow-listed, plus `bla_lint_plaintext_secrets`) recorded in the release
   checklist. It must never gate CI on a tool that is not a dependency and not available to
   contributors.

---

## 6. Hardening milestones

| Id | Milestone | Phase | Acceptance |
|---|---|---|---|
| M-S0 | **Security gate:** triage and remediate the 20 open CodeQL alerts (G1–G5 above); extract `paths.py` and route every containment check through it | 13, before any new engine | Each alert has a recorded disposition; G1 loses request-time deserialization of a request-selected file (safe format allowlist or a model fixed at startup; pickle never accepted from a request); G4 returns a generic message; containment regression tests cover `..`, Windows separators, absolute drives, UNC and symlinks; UNC/device paths outside the root are refused before resolution (a UNC workspace root still works); no alert dismissed without a linked analysis and a passing test; CodeQL configuration unchanged |
| M-S1 | Generalise `upload_guard` into a decoder intake contract (magic bytes, budgets, refusal reasons surfaced in results) | 16 (artifacts) then 17 (images) — the first new decoders | Every refusal path is a typed skip with a reason; fuzz corpus of 30 hostile archives (traversal, bomb, encrypted, wrong magic) produces no crash and no write |
| M-S2 | Serialized-artifact policy enforced in code | 16 | Pickle/`.pt` produce a refusal finding; `safetensors` and JSON parse with caps; an AST check over `leaklens/**/*.py` resolves direct and aliased `torch.load` imports (docs excluded) |
| M-S3 | Audit manifest emitted by `assemble.py` + offline verifier (`leaklens verify`) | 18 | Manifest digest stable across two runs on identical inputs; verifier detects a changed score, changed evidence and a changed detector version; verification uses stdlib only |
| M-S4 | Release allowlist + SHA-pinned actions | 13 | Wheel diff fails the build if an unexpected file ships; all actions pinned; Dependabot keeps pins fresh |
| M-S5 | Image decode budgets | 17 | LeakLens preflight pixel limit set (Pillow's global limit untouched), per-file and per-audit pixel budget enforced, decompression-bomb image refused with a skip |
| M-S6 | Prompt-injection corpus | 18 | A corpus of documents containing instructions aimed at the LLM produces no change to findings/score; the LLM output is discarded or ignored, and a test asserts it |
| M-S7 | Dependency admission rule | every phase | A new dependency requires: license check, maintenance check, install-size note, optional-extra placement, and a labelled benchmark or test — documented in the PR template |

Status (Phase 16, 2026-11-13): the **artifact half of M-S1** and all of **M-S2**
are shipped in `analysis/artifacts/` and recorded in
[PHASE16_ACCEPTANCE.md](PHASE16_ACCEPTANCE.md). One typed intake refuses pickle
and archive input before any parser, with a reason on every skip, and a 30-case
hostile corpus proves no crash and no write. `upload_guard` itself is unchanged:
it is the dataset-upload boundary, not an artifact decoder.

Status (image intake, 2026-11-13): **M-S5 is complete and the image half of M-S1
is complete for the existing intake.** `images.py` enforces per-file compressed
bytes, side, pixel and animation-frame limits before decoding, plus one pixel
budget shared by both splits of an audit; a refused image carries a stable reason
(`too_large`, `too_large_dimensions`, `too_many_pixels`, `too_many_frames`,
`unsupported_format`, `decompression_bomb`, `budget_exhausted`) that is reported
separately from `unreadable`. Pillow's own decompression-bomb limit is not
weakened. See [IMAGE_DECODE_BUDGETS.md](IMAGE_DECODE_BUDGETS.md).

Status (prompt injection, 2026-11-13): **M-S6 is complete for the existing LLM layer and accepted
with findings.** A versioned 47-case synthetic corpus (32 attacks, 15 benign hard negatives)
exercises the real `--llm` finding prompt and `--ai-llm` plan-narration prompt with deterministic
fake providers. Untrusted fields are flattened before rendering, unsafe model claims are dropped,
and every model reply is display-only. Pinned development/holdout digests, category-level metrics
and the measured `A-009` false negative are published. An independent adversarial review at
`b7142aa` reproduced the trust boundaries and metrics; its demonstrated defects were fixed in
`c01481b`. The guard remains a finite denylist, so novel paraphrases and homoglyphs can bypass it,
withheld explanations are silent in the CLI, and the corpus is not evidence of universal injection
resistance. See [PROMPT_INJECTION.md](PROMPT_INJECTION.md).

Status (dependency admission, 2026-11-13): **M-S7 is not implemented.** The pull-request template
has no dependency-admission checklist (license, maintenance, install size, optional-extra placement
and a labelled test or benchmark). No phase has added a mandatory dependency, so the gap has not
blocked a milestone.

Status (release integrity, 2026-11-13): **M-S4 is complete and strengthened.**
The wheel check now requires every module and package resource derived from the
tagged tree, the sdist is validated too, and a `package` CI job builds and
validates both distributions on every pull request ([ADR-026](ARCHITECTURE_DECISIONS.md)).

Status (audit manifest, 2026-11-13): **M-S3 is complete, unsigned by design.**
`--manifest` writes a deterministic digest manifest of the inputs, options,
detector catalogue, findings, policy outcome, score and gate; `leaklens verify`
recomputes it offline with the standard library and names a changed section or a
missing/modified input. It proves content integrity, not authenticity or
provenance. See [AUDIT_MANIFEST.md](AUDIT_MANIFEST.md).

---

## 7. What this strategy refuses

- Obfuscating, packing or bytecode-compiling LeakLens's own Python to "protect" it.
- Claiming any reverse-engineering resistance for a source-distributed MIT package.
- Running user code, models or datasets to "see what happens".
- Any network access in a default audit path.
- Presenting an audit manifest as a compliance certificate or a security guarantee.
- Adding a "security score" alongside the trust score. One score, one meaning, one path.

---

## 8. Honest residual risk

LeakLens cannot make an untrusted archive safe for every downstream consumer, cannot detect a
backdoored dependency by inspection, cannot guarantee that a parsed file is benign, and cannot
prevent a user from running `--host 0.0.0.0` on an untrusted network. It can refuse, cap, record
and report — and the refused/capped cases must remain visible in the result rather than vanishing
into a log line.
