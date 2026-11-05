# Security policy

## Supported versions

| Version | Supported |
| --- | --- |
| Latest on PyPI (`leaklens-ml`) | ✅ Security fixes |
| Older releases | ❌ Please upgrade |

Only the latest release receives fixes. If you are on an older version, upgrade first — fixes are not backported.

## Reporting a vulnerability

Please **do not open a public issue**. Use GitHub's private reporting:

1. Go to the repository's **Security** tab.
2. Click **Report a vulnerability**.
3. Describe the issue clearly: what you observed, how to reproduce it, and the impact.

What to expect:

- **Acknowledgement** within a few days.
- A fix is prioritised by severity; you'll be notified when a patched release is available.
- Please give us a reasonable amount of time to publish a fix before disclosing publicly.

We appreciate responsible reports and will credit reporters in the advisory unless you prefer to remain anonymous.

## Scope and threat model

### Static code scanner (`leaklens --scan`)
- Parses Python and Jupyter files with `ast`. **Scanned code is never executed.**
- The scanner only inspects call patterns (e.g. `fit_transform` before a split). It does not run the notebook or script.

### Dataset audit
- Reads the CSV/Parquet/JSON/Excel/Feather file you point it at. Dataset contents stay on your machine; nothing is uploaded unless you use an optional feature below.

### FastAPI service (`leaklens.api`)
- `POST /predict` loads model files with `joblib` from `./models/` only; absolute and traversal paths outside that directory are rejected. `joblib`/`pickle` deserialisation can still execute arbitrary code.
- **Only load models you trust.** Do not expose `/predict` (or the API) to untrusted users or networks.
- Uploads are capped (8 MiB and 50,000 rows); ZIP/gzip containers are size-checked (256 MiB decompressed) and ZIP entry paths are validated before parsing.
- The API is designed for local or trusted deployment (e.g. Docker on `localhost`), not as a public service.

### Public live-scan API (`leaklens.public_api`)
- A **separate** app for untrusted uploads. It exposes no `/predict`, no model loading and no dataset storage, so the `/predict` pickle risk does not apply.
- Uploads default to 4 MiB, 5,000 rows, 200 columns and 750k cells. Filenames are sanitised; the extension allowlist and real content sniffing (CSV/JSON/XLSX/Parquet/Feather) reject renamed or malformed files. XLSX ZIPs are checked for traversal, encryption, entry count and decompression expansion.
- Request bodies are capped before multipart parsing (including chunked bodies). Parsing is bounded and in-memory; uploads are never written to disk and the upload handle is closed even on failure.
- One scan worker runs at a time; overlapping requests get `503`, and a scan that exceeds its wall-clock budget gets `504`. A bounded per-client rate limiter applies to `POST /inspect`, `POST /audit` and `POST /scan`.
- Error bodies are generic: parser internals, filesystem paths and stack traces are not returned. Responses carry `nosniff`, `DENY` framing, a restrictive CSP, `no-store` and a request ID; CORS is restricted to an origin allowlist.
- Deploy it with `Dockerfile.api` / `render.yaml` and review [docs/LIVE_SCANNING.md](docs/LIVE_SCANNING.md). The repository does **not** deploy a public instance.

### LLM explanations (`--llm`) — opt-in only
- **Disabled by default.** Nothing leaves your machine unless you pass `--llm`.
- Only finding metadata is sent: severity, kind, file name and line, and short diagnostic detail (which may include brief code expressions from the scanned file). **Complete source files, notebooks, dataset rows, and dataset contents are never sent.**
- The default provider is a local Ollama instance (`http://localhost:11434`).
- If you configure a remote OpenAI-compatible endpoint (`--llm-provider openai-compatible`, `LEAKLENS_LLM_BASE_URL`), finding details are sent to that host. Provider URLs must use HTTP/HTTPS, redirects are refused, and the API key (`LEAKLENS_LLM_API_KEY`) is not sent over plain HTTP to non-local hosts.
- Findings are treated as **untrusted diagnostic data, not instructions** — the prompt tells the model this explicitly.
- Optional: install the `taintwatch` extra (`pip install "leaklens-ml[taintwatch]"`) and set `LEAKLENS_TAINTWATCH=1` to additionally gate the provider call with information-flow checks.

### Dependencies and CI
- CI runs `bandit` (static code security checks), `pip-audit` (known-vulnerability scan of dependencies), and `pip-licenses` on every push/pull request.
- A separate CodeQL workflow runs `security-extended` Python analysis and uploads results to GitHub code scanning.
- The release workflow generates a CycloneDX SBOM and uploads it alongside the wheel and sdist.

## Out of scope

- Reports about untrusted scanned code *contents* (e.g. a notebook that contains malicious code) — the scanner never executes it, so the risk is not in LeakLens itself.
- Issues in third-party dependencies: please report those upstream; we'll update the dependency once a fix is available.
- Social-engineering or phishing attempts against the maintainer.
