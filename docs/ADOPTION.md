# Adoption baseline and growth strategy

**Status:** measured baseline, not a forecast. Every number below was read from a
public API on 2026-11-25 and is reproducible with the commands in
[Reproducing](#reproducing). Nothing here is an estimate of users, demand or
revenue.

The point of this document is to stop the project from confusing motion with
adoption. Downloads are not users, stars are not usage, forks are not
contributors, and a seeded backlog is not demand.

## Baseline

| Signal | Value (2026-11-25) | What it can mean | What it cannot mean |
|---|---:|---|---|
| Repository created | 2026-10-20 | The project is five weeks old. | That anyone outside it depends on it. |
| GitHub stars | 3 | A few people bookmarked it. | Usage. |
| Forks / network | 24 | Includes forks created to open a PR. | 24 independent users. |
| Watchers | 1 | The maintainer watches it. | Interest from a community. |
| Contributors (incl. anonymous) | 25 | Most are single-PR drive-by accounts. | A contributor community. |
| Open issues | 441 | A deliberately seeded backlog, not requests. | Demand, or a healthy triage state. |
| Closed issues | 163 | The backlog is being worked. | That outside users filed them. |
| `good first issue` / `help wanted` / `enhancement` / `bug` | 125 / 365 / 380 / 7 | Contribution paths exist at scale. | That they are all actionable today. |
| PyPI releases | 13 (`0.4.0` → `0.14.0`) | Fast iteration; `0.14.0` uploaded 2026-11-20. | Stability. |
| PyPI downloads, last day / week / month | 11 / 141 / 610 | Some automated installs plus a few humans. | 610 unique users. |
| GitHub views, last 14 days | 459 / 19 unique | Near-zero organic discovery. | A funnel. |
| GitHub clones, last 14 days | 1,755 / 234 unique | CI and bots clone far more than people. | 234 interested developers. |
| Top referrers | `github.com` 80, `typst.app` 5 | Almost no external referrer. | Any community traction. |
| Top paths | repo root 151, `/pulls` 55, `/commits/main` 28 | Visitors arrive from PR activity. | A documentation funnel. |

## First-run audit

Observed, from the repository:

- `pip install --upgrade leaklens-ml` has no mandatory extras and supports
  Python 3.10+; the README quick start then runs `leaklens --scan .`.
- There is a planted-leak demo (`examples/make_demo.py` → `leaklens demo.csv
  --target is_fraud`), a walkthrough notebook, and an advanced-example script.
- The documented output includes the trust score, the finding, the severity and
  the exit code, so the first run explains itself.
- Optional engines are opt-in and named by extra (`[sql]`, `[yaml]`, `[images]`,
  `[app]`, `[api]`), and a missing extra raises `ImportError` naming it.
- Integrations that exist today: GitHub Action, pre-commit hook, Docker/Compose,
  FastAPI app, Streamlit UI, SARIF, HTML report.
- The docs site is live at <https://eyadstein.github.io/leaklens/> and is built
  from `docs/` on `main`; CI runs `scripts/build_site.py --check`.

Gaps that a new user meets before they meet the product:

1. The quick start does not show what a *clean* scan of a real repository looks
   like, so a first-time user cannot tell "nothing found" from "not working".
2. There is no single page that says which detector to use for which artifact
   (dataset file vs notebook vs SQL vs images); the information is spread across
   `docs/DETECTORS.md`, the README and eight engine docs.
3. The generated site has no tutorial that starts from a failing CI run and ends
   at a passing one, which is the workflow the exit code is designed for.

## Discovery audit

- PyPI summary, keywords and GitHub topics are present and specific
  (`data-leakage`, `machine-learning`, `mlops`, `static-analysis`).
- PyPI metadata carried only a Python-3 classifier and no changelog/source URL
  until this phase; the patch adds `Development Status :: 4 - Beta`, explicit
  3.10–3.12 classifiers, and `Changelog`/`Source` project URLs (visible on the
  next release, not retroactively on `0.14.0`).
- The README leads with the problem ("find data leakage before it inflates your
  metrics") rather than a feature list, and ships Arabic and Japanese
  translations.
- Search discoverability cannot be measured honestly here: no data source in this
  repository records impressions or search ranking, so no claim is made.

## Contribution surface

- Three issue templates (`bug_report`, `leak_idea`, `release_checklist`), a PR
  template, `CONTRIBUTING.md` with an explicit verify list, and a claiming rule.
- 125 open `good first issue` items is more than a new contributor can navigate.
  The useful number is not 125; it is whether the first three are correct,
  self-contained and still unfixed.
- Review latency is not measured. Do not claim a 48-hour SLA until a sample of
  outside PRs supports it.

## Growth strategy

Four bets, ordered by expected effect per unit of maintainer time. Each has a
signal and a falsifier, so a bet can lose.

### 1. Make the first five minutes undeniable

Improve the quick start so a newcomer sees a real leak, the fix, and a passing
re-run in one page, then the same thing in CI. Add a "clean scan looks like this"
example so silence is interpretable.

- **Signal:** the quick start is executed in CI on every push, and the docs page
  is the top non-repository path in the 14-day traffic report.
- **Falsified if:** after four weeks, views on the tutorial page stay in single
  digits and no issue references it.

### 2. Publish one real case study instead of ten feature claims

Use a real, legally reusable notebook from the reviewed corpus. Show the finding,
the evidence, the human verdict and the corrected pipeline. Redact where needed.

- **Signal:** a reviewer outside the project reproduces the case and the finding
  on the recorded commit.
- **Falsified if:** the case cannot be published under its licence, or the
  detector does not reproduce on the recorded commit.

### 3. Meet teams where they already run

The GitHub Action and pre-commit hook exist. The Action now carries `path`,
`fail-under`, `baseline`, `fail-on-new` and `sarif`, so the documented adoption
path — record the findings you have reviewed, then fail only on new ones — is
reachable from CI without a hand-written `run:` step, and a code-scanning upload
can follow the scan. The README shows the copy-paste workflow; the `baseline`,
`fail-on-new` and `sarif` inputs are unreleased, so they need `@main` until the
next tag. One integration that works beats five adapters that are announced.

- **Signal:** an external repository uses the Action and links back.
- **Falsified if:** no external usage appears after the docs and a single
  announcement, and the Action's own example is the only one in the wild.

### 4. Keep the backlog honest

Do not grow the issue count to look mature. Triage to a small set of
well-specified, currently-actionable issues; archive the rest into a labelled
backlog that does not pretend to be demand. The 700-issue program is a
maintenance tool, not a growth metric.

- **Signal:** the first page of `good first issue` contains only issues that are
  reproducible, scoped and unfixed; median age of open bugs falls.
- **Falsified if:** the open-issue count keeps rising while external engagement
  stays flat.

### What we will not do

No purchased stars, no fabricated testimonials or user counts, no mass-posting to
communities, no claiming a precision number the corpus cannot support, and no
download or fork number presented as a user count.

## Reproducing

```sh
gh api repos/eyadstein/leaklens --jq '{stars:.stargazers_count,forks:.forks_count,watchers:.subscribers_count,open_issues:.open_issues_count,created_at}'
gh api repos/eyadstein/leaklens/traffic/views
gh api repos/eyadstein/leaklens/traffic/clones
gh api repos/eyadstein/leaklens/traffic/popular/referrers
curl -s https://pypi.org/pypi/leaklens-ml/json | python -c "import json,sys;d=json.load(sys.stdin);print(d['info']['version'],len(d['releases']))"
curl -s https://pypistats.org/api/packages/leaklens-ml/recent
```

Traffic endpoints need push access and only cover 14 days; PyPI and pypistats are
public. Re-read all of them before quoting a number in public.
