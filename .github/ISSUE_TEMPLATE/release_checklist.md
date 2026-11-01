---
name: Manual release checklist
about: Track pre-release checks, PyPI publishing and post-release verification
title: "Release vX.Y.Z"
---

## Release details

- Version: `X.Y.Z`
- Git tag: `vX.Y.Z`
- Target commit: <!-- paste commit SHA -->
- Release owner: <!-- GitHub username -->

## Before publishing

- [ ] Confirm the release commit is on `main` and contains the intended changes.
- [ ] Verify `pyproject.toml` has version `X.Y.Z`, matching the planned `vX.Y.Z` tag.
- [ ] Update `CHANGELOG.md` with the release changes and check user-facing documentation.
- [ ] Verify CI passes on the target commit, including the supported Python versions.
- [ ] Run `python -m pytest -q` and `leaklens --scan leaklens`.
- [ ] Run `python -m leaklens.bench` without errors.
- [ ] Run `python -m pip install build` and `python -m build`; check the wheel and source distribution in `dist/`.
- [ ] Review release notes and confirm the planned tag points to the intended commit.

## Publish and verify

> **Publishing a GitHub Release automatically triggers `.github/workflows/release.yml` to build and upload `leaklens-ml` to PyPI.** Do not publish the Release until the pre-release checks are complete.

- [ ] Publish the GitHub Release for `vX.Y.Z` (not a draft) with the reviewed release notes.
- [ ] Check that the GitHub Actions `Release` workflow succeeded, including its `pypi` job.
- [ ] Verify the new `leaklens-ml` version and distributions appear on PyPI.
- [ ] Install the published version in a clean environment and run `leaklens --version`.
- [ ] Record the GitHub Release URL, PyPI URL and any follow-up issues below.

## Release links and notes

- GitHub Release:
- PyPI:
- Follow-up:
