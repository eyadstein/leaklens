---
name: Release checklist
about: Track the manual steps for preparing and publishing a LeakLens release
title: "Release vX.Y.Z"
---

## Release details

- **Version:** 
- **Release commit:** 

## Pre-release checks

- [ ] Confirm the release changes are merged into `main`.
- [ ] Confirm the version in `pyproject.toml`, `leaklens/__init__.py` and `CITATION.cff` matches, and the `CHANGELOG.md` heading is `Unreleased` or the release version.
- [ ] Update `CHANGELOG.md` with the release changes.
- [ ] Run `python scripts/validate_release.py --versions` to verify version consistency across the metadata files and changelog heading.
- [ ] Confirm the CI checks pass on the release commit.
- [ ] Build the wheel and source distribution with `python -m build`.
- [ ] Validate both distributions with `python scripts/validate_release.py --wheel dist/*.whl` and `python scripts/validate_release.py --sdist dist/*.tar.gz`.

## Publish

- [ ] Create a `vX.Y.Z` tag for the release commit on `main`.
- [ ] Create and publish the GitHub release using that tag.
- [ ] Confirm the Release workflow completes successfully and publishes to PyPI.

## Post-release verification

- [ ] Confirm the new version is available on [PyPI](https://pypi.org/project/leaklens-ml/).
- [ ] Install the published package in a clean environment and verify `leaklens --version`.
- [ ] Confirm the release notes and documentation link to the correct version.
