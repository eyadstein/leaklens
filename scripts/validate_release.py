"""Validate a release tag against the package version and workflow ref."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_TAG_RE = re.compile(r"v(\d+)\.(\d+)\.(\d+)")


def package_version(path: Path) -> str:
    """Return ``[project].version`` from a pyproject.toml file."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} has no [project].version")
    return version


def validate(tag: str, version: str, event: str = "release", ref: str = "") -> None:
    """Raise ValueError unless the tag/ref pair is safe to publish."""
    if not tag:
        raise ValueError("release tag is empty")
    match = _TAG_RE.fullmatch(tag)
    if match is None:
        raise ValueError(f"release tag {tag!r} is not in vX.Y.Z format")
    tag_version = ".".join(match.groups())
    if tag_version != version:
        raise ValueError(
            f"release tag {tag!r} does not match package version {version!r}"
        )
    if event == "release":
        expected = f"refs/tags/{tag}"
        if ref and ref != expected:
            raise ValueError(f"release event ref {ref!r} does not match tag {tag!r}")
    elif event == "workflow_dispatch" and ref != "refs/heads/main":
        raise ValueError(f"manual dispatch must run from refs/heads/main (got {ref!r})")


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", required=True, help="release tag, e.g. v0.13.0")
    parser.add_argument(
        "--pyproject",
        default=Path("pyproject.toml"),
        type=Path,
        help="package metadata",
    )
    parser.add_argument("--event", default="release", help="GitHub event name")
    parser.add_argument("--ref", default="", help="GitHub ref, e.g. refs/heads/main")
    args = parser.parse_args(argv)

    try:
        version = package_version(args.pyproject)
        validate(args.tag, version, args.event, args.ref)
    except (OSError, ValueError, tomllib.TOMLDecodeError) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
    print(f"Release tag {args.tag} matches package version {version}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
