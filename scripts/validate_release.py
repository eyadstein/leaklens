"""Validate a release tag against the package version and workflow ref."""

from __future__ import annotations

import argparse
import re
import sys
import zipfile
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


def validate_wheel(names: list[str], version: str) -> None:
    """Raise ValueError unless every wheel member is leaklens code or its dist-info.

    The wheel must not carry tests, docs, the repository metadata or an injected
    file: only ``leaklens/**`` and ``leaklens_ml-<version>.dist-info/**`` ship.
    """
    if not names:
        raise ValueError("wheel archive is empty")
    dist_info = f"leaklens_ml-{version}.dist-info"
    unexpected = []
    for name in names:
        parts = name.split("/")
        safe = not (name.startswith("/") or "\\" in name or ".." in parts)
        if safe and len(parts) > 1 and parts[0] in {"leaklens", dist_info}:
            continue
        unexpected.append(name)
    if unexpected:
        raise ValueError(f"wheel contains unexpected file(s): {', '.join(unexpected)}")


def wheel_version(path: Path) -> str:
    """``leaklens_ml-<version>-py3-none-any.whl`` -> ``<version>``."""
    parts = path.name.split("-")
    if len(parts) < 2 or parts[0] != "leaklens_ml":
        raise ValueError(f"{path.name} is not a leaklens wheel filename")
    return parts[1]


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
    parser.add_argument("--tag", help="release tag, e.g. v0.13.0")
    parser.add_argument(
        "--wheel",
        type=Path,
        help="built wheel whose member list must match the release allowlist",
    )
    parser.add_argument(
        "--pyproject",
        default=Path("pyproject.toml"),
        type=Path,
        help="package metadata",
    )
    parser.add_argument("--event", default="release", help="GitHub event name")
    parser.add_argument("--ref", default="", help="GitHub ref, e.g. refs/heads/main")
    args = parser.parse_args(argv)
    if args.wheel is None and not args.tag:
        parser.error("--tag is required unless --wheel is given")

    if args.wheel is not None:
        try:
            with zipfile.ZipFile(args.wheel) as archive:
                validate_wheel(archive.namelist(), wheel_version(args.wheel))
        except (OSError, ValueError, zipfile.BadZipFile) as exc:
            print(f"::error::{exc}", file=sys.stderr)
            return 1
        print(f"Wheel {args.wheel.name} contains only package and dist-info files")
        return 0

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
