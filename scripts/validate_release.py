"""Validate a release tag and the built distributions before publishing.

Two jobs:

* prove that a tag is publishable - tag format, package version match, and the
  workflow ref that triggered the run;
* prove that a built wheel and source distribution contain exactly the files
  the tagged source says they must, byte for byte, and nothing else.

Expected membership is derived from the checked-out tree and the authoritative
packaging configuration (``pyproject.toml`` package data plus the ``MANIFEST.in``
directives) rather than a hand-maintained file list, so the check tracks the tag
instead of drifting from it. Archives are inspected as data: no member is
imported or executed.
"""

from __future__ import annotations

import argparse
import re
import sys
import tarfile
import zipfile
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

_TAG_RE = re.compile(r"v(\d+)\.(\d+)\.(\d+)")
_PACKAGE = "leaklens"
_PACKAGE_NAME = "leaklens-ml"
_EGG_INFO = _PACKAGE_NAME.replace("-", "_") + ".egg-info/"

# Always written by setuptools, so never treated as unexpected archive members.
_WHEEL_METADATA = ("METADATA", "WHEEL", "RECORD")
_SDIST_METADATA = ("PKG-INFO", "pyproject.toml", "MANIFEST.in")
_SDIST_ROOT_FILES = frozenset({"setup.cfg", "README.md", "LICENSE"})


def package_version(path: Path) -> str:
    """Return ``[project].version`` from a pyproject.toml file."""
    data = tomllib.loads(path.read_text(encoding="utf-8"))
    project = data.get("project")
    version = project.get("version") if isinstance(project, dict) else None
    if not isinstance(version, str) or not version:
        raise ValueError(f"{path} has no [project].version")
    return version


def _manifest_paths(root: Path) -> set[str]:
    """Files selected by ``MANIFEST.in``, repo-relative.

    Supports the include forms that decide what packaging copies
    (``include``, ``recursive-include``, ``graft``) and the forms that remove
    files again (``exclude``, ``global-exclude``, ``prune``). Directives that
    only affect the sdist manifest for other reasons are ignored.
    """
    manifest = root / "MANIFEST.in"
    if not manifest.is_file():
        return set()
    included: set[Path] = set()
    excluded: set[Path] = set()
    for line in manifest.read_text(encoding="utf-8").splitlines():
        words = line.split()
        if not words or words[0].startswith("#"):
            continue
        directive, args = words[0], words[1:]
        if directive == "include":
            matched = {path for pattern in args for path in root.glob(pattern)}
        elif directive == "recursive-include" and len(args) >= 2:
            base = root / args[0]
            matched = {path for pattern in args[1:] for path in base.rglob(pattern)}
        elif directive == "graft" and len(args) == 1:
            matched = set((root / args[0]).rglob("*"))
        elif directive in {"exclude", "global-exclude"}:
            excluded.update(path for pattern in args for path in root.rglob(pattern))
            continue
        elif directive == "prune" and len(args) == 1:
            excluded.update((root / args[0]).rglob("*"))
            continue
        else:
            continue
        included.update(path for path in matched if path.is_file())
    return {
        path.relative_to(root).as_posix()
        for path in included - excluded
        if "__pycache__" not in path.parts
    }


def expected_members(source_root: Path, *, sdist: bool = False) -> list[str]:
    """Every package file the built distributions must contain, repo-relative.

    Derived from ``leaklens/**/*.py`` plus the package data declared in
    ``pyproject.toml`` and the ``MANIFEST.in`` rules (``include-package-data``
    is setuptools' default), so adding a module or a package resource needs no
    change here. ``sdist=True`` also keeps root-level manifest includes, which
    a source distribution ships but a wheel does not.
    """
    root = Path(source_root)
    package = root / _PACKAGE
    if not package.is_dir():
        raise ValueError(f"{package} is not a package directory")

    files = {
        path.relative_to(root).as_posix()
        for path in package.rglob("*.py")
        if "__pycache__" not in path.parts
    }

    config = tomllib.loads((root / "pyproject.toml").read_text(encoding="utf-8"))
    tool = config.get("tool") or {}
    setuptools = tool.get("setuptools") or {}
    for dotted, patterns in (setuptools.get("package-data") or {}).items():
        base = root / str(dotted).replace(".", "/")
        for pattern in patterns:
            files.update(
                path.relative_to(root).as_posix()
                for path in base.glob(str(pattern))
                if path.is_file() and "__pycache__" not in path.parts
            )

    files |= _manifest_paths(root)
    if not sdist:
        files = {name for name in files if name.startswith(_PACKAGE + "/")}
    if not files:
        raise ValueError(f"no files found for package {_PACKAGE}")
    return sorted(files)


def _unsafe(name: str) -> bool:
    """True for absolute, backslash, parent-traversal or empty member names."""
    return not name or name.startswith("/") or "\\" in name or ".." in name.split("/")


def _duplicates(names: list[str]) -> list[str]:
    seen: set[str] = set()
    repeated: set[str] = set()
    for name in names:
        if name in seen:
            repeated.add(name)
        seen.add(name)
    return sorted(repeated)


def _diff(label: str, present: set[str], required: set[str], optional=()) -> None:
    """Raise when required members are missing or unexpected members are present."""
    missing = sorted(required - present)
    extra = sorted(present - required - set(optional))
    if not missing and not extra:
        return
    problems = []
    if missing:
        problems.append(f"missing required file(s): {', '.join(missing)}")
    if extra:
        problems.append(f"unexpected file(s): {', '.join(extra)}")
    raise ValueError(f"{label} " + "; ".join(problems))


def validate_wheel(names: list[str], version: str, required=()) -> None:
    """Raise ValueError unless the wheel is exactly the expected package.

    Every member must be a safe, unique ``leaklens/**`` file or dist-info
    metadata; the dist-info metadata must be present; and the set of
    ``leaklens/**`` members must equal ``required``. Passing ``required`` empty
    keeps the structural checks only.
    """
    files = [name for name in names if not name.endswith("/")]
    if not files:
        raise ValueError("wheel archive is empty")
    dist_info = f"leaklens_ml-{version}.dist-info"
    unexpected = [
        name
        for name in files
        if _unsafe(name) or name.split("/")[0] not in {_PACKAGE, dist_info}
    ]
    if unexpected:
        raise ValueError(f"wheel contains unexpected file(s): {', '.join(unexpected)}")

    repeated = _duplicates(files)
    if repeated:
        raise ValueError(f"wheel contains duplicate file(s): {', '.join(repeated)}")

    present = set(files)
    for meta in _WHEEL_METADATA:
        if f"{dist_info}/{meta}" not in present:
            raise ValueError(f"wheel is missing {dist_info}/{meta}")

    if required:
        package = {name for name in files if name.startswith(_PACKAGE + "/")}
        _diff("wheel", package, set(required))


def validate_sdist(names: list[str], version: str, required=()) -> None:
    """Raise ValueError unless the sdist is exactly the expected source tree.

    Every file must live under ``leaklens_ml-<version>/`` and be either a
    required package file, setuptools metadata, ``tests/**/*.py`` or generated
    egg-info; the required metadata must be present.
    """
    root = f"leaklens_ml-{version}"
    prefix = f"{root}/"
    entries = [name for name in names if not name.endswith("/")]
    if not any(name != root for name in entries):
        raise ValueError("sdist archive is empty")
    stray = [
        name
        for name in entries
        if _unsafe(name) or (name != root and not name.startswith(prefix))
    ]
    if stray:
        raise ValueError(f"sdist contains unexpected file(s): {', '.join(stray)}")

    # Tar directory entries carry no content; drop the root and any member that
    # is only a parent directory of another member.
    relative = [name[len(prefix) :] for name in entries if name != root]
    parents = {
        "/".join(parts[:depth])
        for name in relative
        for parts in [name.split("/")]
        for depth in range(1, len(parts))
    }
    files = [name for name in relative if name not in parents]

    repeated = _duplicates(files)
    if repeated:
        raise ValueError(f"sdist contains duplicate file(s): {', '.join(repeated)}")

    present = set(files)
    for meta in _SDIST_METADATA:
        if meta not in present:
            raise ValueError(f"sdist is missing {prefix}{meta}")

    if required:
        optional = (
            set(_SDIST_METADATA)
            | _SDIST_ROOT_FILES
            | {
                name
                for name in present
                if name.startswith(_EGG_INFO)
                or (name.startswith("tests/") and name.endswith(".py"))
            }
        )
        _diff("sdist", present, set(required), optional)


def validate_metadata(text: str, version: str, name: str = _PACKAGE_NAME) -> None:
    """Raise ValueError unless the header ``Name`` and ``Version`` match.

    Only the header block is read: ``METADATA`` appends the README, which can
    contain lines that look like fields.
    """
    header = re.split(r"\r?\n\r?\n", text, maxsplit=1)[0]
    fields: dict[str, str] = {}
    for line in header.splitlines():
        key, sep, value = line.partition(":")
        if sep and key in {"Name", "Version"}:
            fields[key] = value.strip()
    if fields.get("Name") != name:
        raise ValueError(
            f"metadata name {fields.get('Name')!r} does not match package {name!r}"
        )
    if fields.get("Version") != version:
        raise ValueError(
            f"metadata version {fields.get('Version')!r} does not match {version!r}"
        )


def wheel_version(path: Path) -> str:
    """``leaklens_ml-<version>-py3-none-any.whl`` -> ``<version>``."""
    parts = path.name.split("-")
    if len(parts) < 2 or parts[0] != "leaklens_ml":
        raise ValueError(f"{path.name} is not a leaklens wheel filename")
    return parts[1]


def sdist_version(path: Path) -> str:
    """``leaklens_ml-<version>.tar.gz`` -> ``<version>``."""
    prefix = "leaklens_ml-"
    suffix = ".tar.gz"
    if not path.name.startswith(prefix) or not path.name.endswith(suffix):
        raise ValueError(f"{path.name} is not a leaklens sdist filename")
    version = path.name[len(prefix) : -len(suffix)]
    if not version:
        raise ValueError(f"{path.name} has no version")
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


def _check_contents(source_root: Path, names, read, label: str) -> None:
    """Fail when a packaged member differs from the file in the source tree."""
    for name in names:
        source = source_root / name
        if not source.is_file() or read(name) != source.read_bytes():
            raise ValueError(f"{label} member {name} does not match the source tree")


def _check_wheel(path: Path, source_root: Path) -> tuple[int, int]:
    version = wheel_version(path)
    expected = package_version(source_root / "pyproject.toml")
    if version != expected:
        raise ValueError(
            f"wheel version {version!r} does not match package version {expected!r}"
        )
    members = expected_members(source_root)
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        validate_wheel(names, version, required=members)
        validate_metadata(
            archive.read(f"leaklens_ml-{version}.dist-info/METADATA").decode("utf-8"),
            version,
        )
        _check_contents(source_root, members, archive.read, "wheel")
    return len(members), len(names)


def _check_sdist(path: Path, source_root: Path) -> tuple[int, int]:
    version = sdist_version(path)
    expected = package_version(source_root / "pyproject.toml")
    if version != expected:
        raise ValueError(
            f"sdist version {version!r} does not match package version {expected!r}"
        )
    members = expected_members(source_root, sdist=True)
    prefix = f"leaklens_ml-{version}/"
    with tarfile.open(path) as archive:
        names = archive.getnames()
        validate_sdist(names, version, required=members)

        def read(name: str) -> bytes:
            member = archive.extractfile(prefix + name)
            if member is None:
                raise ValueError(f"sdist member {name} is not a readable file")
            return member.read()

        validate_metadata(read("PKG-INFO").decode("utf-8"), version)
        _check_contents(source_root, members, read, "sdist")
    return len(members), len(names)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tag", help="release tag, e.g. v0.13.0")
    parser.add_argument(
        "--wheel",
        type=Path,
        help="built wheel that must match the source tree exactly",
    )
    parser.add_argument(
        "--sdist",
        type=Path,
        help="built sdist that must match the source tree exactly",
    )
    parser.add_argument(
        "--pyproject",
        default=Path("pyproject.toml"),
        type=Path,
        help="package metadata for --tag mode",
    )
    parser.add_argument(
        "--source-root",
        default=Path(__file__).resolve().parents[1],
        type=Path,
        help="checked-out source tree that defines the expected members",
    )
    parser.add_argument("--event", default="release", help="GitHub event name")
    parser.add_argument("--ref", default="", help="GitHub ref, e.g. refs/heads/main")
    args = parser.parse_args(argv)
    if args.wheel is None and args.sdist is None and not args.tag:
        parser.error("--tag, --wheel or --sdist is required")

    try:
        if args.wheel is not None:
            expected, total = _check_wheel(args.wheel, args.source_root)
            print(
                f"Wheel {args.wheel.name} contains all {expected} expected "
                f"package files and nothing else ({total} members)"
            )
        if args.sdist is not None:
            expected, total = _check_sdist(args.sdist, args.source_root)
            print(
                f"Sdist {args.sdist.name} contains all {expected} expected files "
                f"and no build junk ({total} members)"
            )
        if args.tag:
            version = package_version(args.pyproject)
            validate(args.tag, version, args.event, args.ref)
            print(f"Release tag {args.tag} matches package version {version}")
        return 0
    except (
        OSError,
        ValueError,
        KeyError,
        zipfile.BadZipFile,
        tarfile.TarError,
        tomllib.TOMLDecodeError,
    ) as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
