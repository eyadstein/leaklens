"""SBOM - CycloneDX Software Bill of Materials for LeakLens.

Usage:
    python scripts/generate_sbom.py [OUT]
    python scripts/generate_sbom.py --help

The default output is ``sbom/sbom.json``, deliberately outside ``dist/`` so
the PyPI publisher can never upload it. The generator is standalone: it
needs only the standard library (plus ``tomli`` on Python 3.10) and reads
dependencies and extras directly from ``pyproject.toml``. It does not need
``cyclonedx-py``. The generated file is parsed back and checked for the
core CycloneDX 1.5 keys before it is accepted.

The SBOM lists:
  - the LeakLens package itself (name leaklens-ml, actual version,
    description, MIT license from pyproject)
  - declared runtime dependencies, with installed versions when available
  - optional extras as optional-scope components (the dev-only extra is
    excluded from the release dependency list)

No local machine paths or secrets are written. Generation is reproducible
for a given environment and dependency set.
"""

from __future__ import annotations

import json
import re
import sys
from importlib import metadata
from pathlib import Path

try:
    import tomllib
except ModuleNotFoundError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[1]
PYPROJECT = ROOT / "pyproject.toml"
_REQUIREMENT_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*")


def _project():
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _dependency_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement.strip())
    if match is None:
        raise ValueError(f"cannot parse dependency: {requirement!r}")
    return match.group(0)


def _purl(name: str, version: str | None = None) -> str:
    normalized = name.lower().replace("_", "-")
    return f"pkg:pypi/{normalized}" + (f"@{version}" if version else "")


def _component(name: str, scope: str) -> dict:
    try:
        version = metadata.version(name)
    except metadata.PackageNotFoundError:
        version = None
    component = {
        "type": "library",
        "name": name,
        "scope": scope,
        "purl": _purl(name, version),
        "bom-ref": _purl(name, version),
    }
    if version:
        component["version"] = version
    return component


def build_sbom() -> dict:
    project = _project()
    name = project["name"]
    version = project["version"]
    runtime = [_dependency_name(dep) for dep in project.get("dependencies", [])]
    extras = {
        extra: [_dependency_name(dep) for dep in deps]
        for extra, deps in project.get("optional-dependencies", {}).items()
        if extra != "dev"
    }

    # A package can appear in more than one extra (pyarrow for parquet and
    # feather). Emit one component per package and keep the strictest scope.
    scopes = {dep: "required" for dep in runtime}
    for deps in extras.values():
        for dep in deps:
            scopes.setdefault(dep, "optional")
    components = [_component(dep, scope) for dep, scope in sorted(scopes.items())]
    component_refs = {component["bom-ref"] for component in components}
    runtime_refs = sorted(
        component["bom-ref"] for component in components if component["name"] in runtime
    )

    own_purl = _purl(name, version)
    return {
        "$schema": "http://cyclonedx.org/schema/bom-1.5.schema.json",
        "bomFormat": "CycloneDX",
        "specVersion": "1.5",
        "version": 1,
        "metadata": {
            "component": {
                "type": "application",
                "name": name,
                "version": version,
                "description": project.get("description", ""),
                "licenses": [{"license": {"id": "MIT"}}],
                "purl": own_purl,
                "bom-ref": own_purl,
            },
            "properties": [
                {"name": "ci:PYTHON", "value": project.get("requires-python", "")},
                {"name": "runtime-deps", "value": ",".join(sorted(runtime))},
                {"name": "optional-extras", "value": ",".join(sorted(extras))},
            ],
        },
        "components": components,
        "dependencies": [
            {"ref": own_purl, "dependsOn": runtime_refs},
            *[{"ref": ref, "dependsOn": []} for ref in sorted(component_refs)],
        ],
    }


def _inside_dist(path: Path) -> bool:
    """True when ``path`` resolves inside the repository's ``dist/``."""
    dist = (ROOT / "dist").resolve()
    resolved = path.resolve()
    return resolved == dist or dist in resolved.parents


_USAGE = """usage: generate_sbom.py [OUT]

Write a CycloneDX 1.5 SBOM for leaklens-ml.

positional arguments:
  OUT         output path (default: sbom/sbom.json; must be outside dist/)
"""


def main(argv=None) -> int:
    argv = sys.argv if argv is None else argv
    args = list(argv[1:])
    # An option-shaped argument is a usage error, not a filename: taking it as
    # the output path wrote a file literally named "--help".
    if any(arg in ("-h", "--help") for arg in args):
        print(_USAGE, end="")
        return 0
    if any(arg.startswith("-") for arg in args):
        print(_USAGE, end="", file=sys.stderr)
        bad = next(arg for arg in args if arg.startswith("-"))
        print(f"::error::unknown option: {bad}", file=sys.stderr)
        return 2
    if len(args) > 1:
        print(_USAGE, end="", file=sys.stderr)
        print("::error::only one output path is accepted", file=sys.stderr)
        return 2
    out = Path(args[0]) if args else ROOT / "sbom" / "sbom.json"
    if _inside_dist(out):
        print(
            f"::error::SBOM output {out} must be outside dist/ "
            "so the PyPI publisher never uploads it",
            file=sys.stderr,
        )
        return 1
    bom = build_sbom()
    out.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(bom, indent=2) + "\n"
    out.write_text(text, encoding="utf-8")
    # Validate: parse back as JSON and check core CycloneDX keys.
    reparsed = json.loads(out.read_text(encoding="utf-8"))
    assert reparsed["bomFormat"] == "CycloneDX"
    assert reparsed["specVersion"] == "1.5"
    assert reparsed["metadata"]["component"]["name"] == "leaklens-ml"
    assert all(
        "purl" not in component or component["purl"].startswith("pkg:pypi/")
        for component in reparsed["components"]
    )
    assert all(
        "version" not in component
        or component["purl"].endswith("@" + component["version"])
        for component in reparsed["components"]
    )
    print(
        f"Wrote {out} (package leaklens-ml {reparsed['metadata']['component']['version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
