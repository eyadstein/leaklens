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
  - the transitive dependencies that installed distribution metadata declares,
    with real ``dependsOn`` edges

Only unconditional requirements become edges. An extra- or platform-guarded
requirement (``; python_version < "3.11"``) would describe a different
environment, so it is counted in the ``sbom-conditional-requirements`` property
and left out rather than guessed. A package with no installed metadata is a
component with no outgoing edge, and the ``sbom-graph`` property then reads
``declared-only`` instead of claiming a complete graph.

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
# A resolved environment stays well inside these bounds (the [app]/[api]/[arena]
# closure is ~100 components at depth 5); they exist so a pathological
# environment fails loudly instead of producing an unbounded document.
MAX_COMPONENTS = 500
MAX_DEPTH = 20


def _project():
    with PYPROJECT.open("rb") as fh:
        return tomllib.load(fh)["project"]


def _dependency_name(requirement: str) -> str:
    match = _REQUIREMENT_NAME.match(requirement.strip())
    if match is None:
        raise ValueError(f"cannot parse dependency: {requirement!r}")
    return match.group(0)


def _key(name: str) -> str:
    """PEP 503 distribution key, so one package is one node."""
    return re.sub(r"[-_.]+", "-", name).lower()


def _installed_version(name: str) -> str | None:
    try:
        return metadata.version(name)
    except metadata.PackageNotFoundError:
        return None


def _dependency_graph(declared: list[str]) -> dict:
    """Walk installed dependency metadata from the declared roots, breadth first.

    A requirement without an environment marker becomes an edge; one with a
    marker is counted in ``conditional`` and not graphed, because the edge would
    only hold on some environments. A package that is not installed is still a
    node (its parent declares it) but has no outgoing edges, which makes
    ``complete`` false.
    """
    names = {_key(name): name for name in declared}
    depth = {_key(name): 0 for name in declared}
    edges: dict[str, set[str]] = {}
    pending = sorted(declared, key=_key)
    conditional = 0
    complete = True
    while pending:
        name = pending.pop(0)
        key = _key(name)
        if key in edges:
            continue
        targets: set[str] = set()
        if _installed_version(name) is None:
            complete = False
        else:
            for requirement in sorted(metadata.requires(name) or []):
                dependency, guarded = _dependency_name(requirement), ";" in requirement
                if guarded:
                    conditional += 1
                    continue
                dependency_key = _key(dependency)
                if dependency_key == key or dependency_key in targets:
                    continue
                targets.add(dependency_key)
                if dependency_key not in names:
                    if len(names) >= MAX_COMPONENTS:
                        raise ValueError(
                            f"dependency graph exceeds {MAX_COMPONENTS} "
                            "components; narrow the extras or raise "
                            "MAX_COMPONENTS deliberately"
                        )
                    if depth[key] + 1 > MAX_DEPTH:
                        raise ValueError(
                            f"dependency graph exceeds {MAX_DEPTH} levels; "
                            "raise MAX_DEPTH deliberately"
                        )
                    names[dependency_key] = dependency
                    depth[dependency_key] = depth[key] + 1
                if dependency_key not in edges and dependency_key not in {
                    _key(item) for item in pending
                }:
                    pending.append(dependency)
        edges[key] = targets
    return {
        "names": names,
        "edges": edges,
        "conditional": conditional,
        "complete": complete,
    }


def _required_keys(graph: dict, roots: list[str]) -> set[str]:
    """Keys reachable from the runtime roots; everything else stays optional."""
    seen: set[str] = set()
    stack = [_key(root) for root in roots]
    while stack:
        key = stack.pop()
        if key in seen or key not in graph["names"]:
            continue
        seen.add(key)
        stack.extend(graph["edges"].get(key, ()))
    return seen


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
    # feather), and transitively through several parents. One node per package;
    # the strictest scope (reachable from the runtime roots) wins.
    declared = sorted(
        {*runtime, *(dep for deps in extras.values() for dep in deps)}, key=_key
    )
    graph = _dependency_graph(declared)
    required = _required_keys(graph, runtime)
    components = [
        _component(graph["names"][key], "required" if key in required else "optional")
        for key in sorted(graph["names"])
    ]
    refs = {_key(component["name"]): component["bom-ref"] for component in components}
    dependencies = [
        {
            "ref": refs[key],
            "dependsOn": sorted(refs[target] for target in graph["edges"].get(key, ())),
        }
        for key in sorted(refs)
    ]

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
                {
                    "name": "sbom-graph",
                    "value": "complete" if graph["complete"] else "declared-only",
                },
                {
                    "name": "sbom-conditional-requirements",
                    "value": str(graph["conditional"]),
                },
            ],
        },
        "components": components,
        "dependencies": [
            {
                "ref": own_purl,
                "dependsOn": sorted({refs[_key(dep)] for dep in runtime}),
            },
            *dependencies,
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
    try:
        bom = build_sbom()
    except ValueError as exc:
        print(f"::error::{exc}", file=sys.stderr)
        return 1
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
    # Every dependency ref resolves: an edge to a component that was not
    # emitted would be a broken graph, not a partial one.
    known = {component["bom-ref"] for component in reparsed["components"]}
    assert all(
        dependency["ref"] in known
        or dependency["ref"] == reparsed["metadata"]["component"]["bom-ref"]
        for dependency in reparsed["dependencies"]
    )
    assert all(
        target in known
        for dependency in reparsed["dependencies"]
        for target in dependency["dependsOn"]
    )
    print(
        f"Wrote {out} (package leaklens-ml {reparsed['metadata']['component']['version']})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
