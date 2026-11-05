from __future__ import annotations

import re

from . import __version__
from .fixes import advice

LEVEL = {"high": "error", "medium": "warning", "low": "note"}
LOC = re.compile(r"^(?P<path>.+?)(?:\[cell (?P<cell>\d+)\])?:(?P<line>\d+)$")


def to_sarif(findings, default_uri: str = ".") -> dict:
    rules, seen, results = [], set(), []
    for f in findings:
        if f.kind not in seen:
            seen.add(f.kind)
            rules.append(
                {
                    "id": f.kind,
                    "name": f.kind,
                    "shortDescription": {"text": f.kind.replace("_", " ")},
                    "help": {"text": advice(f.kind) or f.kind},
                }
            )
        m = LOC.match(str(f.column))
        if m:
            uri = m.group("path").replace("\\", "/")
            region = {"startLine": int(m.group("line"))}
            msg = f.detail
            if m.group("cell"):
                msg += (
                    f" (notebook cell {m.group('cell')}; line is relative to the cell)"
                )
        else:
            uri = str(default_uri).replace("\\", "/")
            region = {"startLine": 1}
            msg = f"{f.column}: {f.detail}"
        results.append(
            {
                "ruleId": f.kind,
                "level": LEVEL.get(f.severity, "note"),
                "message": {"text": msg},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": uri},
                            "region": region,
                        }
                    }
                ],
            }
        )
    return {
        "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
        "version": "2.1.0",
        "runs": [
            {
                "tool": {
                    "driver": {
                        "name": "LeakLens",
                        "version": __version__,
                        "informationUri": "https://github.com/eyadstein/leaklens",
                        "rules": rules,
                    }
                },
                "results": results,
            }
        ],
    }
