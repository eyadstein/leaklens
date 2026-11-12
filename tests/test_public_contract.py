"""Frozen catalogue identities and the legacy evidence-source bridge.

The golden fingerprints are Phase 13 M6: one finding per surface, hashed with
the shipped ``fingerprint_for``. Changing any of these silently changes every
baseline and suppression in the field, so the test is intentionally literal.
"""

from __future__ import annotations

from leaklens import manifest
from leaklens.evidence import SOURCES
from leaklens.fingerprint import fingerprint_for
from leaklens.leakage import Finding

# The evidence layer predates the catalogue and spells the dataset surface "data".
EVIDENCE_TO_SURFACE = {
    "data": "dataset",
    "split": "split",
    "code": "code",
    "notebook": "notebook",
    "image": "image",
}

GOLDEN_FINGERPRINTS = {
    "dataset": (
        Finding("high", "target_leakage", "feature", "detail"),
        "41980527b405df2d39076cae671c7c5f0922dda2b173fb754b7628a707062859",
    ),
    "split": (
        Finding("high", "split_overlap", "*", "detail", stable_key="abc"),
        "b2e217efceb6dc40139e580a3ceaf265bfbee86d571f46474485affbc723953c",
    ),
    "code": (
        Finding("low", "parse_error", "a.py", "detail"),
        "c519471296b7d66779e0a10be24027399318878d82faf7e34979821ad357e5a6",
    ),
    "notebook": (
        Finding(
            "medium",
            "notebook_out_of_order_execution",
            "nb.ipynb",
            "detail",
            stable_key="cell-2",
        ),
        "4344875b280c60e97aa049cad947f9c4a2ab9b1453e59a260866122e0d45881a",
    ),
    "image": (
        Finding("high", "image_overlap", "*", "detail", stable_key="dup"),
        "6d7c2370eb1e9379b668a68c3282a7d103e5a8131e5875bbb67065fc3911704f",
    ),
}


def test_modalities_are_frozen():
    assert manifest.MODALITIES == {
        "tabular",
        "code",
        "sql",
        "image",
        "text",
        "artifact",
    }


def test_surfaces_are_frozen():
    assert manifest.SURFACES == {"dataset", "split", "code", "notebook", "image"}


def test_engine_ids_are_frozen():
    assert set(manifest.ENGINES) == {
        "leaklens.python",
        "leaklens.scanner.scan_source",
        "leaklens.ai.experimental",
        "leaklens.intelligence.deterministic",
        "leaklens.pipeline",
        "leaklens.text",
    }


def test_engine_kinds_are_registered():
    for engine in manifest.ENGINES.values():
        assert engine.kinds <= set(manifest.BY_KIND)


def test_legacy_evidence_sources_map_to_catalogue_surfaces():
    assert set(EVIDENCE_TO_SURFACE) == set(SOURCES)
    assert set(EVIDENCE_TO_SURFACE.values()) <= manifest.SURFACES
    assert EVIDENCE_TO_SURFACE["data"] == "dataset"


def test_surface_fingerprints_are_frozen():
    for surface, (finding, expected) in GOLDEN_FINGERPRINTS.items():
        assert manifest.surface_for(finding.kind) == surface
        assert fingerprint_for(finding) == expected, surface
