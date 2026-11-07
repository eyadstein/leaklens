from leaklens.intelligence.confidence import (
    assess_cluster_confidence,
    assess_confidence,
)
from leaklens.intelligence.schemas import Evidence


def make_evidence(identifier, kind="test", supports=True):
    return Evidence(
        evidence_id=identifier,
        source="scanner",
        kind=kind,
        description="Test evidence",
        supports=supports,
    )


def test_empty_evidence_has_very_low_confidence():
    result = assess_confidence([])

    assert result.score == 0.0
    assert result.level == "very_low"
    assert result.supporting_evidence == 0


def test_supporting_evidence_increases_confidence():
    one = assess_confidence([make_evidence("one")])
    several = assess_confidence([make_evidence("one"), make_evidence("two")])

    assert several.score > one.score


def test_contradicting_evidence_reduces_confidence():
    supporting = assess_confidence([make_evidence("one")])
    mixed = assess_confidence(
        [make_evidence("one"), make_evidence("two", supports=False)]
    )

    assert mixed.score < supporting.score
    assert mixed.contradicting_evidence == 1


def test_multiple_supporting_categories_add_diversity_bonus():
    same_kind = assess_confidence([make_evidence("one"), make_evidence("two")])
    different_kinds = assess_confidence(
        [
            make_evidence("one", "pipeline"),
            make_evidence("two", "dataset"),
        ]
    )

    assert different_kinds.score > same_kind.score


def test_confidence_is_deterministic_and_bounded():
    evidence = [make_evidence(str(i)) for i in range(20)]

    first = assess_confidence(evidence)
    second = assess_confidence(evidence)

    assert first == second
    assert 0.0 <= first.score <= 1.0


def test_cluster_confidence_rewards_corroboration_without_certainty():
    single = assess_cluster_confidence([make_evidence("one", "target_leakage")])
    corroborated = assess_cluster_confidence(
        [
            make_evidence("one", "target_leakage"),
            make_evidence("two", "id_like"),
            Evidence(
                "one-detail",
                "dataset",
                "target_leakage",
                "detector detail",
                metadata=(("role", "detector"),),
            ),
        ]
    )

    assert corroborated.score > single.score
    assert corroborated.score <= 0.95
    assert any("independent finding kinds" in reason for reason in corroborated.reasons)


def test_cluster_confidence_empty_is_very_low():
    result = assess_cluster_confidence([])

    assert result.score == 0.0
    assert result.level == "very_low"
