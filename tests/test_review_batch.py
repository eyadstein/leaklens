"""Review batch 01: dossiers are evidence, never labels.

These tests read the committed corpus (``realworld_expansion/``) and never
touch the content-addressed cache, the network, or the frozen ``realworld/``
corpus. They guard one property above all: a dossier can carry a hypothesis,
but only a human decision recorded in ``reviews.jsonl`` can make a candidate
scored ground truth.
"""

import hashlib
import json
import pathlib

import pytest

from leaklens.corpus_expansion import (
    CODE_REF,
    HYPOTHESIS_AUTHOR,
    HYPOTHESIS_CONFIDENCE,
    MECHANISMS,
    SCORED_STATES,
    apply_decisions,
    code_text,
    dossier,
    load_decisions,
    load_manifest,
    redistributable,
    scored_records,
    select_batch,
    signal_bucket,
    validate_decision,
    validate_dossier,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]
EXPANSION = ROOT / "realworld_expansion"
BATCH = EXPANSION / "review_batch_01.jsonl"
BATCH_MD = EXPANSION / "REVIEW_BATCH_01.md"


@pytest.fixture(scope="module")
def dossiers():
    return load_manifest(BATCH)


@pytest.fixture(scope="module")
def manifest():
    return {rec["id"]: rec for rec in load_manifest(EXPANSION / "manifest.jsonl")}


def _decision(**kw):
    decision = {
        "candidate_id": "rwx_cf8cb6a83808",
        "state": "legitimate",
        "rationale": "scaler is fitted on X_train after the split at cell 15",
        "evidence_refs": ["cell 128:4"],
        "mechanism": "legitimate_lookalike",
        "confidence": None,
        "reviewer": "eyad",
        "decided_at": "2026-11-15T12:00:00Z",
        "source": "human",
    }
    decision.update(kw)
    return decision


# --- composition ------------------------------------------------------------


def test_batch_is_thirty_candidates_from_thirty_repositories(dossiers):
    ids = [row["candidate_id"] for row in dossiers]
    repos = [row["provenance"]["repo"] for row in dossiers]
    assert len(ids) == 30
    assert len(set(ids)) == 30
    assert len(set(repos)) == 30, "one candidate per repository is the default"


def test_batch_is_the_queue_selection_not_an_arbitrary_pick(dossiers, manifest):
    chosen = select_batch(list(manifest.values()))
    assert [row["candidate_id"] for row in dossiers] == chosen


def test_batch_covers_sources_signals_licences_and_mechanisms(dossiers):
    assert {row["provenance"]["source_id"] for row in dossiers} == {
        "github_repos",
        "github_search",
        "kaggle_distilkaggle",
    }
    assert {row["scanner"]["signal"] for row in dossiers} == {
        "high",
        "medium",
        "low",
        "none",
    }
    assert len({row["license"]["status"] for row in dossiers}) >= 2
    mechanisms = [row["mechanism"] for row in dossiers]
    assert len(set(mechanisms)) >= 4
    # A batch of only suspects would be useless for measuring false positives.
    assert "none_observed" in mechanisms
    assert "legitimate_lookalike" in mechanisms


def test_no_candidate_is_reused_from_the_frozen_corpus(dossiers, manifest):
    frozen = {
        json.loads(line)["source_repo"]
        for line in (ROOT / "realworld" / "manifest.jsonl")
        .read_text(encoding="utf-8")
        .splitlines()
        if line.strip()
    }
    assert {row["provenance"]["repo"] for row in dossiers}.isdisjoint(frozen)


# --- provenance and stable ids ---------------------------------------------


def test_every_dossier_binds_to_the_manifest_record(dossiers, manifest):
    for row in dossiers:
        rec = manifest[row["candidate_id"]]
        assert validate_dossier(row, rec) == []
        assert row["provenance"]["sha256"] == rec["sha256"]
        assert row["provenance"]["commit_sha"] == rec["commit_sha"]
        assert row["scanner"]["signal"] == signal_bucket(rec)


def test_dossier_ids_are_the_manifest_ids(dossiers, manifest):
    assert {row["candidate_id"] for row in dossiers} <= set(manifest)


def test_references_are_well_formed_and_cited(dossiers):
    for row in dossiers:
        assert row["code_refs"]
        for ref in row["code_refs"]:
            assert CODE_REF.match(ref["ref"]), ref
            assert ref["note"].strip()


# --- decisions stay empty and human ----------------------------------------


def test_every_decision_is_empty(dossiers):
    assert [row["decision"] for row in dossiers] == [None] * len(dossiers)


def test_decision_log_only_ever_holds_human_decisions():
    # An absent log means nobody has decided yet; a present one must be human.
    log = EXPANSION / "reviews.jsonl"
    if log.exists():
        assert all(row.get("source") == "human" for row in load_decisions(log)), (
            "a non-human decision source reached reviews.jsonl"
        )


def test_hypotheses_are_ai_marked_and_never_ground_truth(dossiers):
    for row in dossiers:
        hypothesis = row["hypothesis"]
        assert hypothesis["author"] == HYPOTHESIS_AUTHOR == "ai_unverified"
        assert hypothesis["is_ground_truth"] is False
        assert hypothesis["confidence"] in HYPOTHESIS_CONFIDENCE
        assert row["mechanism"] in MECHANISMS
        # The human-facing fields must be the only place a verdict could live.
        assert "verdict" not in row
        assert "state" not in row


def test_dossier_rejects_an_invented_decision(dossiers, manifest):
    row = json.loads(json.dumps(dossiers[0]))
    rec = manifest[row["candidate_id"]]
    row["decision"] = {"state": "confirmed_leakage", "reviewer": "ai"}
    assert any("decision" in error for error in validate_dossier(row, rec))


def test_dossier_requires_the_full_analysis(dossiers, manifest):
    rec = manifest[dossiers[0]["candidate_id"]]
    partial = {"mechanism": "none_observed", "code_refs": ["cell 0:1"]}
    with pytest.raises(ValueError, match="missing"):
        dossier(rec, partial)


def test_dossier_rejects_code_text_on_a_copyleft_row(dossiers, manifest):
    copyleft = next(row for row in dossiers if row["license"]["status"] != "permissive")
    rec = manifest[copyleft["candidate_id"]]
    row = json.loads(json.dumps(copyleft))
    row["excerpts"] = [{"ref": row["code_refs"][0]["ref"], "text": "print(1)"}]
    errors = validate_dossier(row, rec)
    assert any("non-permissive" in error for error in errors)


# --- licence ---------------------------------------------------------------


def test_excerpts_are_shipped_only_for_permissive_licences(dossiers):
    for row in dossiers:
        allowed = redistributable(row["license"]["spdx"])
        assert row["license"]["redistributable"] is allowed
        if not allowed:
            assert row["excerpts"] == []
        else:
            assert row["excerpts"], f"{row['candidate_id']} lost its excerpts"
            refs = {ref["ref"] for ref in row["code_refs"]}
            assert {item["ref"] for item in row["excerpts"]} <= refs
            assert all(item["text"].strip() for item in row["excerpts"])


# --- scoring gate ----------------------------------------------------------


@pytest.mark.parametrize("state", ["potential", "uncertain", "not_reviewed"])
def test_indefinite_states_never_enter_a_scored_set(manifest, state):
    records = [manifest["rwx_cf8cb6a83808"]]
    assert scored_records(records, [_decision(state=state)]) == []


@pytest.mark.parametrize("state", sorted(SCORED_STATES))
def test_only_definitive_human_states_are_scored(manifest, state):
    records = [manifest["rwx_cf8cb6a83808"]]
    scored = scored_records(records, [_decision(state=state)])
    assert [rec["id"] for rec in scored] == ["rwx_cf8cb6a83808"]
    assert scored[0]["review"]["reviewer"] == "eyad"


def test_ai_authored_decisions_are_rejected():
    errors = validate_decision(_decision(source="ai"))
    assert any("human" in error for error in errors)


@pytest.mark.parametrize("reviewer", ["ai", "unassigned", "anonymous", ""])
def test_placeholder_reviewer_attribution_is_rejected(reviewer):
    assert any(
        "placeholder" in error
        for error in validate_decision(_decision(reviewer=reviewer))
    )


def test_a_definitive_verdict_needs_a_rationale_and_a_reference():
    assert any(
        "rationale" in error for error in validate_decision(_decision(rationale="  "))
    )
    assert any(
        "code reference" in error
        for error in validate_decision(_decision(evidence_refs=[]))
    )


def test_a_fabricated_timestamp_is_rejected():
    errors = validate_decision(_decision(decided_at="just now"))
    assert any("timestamp" in error for error in errors)


# --- the manifest is history ----------------------------------------------


def test_merging_decisions_does_not_rewrite_the_manifest(manifest):
    path = EXPANSION / "manifest.jsonl"
    before = hashlib.sha256(path.read_bytes()).hexdigest()
    records = list(manifest.values())
    apply_decisions(records, [_decision()])
    assert hashlib.sha256(path.read_bytes()).hexdigest() == before


def test_merging_decisions_does_not_mutate_the_input_records(manifest):
    record = manifest["rwx_cf8cb6a83808"]
    assert record["review"] == {"state": "not_reviewed"}
    apply_decisions([record], [_decision()])
    assert record["review"] == {"state": "not_reviewed"}


def test_code_text_is_extracted_from_the_notebook_not_the_dossier():
    """Dossier excerpts must be source, never a summary written by hand."""
    notebook = json.dumps(
        {
            "cells": [
                {"cell_type": "code", "source": ["x = 1\n"]},
                {"cell_type": "markdown", "source": ["prose"]},
            ]
        }
    )
    text = code_text(notebook.encode("utf-8"), "ipynb")
    assert "x = 1" in text
    assert "prose" not in text


# --- the readable artifact -------------------------------------------------


def test_markdown_documents_every_candidate(dossiers):
    markdown = BATCH_MD.read_text(encoding="utf-8")
    for row in dossiers:
        assert f"`{row['candidate_id']}`" in markdown
    assert markdown.count("**Decision.** none recorded") == len(dossiers)
    assert "AI hypothesis (unverified" in markdown
