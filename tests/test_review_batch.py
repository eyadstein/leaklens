"""Review batch 01: dossiers are evidence, never labels.

These tests read the committed corpus (``realworld_expansion/``) and never
touch the content-addressed cache, the network, or the frozen ``realworld/``
corpus. They guard one property above all: a dossier can carry a hypothesis,
but only a human decision recorded in ``reviews.jsonl`` can make a candidate
scored ground truth.
"""

import hashlib
import importlib.util
import json
import pathlib
from types import SimpleNamespace

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
    review_status,
    scored_records,
    select_batch,
    signal_bucket,
    summarize,
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


# --- independent review and adjudication ------------------------------------


def test_roles_materials_and_revision_are_validated():
    assert validate_decision(_decision(role="adjudication")) == []
    assert validate_decision(_decision(materials="blind")) == []
    assert any(
        "role" in error for error in validate_decision(_decision(role="second_pass"))
    )
    assert any(
        "materials" in error
        for error in validate_decision(_decision(materials="partial"))
    )
    assert any(
        "corpus_revision" in error
        for error in validate_decision(_decision(corpus_revision="not-a-revision"))
    )
    assert validate_decision(_decision(corpus_revision="0123456789abcdef")) == []


def test_a_single_review_is_single_reviewed():
    status = review_status([_decision()])
    assert status["rwx_cf8cb6a83808"] == "single_reviewed"


def test_two_distinct_blind_reviewers_that_agree_are_double_reviewed():
    status = review_status(
        [
            _decision(reviewer="eyad", materials="full"),
            _decision(reviewer="reviewer-b", materials="blind"),
        ]
    )
    assert status["rwx_cf8cb6a83808"] == "double_reviewed"


def test_two_reviewers_with_the_same_materials_are_not_independent():
    status = review_status(
        [
            _decision(reviewer="eyad"),
            _decision(reviewer="reviewer-b"),
        ]
    )
    assert status["rwx_cf8cb6a83808"] == "double_reviewed_unblinded"


def test_disagreement_is_disputed_until_adjudicated():
    decisions = [
        _decision(reviewer="eyad", materials="blind"),
        _decision(reviewer="reviewer-b", materials="blind", state="confirmed_leakage"),
    ]
    assert review_status(decisions)["rwx_cf8cb6a83808"] == "disputed"
    decisions.append(
        _decision(
            reviewer="reviewer-c",
            materials="blind",
            role="adjudication",
            state="legitimate",
        )
    )
    assert review_status(decisions)["rwx_cf8cb6a83808"] == "adjudicated"


def test_the_same_reviewer_recording_twice_is_not_double_review():
    decisions = [
        _decision(reviewer="eyad", state="potential"),
        _decision(reviewer="Eyad", state="legitimate"),
    ]
    status = review_status(decisions)["rwx_cf8cb6a83808"]
    assert status == "single_reviewed"


def test_indefinite_decisions_remain_uncertain():
    decisions = [_decision(state="potential"), _decision(state="uncertain")]
    assert review_status(decisions)["rwx_cf8cb6a83808"] == "uncertain"


def test_summary_counts_the_audit_status_of_the_clean_corpus(manifest):
    records = [manifest["rwx_cf8cb6a83808"]]
    decisions = [_decision()]
    stats = summarize(records, [], decisions)
    assert stats["review_status"]["single_reviewed"] == 1
    assert stats["review_status"]["double_reviewed"] == 0
    assert stats["review_status"]["not_reviewed"] == 0


def test_the_committed_corpus_does_not_claim_independence():
    decisions = load_decisions(EXPANSION / "reviews.jsonl")
    reviewers = {decision["reviewer"] for decision in decisions}
    status = review_status(decisions)
    if len(reviewers) == 1:
        assert set(status.values()) <= {"single_reviewed", "uncertain"}
        assert "double_reviewed" not in status.values()
        assert "adjudicated" not in status.values()


def _collect_script():
    path = ROOT / "scripts" / "collect_expansion.py"
    spec = importlib.util.spec_from_file_location("collect_expansion_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_blind_show_does_not_leak_the_recorded_verdict(capsys):
    """A second reviewer must not see the first verdict while reviewing."""
    script = _collect_script()
    ids = [row["candidate_id"] for row in load_manifest(BATCH)]
    records = {
        rec["id"]: rec
        for rec in apply_decisions(
            load_manifest(EXPANSION / "manifest.jsonl"),
            load_decisions(EXPANSION / "reviews.jsonl"),
        )
    }
    ident = next(i for i in ids if records[i]["review"]["state"] != "not_reviewed")
    state = records[ident]["review"]["state"]

    script.command_show(SimpleNamespace(id=ident, context=0, blind=True, side=None))
    blind = capsys.readouterr().out
    assert f"decision: {state} by" not in blind
    assert "withheld until your independent verdict" in blind
    # Derived review state is a disclosure path too: it tells a second reviewer
    # that a verdict exists (and whether it was disputed or adjudicated).
    assert "audit status" not in blind

    script.command_show(SimpleNamespace(id=ident, context=0, blind=False, side=None))
    full = capsys.readouterr().out
    assert f"decision: {state} by" in full


def test_blind_show_withholds_the_authored_interpretation(capsys):
    """The verdict is not the only disclosure: the hypothesis states the answer."""
    script = _collect_script()
    dossier = load_manifest(BATCH)[0]
    script.command_show(
        SimpleNamespace(id=dossier["candidate_id"], context=0, blind=True, side=None)
    )
    blind = capsys.readouterr().out
    assert dossier["workflow"]["summary"] not in blind
    for item in dossier["evidence_against"]:
        assert item not in blind
    for item in dossier["questions"]:
        assert item not in blind
    for ref in dossier["code_refs"]:
        assert ref["note"] not in blind
        assert ref["ref"] in blind
    assert "scanner signal" not in blind
    assert "hypothesis confidence" not in blind


# --- the blind pack --------------------------------------------------------


def _blind_pack():
    script = _collect_script()
    script.command_blind_pack(None)
    rows = load_manifest(EXPANSION / "review_batch_01.blind.jsonl")
    markdown = (EXPANSION / "REVIEW_BATCH_01.blind.md").read_text(encoding="utf-8")
    return rows, markdown


def test_blind_pack_keeps_only_what_a_reviewer_may_see():
    rows, _ = _blind_pack()
    assert len(rows) == len(load_manifest(BATCH))
    assert {row["candidate_id"] for row in rows} == {
        row["candidate_id"] for row in load_manifest(BATCH)
    }
    for row in rows:
        assert set(row) == {
            "candidate_id",
            "provenance",
            "license",
            "code_refs",
            "excerpts",
            "decision",
        }
        assert row["decision"] == {"state": "not_reviewed"}
        assert all(set(ref) == {"ref"} for ref in row["code_refs"])


def test_blind_pack_markdown_states_no_verdict_and_no_hypothesis():
    dossiers = {row["candidate_id"]: row for row in load_manifest(BATCH)}
    rows, markdown = _blind_pack()
    assert "**Scanner signal**" not in markdown
    assert "**Workflow.**" not in markdown
    assert "AI hypothesis (unverified" not in markdown
    assert "Possible mechanism" not in markdown
    for row in rows:
        assert f"`{row['candidate_id']}`" in markdown
        dossier = dossiers[row["candidate_id"]]
        assert dossier["workflow"]["summary"] not in markdown
        assert dossier["why_it_may_apply"] not in markdown
        for item in dossier["evidence_against"]:
            assert item not in markdown
    assert markdown.count("**Decision.** none recorded") == len(rows)


def test_blind_pack_is_deterministic():
    first, _ = _blind_pack()
    second, _ = _blind_pack()
    assert first == second


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


# --- generated views apply the decision log -------------------------------


def test_review_queue_excludes_decided_candidates():
    """A reviewed sample must never be queued for review again."""
    decided = {
        decision["candidate_id"]
        for decision in load_decisions(EXPANSION / "reviews.jsonl")
    }
    queue = load_manifest(EXPANSION / "review_queue.jsonl")
    assert queue
    assert {row["id"] for row in queue}.isdisjoint(decided)
    assert {row["review"]["state"] for row in queue} == {"not_reviewed"}


def test_stats_count_the_recorded_decisions(manifest):
    """stats.json is generated: it must agree with the decision log, not 0."""
    decisions = load_decisions(EXPANSION / "reviews.jsonl")
    records = list(manifest.values())
    assert {decision["candidate_id"] for decision in decisions} <= set(manifest)
    decided = apply_decisions(records, decisions)
    stats = json.loads((EXPANSION / "stats.json").read_text(encoding="utf-8"))
    assert stats["scored_labels"] == len(scored_records(records, decisions))
    assert stats["confirmed_labels"] == sum(
        1 for rec in decided if rec["review"]["state"] == "confirmed_leakage"
    )
    assert stats["reviews"]["not_reviewed"] == sum(
        1 for rec in decided if rec["review"]["state"] == "not_reviewed"
    )


def test_blind_show_suggests_recording_the_decision_as_blind(capsys):
    """The printed command must carry the provenance the reviewer actually used."""
    script = _collect_script()
    ids = [row["candidate_id"] for row in load_manifest(BATCH)]
    records = {
        rec["id"]: rec
        for rec in apply_decisions(
            load_manifest(EXPANSION / "manifest.jsonl"),
            load_decisions(EXPANSION / "reviews.jsonl"),
        )
    }
    ident = next(i for i in ids if records[i]["review"]["state"] == "not_reviewed")

    script.command_show(SimpleNamespace(id=ident, context=0, blind=True, side=None))
    blind = capsys.readouterr().out.split("record one with:")[1].splitlines()[0]
    assert "review --id" in blind
    assert "--blind" in blind

    script.command_show(SimpleNamespace(id=ident, context=0, blind=False, side=None))
    full = capsys.readouterr().out.split("record one with:")[1].splitlines()[0]
    assert "--blind" not in full
