"""Collection pipeline logic for the real-world expansion corpus (Mission 1).

These tests use synthetic fixtures, plus the committed manifest for the
split-integrity invariant: no network, no downloads. The acquisition script
(``scripts/collect_expansion.py``) is exercised by the live run recorded in
``realworld_expansion/``, not here.
"""

import hashlib
import json
import pathlib

import pytest

from leaklens.corpus_expansion import (
    FREEZE_SCHEMA,
    NEAR_THRESHOLD,
    REVIEW_STATES,
    SCORED_STATES,
    SKETCH_SIZE,
    candidate_id,
    code_text,
    content_sha256,
    deduplicate,
    freeze,
    freeze_filename,
    license_status,
    load_manifest,
    queue,
    redistributable,
    review,
    sketch,
    split_role,
    split_roles,
    summarize,
    validate,
    verify_freeze,
    write_manifest,
)

ROOT = pathlib.Path(__file__).resolve().parents[1]


def _record(sid, repo="o/r", path="a.ipynb", sha="a" * 16, **kw):
    rec = {
        "id": candidate_id(sid, repo, path, sha),
        "source_id": sid,
        "source_kind": "github_search",
        "repo": repo,
        "repo_url": f"https://github.com/{repo}",
        "commit_sha": "c" * 40,
        "path": path,
        "file_url": f"https://github.com/{repo}/blob/{'c' * 40}/{path}",
        "collected_at": "2026-11-15T00:00:00Z",
        "license_spdx": "MIT",
        "file_type": path.rsplit(".", 1)[-1],
        "sha256": sha,
        "bytes": 100,
        "code_sha256": sha,
        "findings": [],
        "workflow": {"split": True, "fit": True, "cv": False},
        "eligible_for_review": True,
        "cache_key": f"{sha}.ipynb.gz",
        "duplicate_of": None,
        "code_duplicate_of": None,
        "near_of": None,
        "review": {"state": "not_reviewed"},
    }
    rec.update(kw)
    return rec


# --- identity ---------------------------------------------------------------


def test_candidate_id_is_stable_and_content_addressed():
    a = candidate_id("github_search", "o/r", "a.ipynb", "a" * 16)
    assert a == candidate_id("github_search", "o/r", "a.ipynb", "a" * 16)
    assert a.startswith("rwx_") and len(a) == 16
    # any identity component changes the id
    assert a != candidate_id("github_search", "o/r", "b.ipynb", "a" * 16)
    assert a != candidate_id("github_search", "o/s", "a.ipynb", "a" * 16)
    assert a != candidate_id("kaggle", "o/r", "a.ipynb", "a" * 16)
    assert a != candidate_id("github_search", "o/r", "a.ipynb", "b" * 16)


def test_candidate_id_never_collides_on_case_only_paths():
    assert candidate_id("s", "o/r", "A.ipynb", "h") != candidate_id(
        "s", "o/r", "a.ipynb", "h"
    )


# --- licence and redistribution --------------------------------------------


@pytest.mark.parametrize(
    "spdx", ["MIT", "Apache-2.0", "BSD-3-Clause", "BSD-2-Clause", "ISC", "CC-BY-4.0"]
)
def test_permissive_licences_are_redistributable(spdx):
    assert license_status(spdx) == "permissive"
    assert redistributable(spdx) is True


@pytest.mark.parametrize("spdx", ["GPL-3.0", "AGPL-3.0", "LGPL-2.1"])
def test_copyleft_is_known_but_not_redistributable(spdx):
    assert license_status(spdx) == "copyleft"
    assert redistributable(spdx) is False


@pytest.mark.parametrize("spdx", [None, "", "NOASSERTION", "other", "unknown"])
def test_missing_licence_is_unknown_and_not_redistributable(spdx):
    assert license_status(spdx) == "unknown"
    assert redistributable(spdx) is False


# --- notebook code extraction ----------------------------------------------


def _notebook(*cells, execution_count=7):
    return json.dumps(
        {
            "cells": [
                {"cell_type": "markdown", "source": ["# Title\n", "prose\n"]},
                *[
                    {
                        "cell_type": "code",
                        "source": [line + "\n" for line in cell],
                        "outputs": [
                            {"output_type": "stream", "text": ["SECRET OUTPUT\n"]}
                        ],
                        "execution_count": execution_count,
                    }
                    for cell in cells
                ],
            ]
        }
    ).encode()


def test_code_text_keeps_code_and_drops_prose_and_outputs():
    raw = _notebook(["import pandas as pd", "df = pd.read_csv('x.csv')"])
    text = code_text(raw, "ipynb")
    assert "read_csv" in text
    assert "Title" not in text and "prose" not in text
    assert "SECRET OUTPUT" not in text


def test_code_text_is_stable_across_reexecution():
    # re-running the notebook changes bytes (execution_count) but not the code
    first = _notebook(["x = 1"], execution_count=1)
    again = _notebook(["x = 1"], execution_count=99)
    assert content_sha256(first) != content_sha256(again)
    assert code_text(first, "ipynb") == code_text(again, "ipynb") == "x = 1\n"


def test_code_text_passes_through_python_sources():
    assert "fit_transform" in code_text(b"est.fit_transform(X)\n", "py")


def test_code_text_of_unparseable_notebook_is_empty_not_an_error():
    assert code_text(b"{not json", "ipynb") == ""


# --- deduplication ----------------------------------------------------------


def test_exact_duplicate_is_linked_to_one_canonical_id():
    a = _record("kaggle", repo="k/x", sha="a" * 16)
    b = _record("github_search", repo="o/r", sha="a" * 16)
    deduplicate([a, b])
    canon, other = sorted((a, b), key=lambda r: r["id"])
    assert canon["duplicate_of"] is None
    assert other["duplicate_of"] == canon["id"]


def test_reexecuted_notebook_is_a_code_duplicate_not_a_unique_sample():
    a = _record("github_search", sha="a" * 16, code_sha256="c" * 16)
    b = _record("github_search", path="b.ipynb", sha="b" * 16, code_sha256="c" * 16)
    deduplicate([a, b])
    canon, other = sorted((a, b), key=lambda r: r["id"])
    assert other["duplicate_of"] == canon["id"], "same code must collapse"
    assert other["code_duplicate_of"] == canon["id"]


def test_near_duplicates_are_clustered_and_linked():
    words = " ".join(f"token{i}" for i in range(400))
    a = _record("github_search", path="a.ipynb", sha="a" * 16, code_sha256="1" * 16)
    b = _record("github_search", path="b.ipynb", sha="b" * 16, code_sha256="2" * 16)
    a["sketch"], b["sketch"] = list(sketch(words)), list(sketch(words))
    deduplicate([a, b])
    canon, other = sorted((a, b), key=lambda r: r["id"])
    assert other["duplicate_of"] == canon["id"]
    assert a["near_cluster"] and a["near_cluster"] == b["near_cluster"]
    assert other["near_of"] == canon["id"]


def test_near_duplicate_banding_survives_one_extra_sketch_value():
    """A sorted bottom-k sketch shifts every position after one insertion.

    Banding by position then finds no shared band even at 0.97 similarity, so
    ``deduplicate`` bands by value instead. Without that, near-copies land on
    opposite development/holdout sides.
    """
    words = " ".join(f"token{i}" for i in range(400))
    base = sorted(sketch(words))
    # One extra low value pushes the largest out and shifts every later position.
    shifted = sorted([base[0] - 1, *base[:-1]])
    a = _record("s", repo="o/a", sha="a" * 16, code_sha256="1" * 16)
    b = _record("s", repo="o/b", sha="b" * 16, code_sha256="2" * 16)
    a["sketch"], b["sketch"] = base, shifted
    deduplicate([a, b])
    assert a["near_cluster"] and a["near_cluster"] == b["near_cluster"]
    assert {a["duplicate_of"], b["duplicate_of"]} != {None}


def test_no_near_duplicate_straddles_the_development_holdout_split():
    """Repository hashing alone is not isolation: content must agree.

    A near-copy in development that also sits in the holdout would let detector
    development see holdout material. This checks the committed manifest with an
    independent candidate generator (shared sketch values), not the code under
    test.
    """
    records = load_manifest(ROOT / "realworld_expansion" / "manifest.jsonl")
    sides = split_roles(records)
    sketched = [
        rec
        for rec in records
        if len(rec.get("sketch") or ()) >= SKETCH_SIZE
        and sides[rec["id"]]["side"] == "holdout"
    ]
    development = [
        rec
        for rec in records
        if len(rec.get("sketch") or ()) >= SKETCH_SIZE
        and sides[rec["id"]]["side"] == "development"
    ]
    by_value: dict[int, list[int]] = {}
    for index, rec in enumerate(development):
        for value in rec["sketch"]:
            by_value.setdefault(value, []).append(index)

    def jaccard(left, right):
        a, b = set(left), set(right)
        return len(a & b) / len(a | b)

    for hold in sketched:
        candidates = {i for value in hold["sketch"] for i in by_value.get(value, ())}
        for index in candidates:
            dev = development[index]
            assert jaccard(hold["sketch"], dev["sketch"]) < NEAR_THRESHOLD, (
                f"{hold['repo']} ({hold['id']}) is a near-duplicate of "
                f"{dev['repo']} ({dev['id']}) across the split"
            )


def test_unrelated_notebooks_stay_unique():
    a = _record("s", repo="o/a", sha="a" * 16, code_sha256="1" * 16)
    b = _record("s", repo="o/b", sha="b" * 16, code_sha256="2" * 16)
    a["sketch"] = list(sketch(" ".join(f"alpha{i}" for i in range(300))))
    b["sketch"] = list(sketch(" ".join(f"beta{i}" for i in range(300))))
    deduplicate([a, b])
    assert a["duplicate_of"] is None and b["duplicate_of"] is None
    assert "near_cluster" not in a


def test_deduplication_is_order_independent():
    a = _record("s", repo="o/a", sha="a" * 16)
    b = _record("s", repo="o/b", sha="a" * 16)
    deduplicate([a, b])
    first = (a["duplicate_of"], b["duplicate_of"])
    a["duplicate_of"] = b["duplicate_of"] = None
    deduplicate([b, a])
    assert (a["duplicate_of"], b["duplicate_of"]) == first


def test_short_notebook_sketch_does_not_break_near_duplicate_detection():
    a = _record("s", sha="a" * 16, code_sha256="1" * 16)
    b = _record("s", path="b.ipynb", sha="b" * 16, code_sha256="2" * 16)
    a["sketch"], b["sketch"] = list(sketch("x = 1")), list(sketch("y = 2"))
    deduplicate([a, b])
    assert a["duplicate_of"] is None and b["duplicate_of"] is None


# --- review queue -----------------------------------------------------------


def _queue_records(n=6, per_repo=2):
    return [
        _record(
            "github_search",
            repo=f"o/r{i // per_repo}",
            path=f"n{i}.ipynb",
            sha=f"{i:016d}",
            code_sha256=f"{i:016d}",
        )
        for i in range(n)
    ]


def test_queue_is_deterministic_and_respects_limits():
    recs = _queue_records(8)  # 4 repositories, 2 notebooks each
    first = queue(recs, limit=4, per_repo_cap=1)
    assert first == queue(recs, limit=4, per_repo_cap=1)
    assert len(first) == 4
    # one per repository, so the cap is what limits the queue
    repos = {r["repo"] for r in recs if r["id"] in first}
    assert len(repos) == 4


def test_queue_skips_duplicates_ineligible_and_already_reviewed():
    recs = _queue_records(4)
    recs[0]["duplicate_of"] = recs[1]["id"]
    recs[1]["eligible_for_review"] = False
    recs[2]["review"] = {"state": "legitimate"}
    assert queue(recs, limit=10) == [recs[3]["id"]]


def test_queue_excludes_samples_already_in_the_frozen_corpus():
    recs = _queue_records(2)
    recs[0]["overlaps_frozen"] = True
    assert queue(recs, limit=10) == [recs[1]["id"]]


def test_queue_prefers_flagged_then_workflow_then_diversity():
    quiet = _record("github_search", repo="o/a", sha="a" * 16, code_sha256="a" * 16)
    quiet["workflow"] = {"split": False, "fit": False, "cv": False}
    flagged = _record("github_search", repo="o/b", sha="b" * 16, code_sha256="b" * 16)
    flagged["findings"] = [
        {"severity": "high", "kind": "target_leakage", "column": "x", "detail": ""}
    ]
    chosen = queue([quiet, flagged], limit=1)
    assert chosen == [flagged["id"]]


def test_queue_never_assigns_review_state():
    recs = _queue_records(3)
    for rec in recs:
        rec["findings"] = [{"severity": "high", "kind": "target_leakage"}]
    queue(recs, limit=3)
    assert {r["review"]["state"] for r in recs} == {"not_reviewed"}


def test_queue_limit_is_honoured_when_supply_is_short():
    assert len(queue(_queue_records(2), limit=300)) == 2


# --- review workflow --------------------------------------------------------


def test_review_states_are_the_documented_five():
    assert REVIEW_STATES == (
        "not_reviewed",
        "confirmed_leakage",
        "legitimate",
        "potential",
        "uncertain",
    )
    assert SCORED_STATES == ("confirmed_leakage", "legitimate")


def test_review_rejects_invalid_state():
    rec = _record("s")
    with pytest.raises(ValueError):
        review(rec, state="probably_leaky", evidence="x", explanation="y")


def test_review_requires_evidence_and_explanation():
    rec = _record("s")
    with pytest.raises(ValueError):
        review(rec, state="confirmed_leakage", evidence="", explanation="")
    rec2 = _record("s")
    with pytest.raises(ValueError):
        review(
            rec2, state="confirmed_leakage", evidence="leakage.py:12", explanation=""
        )


def test_review_records_full_provenance():
    rec = _record("s")
    out = review(
        rec,
        state="confirmed_leakage",
        evidence="notebook[cell 3]:2 scaler fitted before split",
        mechanism="preprocessing_scaling",
        explanation="StandardScaler.fit_transform on the full frame precedes the split",
        confidence=0.9,
        reviewer="reviewer-1",
        when="2026-11-15T12:00:00Z",
    )
    assert out["review"] == {
        "state": "confirmed_leakage",
        "evidence": "notebook[cell 3]:2 scaler fitted before split",
        "mechanism": "preprocessing_scaling",
        "explanation": "StandardScaler.fit_transform on the full frame precedes the split",
        "confidence": 0.9,
        "reviewer": "reviewer-1",
        "decided_at": "2026-11-15T12:00:00Z",
    }
    assert out["scored"] is True


def test_potential_and_uncertain_are_never_scored():
    for state in ("potential", "uncertain", "not_reviewed"):
        out = review(
            _record("s"), state=state, evidence="e", explanation="x", mechanism=None
        )
        assert out["scored"] is False


# --- provenance validation --------------------------------------------------


def test_validate_accepts_a_complete_record():
    assert validate([_record("s")]) == []


@pytest.mark.parametrize(
    "field", ["source_id", "repo", "path", "file_url", "sha256", "collected_at"]
)
def test_validate_names_each_missing_provenance_field(field):
    rec = _record("s")
    rec[field] = ""
    errors = validate([rec])
    assert errors and field in errors[0]


def test_validate_rejects_github_record_without_commit_sha():
    rec = _record("s")  # github by default
    rec["commit_sha"] = ""
    assert any("commit_sha" in e for e in validate([rec]))


def test_validate_rejects_unknown_license_without_flag():
    rec = _record("s", license_spdx=None)
    assert any("license" in e.lower() for e in validate([rec]))


# --- manifest and statistics ------------------------------------------------


def test_manifest_roundtrip_is_stable(tmp_path):
    recs = _queue_records(3)
    path = tmp_path / "manifest.jsonl"
    write_manifest(path, recs)
    assert load_manifest(path) == recs
    write_manifest(path, recs)
    assert load_manifest(path) == recs
    assert path.read_text(encoding="utf-8").count("\n") == 3


def test_summary_counts_unique_samples_and_repositories():
    a = _record("github_search", repo="o/a", sha="a" * 16, code_sha256="a" * 16)
    mirror = _record(
        "kaggle", repo="k/x", sha="a" * 16, code_sha256="a" * 16, source_kind="kaggle"
    )
    b = _record("github_scan", repo="o/b", sha="b" * 16, code_sha256="b" * 16)
    b["license_spdx"] = None
    recs = [a, mirror, b]
    deduplicate(recs)
    s = summarize(recs)
    assert s["candidates"] == 3
    assert s["unique_samples"] == 2
    assert s["repositories"] == 3
    # kernels and repositories are never added together
    assert s["github_repositories"] == 2
    assert s["kaggle_kernels"] == 1
    assert s["sources"] == {"github_scan": 1, "github_search": 1, "kaggle": 1}
    assert s["licenses"]["permissive"] == 2
    assert s["licenses"]["unknown"] == 1
    assert s["redistributable"] == 2
    assert s["duplicates"] == 1


def test_summary_never_reports_unreviewed_as_confirmed():
    recs = _queue_records(3)
    for rec in recs:
        rec["findings"] = [{"severity": "high", "kind": "target_leakage"}]
    s = summarize(recs)
    assert s["reviews"]["confirmed_leakage"] == 0
    assert s["reviews"]["not_reviewed"] == 3


# --- development / holdout separation --------------------------------------


def _repo_on_side(side, prefix="o"):
    for number in range(200):
        repo = f"{prefix}/side{number}"
        if split_role({"repo": repo}) == side:
            return repo
    raise AssertionError(f"no repository hashed to {side}")


def _side_records():
    return [
        _record(
            "github_search",
            repo=f"o/r{i // 2}",
            path=f"n{i}.ipynb",
            sha=f"{i:016d}",
            code_sha256=f"{i:016d}",
        )
        for i in range(8)
    ]


def test_split_role_is_stable_and_reads_only_the_repository():
    rec = _record("github_search", repo="o/a")
    assert split_role(rec) == split_role(rec)
    assert split_role(rec) == split_role({"repo": "o/a", "id": "other"})
    assert split_role({"repo": ""}) in ("development", "holdout")
    first = split_roles(_side_records())
    second = split_roles(_side_records())
    assert first == second
    for rec in _side_records():
        assert first[rec["id"]]["side"] == split_role(rec)
        assert first[rec["id"]]["reason"] == "repo-hash"


@pytest.mark.parametrize("field", ["duplicate_of", "code_duplicate_of", "near_of"])
def test_duplicate_and_near_duplicate_pairs_stay_on_one_side(field):
    development = _repo_on_side("development")
    holdout = _repo_on_side("holdout")
    first = _record("github_search", repo=development, sha="a" * 16)
    second = _record("github_search", repo=holdout, path="b.ipynb", sha="b" * 16)
    second[field] = first["id"]
    sides = split_roles([first, second])
    # the two repositories hash to opposite sides, so the link is what binds them
    assert split_role(first) != split_role(second)
    assert sides[first["id"]] == sides[second["id"]]
    assert sides[first["id"]] == {"side": "development", "reason": "cluster-straddle"}


def test_adding_an_unrelated_repository_never_flips_an_existing_side():
    recs = _side_records()
    before = split_roles(recs)
    added = _record("github_search", repo="o/new", path="new.ipynb", sha="f" * 16)
    after = split_roles([*recs, added])
    for rec in recs:
        assert after[rec["id"]] == before[rec["id"]]
    assert after[added["id"]]["side"] == split_role(added)


def test_stats_split_counts_sum_to_unique_samples():
    recs = _side_records()
    recs[0]["duplicate_of"] = recs[1]["id"]
    summary = summarize(recs)
    assert summary["unique_samples"] == len(recs) - 1
    assert sum(summary["splits"].values()) == summary["unique_samples"]


def test_queue_can_be_restricted_to_one_side():
    development = _repo_on_side("development")
    holdout = _repo_on_side("holdout")
    recs = [
        _record("github_search", repo=development),
        _record("github_search", repo=holdout, path="b.ipynb", sha="b" * 16),
    ]
    sides = split_roles(recs)
    assert {sides[rec["id"]]["side"] for rec in recs} == {"development", "holdout"}
    for side in ("development", "holdout"):
        chosen = queue(recs, limit=50, side=side)
        assert len(chosen) == 1
        assert sides[chosen[0]]["side"] == side
        assert chosen == queue(recs, limit=50, side=side)


# --- label-leakage guard ---------------------------------------------------


def test_detectors_never_import_collection_metadata():
    """Collection/label metadata must not become a detector input."""
    import pathlib

    root = pathlib.Path("leaklens")
    offenders = []
    for path in sorted(root.rglob("*.py")):
        if path.name == "corpus_expansion.py":
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        if "corpus_expansion" in text or "realworld_expansion" in text:
            offenders.append(str(path))
    assert offenders == []


# --- frozen case-set snapshots ---------------------------------------------


def _freeze_corpus():
    return [
        _record("s1", repo=_repo_on_side("development", "freeze-dev"), sha="1" * 64),
        _record("s2", repo=_repo_on_side("holdout", "freeze-hold"), sha="2" * 64),
    ]


def test_freeze_is_deterministic_and_records_no_verdict():
    records = _freeze_corpus()
    stamp = {"version": "0.14.0", "commit": "abc", "split_rule": "repo-hash+cluster"}
    first = freeze(records, stamp)
    second = freeze(list(reversed(records)), dict(stamp, version="9.9.9"))
    assert first["digest"] == second["digest"]
    assert first["cases"] == second["cases"]
    assert set(first["cases"][0]) == {"id", "repo", "path", "sha256", "side", "reason"}
    assert verify_freeze(records, first) == []


def test_verify_freeze_reports_added_removed_changed_and_moved_cases():
    records = _freeze_corpus()
    frozen = freeze(records, {"version": "0.14.0", "commit": "abc"})
    assert verify_freeze(records, frozen) == []

    missing = verify_freeze(records[1:], frozen)
    assert any("missing from the manifest" in problem for problem in missing)

    extra = verify_freeze([*records, _record("s3", repo="new/repo")], frozen)
    assert any("not in the freeze" in problem for problem in extra)

    moved = [dict(rec) for rec in records]
    moved[0]["repo"] = moved[1]["repo"]
    drift = verify_freeze(moved, frozen)
    assert "side moved development -> holdout" in " ".join(drift)

    tampered = json.loads(json.dumps(frozen))
    tampered["cases"][0]["sha256"] = "f" * 64
    assert "the digest does not match the frozen cases" in verify_freeze(
        records, tampered
    )


def test_freeze_refuses_duplicate_record_ids():
    records = _freeze_corpus()
    with pytest.raises(ValueError, match="duplicate"):
        freeze([*records, records[0]], {"version": "0.14.0", "commit": "abc"})


def test_verify_freeze_rejects_malformed_snapshots_without_raising():
    records = _freeze_corpus()
    frozen = freeze(records, {"version": "0.14.0", "commit": "abc"})
    malformed = (
        [],
        {"schema": FREEZE_SCHEMA},
        {"schema": FREEZE_SCHEMA, "cases": {}},
        {"schema": FREEZE_SCHEMA, "cases": ["not an object"]},
        {"schema": FREEZE_SCHEMA, "cases": [{"id": "only-an-id"}]},
        {
            "schema": FREEZE_SCHEMA,
            "cases": [
                {
                    "id": records[0]["id"],
                    "repo": records[0]["repo"],
                    "path": records[0]["path"],
                    "sha256": records[0]["sha256"],
                    "side": "bogus",
                    "reason": "repo-hash",
                }
            ],
        },
    )
    for broken in malformed:
        problems = verify_freeze(records, broken)
        assert problems, broken
    assert verify_freeze(records, "not a mapping") == ["the freeze is not an object"]
    assert verify_freeze(records, frozen) == []


def test_verify_freeze_rejects_a_duplicate_case_id():
    records = _freeze_corpus()
    frozen = freeze(records, {"version": "0.14.0", "commit": "abc"})
    duplicated = dict(frozen, cases=[frozen["cases"][0], frozen["cases"][0]])
    payload = json.dumps(duplicated["cases"], sort_keys=True, separators=(",", ":"))
    duplicated["digest"] = hashlib.sha256(payload.encode()).hexdigest()
    problems = verify_freeze(records, duplicated)
    assert any("duplicated in the freeze" in problem for problem in problems)


def test_verify_freeze_reports_a_changed_side_reason():
    records = _freeze_corpus()
    frozen = freeze(records, {"version": "0.14.0", "commit": "abc"})
    tampered = json.loads(json.dumps(frozen))
    tampered["cases"][0]["reason"] = "cluster-straddle"
    payload = json.dumps(tampered["cases"], sort_keys=True, separators=(",", ":"))
    tampered["digest"] = hashlib.sha256(payload.encode()).hexdigest()
    assert any(
        "side reason changed" in problem for problem in verify_freeze(records, tampered)
    )


def test_freeze_label_cannot_escape_the_snapshot_directory():
    for label in ("", "../escape", "a/b", "a\\b", ".hidden", "a b"):
        with pytest.raises(ValueError):
            freeze_filename(label)
    assert freeze_filename("2026-11-26") == "2026-11-26.json"
    assert freeze_filename("dev_2.1") == "dev_2.1.json"
