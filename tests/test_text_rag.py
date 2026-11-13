"""Static RAG wiring analysis: shared index roots and clean layouts."""

from __future__ import annotations

import textwrap

from leaklens.analysis.text.rag import scan_rag
from leaklens.leakage import score

SHARED_INDEX = """
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

def build(embeddings):
    loader = DirectoryLoader("data")
    docs = loader.load()
    return FAISS.from_documents(docs, embeddings)
"""

CLEAN_INDEX = """
from langchain_community.document_loaders import DirectoryLoader
from langchain_community.vectorstores import FAISS

def build(embeddings):
    loader = DirectoryLoader("data/train")
    docs = loader.load()
    return FAISS.from_documents(docs, embeddings)
"""

NO_INDEX = """
from langchain_community.document_loaders import DirectoryLoader

def load():
    return DirectoryLoader("data").load()
"""


def _repo(tmp_path, source=SHARED_INDEX, name="rag/index.py"):
    path = tmp_path / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(textwrap.dedent(source), encoding="utf-8")
    (tmp_path / "data" / "eval").mkdir(parents=True, exist_ok=True)
    (tmp_path / "data" / "train").mkdir(parents=True, exist_ok=True)
    return tmp_path


def test_a_shared_loader_root_containing_the_eval_path_is_high(tmp_path):
    root = _repo(tmp_path)
    findings, hits = scan_rag(root, ["data/eval"])
    assert len(findings) == 1
    finding = findings[0]
    assert finding.kind == "rag_index_leakage"
    assert finding.severity == "high"
    assert finding.diagnostic is True
    assert finding.column == "rag/index.py"
    assert score(findings) == 100
    assert [hit.path for hit in hits] == ["data"]
    blob = " ".join(
        str(value) for item in finding.evidence for value in item.to_dict().values()
    )
    assert "data/eval" in blob
    assert "FAISS" in blob or "from_documents" in blob
    assert 'DirectoryLoader("data")' not in blob


def test_a_loader_root_that_excludes_eval_is_clean(tmp_path):
    root = _repo(tmp_path, source=CLEAN_INDEX)
    findings, hits = scan_rag(root, ["data/eval"])
    assert findings == [] and hits == []


def test_a_loader_without_an_index_is_clean(tmp_path):
    root = _repo(tmp_path, source=NO_INDEX)
    assert scan_rag(root, ["data/eval"])[0] == []


def test_malformed_python_is_skipped_not_raised(tmp_path):
    root = _repo(tmp_path, source="def broken(:\n", name="rag/broken.py")
    findings, hits = scan_rag(root, ["data/eval"])
    assert findings == [] and hits == []


def test_eval_paths_with_traversal_never_match(tmp_path):
    root = _repo(tmp_path)
    assert scan_rag(root, ["../eval"])[0] == []
    assert scan_rag(root, ["data/../../data/eval"])[0] == []


def test_findings_are_sorted_and_stable(tmp_path):
    root = _repo(tmp_path)
    _repo(root, name="rag/other.py")
    findings, _ = scan_rag(root, ["data/eval"])
    assert [finding.column for finding in findings] == ["rag/index.py", "rag/other.py"]
    again, _ = scan_rag(root, ["data/eval"])
    assert [finding.fingerprint for finding in findings] == [
        finding.fingerprint for finding in again
    ]


def test_file_cap_and_non_python_files(tmp_path):
    root = _repo(tmp_path)
    (root / "rag" / "notes.txt").write_text(SHARED_INDEX, encoding="utf-8")
    findings, _ = scan_rag(root, ["data/eval"])
    assert [finding.column for finding in findings] == ["rag/index.py"]
    assert scan_rag(root, ["data/eval"], max_files=0)[0] == []
